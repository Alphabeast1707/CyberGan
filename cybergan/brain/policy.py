"""
CyberGAN — Brain: Production Policy Network
Loads the trained Blue Agent policy and runs inference for real-time
defense decisions. Falls back to heuristic rules if no model is available.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import structlog

from cybergan.config import BrainConfig
from cybergan.brain.action_space import DEFENSE_ACTIONS, DefenseAction

logger = structlog.get_logger(__name__)

NUM_DEFENSE_ACTIONS = len(DEFENSE_ACTIONS)


class _ArenaBluePolicyAdapter:
    """
    Wraps the arena-trained BluePolicy for production use.

    Handles the obs_dim mismatch between the arena environment
    (which encodes the full network graph) and the production agent
    (which encodes real security telemetry). Uses zero-padding / truncation
    to bridge the two spaces.

    Action mapping: arena has 2 actions (patch/isolate), mapped to
    production defense actions via risk-weighted selection.
    """

    def __init__(self, arena_policy, arena_obs_dim: int, prod_obs_dim: int):
        self.arena_policy = arena_policy
        self.arena_obs_dim = arena_obs_dim
        self.prod_obs_dim = prod_obs_dim
        self._device = next(arena_policy.parameters()).device

        # Map arena action indices → production defense action indices
        # Arena: 0=patch/defend, 1=isolate
        # Production: we spread across multiple defensive options
        self._arena_to_prod = {
            0: [0, 1, 2, 3],   # defend  → monitor, alert, rate_limit, block_ip
            1: [4, 5, 6],      # isolate → firewall_block, kill_process, isolate_service
        }

    def __call__(self, obs_t: "torch.Tensor", mask_t=None):
        """Forward pass with obs dimension adaptation."""
        import torch
        batch = obs_t.shape[0]
        arena_dim = self.arena_obs_dim

        # Pad or truncate production obs to arena obs_dim
        if self.prod_obs_dim < arena_dim:
            pad = torch.zeros(batch, arena_dim - self.prod_obs_dim,
                              device=self.arena_policy.trunk[0].weight.device)
            adapted = torch.cat([obs_t.to(self._device), pad], dim=1)
        else:
            adapted = obs_t[:, :arena_dim].to(self._device)

        with torch.no_grad():
            dists, value = self.arena_policy.forward(adapted)
            # dists is a list of distributions, first is action_type
            arena_dist = dists[0] if isinstance(dists, (list, tuple)) else dists
            arena_probs = arena_dist.probs  # shape (batch, 2)

            # Expand from 2 arena actions to NUM_DEFENSE_ACTIONS production actions
            prod_probs = torch.zeros(batch, NUM_DEFENSE_ACTIONS, device=self._device)
            for arena_idx, prod_indices in self._arena_to_prod.items():
                if arena_idx < arena_probs.shape[1]:
                    # Distribute this arena action's probability across mapped prod actions
                    share = arena_probs[:, arena_idx:arena_idx+1] / len(prod_indices)
                    for pi in prod_indices:
                        if pi < NUM_DEFENSE_ACTIONS:
                            prod_probs[:, pi] += share.squeeze(1)

            from torch.distributions import Categorical
            prod_dist = Categorical(probs=prod_probs.clamp(min=1e-8))
            return prod_dist, value

    def to(self, device):
        self.arena_policy.to(device)
        self._device = device
        return self

    def eval(self):
        self.arena_policy.eval()
        return self



class ProductionPolicy(nn.Module):
    """
    Neural network policy for production defense decisions.

    Takes a security state observation vector and outputs a probability
    distribution over defensive actions. Uses the same architecture as
    the arena-trained Blue Agent but with the production action space.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Actor: outputs action probabilities
        self.action_head = nn.Linear(hidden_dim // 2, NUM_DEFENSE_ACTIONS)

        # Critic: outputs state value estimate
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Categorical, torch.Tensor]:
        """Forward pass returning action distribution and value estimate."""
        features = self.trunk(obs)
        logits = self.action_head(features)

        if action_mask is not None:
            mask = torch.clamp(action_mask, min=1e-8)
            logits = logits + mask.log()

        dist = Categorical(logits=logits)
        value = self.value_head(features)
        return dist, value


class Brain:
    """
    RL decision engine for production defense.

    Wraps the policy network for inference and provides fallback
    heuristic rules when no trained model is available.
    """

    def __init__(self, config: BrainConfig, obs_dim: int):
        self.config = config
        self.obs_dim = obs_dim
        self.policy: Optional[ProductionPolicy] = None
        self.model_loaded = False

        # Initialize device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        # Try to load model
        self._load_model()

    def _load_model(self):
        """
        Load the trained policy model from checkpoint.

        Tries two architectures in order:
        1. Arena BluePolicy (from training) — preferred, uses trained weights directly
        2. ProductionPolicy (old format) — legacy fallback

        This bridges the gap between the arena training output and the
        production agent without requiring a separate export step.
        """
        if not os.path.exists(self.config.model_path):
            logger.warning("brain.no_model", path=self.config.model_path)
            if self.config.fallback_to_heuristic:
                logger.info("brain.fallback_to_heuristic")
            return

        try:
            state_dict = torch.load(
                self.config.model_path,
                map_location=self.device,
                weights_only=False,
            )

            # ── Try 1: Arena BluePolicy ─────────────────────────────
            # The arena BluePolicy takes (obs_dim_arena, num_nodes, num_vulns)
            # We can run it in production with a zero-padded/truncated obs.
            try:
                import sys, os as _os
                _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                from agents.blue.policy import BluePolicy as ArenaBluePolicy

                # Infer arena obs_dim from checkpoint's first linear layer
                w = state_dict.get("trunk.0.weight")
                if w is not None:
                    arena_obs_dim = w.shape[1]

                    # Instantiate with matching obs_dim.
                    # BluePolicy(num_nodes, num_vulns) where obs_dim depends on these.
                    # We'll search for (num_nodes, num_vulns) where get_obs_dim matches.
                    from agents.blue.observer import get_obs_dim as _get_obs_dim
                    _dummy = None
                    for nn_ in range(3, 20):
                        for nv in range(1, 10):
                            if _get_obs_dim(nn_, nv, 7) == arena_obs_dim:
                                _dummy = ArenaBluePolicy(nn_, nv)
                                break
                        if _dummy is not None:
                            break

                    if _dummy is None:
                        # Fallback: use N=5, V=3 and patch the first layer
                        _dummy = ArenaBluePolicy(5, 3)
                        _dummy.trunk[0] = nn.Linear(arena_obs_dim, 256)

                    _dummy.load_state_dict(state_dict, strict=False)
                    _dummy.to(self.device)
                    _dummy.eval()

                    self.policy = _ArenaBluePolicyAdapter(_dummy, arena_obs_dim, self.obs_dim)
                    self.model_loaded = True
                    logger.info(
                        "brain.model_loaded",
                        path=self.config.model_path,
                        architecture="arena_blue_policy",
                        arena_obs_dim=arena_obs_dim,
                        prod_obs_dim=self.obs_dim,
                    )
                    return

            except Exception as _e:
                logger.debug("brain.arena_policy_load_failed", reason=str(_e))

            # ── Try 2: ProductionPolicy (same obs_dim) ───────────────
            self.policy = ProductionPolicy(self.obs_dim)
            self.policy.load_state_dict(state_dict)
            self.policy.to(self.device)
            self.policy.eval()
            self.model_loaded = True
            logger.info("brain.model_loaded", path=self.config.model_path,
                        architecture="production_policy")

        except Exception as e:
            logger.error("brain.model_load_error", error=str(e))
            self.policy = None
            self.model_loaded = False

    def decide(
        self,
        observation: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        risk_score: float = 0.0,
        event_type: str = "",
    ) -> tuple[DefenseAction, float, dict]:
        """
        Make a defense decision given the current security state.

        Args:
            observation: Feature vector from the FeatureExtractor
            action_mask: Binary mask of valid actions
            risk_score: Current risk score (0-100)
            event_type: Type of the triggering event

        Returns:
            (action, confidence, metadata)
        """
        if self.model_loaded and self.policy is not None:
            return self._decide_rl(observation, action_mask)
        else:
            return self._decide_heuristic(risk_score, event_type)

    def _decide_rl(
        self,
        observation: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> tuple[DefenseAction, float, dict]:
        """Make decision using the RL policy."""
        obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        mask_t = None
        if action_mask is not None:
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, value = self.policy(obs_t, mask_t)

            if self.config.action_selection == "deterministic":
                action_idx = dist.probs.argmax(dim=-1).item()
            else:
                # Temperature-scaled sampling
                logits = dist.logits / max(self.config.temperature, 0.01)
                tempered_dist = Categorical(logits=logits)
                action_idx = tempered_dist.sample().item()

            confidence = dist.probs[0, action_idx].item()

        action = DEFENSE_ACTIONS[action_idx]
        metadata = {
            "method": "rl",
            "value_estimate": value.item(),
            "action_probs": dist.probs[0].cpu().numpy().tolist(),
            "temperature": self.config.temperature,
        }

        return action, confidence, metadata

    def _decide_heuristic(
        self,
        risk_score: float,
        event_type: str,
    ) -> tuple[DefenseAction, float, dict]:
        """
        Fallback heuristic decision rules.
        Used when no trained model is available.
        """
        # Map event types to recommended actions
        heuristic_map = {
            # Web attacks → block IP + alert
            "sql_injection": "block_ip",
            "xss": "block_ip",
            "command_injection": "block_ip",
            "rce": "block_ip",
            "lfi": "block_ip",
            "rfi": "block_ip",
            "xxe": "block_ip",
            "ssrf": "block_ip",
            "directory_traversal": "block_ip",
            "web_shell": "block_ip",
            "insecure_deserialization": "block_ip",

            # Auth attacks → rate limit + block
            "brute_force_detected": "block_ip",
            "brute_force": "rate_limit",
            "session_hijack": "block_ip",
            "cookie_poisoning": "block_ip",

            # Network attacks → firewall
            "port_scan": "rate_limit",
            "syn_flood": "firewall_block",
            "udp_flood": "firewall_block",
            "ddos_indicator": "firewall_block",
            "connection_spike": "rate_limit",

            # Process attacks → kill + alert
            "cryptominer": "kill_process",
            "reverse_shell": "kill_process",
            "suspicious_process": "alert",

            # Persistence → alert (needs manual review)
            "ssh_key_injection": "alert",
            "crontab_change": "alert",
            "backdoor": "alert",
            "suid_set": "alert",

            # Data → alert
            "data_exfiltration": "block_ip",
            "log_tampering": "alert",
            "ransomware": "isolate_service",
        }

        action_name = heuristic_map.get(event_type, "alert" if risk_score > 50 else "monitor")

        # Find matching action
        action = None
        for a in DEFENSE_ACTIONS:
            if a.name == action_name:
                action = a
                break

        if action is None:
            action = DEFENSE_ACTIONS[0]  # Default to monitor

        confidence = min(risk_score / 100.0, 0.95)
        metadata = {
            "method": "heuristic",
            "rule": f"{event_type} → {action_name}",
            "risk_score": risk_score,
        }

        return action, confidence, metadata

    def update_model(self, new_model_path: str):
        """Hot-reload a new model checkpoint."""
        old_path = self.config.model_path
        self.config.model_path = new_model_path
        self._load_model()
        if not self.model_loaded:
            self.config.model_path = old_path
            logger.warning("brain.update_failed", path=new_model_path)
