"""
CyberGAN — Red Agent: PPO Actor-Critic Policy Network
A neural network that outputs a probability distribution over actions
and a value estimate for the current state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from agents.red.observer import flatten_red_obs, get_obs_dim
from agents.red.actions import NUM_RED_ACTIONS


class RedPolicy(nn.Module):
    """
    Actor-Critic network for the Red (attacker) agent.

    Architecture:
      Shared trunk → Actor head (per action dimension) + Critic head

    The action is MultiDiscrete: [action_type, target_node, target_vuln]
    so we output separate categorical distributions for each.
    """

    def __init__(
        self,
        num_nodes: int,
        num_vulns: int,
        hidden_dim: int = 256,
        num_opponent_actions: int = 7,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vulns = max(num_vulns, 1)
        self.obs_dim = get_obs_dim(num_nodes, self.num_vulns, num_opponent_actions)

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor heads (one per action dimension)
        self.action_type_head = nn.Linear(hidden_dim, NUM_RED_ACTIONS)
        self.target_node_head = nn.Linear(hidden_dim, num_nodes)
        self.target_vuln_head = nn.Linear(hidden_dim, self.num_vulns)

        # Critic head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[list[Categorical], torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Flattened observation tensor [batch, obs_dim]
            action_mask: Optional dict of masks for each action dimension

        Returns:
            distributions: List of Categorical distributions [action_type, target_node, target_vuln]
            value: Value estimate [batch, 1]
        """
        features = self.trunk(obs)

        # Action type logits
        at_logits = self.action_type_head(features)
        tn_logits = self.target_node_head(features)
        tv_logits = self.target_vuln_head(features)

        # Apply action masks (set invalid actions to -inf)
        if action_mask is not None:
            if "action_type" in action_mask:
                at_logits = at_logits + (action_mask["action_type"].log() + 1e-8)
            if "target_node" in action_mask:
                mask = action_mask["target_node"]
                mask = torch.clamp(mask, min=1e-8)
                tn_logits = tn_logits + mask.log()
            if "target_vuln" in action_mask:
                mask = action_mask["target_vuln"]
                mask = torch.clamp(mask, min=1e-8)
                tv_logits = tv_logits + mask.log()

        dist_at = Categorical(logits=at_logits)
        dist_tn = Categorical(logits=tn_logits)
        dist_tv = Categorical(logits=tv_logits)

        value = self.value_head(features)

        return [dist_at, dist_tn, dist_tv], value

    def get_action(
        self,
        obs_np: dict[str, np.ndarray],
        action_mask: dict[str, np.ndarray] | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """
        Select an action given a raw observation dict.

        Returns:
            action: numpy array [action_type, target_node, target_vuln]
            log_prob: sum of log probs across dimensions
            value: critic value estimate
        """
        obs_flat = flatten_red_obs(obs_np)
        obs_t = torch.FloatTensor(obs_flat).unsqueeze(0)

        mask_t = None
        if action_mask is not None:
            mask_t = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in action_mask.items()}

        with torch.no_grad():
            dists, value = self.forward(obs_t, mask_t)

        if deterministic:
            actions = [d.probs.argmax(dim=-1) for d in dists]
        else:
            actions = [d.sample() for d in dists]

        log_probs = sum(d.log_prob(a) for d, a in zip(dists, actions))

        action_np = np.array([a.item() for a in actions], dtype=np.int64)
        return action_np, log_probs.item(), value.item()

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log_probs, value, and entropy for stored (obs, action) pairs.
        Used during PPO update.

        Args:
            obs: [batch, obs_dim]
            actions: [batch, 3] — (action_type, target_node, target_vuln)

        Returns:
            log_probs: [batch]
            values: [batch]
            entropy: [batch]
        """
        dists, values = self.forward(obs, action_mask)

        log_probs = torch.zeros(obs.shape[0], device=obs.device)
        entropy = torch.zeros(obs.shape[0], device=obs.device)

        for i, dist in enumerate(dists):
            a = actions[:, i].long()
            log_probs += dist.log_prob(a)
            entropy += dist.entropy()

        return log_probs, values.squeeze(-1), entropy
