"""
CyberGAN — Blue Agent: PPO Actor-Critic Policy Network
Symmetric architecture to Red, but with 2-dim action: (action_type, target_node).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from agents.blue.observer import flatten_blue_obs, get_obs_dim
from agents.blue.actions import NUM_BLUE_ACTIONS


class BluePolicy(nn.Module):
    """
    Actor-Critic network for the Blue (defender) agent.

    Action is MultiDiscrete: [action_type, target_node]
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

        # Actor heads
        self.action_type_head = nn.Linear(hidden_dim, NUM_BLUE_ACTIONS)
        self.target_node_head = nn.Linear(hidden_dim, num_nodes)

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
        features = self.trunk(obs)

        at_logits = self.action_type_head(features)
        tn_logits = self.target_node_head(features)

        if action_mask is not None:
            if "action_type" in action_mask:
                mask = torch.clamp(action_mask["action_type"], min=1e-8)
                at_logits = at_logits + mask.log()
            if "target_node" in action_mask:
                mask = torch.clamp(action_mask["target_node"], min=1e-8)
                tn_logits = tn_logits + mask.log()

        dist_at = Categorical(logits=at_logits)
        dist_tn = Categorical(logits=tn_logits)

        value = self.value_head(features)
        return [dist_at, dist_tn], value

    def get_action(
        self,
        obs_np: dict[str, np.ndarray],
        action_mask: dict[str, np.ndarray] | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        obs_flat = flatten_blue_obs(obs_np)
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
        dists, values = self.forward(obs, action_mask)

        log_probs = torch.zeros(obs.shape[0], device=obs.device)
        entropy = torch.zeros(obs.shape[0], device=obs.device)

        for i, dist in enumerate(dists):
            a = actions[:, i].long()
            log_probs += dist.log_prob(a)
            entropy += dist.entropy()

        return log_probs, values.squeeze(-1), entropy
