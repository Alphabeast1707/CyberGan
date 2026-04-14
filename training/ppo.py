"""
CyberGAN — Training: PPO Algorithm
Proximal Policy Optimization for both Red and Blue agents.
Shared algorithm with agent-specific policy networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from training.buffer import RolloutBuffer


class PPO:
    """
    Proximal Policy Optimization (clip variant).

    Used to train both Red and Blue policy networks.
    Each agent gets its own PPO instance with its own optimizer.
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.policy.to(device)

        # Training stats
        self._stats: dict = {}

    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Run PPO optimization on collected rollout data.

        Args:
            buffer: RolloutBuffer with computed GAE advantages

        Returns:
            Training statistics dict
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        # Normalize advantages
        advantages = buffer.advantages[:buffer.size]
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        buffer.advantages[:buffer.size] = (advantages - adv_mean) / adv_std

        for _ in range(self.ppo_epochs):
            for batch in buffer.sample_batches(self.batch_size):
                obs = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages_batch = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                # Evaluate current policy
                new_log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Check for NaN
                if torch.isnan(loss):
                    continue

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Stats
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    total_approx_kl += approx_kl
                    num_updates += 1

        num_updates = max(num_updates, 1)
        self._stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "num_updates": num_updates,
        }
        return self._stats

    def get_stats(self) -> dict:
        return self._stats
