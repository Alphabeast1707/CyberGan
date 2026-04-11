"""
CyberGAN — Training: Rollout Buffer
Stores experience tuples from self-play episodes for PPO training.
Supports GAE (Generalized Advantage Estimation) computation.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Generator


class RolloutBuffer:
    """
    Stores transitions (obs, action, reward, value, log_prob, done)
    collected during self-play rollouts.

    Computes GAE-λ advantages and returns for PPO updates.
    """

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a single transition to the buffer."""
        if self.pos >= self.buffer_size:
            # Wrap around
            self.pos = 0
            self.full = True

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = float(done)
        self.pos += 1

    def compute_gae(self, last_value: float = 0.0):
        """
        Compute Generalized Advantage Estimation.
        Must be called after collecting a full rollout, before sampling.
        """
        size = self.pos if not self.full else self.buffer_size
        last_gae = 0.0

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae

        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def sample_batches(self, batch_size: int) -> Generator[dict[str, torch.Tensor], None, None]:
        """
        Yield random minibatches for PPO optimization.
        """
        size = self.pos if not self.full else self.buffer_size
        indices = np.random.permutation(size)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_idx = indices[start:end]

            yield {
                "observations": torch.FloatTensor(self.observations[batch_idx]),
                "actions": torch.LongTensor(self.actions[batch_idx]),
                "old_log_probs": torch.FloatTensor(self.log_probs[batch_idx]),
                "advantages": torch.FloatTensor(self.advantages[batch_idx]),
                "returns": torch.FloatTensor(self.returns[batch_idx]),
                "values": torch.FloatTensor(self.values[batch_idx]),
            }

    def reset(self):
        """Clear the buffer for a new epoch."""
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.pos if not self.full else self.buffer_size
