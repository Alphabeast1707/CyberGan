"""
CyberGAN — Healing: Online Learner
Reinforcement learning update from real production incidents.
When the agent takes an action and the outcome is observed,
a reward signal is generated and the policy is updated.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

import structlog

from cybergan.config import OnlineLearningConfig

logger = structlog.get_logger(__name__)


@dataclass
class Experience:
    """A single experience tuple from production."""
    timestamp: float
    observation: np.ndarray
    action_id: int
    reward: float
    next_observation: np.ndarray
    done: bool = False
    event_type: str = ""
    action_name: str = ""
    outcome: str = ""  # "blocked", "missed", "false_positive"


class ExperienceBuffer:
    """Stores production experiences for online learning."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque[Experience] = deque(maxlen=max_size)
        self._pending_experiences: dict[str, dict] = {}

    def start_experience(self, experience_id: str, observation: np.ndarray, action_id: int,
                          event_type: str = "", action_name: str = ""):
        """Record the start of an action (before outcome is known)."""
        self._pending_experiences[experience_id] = {
            "timestamp": time.time(),
            "observation": observation.copy(),
            "action_id": action_id,
            "event_type": event_type,
            "action_name": action_name,
        }

    def complete_experience(self, experience_id: str, reward: float,
                             next_observation: np.ndarray, outcome: str = ""):
        """Complete an experience with the observed outcome."""
        pending = self._pending_experiences.pop(experience_id, None)
        if pending is None:
            return

        exp = Experience(
            timestamp=pending["timestamp"],
            observation=pending["observation"],
            action_id=pending["action_id"],
            reward=reward,
            next_observation=next_observation.copy(),
            event_type=pending["event_type"],
            action_name=pending["action_name"],
            outcome=outcome,
        )
        self._buffer.append(exp)

    def add_direct(self, observation: np.ndarray, action_id: int, reward: float,
                    next_observation: np.ndarray, event_type: str = ""):
        """Add a complete experience directly."""
        self._buffer.append(Experience(
            timestamp=time.time(),
            observation=observation.copy(),
            action_id=action_id,
            reward=reward,
            next_observation=next_observation.copy(),
            event_type=event_type,
        ))

    @property
    def size(self) -> int:
        return len(self._buffer)

    def sample_batch(self, batch_size: int) -> list[Experience]:
        """Sample a random batch of experiences."""
        if self.size < batch_size:
            return list(self._buffer)
        indices = np.random.choice(self.size, batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def get_recent(self, count: int = 10) -> list[Experience]:
        return list(self._buffer)[-count:]


class OnlineLearner:
    """
    Online RL policy updater using production experiences.

    Periodically takes accumulated (state, action, reward) tuples from
    production incidents and performs policy gradient updates to improve
    the Blue agent's decision-making.

    Implements a simplified policy gradient update:
    - Positive reward for successfully blocked attacks
    - Negative reward for missed attacks
    - Mild negative for false positives (unnecessary actions)
    """

    def __init__(self, config: OnlineLearningConfig, policy=None):
        self.config = config
        self.policy = policy
        self.experience_buffer = ExperienceBuffer()
        self._update_count = 0
        self._total_reward = 0.0

        # Optimizer (will be initialized when policy is set)
        self._optimizer = None

    def set_policy(self, policy):
        """Set/update the policy to learn on."""
        self.policy = policy
        if policy is not None:
            self._optimizer = torch.optim.Adam(
                policy.parameters(), lr=self.config.learning_rate
            )

    def record_outcome(
        self,
        observation: np.ndarray,
        action_id: int,
        outcome: str,
        next_observation: Optional[np.ndarray] = None,
    ):
        """
        Record the outcome of a defensive action.

        Args:
            observation: State when action was taken
            action_id: Which action was taken
            outcome: "blocked", "missed", "false_positive"
            next_observation: State after action
        """
        # Compute reward based on outcome
        if outcome == "blocked":
            reward = self.config.reward_for_blocked_attack
        elif outcome == "missed":
            reward = self.config.reward_for_missed_attack
        elif outcome == "false_positive":
            reward = self.config.reward_for_false_positive
        else:
            reward = 0.0

        if next_observation is None:
            next_observation = observation

        self.experience_buffer.add_direct(
            observation=observation,
            action_id=action_id,
            reward=reward,
            next_observation=next_observation,
            event_type=outcome,
        )

        self._total_reward += reward

    def should_update(self) -> bool:
        """Check if we have enough experience for a policy update."""
        return (
            self.config.enabled
            and self.policy is not None
            and self.experience_buffer.size >= self.config.min_experience_for_update
        )

    def update(self) -> dict:
        """
        Perform a policy gradient update using accumulated experiences.

        Returns:
            Training statistics dict
        """
        if not self.should_update():
            return {}

        batch = self.experience_buffer.sample_batch(
            min(self.experience_buffer.size, 256)
        )

        # Convert to tensors
        obs = torch.FloatTensor(np.array([e.observation for e in batch]))
        actions = torch.LongTensor([e.action_id for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])

        # Normalize rewards
        if rewards.std() > 1e-6:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Policy gradient update
        self.policy.train()
        dist, values = self.policy(obs)

        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * rewards).mean()

        # Value loss (if we have value targets)
        value_loss = torch.nn.functional.mse_loss(values.squeeze(), rewards)

        # Entropy bonus
        entropy = dist.entropy().mean()

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self._optimizer.step()

        self.policy.eval()
        self._update_count += 1

        stats = {
            "update_count": self._update_count,
            "batch_size": len(batch),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "mean_reward": rewards.mean().item(),
            "buffer_size": self.experience_buffer.size,
        }

        logger.info("online_learner.update", **stats)
        return stats

    def get_stats(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "updates": self._update_count,
            "buffer_size": self.experience_buffer.size,
            "total_reward": self._total_reward,
            "has_policy": self.policy is not None,
        }
