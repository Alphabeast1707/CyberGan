"""
CyberGAN — League: Opponent Pool
Manages a pool of historical policy checkpoints for self-play training.
When an agent exceeds a win rate threshold, its weights are snapshotted
into the pool to serve as future sparring partners.
"""

from __future__ import annotations

import os
import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PolicyCheckpoint:
    """A frozen snapshot of a policy at a specific epoch."""
    epoch: int
    elo_at_save: float
    win_rate_at_save: float
    state_dict: dict
    agent_type: str  # "red" or "blue"

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "elo": round(self.elo_at_save, 1),
            "win_rate": round(self.win_rate_at_save, 3),
            "agent": self.agent_type,
        }


class OpponentPool:
    """
    Maintains a league of historical opponents for self-play.

    When win_rate > threshold, the current policy is snapshotted.
    Opponents are sampled with a bias towards recent + strongest.
    """

    def __init__(self, max_size: int = 10, save_threshold: float = 0.6):
        self.max_size = max_size
        self.save_threshold = save_threshold
        self.red_pool: list[PolicyCheckpoint] = []
        self.blue_pool: list[PolicyCheckpoint] = []

    def maybe_save(
        self,
        policy: nn.Module,
        agent_type: str,
        epoch: int,
        elo: float,
        win_rate: float,
    ) -> bool:
        """
        Save a checkpoint if win rate exceeds threshold.
        Returns True if saved.
        """
        if win_rate < self.save_threshold:
            return False

        ckpt = PolicyCheckpoint(
            epoch=epoch,
            elo_at_save=elo,
            win_rate_at_save=win_rate,
            state_dict=copy.deepcopy(policy.state_dict()),
            agent_type=agent_type,
        )

        pool = self.red_pool if agent_type == "red" else self.blue_pool

        pool.append(ckpt)
        if len(pool) > self.max_size:
            # Remove the weakest (lowest ELO) checkpoint
            pool.sort(key=lambda c: c.elo_at_save)
            pool.pop(0)

        return True

    def sample_opponent(self, agent_type: str) -> Optional[dict]:
        """
        Sample an opponent's state_dict from the pool.
        Biased towards recent + strong opponents.

        Args:
            agent_type: "red" to get a Red opponent, "blue" for Blue

        Returns:
            state_dict or None if pool is empty
        """
        pool = self.red_pool if agent_type == "red" else self.blue_pool
        if not pool:
            return None

        # Weight recent opponents more heavily
        import random
        weights = [i + 1 for i in range(len(pool))]  # linear increasing
        ckpt = random.choices(pool, weights=weights, k=1)[0]
        return ckpt.state_dict

    def save_to_disk(self, directory: str):
        """Save all checkpoints to disk."""
        os.makedirs(directory, exist_ok=True)
        for pool, name in [(self.red_pool, "red"), (self.blue_pool, "blue")]:
            for i, ckpt in enumerate(pool):
                path = os.path.join(directory, f"{name}_epoch{ckpt.epoch}.pt")
                torch.save(ckpt.state_dict, path)

    def get_pool_info(self) -> dict:
        return {
            "red_pool_size": len(self.red_pool),
            "blue_pool_size": len(self.blue_pool),
            "red_checkpoints": [c.to_dict() for c in self.red_pool],
            "blue_checkpoints": [c.to_dict() for c in self.blue_pool],
        }
