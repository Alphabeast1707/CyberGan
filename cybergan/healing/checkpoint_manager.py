"""
CyberGAN — Healing: Checkpoint Manager
Model versioning, rollback, and performance tracking.
"""

from __future__ import annotations

import os
import time

import torch
import structlog

from cybergan.config import CheckpointManagerConfig

logger = structlog.get_logger(__name__)


class CheckpointManager:
    """Manages policy model checkpoints with auto-rollback."""

    def __init__(self, config: CheckpointManagerConfig, checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self._checkpoints: list[dict] = []
        self._best_performance: float = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, policy, epoch: int, performance: float) -> str:
        """Save a policy checkpoint."""
        filename = f"blue_epoch_{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save(policy.state_dict(), path)

        self._checkpoints.append({
            "epoch": epoch,
            "path": path,
            "performance": performance,
            "timestamp": time.time(),
        })

        # Update best
        if performance > self._best_performance:
            self._best_performance = performance
            best_path = os.path.join(self.checkpoint_dir, "blue_production.pt")
            torch.save(policy.state_dict(), best_path)
            logger.info("checkpoint.new_best", epoch=epoch, performance=performance)

        # Cleanup old checkpoints
        if len(self._checkpoints) > self.config.max_checkpoints:
            old = self._checkpoints.pop(0)
            if os.path.exists(old["path"]):
                os.remove(old["path"])

        logger.info("checkpoint.saved", path=path, epoch=epoch)
        return path

    def should_rollback(self, current_performance: float) -> bool:
        """Check if we should rollback to a previous checkpoint."""
        if not self.config.auto_rollback_on_regression:
            return False
        if self._best_performance == 0:
            return False
        regression = (self._best_performance - current_performance) / self._best_performance
        return regression > self.config.regression_threshold

    def get_best_path(self) -> str:
        """Get path to the best performing checkpoint."""
        return os.path.join(self.checkpoint_dir, "blue_production.pt")

    def get_stats(self) -> dict:
        return {
            "total_checkpoints": len(self._checkpoints),
            "best_performance": self._best_performance,
        }
