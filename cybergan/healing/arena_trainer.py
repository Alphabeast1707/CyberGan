"""
CyberGAN — Healing: Arena Trainer
Runs adversarial Red vs Blue training in the background.
New attack patterns from production are added to the Red Agent's repertoire.
"""

from __future__ import annotations

import structlog

from cybergan.config import ArenaTrainingConfig

logger = structlog.get_logger(__name__)


class ArenaTrainer:
    """Background adversarial training manager."""

    def __init__(self, config: ArenaTrainingConfig):
        self.config = config
        self._epoch = 0
        self._running = False

    async def start(self):
        """Start background training loop."""
        if not self.config.enabled:
            return
        self._running = True
        logger.info("arena_trainer.start")

    def stop(self):
        self._running = False

    def add_attack_pattern(self, pattern: dict):
        """Add a new attack pattern observed in production to the arena."""
        logger.info("arena_trainer.new_pattern", pattern=pattern.get("type", "unknown"))

    def get_stats(self) -> dict:
        return {
            "epoch": self._epoch,
            "running": self._running,
        }
