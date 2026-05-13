"""
CyberGAN — Brain: Action Masker
Context-aware filtering of valid defensive actions.
"""

from __future__ import annotations

import time

import numpy as np

from cybergan.brain.action_space import DEFENSE_ACTIONS, ActionRisk


class ActionMasker:
    """
    Determines which defensive actions are valid given:
    - Current server state
    - Agent mode (advisory/autonomous/hybrid)
    - Action cooldowns
    - Risk thresholds
    """

    def __init__(self, mode: str = "hybrid", confidence_threshold: float = 0.7):
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self._cooldowns: dict[int, float] = {}

    def get_mask(self, risk_score: float = 0.0) -> np.ndarray:
        """
        Generate a binary mask of valid actions.

        Args:
            risk_score: Current risk score (0-100)

        Returns:
            Binary array where 1 = valid, 0 = masked
        """
        mask = np.ones(len(DEFENSE_ACTIONS), dtype=np.float32)
        now = time.time()

        for action in DEFENSE_ACTIONS:
            # Cooldown check
            last_use = self._cooldowns.get(action.id, 0)
            if now - last_use < action.cooldown_s:
                mask[action.id] = 0.0
                continue

            # Mode-based masking
            if self.mode == "advisory":
                # Only allow monitoring and alerting in advisory mode
                if action.risk > ActionRisk.NONE:
                    mask[action.id] = 0.0

            elif self.mode == "hybrid":
                # Block high-risk actions unless risk score warrants them
                if action.risk >= ActionRisk.HIGH and risk_score < 70:
                    mask[action.id] = 0.0
                if action.requires_approval:
                    mask[action.id] = 0.0  # Require manual approval

        # Always allow monitor and alert
        mask[0] = 1.0  # monitor
        mask[1] = 1.0  # alert

        return mask

    def record_action(self, action_id: int):
        """Record that an action was taken (for cooldown tracking)."""
        self._cooldowns[action_id] = time.time()
