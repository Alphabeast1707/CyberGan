"""
CyberGAN — Brain: Observer
Builds real-time observation vectors from server state.
Maps real network state to the same format the arena-trained policy expects.
"""

from __future__ import annotations

import numpy as np

from cybergan.analysis.feature_extractor import FeatureExtractor, SecurityState


class ProductionObserver:
    """
    Builds observation vectors for the production policy from live server state.

    Bridges between the FeatureExtractor (which aggregates raw events)
    and the policy network (which expects a fixed-size vector).
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    def observe(self) -> tuple[np.ndarray, SecurityState]:
        """
        Get current observation from feature extractor.

        Returns:
            (observation_vector, security_state)
        """
        state = self.feature_extractor.extract()
        return state.observation, state

    def get_observation_dim(self) -> int:
        return self.feature_extractor.get_observation_dim()
