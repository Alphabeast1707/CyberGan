"""
CyberGAN — Blue Agent: Observation Builder
Flattens the dict observation from the arena into a single tensor for the policy network.
"""

from __future__ import annotations

import numpy as np


def flatten_blue_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten the dict observation into a single 1D vector for the neural network.
    Same structure as Red obs (symmetric for simplicity).
    """
    parts = [
        obs["adjacency"].flatten(),
        obs["node_states"].flatten(),
        obs["vuln_map"].flatten(),
        obs["owned_mask"].flatten().astype(np.float32),
        obs["health"].flatten(),
        obs["step_ratio"].flatten(),
        obs["last_opponent_action"].flatten(),
    ]
    return np.concatenate(parts)


def get_obs_dim(num_nodes: int, num_vulns: int, num_opponent_actions: int = 7) -> int:
    """Calculate the total flattened observation dimension."""
    return (
        num_nodes * num_nodes
        + num_nodes
        + num_nodes * num_vulns
        + num_nodes
        + num_nodes
        + 1
        + num_opponent_actions
    )
