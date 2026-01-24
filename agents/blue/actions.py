"""
CyberGAN — Blue Agent: Action Types & Masking
Defines the Blue (defender) action space and valid action masks.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arena.network import NetworkGraph


class BlueActionType(IntEnum):
    """Blue agent action types aligned with arena/env.py BLUE_ACTIONS."""
    MONITOR = 0
    PATCH = 1
    ISOLATE = 2
    RESTORE = 3
    HONEYPOT = 4
    FIREWALL = 5
    BLOCK = 6


NUM_BLUE_ACTIONS = len(BlueActionType)


def compute_blue_action_mask(network: "NetworkGraph") -> dict[str, np.ndarray]:
    """
    Compute a binary mask of valid (action_type, target_node) combinations.

    Returns: dict with keys "action_type", "target_node"
    """
    N = network.num_nodes
    node_ids = list(network.nodes.keys())

    action_mask = np.ones(NUM_BLUE_ACTIONS, dtype=np.float32)  # All actions generally valid
    node_mask = np.ones(N, dtype=np.float32)  # Can target any node

    has_unpatched = False
    has_compromised = False
    has_restorable = False

    for i, nid in enumerate(node_ids):
        node = network.nodes[nid]

        if node.unpatched_vulns:
            has_unpatched = True
        if node.is_compromised:
            has_compromised = True
        if node.is_compromised or node.is_isolated:
            has_restorable = True

    # Mask out useless actions
    if not has_unpatched:
        action_mask[BlueActionType.PATCH] = 0.0
    if not has_compromised:
        action_mask[BlueActionType.ISOLATE] = 0.0
        action_mask[BlueActionType.BLOCK] = 0.0
    if not has_restorable:
        action_mask[BlueActionType.RESTORE] = 0.0

    return {
        "action_type": action_mask,
        "target_node": node_mask,
    }
