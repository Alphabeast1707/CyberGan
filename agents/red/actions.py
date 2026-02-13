"""
CyberGAN — Red Agent: Action Types & Masking
Defines the Red (attacker) action space and valid action masks.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arena.network import NetworkGraph


class RedActionType(IntEnum):
    """Red agent action types aligned with arena/env.py RED_ACTIONS."""
    SCAN = 0
    EXPLOIT = 1
    ESCALATE = 2
    PIVOT = 3
    EXFILTRATE = 4
    PERSIST = 5


NUM_RED_ACTIONS = len(RedActionType)


def compute_red_action_mask(
    network: "NetworkGraph",
    scanned_nodes: set[str],
    credentials: set[str],
) -> np.ndarray:
    """
    Compute a binary mask of valid (action_type, target_node, target_vuln) combinations.

    Returns: dict with keys "action_type", "target_node", "target_vuln"
             each containing a binary mask array.
    """
    N = network.num_nodes
    V = max(network.num_vulns, 1)
    node_ids = list(network.nodes.keys())

    # Action type mask: which action categories are valid at all?
    action_mask = np.zeros(NUM_RED_ACTIONS, dtype=np.float32)

    # Node mask: which nodes can be targeted?
    node_mask = np.zeros(N, dtype=np.float32)

    # Vuln mask: which vulns can be targeted?
    vuln_mask = np.zeros(V, dtype=np.float32)

    has_compromised = False
    has_unscanned = False
    has_exploitable = False

    for i, nid in enumerate(node_ids):
        node = network.nodes[nid]

        if node.is_isolated:
            continue

        # Can scan any non-isolated node
        if nid not in scanned_nodes:
            has_unscanned = True
            node_mask[i] = 1.0

        # Can exploit scanned nodes with unpatched vulns
        if nid in scanned_nodes and node.exploitable_vulns:
            has_exploitable = True
            node_mask[i] = 1.0

        # Can escalate/pivot/exfil/persist on compromised nodes
        if node.is_compromised:
            has_compromised = True
            node_mask[i] = 1.0

    # Enable action types based on state
    action_mask[RedActionType.SCAN] = 1.0  # Can always scan
    if has_exploitable:
        action_mask[RedActionType.EXPLOIT] = 1.0
    if has_compromised:
        action_mask[RedActionType.ESCALATE] = 1.0
        action_mask[RedActionType.PIVOT] = 1.0
        action_mask[RedActionType.EXFILTRATE] = 1.0
        action_mask[RedActionType.PERSIST] = 1.0

    # Vuln mask: all valid (we resolve at exec time)
    vuln_mask[:] = 1.0

    return {
        "action_type": action_mask,
        "target_node": node_mask,
        "target_vuln": vuln_mask,
    }
