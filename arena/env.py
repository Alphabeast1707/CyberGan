"""
CyberGAN — Arena: Gymnasium Environment
Turn-based two-agent cyber battle environment.

Each step consists of:
  1. Red agent takes an action (attack)
  2. Blue agent takes an action (defend)
  3. Environment returns (obs_red, obs_blue, reward_red, reward_blue, terminated, info)
"""

from __future__ import annotations

import random
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import yaml

from arena.network import NetworkGraph, NodeState
from arena.vulnerabilities import EXPLOIT_CATALOG, get_exploit
from arena.reward import RedRewardEngine, BlueRewardEngine


# ──────────────────────────────────────────────────────────────
# Action type enumerations (kept in sync with agents/*/actions.py)
# ──────────────────────────────────────────────────────────────

RED_ACTIONS = ["scan", "exploit", "escalate", "pivot", "exfiltrate", "persist"]
BLUE_ACTIONS = ["monitor", "patch", "isolate", "restore", "honeypot", "firewall", "block"]


class CyberGANEnv(gym.Env):
    """
    CyberGAN Gymnasium environment.

    Two agents take turns:
      - Red picks (action_type, target_node, target_vuln)
      - Blue picks (action_type, target_node)

    The env manages state transitions, reward computation,
    and termination conditions.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config_path: str = "config.yaml", max_steps: int = 64):
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.network = NetworkGraph.from_dict(self.config.get("network", {}))
        self.max_steps = max_steps

        N = self.network.num_nodes
        V = max(self.network.num_vulns, 1)

        # ── Observation spaces ──
        self.observation_space = gym.spaces.Dict({
            "adjacency": gym.spaces.Box(0, 1, shape=(N, N), dtype=np.float32),
            "node_states": gym.spaces.Box(0, 5, shape=(N,), dtype=np.float32),
            "vuln_map": gym.spaces.Box(0, 10, shape=(N, V), dtype=np.float32),
            "owned_mask": gym.spaces.MultiBinary(N),
            "health": gym.spaces.Box(0, 1, shape=(N,), dtype=np.float32),
            "step_ratio": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "last_opponent_action": gym.spaces.Box(0, 1, shape=(max(len(RED_ACTIONS), len(BLUE_ACTIONS)),), dtype=np.float32),
        })

        # ── Action spaces ──
        self.red_action_space = gym.spaces.MultiDiscrete([len(RED_ACTIONS), N, V])
        self.blue_action_space = gym.spaces.MultiDiscrete([len(BLUE_ACTIONS), N])

        # ── Reward engines ──
        self.red_reward_engine = RedRewardEngine()
        self.blue_reward_engine = BlueRewardEngine()

        # ── Internal state ──
        self.current_step = 0
        self.red_score = 0.0
        self.blue_score = 0.0
        self._last_red_action: Optional[np.ndarray] = None
        self._last_blue_action: Optional[np.ndarray] = None
        self._event_log: list[dict] = []

        # Build adjacency matrix once
        self._adjacency_matrix = self._build_adjacency()

        # Track which nodes Red has "discovered" (scanned)
        self._red_scanned: set[str] = set()
        self._red_credentials: set[str] = set()

    # ──────────────────────────────────────────────────────────
    # Core Gym interface
    # ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        self.network.reset()
        self.current_step = 0
        self.red_score = 0.0
        self.blue_score = 0.0
        self._last_red_action = None
        self._last_blue_action = None
        self._event_log.clear()
        self._red_scanned.clear()
        self._red_credentials.clear()
        self.red_reward_engine.reset()
        self.blue_reward_engine.reset()

        # Red starts by knowing the gateway exists
        node_ids = list(self.network.nodes.keys())
        if node_ids:
            entry = node_ids[0]  # gateway
            self._red_scanned.add(entry)
            self.network.nodes[entry].state = NodeState.SCANNED

        obs_red = self._get_red_obs()
        obs_blue = self._get_blue_obs()
        info = {"red_obs": obs_red, "blue_obs": obs_blue}
        return obs_red, info

    def step_red(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute the Red agent's action.
        Returns (obs_blue, red_reward, terminated, truncated, info).
        Blue gets the observation to make its move.
        """
        self._last_red_action = action
        action_type_idx, target_idx, vuln_idx = int(action[0]), int(action[1]), int(action[2])
        action_type = RED_ACTIONS[min(action_type_idx, len(RED_ACTIONS) - 1)]
        target_node = self.network.get_node_by_index(target_idx % self.network.num_nodes)

        reward = 0.0
        event = {"step": self.current_step, "agent": "red", "action": action_type}

        if target_node is None:
            reward = self.red_reward_engine.compute("invalid", "none", False)
            event["result"] = "invalid_target"
            self._event_log.append(event)
            return self._get_blue_obs(), reward, False, False, {"event": event}

        event["target"] = target_node.id

        if action_type == "scan":
            reward = self._red_scan(target_node, event)
        elif action_type == "exploit":
            reward = self._red_exploit(target_node, vuln_idx, event)
        elif action_type == "escalate":
            reward = self._red_escalate(target_node, vuln_idx, event)
        elif action_type == "pivot":
            reward = self._red_pivot(target_node, event)
        elif action_type == "exfiltrate":
            reward = self._red_exfiltrate(target_node, event)
        elif action_type == "persist":
            reward = self._red_persist(target_node, event)
        else:
            reward = self.red_reward_engine.compute("invalid", target_node.id, False)
            event["result"] = "unknown_action"

        self.red_score += reward
        self._event_log.append(event)

        obs_blue = self._get_blue_obs()
        return obs_blue, reward, False, False, {"event": event}

    def step_blue(self, action: np.ndarray) -> tuple[dict, dict, float, float, bool, bool, dict]:
        """
        Execute the Blue agent's action.
        Returns (obs_red, obs_blue, red_penalty, blue_reward, terminated, truncated, info).
        """
        self._last_blue_action = action
        action_type_idx, target_idx = int(action[0]), int(action[1])
        action_type = BLUE_ACTIONS[min(action_type_idx, len(BLUE_ACTIONS) - 1)]
        target_node = self.network.get_node_by_index(target_idx % self.network.num_nodes)

        blue_reward = 0.0
        red_penalty = 0.0
        event = {"step": self.current_step, "agent": "blue", "action": action_type}

        if target_node is None:
            blue_reward = self.blue_reward_engine.compute("invalid", "none", False)
            event["result"] = "invalid_target"
            self._event_log.append(event)
            self.current_step += 1
            terminated = self._check_terminated()
            truncated = self.current_step >= self.max_steps
            return self._get_red_obs(), self._get_blue_obs(), red_penalty, blue_reward, terminated, truncated, {"event": event}

        event["target"] = target_node.id
        stats = self.network.get_stats()
        healthy = stats["total_nodes"] - stats["compromised"] - stats["isolated"]

        if action_type == "monitor":
            blue_reward = self._blue_monitor(target_node, event, healthy, stats["total_nodes"])
        elif action_type == "patch":
            blue_reward = self._blue_patch(target_node, event, healthy, stats["total_nodes"])
        elif action_type == "isolate":
            blue_reward, red_penalty = self._blue_isolate(target_node, event, healthy, stats["total_nodes"])
        elif action_type == "restore":
            blue_reward = self._blue_restore(target_node, event, healthy, stats["total_nodes"])
        elif action_type == "honeypot":
            blue_reward = self._blue_honeypot(target_node, event, healthy, stats["total_nodes"])
        elif action_type == "firewall":
            blue_reward = self._blue_firewall(target_node, event, healthy, stats["total_nodes"])
        elif action_type == "block":
            blue_reward = self._blue_block(target_node, event, healthy, stats["total_nodes"])

        self.blue_score += blue_reward
        self._event_log.append(event)
        self.current_step += 1

        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps

        obs_red = self._get_red_obs()
        obs_blue = self._get_blue_obs()
        info = {
            "event": event,
            "red_score": self.red_score,
            "blue_score": self.blue_score,
            "stats": stats,
            "step": self.current_step,
        }
        return obs_red, obs_blue, red_penalty, blue_reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────
    # Red agent action implementations
    # ──────────────────────────────────────────────────────────

    def _red_scan(self, node, event) -> float:
        """Scan a node to discover its services and vulnerabilities."""
        if node.id in self._red_scanned:
            event["result"] = "already_scanned"
            return self.red_reward_engine.compute("scan", node.id, True)

        self._red_scanned.add(node.id)
        if node.state == NodeState.UNKNOWN:
            node.state = NodeState.SCANNED
        if node.unpatched_vulns:
            node.state = NodeState.VULNERABLE
        event["result"] = "discovered"
        event["vulns_found"] = len(node.unpatched_vulns)

        # Also discover neighbors
        for neighbor_id in self.network.neighbors(node.id):
            self._red_scanned.add(neighbor_id)
            n = self.network.nodes.get(neighbor_id)
            if n and n.state == NodeState.UNKNOWN:
                n.state = NodeState.SCANNED

        return self.red_reward_engine.compute("scan", node.id, True)

    def _red_exploit(self, node, vuln_idx, event) -> float:
        """Attempt to exploit a vulnerability on a node."""
        if node.is_isolated:
            event["result"] = "node_isolated"
            return self.red_reward_engine.compute("exploit", node.id, False, was_blocked=True)

        # Find the target vulnerability
        vulns = node.vulnerabilities
        if not vulns or vuln_idx >= len(vulns):
            event["result"] = "no_vuln"
            return self.red_reward_engine.compute("exploit", node.id, False)

        vuln = vulns[vuln_idx % len(vulns)]
        if vuln.is_patched:
            event["result"] = "patched"
            return self.red_reward_engine.compute("exploit", node.id, False, was_blocked=True)

        # Look up exploit template
        exploit = get_exploit(vuln.cve)
        success_rate = exploit.base_success_rate if exploit else 0.4

        # Honeypot detection
        hit_honeypot = node.has_honeypot
        if hit_honeypot:
            event["result"] = "honeypot_trapped"
            return self.red_reward_engine.compute(
                "exploit", node.id, False, cve=vuln.cve, hit_honeypot=True
            )

        # Roll for success
        roll = random.random()
        success = roll < success_rate

        if success:
            vuln.is_exploited = True
            vuln.times_exploited += 1
            node.state = NodeState.COMPROMISED
            if exploit and exploit.yields_credentials:
                self._red_credentials.add(node.id)
            event["result"] = "exploited"
            event["cve"] = vuln.cve
        else:
            event["result"] = "failed"
            event["cve"] = vuln.cve

        return self.red_reward_engine.compute(
            "exploit", node.id, success,
            cve=vuln.cve,
            node_already_owned=False,
            yields_root=exploit.yields_root if exploit else False,
            yields_credentials=exploit.yields_credentials if exploit else False,
        )

    def _red_escalate(self, node, vuln_idx, event) -> float:
        """Attempt privilege escalation on a compromised node."""
        if not node.is_compromised:
            event["result"] = "not_compromised"
            return self.red_reward_engine.compute("escalate", node.id, False)

        vulns = [v for v in node.vulnerabilities if not v.is_patched and not v.is_exploited]
        if not vulns:
            event["result"] = "no_local_vuln"
            return self.red_reward_engine.compute("escalate", node.id, False)

        vuln = vulns[vuln_idx % len(vulns)]
        exploit = get_exploit(vuln.cve)
        success_rate = (exploit.base_success_rate if exploit else 0.3) * 0.8  # harder than remote

        success = random.random() < success_rate
        if success:
            vuln.is_exploited = True
            vuln.times_exploited += 1
            event["result"] = "escalated"
        else:
            event["result"] = "failed"

        return self.red_reward_engine.compute(
            "escalate", node.id, success,
            cve=vuln.cve,
            node_already_owned=True,
            yields_root=exploit.yields_root if exploit and success else False,
        )

    def _red_pivot(self, node, event) -> float:
        """Attempt lateral movement from a compromised node to a neighbor."""
        if not node.is_compromised:
            event["result"] = "not_compromised"
            return self.red_reward_engine.compute("pivot", node.id, False)

        neighbors = self.network.neighbors(node.id)
        uncompromised = [n for n in neighbors if not self.network.nodes[n].is_compromised and not self.network.nodes[n].is_isolated]
        if not uncompromised:
            event["result"] = "no_targets"
            return self.red_reward_engine.compute("pivot", node.id, False)

        target_id = random.choice(uncompromised)
        target = self.network.nodes[target_id]

        # Pivot success depends on credentials
        has_creds = node.id in self._red_credentials
        success_rate = 0.6 if has_creds else 0.3

        success = random.random() < success_rate
        if success:
            target.state = NodeState.COMPROMISED
            self._red_scanned.add(target_id)
            event["result"] = "pivoted"
            event["new_node"] = target_id
        else:
            event["result"] = "failed"

        return self.red_reward_engine.compute(
            "pivot", target_id, success, is_lateral=True,
        )

    def _red_exfiltrate(self, node, event) -> float:
        """Attempt data exfiltration from a compromised node."""
        if not node.is_compromised:
            event["result"] = "not_compromised"
            return self.red_reward_engine.compute("exfiltrate", node.id, False)

        success = random.random() < 0.7
        event["result"] = "exfiltrated" if success else "failed"
        return self.red_reward_engine.compute(
            "exfiltrate", node.id, success, is_exfiltration=True,
        )

    def _red_persist(self, node, event) -> float:
        """Install persistence (backdoor) on a compromised node."""
        if not node.is_compromised:
            event["result"] = "not_compromised"
            return self.red_reward_engine.compute("persist", node.id, False)
        if node.has_backdoor:
            event["result"] = "already_persistent"
            return self.red_reward_engine.compute("persist", node.id, False)

        node.has_backdoor = True
        event["result"] = "backdoor_installed"
        return self.red_reward_engine.compute(
            "persist", node.id, True, is_persistence=True,
        )

    # ──────────────────────────────────────────────────────────
    # Blue agent action implementations
    # ──────────────────────────────────────────────────────────

    def _blue_monitor(self, node, event, healthy, total) -> float:
        """Monitor a node for suspicious activity."""
        detected = node.is_compromised or node.has_backdoor
        event["result"] = "threat_detected" if detected else "all_clear"
        return self.blue_reward_engine.compute(
            "monitor", node.id, True,
            attack_happened=detected,
            attack_detected=detected,
            num_healthy_nodes=healthy,
            total_nodes=total,
        )

    def _blue_patch(self, node, event, healthy, total) -> float:
        """Patch an unpatched vulnerability on a node."""
        unpatched = node.unpatched_vulns
        if not unpatched:
            event["result"] = "nothing_to_patch"
            return self.blue_reward_engine.compute(
                "patch", node.id, False, num_healthy_nodes=healthy, total_nodes=total,
            )
        # Patch the highest-severity unpatched vuln
        unpatched.sort(key=lambda v: v.severity, reverse=True)
        vuln = unpatched[0]
        vuln.is_patched = True
        event["result"] = "patched"
        event["cve"] = vuln.cve
        return self.blue_reward_engine.compute(
            "patch", node.id, True, num_healthy_nodes=healthy, total_nodes=total,
        )

    def _blue_isolate(self, node, event, healthy, total) -> tuple[float, float]:
        """Isolate (quarantine) a node from the network."""
        was_compromised = node.is_compromised
        was_clean = not was_compromised and node.state != NodeState.ISOLATED

        if node.is_isolated:
            event["result"] = "already_isolated"
            return self.blue_reward_engine.compute(
                "isolate", node.id, False, num_healthy_nodes=healthy, total_nodes=total,
            ), 0.0

        node.state = NodeState.ISOLATED
        # Block all edges to/from this node
        for edge in self.network.edges:
            if edge.source == node.id or edge.target == node.id:
                edge.is_blocked = True

        event["result"] = "isolated"
        red_penalty = -2.0 if was_compromised else 0.0
        blue_reward = self.blue_reward_engine.compute(
            "isolate", node.id, True,
            node_was_compromised=was_compromised,
            node_was_clean=was_clean,
            num_healthy_nodes=healthy,
            total_nodes=total,
        )
        return blue_reward, red_penalty

    def _blue_restore(self, node, event, healthy, total) -> float:
        """Restore a compromised or isolated node to a clean state."""
        if node.state not in (NodeState.COMPROMISED, NodeState.ISOLATED):
            event["result"] = "nothing_to_restore"
            return self.blue_reward_engine.compute(
                "restore", node.id, False, num_healthy_nodes=healthy, total_nodes=total,
            )

        node.state = NodeState.SCANNED
        node.has_backdoor = False
        # Unblock edges
        for edge in self.network.edges:
            if edge.source == node.id or edge.target == node.id:
                edge.is_blocked = False

        event["result"] = "restored"
        return self.blue_reward_engine.compute(
            "restore", node.id, True, num_healthy_nodes=healthy, total_nodes=total,
        )

    def _blue_honeypot(self, node, event, healthy, total) -> float:
        """Deploy a honeypot on a node to trap Red."""
        if node.has_honeypot:
            event["result"] = "already_deployed"
            return self.blue_reward_engine.compute(
                "honeypot", node.id, False, num_healthy_nodes=healthy, total_nodes=total,
            )
        node.has_honeypot = True
        event["result"] = "deployed"
        return self.blue_reward_engine.compute(
            "honeypot", node.id, True, num_healthy_nodes=healthy, total_nodes=total,
        )

    def _blue_firewall(self, node, event, healthy, total) -> float:
        """Harden firewall rules on edges leading to this node."""
        for edge in self.network.edges:
            if edge.target == node.id and not edge.is_blocked:
                edge.firewall_ports = []  # Close all ports
        event["result"] = "hardened"
        return self.blue_reward_engine.compute(
            "firewall", node.id, True, num_healthy_nodes=healthy, total_nodes=total,
        )

    def _blue_block(self, node, event, healthy, total) -> float:
        """Block the most recent attack if one occurred on this node."""
        last_red = self._event_log[-1] if self._event_log else None
        attack_happened = (
            last_red and last_red.get("agent") == "red"
            and last_red.get("target") == node.id
            and last_red.get("result") in ("exploited", "pivoted", "escalated")
        )
        if attack_happened:
            # Reverse the compromise
            node.state = NodeState.SCANNED
            event["result"] = "attack_blocked"
        else:
            event["result"] = "no_attack_to_block"

        return self.blue_reward_engine.compute(
            "block", node.id, True,
            attack_happened=attack_happened,
            num_healthy_nodes=healthy,
            total_nodes=total,
        )

    # ──────────────────────────────────────────────────────────
    # Observation builders
    # ──────────────────────────────────────────────────────────

    def _get_red_obs(self) -> dict[str, np.ndarray]:
        """Build observation from Red agent's perspective."""
        N = self.network.num_nodes
        V = max(self.network.num_vulns, 1)

        node_states = np.zeros(N, dtype=np.float32)
        vuln_map = np.zeros((N, V), dtype=np.float32)
        owned = np.zeros(N, dtype=np.int8)
        health = np.zeros(N, dtype=np.float32)

        for i, (nid, node) in enumerate(self.network.nodes.items()):
            node_states[i] = node.state.value
            health[i] = node.health_score()
            if node.is_compromised:
                owned[i] = 1
            # Red can only see vulns on scanned nodes
            if nid in self._red_scanned:
                for j, v in enumerate(node.vulnerabilities[:V]):
                    if not v.is_patched:
                        vuln_map[i, j] = v.severity

        last_opp = np.zeros(max(len(RED_ACTIONS), len(BLUE_ACTIONS)), dtype=np.float32)
        if self._last_blue_action is not None:
            idx = int(self._last_blue_action[0]) % len(BLUE_ACTIONS)
            last_opp[idx] = 1.0

        return {
            "adjacency": self._adjacency_matrix.copy(),
            "node_states": node_states,
            "vuln_map": vuln_map,
            "owned_mask": owned,
            "health": health,
            "step_ratio": np.array([self.current_step / max(self.max_steps, 1)], dtype=np.float32),
            "last_opponent_action": last_opp,
        }

    def _get_blue_obs(self) -> dict[str, np.ndarray]:
        """Build observation from Blue agent's perspective."""
        N = self.network.num_nodes
        V = max(self.network.num_vulns, 1)

        node_states = np.zeros(N, dtype=np.float32)
        vuln_map = np.zeros((N, V), dtype=np.float32)
        compromised = np.zeros(N, dtype=np.int8)
        health = np.zeros(N, dtype=np.float32)

        for i, (nid, node) in enumerate(self.network.nodes.items()):
            node_states[i] = node.state.value
            health[i] = node.health_score()
            if node.is_compromised:
                compromised[i] = 1
            # Blue sees all vulns (defender has full asset inventory)
            for j, v in enumerate(node.vulnerabilities[:V]):
                vuln_map[i, j] = v.severity if not v.is_patched else 0.0

        last_opp = np.zeros(max(len(RED_ACTIONS), len(BLUE_ACTIONS)), dtype=np.float32)
        if self._last_red_action is not None:
            idx = int(self._last_red_action[0]) % len(RED_ACTIONS)
            last_opp[idx] = 1.0

        return {
            "adjacency": self._adjacency_matrix.copy(),
            "node_states": node_states,
            "vuln_map": vuln_map,
            "owned_mask": compromised,
            "health": health,
            "step_ratio": np.array([self.current_step / max(self.max_steps, 1)], dtype=np.float32),
            "last_opponent_action": last_opp,
        }

    def _build_adjacency(self) -> np.ndarray:
        """Build the adjacency matrix for the network."""
        N = self.network.num_nodes
        adj = np.zeros((N, N), dtype=np.float32)
        node_ids = list(self.network.nodes.keys())
        for edge in self.network.edges:
            if edge.source in node_ids and edge.target in node_ids:
                i = node_ids.index(edge.source)
                j = node_ids.index(edge.target)
                adj[i, j] = 1.0
        return adj

    def _check_terminated(self) -> bool:
        """Check if the episode should terminate."""
        stats = self.network.get_stats()
        # Terminate if all nodes compromised (Red wins decisively)
        if stats["compromised"] == stats["total_nodes"]:
            return True
        # Terminate if all vulns patched (Blue wins decisively)
        if stats["vuln_coverage"] >= 1.0:
            return True
        return False

    def get_event_log(self) -> list[dict]:
        return list(self._event_log)

    def get_scores(self) -> dict:
        return {"red": self.red_score, "blue": self.blue_score}
