"""
CyberGAN — Arena: Reward Engines
Asymmetric reward functions for Red (attacker) and Blue (defender) agents.
Includes dense shaping rewards and novelty bonuses to guide co-evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardEvent:
    """A single reward event with metadata for logging."""
    agent: str           # "red" or "blue"
    event_type: str      # e.g. "exploit_success", "patch_vuln"
    reward: float
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "event": self.event_type,
            "reward": self.reward,
            "details": self.details,
        }


class RedRewardEngine:
    """
    Computes rewards for the Red (attacker) agent.
    Encourages: novel exploits, deep penetration, stealth.
    Penalizes: detection, failed attacks, repeating same path.
    """

    # Reward constants
    EXPLOIT_NEW_NODE = 10.0
    EXPLOIT_OWNED_NODE = 1.0
    PRIVILEGE_ESCALATION = 5.0
    LATERAL_MOVEMENT = 8.0
    DATA_EXFILTRATION = 15.0
    PERSISTENCE_INSTALLED = 3.0
    CREDENTIAL_STOLEN = 4.0
    DETECTED_BY_BLUE = -5.0
    BLOCKED_BY_BLUE = -3.0
    INVALID_ACTION = -0.1
    SCAN_REWARD = 0.5
    NOVELTY_MULTIPLIER = 2.0
    STEP_PENALTY = -0.05        # Small cost per step (time pressure)

    def __init__(self):
        self._attack_history: set[tuple[str, str]] = set()  # (node_id, cve) pairs seen
        self._events: list[RewardEvent] = []

    def reset(self):
        self._attack_history.clear()
        self._events.clear()

    def compute(
        self,
        action_type: str,
        target_node: str,
        success: bool,
        cve: Optional[str] = None,
        was_detected: bool = False,
        was_blocked: bool = False,
        node_already_owned: bool = False,
        yields_root: bool = False,
        yields_credentials: bool = False,
        is_lateral: bool = False,
        is_exfiltration: bool = False,
        is_persistence: bool = False,
        hit_honeypot: bool = False,
    ) -> float:
        """Compute reward for a Red agent action."""
        reward = self.STEP_PENALTY  # base per-step cost

        if action_type == "scan":
            reward += self.SCAN_REWARD
            self._log("red", "scan", reward, f"scanned {target_node}")
            return reward

        if not success:
            if was_blocked:
                reward += self.BLOCKED_BY_BLUE
                self._log("red", "blocked", reward, f"blocked on {target_node}")
            else:
                reward += self.INVALID_ACTION
                self._log("red", "failed", reward, f"failed on {target_node}")
            return reward

        # Successful exploit
        novelty = self._novelty_bonus(target_node, cve)

        if node_already_owned:
            reward += self.EXPLOIT_OWNED_NODE
        else:
            reward += self.EXPLOIT_NEW_NODE

        if yields_root:
            reward += self.PRIVILEGE_ESCALATION
        if yields_credentials:
            reward += self.CREDENTIAL_STOLEN
        if is_lateral:
            reward += self.LATERAL_MOVEMENT
        if is_exfiltration:
            reward += self.DATA_EXFILTRATION
        if is_persistence:
            reward += self.PERSISTENCE_INSTALLED

        # Novelty bonus
        reward += novelty * self.NOVELTY_MULTIPLIER

        # Penalties
        if was_detected:
            reward += self.DETECTED_BY_BLUE
        if hit_honeypot:
            reward += self.DETECTED_BY_BLUE * 2  # honeypot = extra bad

        self._log("red", "exploit_success", reward, f"exploited {target_node} via {cve}")
        return reward

    def _novelty_bonus(self, node_id: str, cve: Optional[str]) -> float:
        """Reward novel attack paths more than repeated ones."""
        key = (node_id, cve or "none")
        if key in self._attack_history:
            return 0.0
        self._attack_history.add(key)
        return 1.0

    def _log(self, agent: str, event: str, reward: float, details: str):
        self._events.append(RewardEvent(agent, event, reward, details))

    def get_events(self) -> list[RewardEvent]:
        return list(self._events)


class BlueRewardEngine:
    """
    Computes rewards for the Blue (defender) agent.
    Encourages: early detection, patching, containment, uptime.
    Penalizes: missed attacks, false positives, node compromise.
    """

    # Reward constants
    DETECT_ATTACK = 5.0
    PATCH_VULN = 3.0
    ISOLATE_COMPROMISED = 4.0
    RESTORE_NODE = 6.0
    DEPLOY_HONEYPOT = 2.0
    HARDEN_FIREWALL = 1.5
    FALSE_POSITIVE_ISOLATE = -4.0   # Isolated a clean node
    NODE_COMPROMISED = -10.0
    DATA_EXFILTRATED = -15.0
    UPTIME_BONUS = 0.5              # Per step, per healthy node
    INVALID_ACTION = -0.1
    STEP_PENALTY = -0.02            # Mild time pressure

    def __init__(self):
        self._events: list[RewardEvent] = []

    def reset(self):
        self._events.clear()

    def compute(
        self,
        action_type: str,
        target_node: str,
        success: bool,
        node_was_compromised: bool = False,
        node_was_clean: bool = False,
        attack_happened: bool = False,
        attack_detected: bool = False,
        data_exfiltrated: bool = False,
        num_healthy_nodes: int = 0,
        total_nodes: int = 1,
    ) -> float:
        """Compute reward for a Blue agent action."""
        reward = self.STEP_PENALTY

        # Uptime bonus — proportional to healthy nodes
        uptime = (num_healthy_nodes / max(total_nodes, 1)) * self.UPTIME_BONUS
        reward += uptime

        if action_type == "monitor":
            if attack_detected:
                reward += self.DETECT_ATTACK
                self._log("blue", "detect", reward, f"detected attack on {target_node}")
            return reward

        if action_type == "patch":
            if success:
                reward += self.PATCH_VULN
                self._log("blue", "patch", reward, f"patched {target_node}")
            else:
                reward += self.INVALID_ACTION
                self._log("blue", "patch_fail", reward, f"nothing to patch on {target_node}")
            return reward

        if action_type == "isolate":
            if node_was_compromised:
                reward += self.ISOLATE_COMPROMISED
                self._log("blue", "isolate", reward, f"isolated compromised {target_node}")
            elif node_was_clean:
                reward += self.FALSE_POSITIVE_ISOLATE
                self._log("blue", "false_positive", reward, f"isolated clean {target_node}")
            return reward

        if action_type == "restore":
            if success:
                reward += self.RESTORE_NODE
                self._log("blue", "restore", reward, f"restored {target_node}")
            else:
                reward += self.INVALID_ACTION
                self._log("blue", "restore_fail", reward, f"nothing to restore on {target_node}")
            return reward

        if action_type == "honeypot":
            reward += self.DEPLOY_HONEYPOT
            self._log("blue", "honeypot", reward, f"deployed honeypot on {target_node}")
            return reward

        if action_type == "firewall":
            reward += self.HARDEN_FIREWALL
            self._log("blue", "firewall", reward, f"hardened firewall for {target_node}")
            return reward

        if action_type == "block":
            if attack_happened:
                reward += self.DETECT_ATTACK
                self._log("blue", "block", reward, f"blocked attack on {target_node}")
            else:
                reward += self.INVALID_ACTION
                self._log("blue", "block_miss", reward, f"no attack to block on {target_node}")
            return reward

        # Negative events (computed externally, injected into reward)
        if data_exfiltrated:
            reward += self.DATA_EXFILTRATED
        if node_was_compromised and not attack_detected:
            reward += self.NODE_COMPROMISED

        self._log("blue", action_type, reward, f"action on {target_node}")
        return reward

    def _log(self, agent: str, event: str, reward: float, details: str):
        self._events.append(RewardEvent(agent, event, reward, details))

    def get_events(self) -> list[RewardEvent]:
        return list(self._events)
