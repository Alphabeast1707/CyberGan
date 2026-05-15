"""
CyberGAN — Brain: Defense Action Space
Defines all available defensive actions the agent can take in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActionRisk(IntEnum):
    """Risk level of a defensive action (higher = more disruptive)."""
    NONE = 0        # No risk (monitoring, alerting)
    LOW = 1         # Low risk (rate limiting, WAF rules)
    MEDIUM = 2      # Medium risk (IP blocking, firewall rules)
    HIGH = 3        # High risk (process killing, service isolation)
    CRITICAL = 4    # Critical risk (system changes, patching)


@dataclass
class DefenseAction:
    """A defensive action the agent can take."""
    id: int
    name: str
    description: str
    risk: ActionRisk
    requires_target: bool = False  # Needs a target (IP, PID, service)
    requires_approval: bool = False  # Needs human approval in hybrid mode
    cooldown_s: int = 10  # Minimum seconds between invocations
    reversible: bool = True  # Can be undone


# All available defensive actions
DEFENSE_ACTIONS: list[DefenseAction] = [
    DefenseAction(
        id=0, name="monitor",
        description="Continue monitoring without taking action",
        risk=ActionRisk.NONE, cooldown_s=0,
    ),
    DefenseAction(
        id=1, name="alert",
        description="Send alert notification to security team",
        risk=ActionRisk.NONE, cooldown_s=30,
    ),
    DefenseAction(
        id=2, name="rate_limit",
        description="Apply rate limiting to source IP",
        risk=ActionRisk.LOW, requires_target=True, cooldown_s=5,
    ),
    DefenseAction(
        id=3, name="block_ip",
        description="Block source IP address via firewall",
        risk=ActionRisk.MEDIUM, requires_target=True, cooldown_s=5,
    ),
    DefenseAction(
        id=4, name="firewall_block",
        description="Add firewall rule to block traffic pattern",
        risk=ActionRisk.MEDIUM, cooldown_s=10,
    ),
    DefenseAction(
        id=5, name="waf_rule",
        description="Deploy WAF rule to filter malicious requests",
        risk=ActionRisk.LOW, cooldown_s=5,
    ),
    DefenseAction(
        id=6, name="kill_process",
        description="Terminate a suspicious or malicious process",
        risk=ActionRisk.HIGH, requires_target=True, cooldown_s=5,
    ),
    DefenseAction(
        id=7, name="isolate_service",
        description="Isolate a compromised service (stop/restart)",
        risk=ActionRisk.HIGH, requires_target=True, requires_approval=True,
        cooldown_s=30, reversible=True,
    ),
    DefenseAction(
        id=8, name="patch_config",
        description="Apply security hardening configuration",
        risk=ActionRisk.CRITICAL, requires_approval=True,
        cooldown_s=60, reversible=True,
    ),
    DefenseAction(
        id=9, name="deploy_honeypot",
        description="Deploy honeypot service on unused port",
        risk=ActionRisk.LOW, cooldown_s=60,
    ),
    DefenseAction(
        id=10, name="rotate_credentials",
        description="Rotate compromised credentials/API keys",
        risk=ActionRisk.CRITICAL, requires_approval=True,
        cooldown_s=300, reversible=False,
    ),
    DefenseAction(
        id=11, name="snapshot_forensics",
        description="Capture forensic snapshot of current state",
        risk=ActionRisk.NONE, cooldown_s=60,
    ),
    DefenseAction(
        id=12, name="geo_block",
        description="Block traffic from suspicious geographic regions",
        risk=ActionRisk.MEDIUM, cooldown_s=30,
    ),
    DefenseAction(
        id=13, name="enable_mfa",
        description="Enforce multi-factor authentication",
        risk=ActionRisk.MEDIUM, requires_approval=True,
        cooldown_s=300, reversible=True,
    ),
    DefenseAction(
        id=14, name="quarantine_file",
        description="Quarantine suspicious file",
        risk=ActionRisk.HIGH, requires_target=True, cooldown_s=5,
    ),
    DefenseAction(
        id=15, name="restore_from_backup",
        description="Restore compromised files from backup",
        risk=ActionRisk.CRITICAL, requires_approval=True,
        cooldown_s=300, reversible=False,
    ),
]


def get_action_by_name(name: str) -> DefenseAction | None:
    """Look up a defense action by name."""
    for action in DEFENSE_ACTIONS:
        if action.name == name:
            return action
    return None


def get_action_by_id(action_id: int) -> DefenseAction | None:
    """Look up a defense action by ID."""
    if 0 <= action_id < len(DEFENSE_ACTIONS):
        return DEFENSE_ACTIONS[action_id]
    return None
