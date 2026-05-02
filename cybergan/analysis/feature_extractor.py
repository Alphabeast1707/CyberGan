"""
CyberGAN — Analysis: Feature Extractor
Converts raw security events into observation vectors for the RL brain.
Maintains a sliding window of recent events and computes a fixed-size
state representation.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


# Feature vector dimensions
NUM_ATTACK_TYPES = 35       # From AttackType enum
NUM_SEVERITY_LEVELS = 5     # critical, high, medium, low, info
NUM_SYSTEM_FEATURES = 10    # CPU, RAM, disk, net, connections, etc.
NUM_TREND_FEATURES = 10     # Rate-of-change features
NUM_IP_FEATURES = 5         # Top attacker IP stats
OBSERVATION_DIM = NUM_ATTACK_TYPES + NUM_SEVERITY_LEVELS + NUM_SYSTEM_FEATURES + NUM_TREND_FEATURES + NUM_IP_FEATURES


@dataclass
class SecurityState:
    """Current security posture of the system."""
    timestamp: float = 0.0

    # Event counts by type (sliding window)
    attack_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)

    # System metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_mbps: float = 0.0
    active_connections: int = 0

    # Threat summary
    total_events: int = 0
    critical_events: int = 0
    unique_attackers: int = 0
    top_attack_type: str = ""
    risk_score: float = 0.0

    # RL observation vector
    observation: np.ndarray = field(default_factory=lambda: np.zeros(OBSERVATION_DIM, dtype=np.float32))


class FeatureExtractor:
    """
    Transforms raw security events into a fixed-size observation vector
    for the RL policy network.

    Maintains a sliding window of events and computes:
    - Attack type distribution (how many of each attack in window)
    - Severity distribution
    - System health metrics (normalized)
    - Trend features (rate of change)
    - Top attacker information
    """

    def __init__(self, window_s: int = 60):
        self.window_s = window_s
        self._events: deque = deque()
        self._attack_history: deque = deque()
        self._ip_counter: dict[str, int] = defaultdict(int)
        self._last_state: SecurityState = SecurityState()

        # System metrics (updated by metrics events)
        self._cpu = 0.0
        self._memory = 0.0
        self._disk = 0.0
        self._network_mbps = 0.0
        self._connections = 0

    def add_event(self, event) -> None:
        """Add a security event to the window."""
        now = time.time()
        self._events.append((now, event))
        self._prune_old_events()

        # Track source IPs
        source_ip = getattr(event, "source_ip", "")
        if source_ip:
            self._ip_counter[source_ip] += 1

    def update_metrics(self, metrics) -> None:
        """Update system metrics from a SystemMetrics snapshot."""
        self._cpu = getattr(metrics, "cpu_percent", 0)
        self._memory = getattr(metrics, "memory_percent", 0)
        self._disk = getattr(metrics, "disk_percent", 0)
        self._connections = getattr(metrics, "network_connections", 0)

    def extract(self) -> SecurityState:
        """Extract the current state as a SecurityState with observation vector."""
        self._prune_old_events()

        state = SecurityState(timestamp=time.time())

        # Count events by type and severity
        attack_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for _, event in self._events:
            event_type = getattr(event, "event_type", "unknown")
            severity = getattr(event, "severity", "info")
            attack_counts[event_type] += 1
            severity_counts[severity] += 1

        state.attack_counts = dict(attack_counts)
        state.severity_counts = dict(severity_counts)
        state.total_events = len(self._events)
        state.critical_events = severity_counts.get("critical", 0)
        state.unique_attackers = len(self._ip_counter)

        # System metrics
        state.cpu_percent = self._cpu
        state.memory_percent = self._memory
        state.disk_percent = self._disk
        state.network_mbps = self._network_mbps
        state.active_connections = self._connections

        # Top attack type
        if attack_counts:
            state.top_attack_type = max(attack_counts, key=attack_counts.get)

        # Build observation vector
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        idx = 0

        # Attack type counts (normalized by window size)
        attack_type_names = [
            "sql_injection", "xss", "csrf", "rce", "lfi", "rfi",
            "command_injection", "directory_traversal", "xxe", "ssrf",
            "insecure_deserialization", "brute_force", "session_hijack",
            "cookie_poisoning", "api_abuse", "port_scan", "syn_flood",
            "udp_flood", "dns_poisoning", "arp_spoofing", "cryptojacking",
            "web_shell", "backdoor", "reverse_shell", "privilege_escalation",
            "data_exfiltration", "log_tampering", "ssh_key_injection",
            "ransomware", "rootkit", "failed_login", "brute_force_detected",
            "suspicious_process", "high_cpu", "connection_spike",
        ]
        max_events = max(state.total_events, 1)
        for i, atype in enumerate(attack_type_names[:NUM_ATTACK_TYPES]):
            obs[idx + i] = attack_counts.get(atype, 0) / max_events

        idx += NUM_ATTACK_TYPES

        # Severity distribution
        sev_names = ["critical", "high", "medium", "low", "info"]
        for i, sev in enumerate(sev_names):
            obs[idx + i] = severity_counts.get(sev, 0) / max_events

        idx += NUM_SEVERITY_LEVELS

        # System metrics (normalized to 0-1)
        obs[idx] = self._cpu / 100.0
        obs[idx + 1] = self._memory / 100.0
        obs[idx + 2] = self._disk / 100.0
        obs[idx + 3] = min(self._network_mbps / 1000.0, 1.0)
        obs[idx + 4] = min(self._connections / 1000.0, 1.0)
        obs[idx + 5] = min(state.total_events / 100.0, 1.0)
        obs[idx + 6] = min(state.critical_events / 10.0, 1.0)
        obs[idx + 7] = min(state.unique_attackers / 50.0, 1.0)
        obs[idx + 8] = 1.0 if state.critical_events > 0 else 0.0  # Under active attack
        obs[idx + 9] = min(len(self._events) / 50.0, 1.0)  # Event density

        idx += NUM_SYSTEM_FEATURES

        # Trend features (rate of change from last state)
        if self._last_state.timestamp > 0:
            dt = max(state.timestamp - self._last_state.timestamp, 0.1)
            obs[idx] = (state.total_events - self._last_state.total_events) / dt  # Events/sec
            obs[idx + 1] = (state.critical_events - self._last_state.critical_events) / dt
            obs[idx + 2] = (self._cpu - self._last_state.cpu_percent) / 100.0  # CPU delta
            obs[idx + 3] = (self._memory - self._last_state.memory_percent) / 100.0
            obs[idx + 4] = (self._connections - self._last_state.active_connections) / max(self._connections, 1)
        # idx 5-9 reserved for future trend features

        idx += NUM_TREND_FEATURES

        # Top attacker IP features
        if self._ip_counter:
            sorted_ips = sorted(self._ip_counter.values(), reverse=True)
            for i in range(min(NUM_IP_FEATURES, len(sorted_ips))):
                obs[idx + i] = min(sorted_ips[i] / 100.0, 1.0)

        state.observation = obs
        self._last_state = state
        self._attack_history.append(state.total_events)
        if len(self._attack_history) > 100:
            self._attack_history.popleft()

        return state

    def _prune_old_events(self):
        """Remove events older than the window."""
        cutoff = time.time() - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def get_observation_dim(self) -> int:
        """Return the observation vector dimension."""
        return OBSERVATION_DIM
