"""
CyberGAN — Analysis: Threat Classifier
Maps detected events to MITRE ATT&CK technique IDs and kill chain stages.
Tracks attack progression through the kill chain.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import structlog

from cybergan.config import ThreatClassifierConfig

logger = structlog.get_logger(__name__)


# MITRE ATT&CK Technique Mapping
ATTACK_TO_MITRE: dict[str, list[str]] = {
    # Web Application
    "sql_injection": ["T1190", "T1059"],
    "xss": ["T1059.007", "T1189"],
    "csrf": ["T1190"],
    "rce": ["T1059", "T1203"],
    "lfi": ["T1005", "T1083"],
    "rfi": ["T1505.003", "T1059"],
    "command_injection": ["T1059", "T1059.004"],
    "directory_traversal": ["T1083", "T1005"],
    "xxe": ["T1059", "T1190"],
    "ssrf": ["T1090", "T1190"],
    "insecure_deserialization": ["T1059", "T1203"],
    "web_shell": ["T1505.003"],
    "api_abuse": ["T1190", "T1498"],

    # Authentication
    "brute_force": ["T1110", "T1110.001"],
    "brute_force_detected": ["T1110"],
    "failed_login": ["T1110"],
    "credential_stuffing": ["T1110.004"],
    "session_hijack": ["T1563", "T1539"],
    "cookie_poisoning": ["T1539"],
    "default_credentials": ["T1078.001"],
    "weak_password": ["T1110"],

    # Network
    "port_scan": ["T1046"],
    "syn_flood": ["T1498", "T1499"],
    "udp_flood": ["T1498", "T1499"],
    "connection_spike": ["T1498"],
    "ddos_indicator": ["T1498", "T1499"],
    "dns_poisoning": ["T1557", "T1584.002"],
    "arp_spoofing": ["T1557.002"],
    "suspicious_outbound": ["T1071", "T1041"],
    "network_spike": ["T1498"],

    # Malware
    "cryptojacking": ["T1496"],
    "ransomware": ["T1486"],
    "rootkit": ["T1014", "T1547"],
    "backdoor": ["T1543", "T1547"],
    "reverse_shell": ["T1059", "T1571"],
    "suspicious_process": ["T1059"],
    "cryptominer": ["T1496"],

    # Privilege & Persistence
    "privilege_escalation": ["T1548", "T1068"],
    "sudo_failure": ["T1548.003"],
    "sudo_command": ["T1548.003"],
    "su_failure": ["T1548.003"],
    "ssh_key_injection": ["T1098.004"],
    "crontab_change": ["T1053.003"],
    "suid_set": ["T1548.001"],

    # Data
    "data_exfiltration": ["T1041", "T1048"],
    "log_tampering": ["T1070", "T1070.002"],

    # File
    "file_created": ["T1105"],
    "file_modified": ["T1565"],
    "file_deleted": ["T1070.004"],
    "permission_changed": ["T1222"],

    # Recon
    "banner_grab": ["T1595.002"],
    "service_enumeration": ["T1046"],

    # System
    "high_cpu": ["T1496"],
    "high_memory": ["T1499"],
    "oom_kill": ["T1499"],
    "cpu_alert": ["T1496"],
    "memory_alert": ["T1499"],
    "disk_alert": ["T1485"],
    "segfault": ["T1203"],
}

# Kill Chain Stages
KILL_CHAIN = [
    "reconnaissance",
    "weaponization",
    "delivery",
    "exploitation",
    "installation",
    "command_and_control",
    "actions_on_objectives",
]

EVENT_TO_KILL_CHAIN: dict[str, str] = {
    "port_scan": "reconnaissance",
    "banner_grab": "reconnaissance",
    "service_enumeration": "reconnaissance",
    "brute_force": "delivery",
    "brute_force_detected": "delivery",
    "sql_injection": "exploitation",
    "xss": "exploitation",
    "rce": "exploitation",
    "command_injection": "exploitation",
    "lfi": "exploitation",
    "rfi": "exploitation",
    "xxe": "exploitation",
    "ssrf": "exploitation",
    "directory_traversal": "exploitation",
    "privilege_escalation": "exploitation",
    "web_shell": "installation",
    "backdoor": "installation",
    "ssh_key_injection": "installation",
    "crontab_change": "installation",
    "reverse_shell": "command_and_control",
    "suspicious_outbound": "command_and_control",
    "cryptominer": "command_and_control",
    "data_exfiltration": "actions_on_objectives",
    "ransomware": "actions_on_objectives",
    "log_tampering": "actions_on_objectives",
    "cryptojacking": "actions_on_objectives",
}


@dataclass
class ThreatClassification:
    """Classification result for a security event."""
    event_type: str
    mitre_techniques: list[str]
    kill_chain_stage: str
    is_attack: bool
    confidence: float
    details: dict = field(default_factory=dict)


@dataclass
class AttackProgression:
    """Tracks an attacker's progression through the kill chain."""
    source_ip: str
    stages_hit: set[str] = field(default_factory=set)
    technique_ids: set[str] = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0
    event_count: int = 0

    @property
    def furthest_stage(self) -> str:
        """Return the furthest kill chain stage reached."""
        for stage in reversed(KILL_CHAIN):
            if stage in self.stages_hit:
                return stage
        return "unknown"

    @property
    def is_advanced(self) -> bool:
        """Is this an advanced persistent threat (multiple stages)?"""
        return len(self.stages_hit) >= 3


class ThreatClassifier:
    """
    Classifies security events into MITRE ATT&CK techniques
    and tracks attack progression through the kill chain.
    """

    def __init__(self, config: ThreatClassifierConfig):
        self.config = config
        self._progressions: dict[str, AttackProgression] = defaultdict(
            lambda: AttackProgression(source_ip="")
        )

    def classify(self, event) -> ThreatClassification:
        """Classify a security event."""
        event_type = getattr(event, "event_type", "unknown")
        source_ip = getattr(event, "source_ip", "")

        mitre_techniques = ATTACK_TO_MITRE.get(event_type, [])
        kill_chain_stage = EVENT_TO_KILL_CHAIN.get(event_type, "unknown")
        is_attack = bool(mitre_techniques)
        confidence = 0.9 if mitre_techniques else 0.3

        # Track progression
        if source_ip and is_attack:
            prog = self._progressions[source_ip]
            prog.source_ip = source_ip
            prog.stages_hit.add(kill_chain_stage)
            prog.technique_ids.update(mitre_techniques)
            if prog.first_seen == 0:
                prog.first_seen = time.time()
            prog.last_seen = time.time()
            prog.event_count += 1

        return ThreatClassification(
            event_type=event_type,
            mitre_techniques=mitre_techniques,
            kill_chain_stage=kill_chain_stage,
            is_attack=is_attack,
            confidence=confidence,
            details={
                "source_ip": source_ip,
                "progression": self.get_progression(source_ip) if source_ip else None,
            },
        )

    def get_progression(self, source_ip: str) -> dict | None:
        """Get attack progression for an IP."""
        prog = self._progressions.get(source_ip)
        if not prog:
            return None
        return {
            "source_ip": source_ip,
            "stages_hit": list(prog.stages_hit),
            "furthest_stage": prog.furthest_stage,
            "techniques": list(prog.technique_ids),
            "event_count": prog.event_count,
            "is_advanced": prog.is_advanced,
            "duration_s": prog.last_seen - prog.first_seen if prog.first_seen else 0,
        }

    def get_active_threats(self, max_age_s: int = 3600) -> list[dict]:
        """Get all active threat progressions."""
        now = time.time()
        threats = []
        for ip, prog in self._progressions.items():
            if now - prog.last_seen < max_age_s:
                threats.append(self.get_progression(ip))
        return [t for t in threats if t is not None]
