"""
CyberGAN — Analysis: Risk Scorer
Computes composite risk scores combining severity, confidence, and impact.
"""

from __future__ import annotations

from dataclasses import dataclass

from cybergan.config import RiskScorerConfig


SEVERITY_SCORES = {
    "critical": 100,
    "high": 75,
    "medium": 50,
    "low": 25,
    "info": 10,
    "warning": 50,
}


@dataclass
class RiskAssessment:
    """Risk assessment for a security event."""
    score: float           # 0-100 composite risk score
    level: str             # critical, high, medium, low
    severity_score: float
    confidence_score: float
    impact_score: float
    should_act: bool       # Whether autonomous action is warranted
    should_alert: bool     # Whether to send alert


class RiskScorer:
    """
    Computes composite risk scores for security events.

    Score = severity_weight * severity + confidence_weight * confidence + impact_weight * impact

    The score determines the response urgency and whether to act autonomously.
    """

    # Impact factors by event type
    IMPACT_FACTORS = {
        "sql_injection": 90,
        "rce": 100,
        "command_injection": 100,
        "reverse_shell": 100,
        "ransomware": 100,
        "data_exfiltration": 95,
        "privilege_escalation": 90,
        "cryptominer": 70,
        "cryptojacking": 70,
        "web_shell": 90,
        "backdoor": 90,
        "rootkit": 95,
        "ssh_key_injection": 85,
        "brute_force_detected": 75,
        "syn_flood": 80,
        "ddos_indicator": 85,
        "xss": 70,
        "csrf": 60,
        "lfi": 80,
        "rfi": 85,
        "xxe": 80,
        "ssrf": 75,
        "directory_traversal": 70,
        "insecure_deserialization": 80,
        "session_hijack": 75,
        "cookie_poisoning": 60,
        "api_abuse": 55,
        "port_scan": 30,
        "connection_spike": 60,
        "network_spike": 65,
        "log_tampering": 80,
        "suspicious_outbound": 70,
        "suspicious_process": 60,
        "crontab_change": 65,
        "suid_set": 85,
        "file_created": 40,
        "file_modified": 50,
        "file_deleted": 60,
        "permission_changed": 55,
        "cpu_alert": 45,
        "memory_alert": 45,
        "disk_alert": 40,
        "failed_login": 20,
    }

    def __init__(self, config: RiskScorerConfig):
        self.config = config

    def score(
        self,
        event_type: str,
        severity: str = "medium",
        confidence: float = 0.7,
        context: dict | None = None,
    ) -> RiskAssessment:
        """
        Compute risk score for a security event.

        Args:
            event_type: Type of security event
            severity: Severity level string
            confidence: Detection confidence (0-1)
            context: Additional context (e.g., is_advanced_threat)

        Returns:
            RiskAssessment with composite score and recommended actions
        """
        # Severity component
        severity_raw = SEVERITY_SCORES.get(severity, 50)
        severity_score = severity_raw

        # Confidence component
        confidence_score = confidence * 100

        # Impact component
        impact_raw = self.IMPACT_FACTORS.get(event_type, 50)
        impact_score = impact_raw

        # Context modifiers
        if context:
            # APT / advanced threat escalation
            if context.get("is_advanced"):
                impact_score = min(impact_score * 1.3, 100)
            # Multiple stages hit
            stages = context.get("stages_hit", 0)
            if stages >= 3:
                impact_score = min(impact_score * 1.2, 100)

        # Composite score
        score = (
            self.config.severity_weight * severity_score
            + self.config.confidence_weight * confidence_score
            + self.config.impact_weight * impact_score
        )

        # Determine level
        if score >= self.config.critical_threshold:
            level = "critical"
        elif score >= self.config.high_threshold:
            level = "high"
        elif score >= self.config.medium_threshold:
            level = "medium"
        else:
            level = "low"

        return RiskAssessment(
            score=round(score, 1),
            level=level,
            severity_score=severity_score,
            confidence_score=confidence_score,
            impact_score=impact_score,
            should_act=score >= self.config.high_threshold,
            should_alert=score >= self.config.medium_threshold,
        )
