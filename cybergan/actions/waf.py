"""
CyberGAN — Actions: WAF (Web Application Firewall)
Manages WAF rules for SQLi, XSS, CSRF protection.
In production, this generates rules for ModSecurity, nginx, or custom middleware.
"""

from __future__ import annotations

import structlog

from cybergan.config import WAFConfig

logger = structlog.get_logger(__name__)


class WAFManager:
    """Web Application Firewall rule manager."""

    def __init__(self, config: WAFConfig):
        self.config = config
        self._active_rules: list[dict] = []

    async def deploy_rule(self, rule_type: str, pattern: str = "", target: str = "") -> bool:
        """Deploy a WAF rule."""
        rule = {"type": rule_type, "pattern": pattern, "target": target, "active": True}
        self._active_rules.append(rule)
        logger.info("waf.rule_deployed", type=rule_type, target=target)
        return True

    def get_active_rules(self) -> list[dict]:
        return self._active_rules
