"""
CyberGAN — Actions: IP Blocker
IP blocking with TTL-based auto-unblock, tracking block history,
and progressive blocking (temporary → permanent).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import structlog

from cybergan.config import IPBlockerConfig
from cybergan.actions.firewall import FirewallManager

logger = structlog.get_logger(__name__)


@dataclass
class BlockRecord:
    """Record of an IP block action."""
    ip: str
    blocked_at: float
    expires_at: float
    reason: str
    block_count: int = 1
    is_permanent: bool = False


class IPBlocker:
    """
    IP blocking manager with progressive escalation.

    First offenses get temporary blocks. Repeat offenders are
    permanently blocked after exceeding the threshold.
    """

    def __init__(self, config: IPBlockerConfig, firewall: FirewallManager):
        self.config = config
        self.firewall = firewall
        self._blocks: dict[str, BlockRecord] = {}
        self._block_counts: dict[str, int] = defaultdict(int)

    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        if ip in self.config.whitelisted_ips:
            return True
        # TODO: CIDR matching
        return False

    async def block(self, ip: str, reason: str = "", duration_s: int = 0) -> bool:
        """Block an IP address."""
        if self.is_whitelisted(ip):
            logger.info("ip_blocker.whitelisted", ip=ip)
            return False

        self._block_counts[ip] += 1
        count = self._block_counts[ip]

        # Progressive blocking
        if count >= self.config.permanent_block_threshold:
            is_permanent = True
            actual_duration = 0  # 0 = permanent
        else:
            is_permanent = False
            actual_duration = duration_s or self.config.auto_block_duration_s

        now = time.time()
        record = BlockRecord(
            ip=ip,
            blocked_at=now,
            expires_at=now + actual_duration if actual_duration else 0,
            reason=reason,
            block_count=count,
            is_permanent=is_permanent,
        )
        self._blocks[ip] = record

        await self.firewall.block_ip(ip, reason=reason, duration_s=actual_duration)

        logger.info(
            "ip_blocker.blocked",
            ip=ip, reason=reason, count=count,
            permanent=is_permanent,
            duration_s=actual_duration,
        )
        return True

    async def unblock(self, ip: str) -> bool:
        """Manually unblock an IP."""
        self._blocks.pop(ip, None)
        return await self.firewall.unblock_ip(ip)

    def get_blocked(self) -> list[dict]:
        """Get all currently blocked IPs."""
        return [
            {
                "ip": r.ip,
                "reason": r.reason,
                "blocked_at": r.blocked_at,
                "block_count": r.block_count,
                "permanent": r.is_permanent,
            }
            for r in self._blocks.values()
        ]
