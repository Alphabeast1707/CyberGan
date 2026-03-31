"""
CyberGAN — Actions: Firewall Manager
Manages iptables/nftables/ufw rules for IP blocking, port blocking,
rate limiting, and traffic filtering.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field

import structlog

from cybergan.config import FirewallConfig

logger = structlog.get_logger(__name__)


@dataclass
class FirewallRule:
    """A firewall rule managed by CyberGAN."""
    ip: str
    action: str  # DROP, REJECT, ACCEPT
    chain: str = "INPUT"
    protocol: str = ""
    port: int = 0
    created_at: float = 0.0
    expires_at: float = 0.0
    reason: str = ""
    rule_id: str = ""


class FirewallManager:
    """
    Manages firewall rules via iptables/nftables.

    Operations:
    - Block/unblock IP addresses
    - Block/unblock ports
    - Rate limiting rules
    - Custom chain management
    - Auto-cleanup of expired rules
    """

    def __init__(self, config: FirewallConfig):
        self.config = config
        self._rules: dict[str, FirewallRule] = {}
        self._blocked_ips: set[str] = set()
        self._initialized = False

    async def initialize(self):
        """Initialize the firewall. Detects backend and permission level."""
        import platform
        self._is_macos = platform.system() == "Darwin"

        if self.config.backend == "pf" or self._is_macos:
            # macOS pf backend
            success = await self._run("pfctl -s rules 2>/dev/null | head -1")
            if success:
                self._initialized = True
                logger.info("firewall.initialized", backend="pf", note="macOS pf active")
            else:
                logger.info("firewall.dry_run",
                            msg="pf requires root. Running in tracking-only mode.",
                            note="IPs are tracked in memory, real blocking needs sudo")
        else:
            # Linux iptables
            try:
                await self._run(f"iptables -N {self.config.chain} 2>/dev/null || true")
                await self._run(
                    f"iptables -C INPUT -j {self.config.chain} 2>/dev/null || "
                    f"iptables -I INPUT 1 -j {self.config.chain}"
                )
                self._initialized = True
                logger.info("firewall.initialized", chain=self.config.chain, backend="iptables")
            except Exception as e:
                logger.warning("firewall.init_failed", error=str(e),
                               msg="iptables requires root. Running in dry-run mode.")

    async def block_ip(self, ip: str, reason: str = "", duration_s: int = 0) -> bool:
        """Block an IP address. Uses pf on macOS, iptables on Linux."""
        if ip in self._blocked_ips:
            return True
        if len(self._blocked_ips) >= self.config.max_blocked_ips:
            logger.warning("firewall.max_blocked", count=len(self._blocked_ips))
            return False

        now = time.time()
        expires = now + (duration_s or self.config.auto_cleanup_hours * 3600)

        rule = FirewallRule(
            ip=ip, action="DROP", chain=self.config.chain,
            created_at=now, expires_at=expires, reason=reason,
            rule_id=f"block_{ip}",
        )

        # Try actual firewall — different commands per OS
        if getattr(self, "_is_macos", False) or self.config.backend == "pf":
            # macOS: use pfctl (requires root)
            success = await self._run(
                f'echo "block drop from {ip} to any" | pfctl -a cybergan -f - 2>/dev/null'
            )
        else:
            # Linux: iptables
            success = await self._run(
                f"iptables -A {self.config.chain} -s {ip} -j DROP"
            )

        # Always track in memory (dry-run fallback)
        self._rules[rule.rule_id] = rule
        self._blocked_ips.add(ip)

        if success:
            logger.info("firewall.ip_blocked", ip=ip, reason=reason, duration_s=duration_s,
                        backend=self.config.backend)
        else:
            logger.info("firewall.ip_tracked", ip=ip, reason=reason,
                        note="Tracked in memory (no root). On Linux with root, would be blocked by iptables.")
        return True

    async def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address."""
        success = await self._run(
            f"iptables -D {self.config.chain} -s {ip} -j DROP 2>/dev/null"
        )
        self._blocked_ips.discard(ip)
        self._rules.pop(f"block_{ip}", None)
        logger.info("firewall.ip_unblocked", ip=ip)
        return True

    async def rate_limit_ip(self, ip: str, limit: str = "10/minute") -> bool:
        """Apply rate limiting to an IP."""
        success = await self._run(
            f"iptables -A {self.config.chain} -s {ip} "
            f"-m hashlimit --hashlimit-above {limit} "
            f"--hashlimit-mode srcip --hashlimit-name cybergan_{ip.replace('.', '_')} "
            f"-j DROP"
        )
        logger.info("firewall.rate_limited", ip=ip, limit=limit)
        return True

    async def block_port(self, port: int, protocol: str = "tcp") -> bool:
        """Block a specific port."""
        success = await self._run(
            f"iptables -A {self.config.chain} -p {protocol} --dport {port} -j DROP"
        )
        logger.info("firewall.port_blocked", port=port, protocol=protocol)
        return True

    async def cleanup_expired(self):
        """Remove expired firewall rules."""
        now = time.time()
        expired = [
            rule_id for rule_id, rule in self._rules.items()
            if rule.expires_at > 0 and now > rule.expires_at
        ]
        for rule_id in expired:
            rule = self._rules[rule_id]
            await self.unblock_ip(rule.ip)
            logger.info("firewall.rule_expired", ip=rule.ip, rule_id=rule_id)

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked_ips

    def get_blocked_ips(self) -> list[str]:
        return list(self._blocked_ips)

    def get_rules(self) -> list[dict]:
        return [
            {
                "ip": r.ip, "action": r.action, "reason": r.reason,
                "created_at": r.created_at, "expires_at": r.expires_at,
            }
            for r in self._rules.values()
        ]

    async def _run(self, cmd: str) -> bool:
        """Execute a firewall command."""
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = stderr.decode().strip()
                if "Permission denied" in err or "Operation not permitted" in err:
                    logger.debug("firewall.no_permission", cmd=cmd[:80])
                    return False
                logger.debug("firewall.cmd_error", cmd=cmd[:80], error=err)
                return False
            return True
        except Exception as e:
            logger.debug("firewall.cmd_exception", error=str(e))
            return False
