"""
CyberGAN — Actions: Honeypot
Deploy lightweight honeypot services on unused ports.
"""

from __future__ import annotations

import asyncio

import structlog

from cybergan.config import HoneypotConfig

logger = structlog.get_logger(__name__)


class HoneypotManager:
    """Deploy and manage honeypot services."""

    def __init__(self, config: HoneypotConfig):
        self.config = config
        self._active_honeypots: dict[int, dict] = {}

    async def deploy(self, port: int) -> bool:
        """Deploy a honeypot on a port."""
        if not self.config.enabled:
            return False

        self._active_honeypots[port] = {"port": port, "interactions": 0}
        logger.info("honeypot.deployed", port=port)
        return True

    async def deploy_all(self):
        """Deploy honeypots on all configured ports."""
        for port in self.config.ports:
            await self.deploy(port)

    def get_active(self) -> list[dict]:
        return list(self._active_honeypots.values())
