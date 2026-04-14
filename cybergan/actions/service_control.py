"""
CyberGAN — Actions: Service Control
Service restart, isolation, and management via systemd or Docker.
"""

from __future__ import annotations

import asyncio

import structlog

from cybergan.config import ServiceControlConfig

logger = structlog.get_logger(__name__)


class ServiceController:
    """Manage services via systemd or Docker."""

    def __init__(self, config: ServiceControlConfig):
        self.config = config
        self._restart_counts: dict[str, int] = {}

    async def restart_service(self, service_name: str) -> bool:
        """Restart a systemd service."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "restart", service_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info("service_control.restarted", service=service_name)
                return True
            else:
                logger.error("service_control.restart_failed",
                             service=service_name, error=stderr.decode())
                return False
        except Exception as e:
            logger.error("service_control.error", error=str(e))
            return False

    async def stop_service(self, service_name: str) -> bool:
        """Stop a service."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "stop", service_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            logger.info("service_control.stopped", service=service_name)
            return proc.returncode == 0
        except Exception as e:
            logger.error("service_control.error", error=str(e))
            return False

    async def get_service_status(self, service_name: str) -> str:
        """Check service status."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "is-active", service_name,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()
        except Exception:
            return "unknown"
