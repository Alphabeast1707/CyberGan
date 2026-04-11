"""
CyberGAN — Actions: Process Control
Process kill, containment, and resource limiting.
"""

from __future__ import annotations

import structlog

from cybergan.config import ProcessControlConfig

logger = structlog.get_logger(__name__)


class ProcessController:
    """Kill, freeze, and resource-limit suspicious processes."""

    def __init__(self, config: ProcessControlConfig):
        self.config = config
        self._killed_pids: list[dict] = []

    async def kill_process(self, pid: int, reason: str = "") -> bool:
        """Kill a suspicious process by PID."""
        try:
            import psutil
            proc = psutil.Process(pid)
            name = proc.name()
            cmdline = " ".join(proc.cmdline()[:5])

            proc.kill()
            self._killed_pids.append({
                "pid": pid, "name": name, "cmdline": cmdline,
                "reason": reason, "success": True,
            })
            logger.info("process_control.killed", pid=pid, name=name, reason=reason)
            return True
        except Exception as e:
            logger.error("process_control.kill_failed", pid=pid, error=str(e))
            return False

    async def suspend_process(self, pid: int) -> bool:
        """Suspend (freeze) a process."""
        try:
            import psutil
            proc = psutil.Process(pid)
            proc.suspend()
            logger.info("process_control.suspended", pid=pid, name=proc.name())
            return True
        except Exception as e:
            logger.error("process_control.suspend_failed", pid=pid, error=str(e))
            return False

    def get_killed(self) -> list[dict]:
        return self._killed_pids
