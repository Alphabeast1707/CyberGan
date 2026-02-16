"""
CyberGAN — Perception: Process Monitor
Real-time process monitoring for security anomalies.
Detects cryptojacking, reverse shells, privilege escalation,
and suspicious process behavior.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil
import structlog

from cybergan.config import ProcessMonitorConfig

logger = structlog.get_logger(__name__)


@dataclass
class ProcessEvent:
    """A process-related security event."""
    timestamp: float
    event_type: str  # cryptominer, reverse_shell, high_cpu, high_memory, suspicious_process
    severity: str = "warning"
    pid: int = 0
    process_name: str = ""
    username: str = ""
    command_line: str = ""
    details: dict = field(default_factory=dict)


class ProcessMonitor:
    """
    Real-time process monitor.

    Periodically scans running processes to detect:
    - Cryptominers (high CPU + known process names/patterns)
    - Reverse shells (suspicious network-connected shells)
    - Privilege escalation (new root processes from unexpected sources)
    - Resource abuse (abnormal CPU/memory consumption)
    - Suspicious process names or command lines
    """

    REVERSE_SHELL_PATTERNS = [
        re.compile(r"bash\s+-i\s+>[\s&]*/dev/tcp"),
        re.compile(r"nc\s+-[^\s]*e\s+/bin/(?:ba)?sh"),
        re.compile(r"python[23]?\s+-c\s+.*(?:socket|subprocess)"),
        re.compile(r"perl\s+-e\s+.*(?:socket|exec)"),
        re.compile(r"php\s+-r\s+.*(?:fsockopen|exec)"),
        re.compile(r"ruby\s+-r\s*socket"),
        re.compile(r"socat\s+.*exec:"),
        re.compile(r"mkfifo\s+.*nc\b"),
    ]

    CRYPTOMINER_INDICATORS = [
        re.compile(r"stratum\+tcp://", re.IGNORECASE),
        re.compile(r"--algo\s+(?:randomx|cryptonight|equihash|ethash)", re.IGNORECASE),
        re.compile(r"--pool\s+", re.IGNORECASE),
        re.compile(r"-o\s+stratum", re.IGNORECASE),
    ]

    def __init__(self, config: ProcessMonitorConfig):
        self.config = config
        self._known_pids: set[int] = set()
        self._process_baseline: dict[int, dict] = {}
        self._running = False

    async def start(self, event_queue: asyncio.Queue):
        """Start monitoring processes."""
        self._running = True
        logger.info("process_monitor.start")

        # Build initial process baseline
        self._build_baseline()

        while self._running:
            try:
                events = self._scan_processes()
                for event in events:
                    await event_queue.put(event)
            except Exception as e:
                logger.error("process_monitor.error", error=str(e))

            await asyncio.sleep(self.config.poll_interval_s)

    def stop(self):
        self._running = False

    def _build_baseline(self):
        """Build initial process baseline."""
        for proc in psutil.process_iter(["pid", "name", "username", "cpu_percent"]):
            try:
                self._known_pids.add(proc.info["pid"])
                self._process_baseline[proc.info["pid"]] = {
                    "name": proc.info["name"],
                    "username": proc.info["username"],
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def _scan_processes(self) -> list[ProcessEvent]:
        """Scan all running processes for anomalies."""
        events = []
        now = time.time()
        current_pids: set[int] = set()

        for proc in psutil.process_iter([
            "pid", "name", "username", "cpu_percent", "memory_info",
            "cmdline", "create_time",
        ]):
            try:
                info = proc.info
                pid = info["pid"]
                current_pids.add(pid)
                name = info.get("name", "") or ""
                username = info.get("username", "") or ""
                cmdline = " ".join(info.get("cmdline") or [])
                cpu = info.get("cpu_percent", 0) or 0
                mem_info = info.get("memory_info")
                mem_mb = (mem_info.rss / 1024 / 1024) if mem_info else 0

                # ── Check suspicious process names ──
                name_lower = name.lower()
                if name_lower in [n.lower() for n in self.config.suspicious_process_names]:
                    events.append(ProcessEvent(
                        timestamp=now,
                        event_type="suspicious_process",
                        severity="high",
                        pid=pid,
                        process_name=name,
                        username=username,
                        command_line=cmdline,
                        details={"reason": f"Known suspicious process: {name}"},
                    ))

                # ── Check for reverse shells ──
                if self.config.detect_reverse_shells and cmdline:
                    for pattern in self.REVERSE_SHELL_PATTERNS:
                        if pattern.search(cmdline):
                            events.append(ProcessEvent(
                                timestamp=now,
                                event_type="reverse_shell",
                                severity="critical",
                                pid=pid,
                                process_name=name,
                                username=username,
                                command_line=cmdline,
                                details={"pattern": pattern.pattern},
                            ))
                            break

                # ── Check for cryptominers ──
                if self.config.detect_cryptominers:
                    is_miner = False
                    # High CPU + known miner name
                    if cpu > 80 and name_lower in [
                        "xmrig", "minerd", "cpuminer", "cryptonight",
                        "minergate", "nicehash", "ethminer",
                    ]:
                        is_miner = True

                    # Check command line for mining indicators
                    if cmdline:
                        for pattern in self.CRYPTOMINER_INDICATORS:
                            if pattern.search(cmdline):
                                is_miner = True
                                break

                    if is_miner:
                        events.append(ProcessEvent(
                            timestamp=now,
                            event_type="cryptominer",
                            severity="critical",
                            pid=pid,
                            process_name=name,
                            username=username,
                            command_line=cmdline,
                            details={"cpu_percent": cpu},
                        ))

                # ── High CPU usage (non-miner) ──
                if cpu > self.config.cpu_threshold_percent:
                    events.append(ProcessEvent(
                        timestamp=now,
                        event_type="high_cpu",
                        severity="warning",
                        pid=pid,
                        process_name=name,
                        username=username,
                        details={"cpu_percent": cpu, "threshold": self.config.cpu_threshold_percent},
                    ))

                # ── High memory usage ──
                if mem_mb > self.config.memory_threshold_mb:
                    events.append(ProcessEvent(
                        timestamp=now,
                        event_type="high_memory",
                        severity="warning",
                        pid=pid,
                        process_name=name,
                        username=username,
                        details={"memory_mb": round(mem_mb, 1), "threshold": self.config.memory_threshold_mb},
                    ))

                # ── New process with network connections (potential C2) ──
                if pid not in self._known_pids:
                    try:
                        connections = proc.net_connections(kind="inet")
                    except (AttributeError, psutil.AccessDenied, PermissionError, OSError):
                        try:
                            connections = proc.connections(kind="inet")
                        except Exception:
                            connections = []
                    if connections and username == "root":
                        events.append(ProcessEvent(
                            timestamp=now,
                            event_type="new_root_network_process",
                            severity="warning",
                            pid=pid,
                            process_name=name,
                            username=username,
                            command_line=cmdline,
                            details={"connections": len(connections)},
                        ))

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        self._known_pids = current_pids
        return events
