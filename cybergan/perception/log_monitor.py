"""
CyberGAN — Perception: Log Monitor
Real-time log file monitoring for security events.
Parses syslog, auth.log, kern.log, and application logs
to detect brute force, privilege escalation, suspicious commands, etc.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import structlog

from cybergan.config import LogMonitorConfig

logger = structlog.get_logger(__name__)


@dataclass
class LogEvent:
    """A parsed security-relevant log event."""
    timestamp: float
    source_file: str
    raw_line: str
    event_type: str  # e.g., "failed_login", "sudo_command", "process_start"
    severity: str = "info"  # info, warning, high, critical
    source_ip: str = ""
    username: str = ""
    command: str = ""
    details: dict = field(default_factory=dict)


# ── Log Pattern Definitions ──────────────────────────────────

LOG_PATTERNS = {
    "failed_login_ssh": {
        "pattern": re.compile(
            r"Failed password for (?:invalid user )?(\S+) from (\S+) port (\d+)"
        ),
        "event_type": "failed_login",
        "severity": "warning",
        "extract": lambda m: {"username": m.group(1), "source_ip": m.group(2), "port": m.group(3)},
    },
    "invalid_user": {
        "pattern": re.compile(r"Invalid user (\S+) from (\S+)"),
        "event_type": "failed_login",
        "severity": "warning",
        "extract": lambda m: {"username": m.group(1), "source_ip": m.group(2)},
    },
    "accepted_login": {
        "pattern": re.compile(r"Accepted (\S+) for (\S+) from (\S+) port (\d+)"),
        "event_type": "successful_login",
        "severity": "info",
        "extract": lambda m: {
            "auth_method": m.group(1), "username": m.group(2),
            "source_ip": m.group(3), "port": m.group(4),
        },
    },
    "sudo_command": {
        "pattern": re.compile(r"sudo:\s+(\S+)\s+:\s+.*COMMAND=(.+)$"),
        "event_type": "sudo_command",
        "severity": "info",
        "extract": lambda m: {"username": m.group(1), "command": m.group(2)},
    },
    "sudo_failed": {
        "pattern": re.compile(r"sudo:\s+(\S+)\s+:\s+.*authentication failure"),
        "event_type": "sudo_failure",
        "severity": "high",
        "extract": lambda m: {"username": m.group(1)},
    },
    "su_failed": {
        "pattern": re.compile(r"FAILED su for (\S+) by (\S+)"),
        "event_type": "su_failure",
        "severity": "high",
        "extract": lambda m: {"target_user": m.group(1), "source_user": m.group(2)},
    },
    "session_opened": {
        "pattern": re.compile(r"pam_unix\(.*\):\s+session opened for user (\S+)"),
        "event_type": "session_opened",
        "severity": "info",
        "extract": lambda m: {"username": m.group(1)},
    },
    "connection_closed": {
        "pattern": re.compile(r"Connection closed by (?:authenticating user )?(\S+)? ?(\S+) port"),
        "event_type": "connection_closed",
        "severity": "info",
        "extract": lambda m: {"username": m.group(1) or "", "source_ip": m.group(2)},
    },
    "kernel_segfault": {
        "pattern": re.compile(r"segfault at .* ip .* sp .* error"),
        "event_type": "segfault",
        "severity": "high",
        "extract": lambda m: {},
    },
    "oom_kill": {
        "pattern": re.compile(r"Out of memory: Killed process (\d+) \((\S+)\)"),
        "event_type": "oom_kill",
        "severity": "warning",
        "extract": lambda m: {"pid": m.group(1), "process": m.group(2)},
    },
    "crontab_edit": {
        "pattern": re.compile(r"crontab\[\d+\].*\((\S+)\) (REPLACE|DELETE|LIST)"),
        "event_type": "crontab_change",
        "severity": "warning",
        "extract": lambda m: {"username": m.group(1), "action": m.group(2)},
    },
    "service_start": {
        "pattern": re.compile(r"systemd\[\d+\]:\s+Started (.+)"),
        "event_type": "service_start",
        "severity": "info",
        "extract": lambda m: {"service": m.group(1)},
    },
    "service_stop": {
        "pattern": re.compile(r"systemd\[\d+\]:\s+Stopped (.+)"),
        "event_type": "service_stop",
        "severity": "info",
        "extract": lambda m: {"service": m.group(1)},
    },
}


class BruteForceTracker:
    """Tracks failed login attempts per IP for brute force detection."""

    def __init__(self, threshold: int = 5, window_s: int = 300):
        self.threshold = threshold
        self.window_s = window_s
        self._attempts: dict[str, list[float]] = defaultdict(list)
        self._blocked: set[str] = set()

    def record_attempt(self, ip: str) -> bool:
        """Record a failed attempt. Returns True if threshold exceeded."""
        now = time.time()
        attempts = self._attempts[ip]
        # Prune old attempts outside window
        attempts[:] = [t for t in attempts if now - t < self.window_s]
        attempts.append(now)

        if len(attempts) >= self.threshold and ip not in self._blocked:
            self._blocked.add(ip)
            return True
        return False

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked

    def clear_block(self, ip: str):
        self._blocked.discard(ip)
        self._attempts.pop(ip, None)


class LogMonitor:
    """
    Real-time log file monitor.

    Tails configured log files, parses lines against known patterns,
    and emits security events for the analysis pipeline.
    """

    def __init__(self, config: LogMonitorConfig):
        self.config = config
        self.brute_force_tracker = BruteForceTracker(
            threshold=config.brute_force_threshold,
            window_s=config.brute_force_window_s,
        )
        self._file_positions: dict[str, int] = {}
        self._running = False

    async def start(self, event_queue: asyncio.Queue):
        """Start monitoring all configured log files."""
        self._running = True

        # On macOS, also stream from the system log via `log stream`
        import platform
        if platform.system() == "Darwin" and getattr(self.config, "use_macos_log_stream", True):
            asyncio.ensure_future(self._macos_log_stream(event_queue))
            logger.info("log_monitor.macos_stream_started",
                        msg="Streaming macOS system log in real time")

        logger.info("log_monitor.start", files=self._get_watch_files())

        # Initialize file positions to end of file (don't process history)
        for path in self._get_watch_files():
            if os.path.exists(path):
                self._file_positions[path] = os.path.getsize(path)

        while self._running:
            for path in self._get_watch_files():
                try:
                    events = self._read_new_lines(path)
                    for event in events:
                        await event_queue.put(event)

                        # Check for brute force
                        if event.event_type == "failed_login" and event.source_ip:
                            is_brute = self.brute_force_tracker.record_attempt(event.source_ip)
                            if is_brute:
                                brute_event = LogEvent(
                                    timestamp=time.time(),
                                    source_file=path,
                                    raw_line=f"Brute force detected from {event.source_ip}",
                                    event_type="brute_force_detected",
                                    severity="critical",
                                    source_ip=event.source_ip,
                                    details={"threshold": self.config.brute_force_threshold},
                                )
                                await event_queue.put(brute_event)
                except Exception as e:
                    logger.error("log_monitor.read_error", path=path, error=str(e))

            await asyncio.sleep(self.config.poll_interval_ms / 1000.0)

    async def _macos_log_stream(self, event_queue: asyncio.Queue):
        """
        Stream macOS system logs in real time using `log stream`.
        Captures SSH, sudo, security framework, and login events.
        This is the macOS equivalent of tailing /var/log/auth.log on Linux.
        """
        predicate = (
            'process == "sshd" OR '
            'process == "sudo" OR '
            'process == "su" OR '
            'process == "login" OR '
            'process == "securityd" OR '
            'process == "com.apple.security" OR '
            'process == "authd" OR '
            'category == "network" OR '
            'subsystem == "com.apple.login" OR '
            'subsystem CONTAINS "security" OR '
            'eventMessage CONTAINS "Failed password" OR '
            'eventMessage CONTAINS "authentication failure" OR '
            'eventMessage CONTAINS "Invalid user"'
        )
        cmd = ["log", "stream", "--predicate", predicate, "--style", "compact", "--color", "none"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            logger.info("log_monitor.macos_stream_active", pid=proc.pid)

            async for raw_line in proc.stdout:
                if not self._running:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                event = self._parse_line("macos_log_stream", line)
                if event:
                    await event_queue.put(event)
                    if event.event_type == "failed_login" and event.source_ip:
                        if self.brute_force_tracker.record_attempt(event.source_ip):
                            await event_queue.put(LogEvent(
                                timestamp=time.time(),
                                source_file="macos_log_stream",
                                raw_line=f"Brute force from {event.source_ip}",
                                event_type="brute_force_detected",
                                severity="critical",
                                source_ip=event.source_ip,
                                details={"threshold": self.config.brute_force_threshold},
                            ))
        except FileNotFoundError:
            logger.warning("log_monitor.macos_log_not_found",
                           msg="`log` command not available. File polling only.")
        except Exception as e:
            logger.error("log_monitor.macos_stream_error", error=str(e))



    def stop(self):
        self._running = False

    def _get_watch_files(self) -> list[str]:
        return self.config.watch_files + self.config.custom_logs

    def _read_new_lines(self, path: str) -> list[LogEvent]:
        """Read new lines from a log file since last check."""
        if not os.path.exists(path):
            return []

        current_size = os.path.getsize(path)
        last_pos = self._file_positions.get(path, 0)

        # Handle log rotation (file shrunk)
        if current_size < last_pos:
            last_pos = 0

        if current_size <= last_pos:
            return []

        events = []
        try:
            with open(path, "r", errors="replace") as f:
                f.seek(last_pos)
                lines = f.readlines()
                self._file_positions[path] = f.tell()

                for line in lines[:self.config.max_lines_per_batch]:
                    event = self._parse_line(path, line.strip())
                    if event:
                        events.append(event)
        except PermissionError:
            logger.warning("log_monitor.permission_denied", path=path)
        except Exception as e:
            logger.error("log_monitor.parse_error", path=path, error=str(e))

        return events

    def _parse_line(self, source_file: str, line: str) -> Optional[LogEvent]:
        """Parse a log line against known patterns."""
        if not line:
            return None

        for pattern_name, pattern_def in LOG_PATTERNS.items():
            match = pattern_def["pattern"].search(line)
            if match:
                extracted = pattern_def["extract"](match)
                return LogEvent(
                    timestamp=time.time(),
                    source_file=source_file,
                    raw_line=line,
                    event_type=pattern_def["event_type"],
                    severity=pattern_def["severity"],
                    source_ip=extracted.get("source_ip", ""),
                    username=extracted.get("username", ""),
                    command=extracted.get("command", ""),
                    details=extracted,
                )

        return None
