"""
CyberGAN — Perception: File Integrity Monitor
Watches critical filesystem paths for unauthorized changes.
Detects web shells, config tampering, rootkit indicators,
SUID changes, and crontab modifications.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from cybergan.config import FileMonitorConfig

logger = structlog.get_logger(__name__)


@dataclass
class FileEvent:
    """A file system security event."""
    timestamp: float
    event_type: str  # file_created, file_modified, file_deleted, permission_changed, suid_set
    path: str
    severity: str = "info"
    old_hash: str = ""
    new_hash: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class FileState:
    """Cached state of a tracked file."""
    path: str
    hash: str
    mtime: float
    size: int
    mode: int
    is_suid: bool


class FileMonitor:
    """
    File integrity monitor.

    Periodically scans critical directories and compares file states
    to detect unauthorized changes. Uses hash-based comparison for
    content changes and stat-based checks for permission changes.
    """

    DANGEROUS_EXTENSIONS = {
        ".php", ".jsp", ".asp", ".aspx", ".sh", ".py", ".pl",
        ".cgi", ".phtml", ".php5", ".php7",
    }

    CRITICAL_FILES = {
        "/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/hosts",
        "/etc/ssh/sshd_config", "/etc/crontab", "/etc/resolv.conf",
        "/root/.ssh/authorized_keys", "/root/.bashrc",
    }

    def __init__(self, config: FileMonitorConfig):
        self.config = config
        self._baseline: dict[str, FileState] = {}
        self._initialized = False
        self._running = False

    async def start(self, event_queue: asyncio.Queue):
        """Start file integrity monitoring."""
        self._running = True
        logger.info("file_monitor.start", paths=self.config.watch_paths)

        # Build initial baseline
        await self._build_baseline()
        self._initialized = True

        while self._running:
            try:
                events = await self._scan()
                for event in events:
                    await event_queue.put(event)
            except Exception as e:
                logger.error("file_monitor.scan_error", error=str(e))

            await asyncio.sleep(5)  # Check every 5 seconds

    def stop(self):
        self._running = False

    async def _build_baseline(self):
        """Build initial file state baseline."""
        logger.info("file_monitor.building_baseline")
        for watch_path in self.config.watch_paths:
            if os.path.isdir(watch_path):
                for root, dirs, files in os.walk(watch_path):
                    # Skip excluded patterns
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        if self._should_exclude(fpath):
                            continue
                        state = self._get_file_state(fpath)
                        if state:
                            self._baseline[fpath] = state
            elif os.path.isfile(watch_path):
                state = self._get_file_state(watch_path)
                if state:
                    self._baseline[watch_path] = state

        logger.info("file_monitor.baseline_built", file_count=len(self._baseline))

    async def _scan(self) -> list[FileEvent]:
        """Scan for changes since baseline."""
        events = []
        current_files: set[str] = set()
        now = time.time()

        for watch_path in self.config.watch_paths:
            if os.path.isdir(watch_path):
                for root, dirs, files in os.walk(watch_path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        if self._should_exclude(fpath):
                            continue
                        current_files.add(fpath)

                        new_state = self._get_file_state(fpath)
                        if not new_state:
                            continue

                        old_state = self._baseline.get(fpath)

                        if old_state is None:
                            # New file detected
                            severity = self._classify_new_file(fpath)
                            events.append(FileEvent(
                                timestamp=now,
                                event_type="file_created",
                                path=fpath,
                                severity=severity,
                                new_hash=new_state.hash,
                                details={"size": new_state.size, "mode": oct(new_state.mode)},
                            ))
                            self._baseline[fpath] = new_state

                        elif new_state.hash != old_state.hash:
                            # File content changed
                            severity = "critical" if fpath in self.CRITICAL_FILES else "warning"
                            events.append(FileEvent(
                                timestamp=now,
                                event_type="file_modified",
                                path=fpath,
                                severity=severity,
                                old_hash=old_state.hash,
                                new_hash=new_state.hash,
                                details={
                                    "size_delta": new_state.size - old_state.size,
                                    "old_mtime": old_state.mtime,
                                    "new_mtime": new_state.mtime,
                                },
                            ))
                            self._baseline[fpath] = new_state

                        elif new_state.mode != old_state.mode:
                            # Permissions changed
                            is_suid_change = new_state.is_suid != old_state.is_suid
                            severity = "critical" if is_suid_change else "warning"
                            event_type = "suid_set" if is_suid_change and new_state.is_suid else "permission_changed"
                            events.append(FileEvent(
                                timestamp=now,
                                event_type=event_type,
                                path=fpath,
                                severity=severity,
                                details={
                                    "old_mode": oct(old_state.mode),
                                    "new_mode": oct(new_state.mode),
                                    "suid_added": is_suid_change and new_state.is_suid,
                                },
                            ))
                            self._baseline[fpath] = new_state

            elif os.path.isfile(watch_path):
                current_files.add(watch_path)
                # Same logic for single files
                new_state = self._get_file_state(watch_path)
                if new_state:
                    old_state = self._baseline.get(watch_path)
                    if old_state and new_state.hash != old_state.hash:
                        events.append(FileEvent(
                            timestamp=now,
                            event_type="file_modified",
                            path=watch_path,
                            severity="critical",
                            old_hash=old_state.hash,
                            new_hash=new_state.hash,
                        ))
                        self._baseline[watch_path] = new_state

        # Check for deleted files
        for fpath in list(self._baseline.keys()):
            if fpath not in current_files and not os.path.exists(fpath):
                severity = "critical" if fpath in self.CRITICAL_FILES else "warning"
                events.append(FileEvent(
                    timestamp=now,
                    event_type="file_deleted",
                    path=fpath,
                    severity=severity,
                    old_hash=self._baseline[fpath].hash,
                ))
                del self._baseline[fpath]

        return events

    def _get_file_state(self, path: str) -> Optional[FileState]:
        """Get current state of a file."""
        try:
            st = os.stat(path)
            file_hash = self._hash_file(path) if st.st_size < 50 * 1024 * 1024 else ""  # Skip files > 50MB
            return FileState(
                path=path,
                hash=file_hash,
                mtime=st.st_mtime,
                size=st.st_size,
                mode=st.st_mode,
                is_suid=bool(st.st_mode & stat.S_ISUID),
            )
        except (OSError, PermissionError):
            return None

    def _hash_file(self, path: str) -> str:
        """Compute SHA256 hash of a file."""
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()
        except (OSError, PermissionError):
            return ""

    def _should_exclude(self, path: str) -> bool:
        """Check if a file should be excluded from monitoring."""
        import fnmatch
        basename = os.path.basename(path)
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(basename, pattern):
                return True
        return False

    def _classify_new_file(self, path: str) -> str:
        """Classify the severity of a newly created file."""
        ext = os.path.splitext(path)[1].lower()

        # Web shell indicators
        if ext in self.DANGEROUS_EXTENSIONS and "/var/www" in path:
            return "critical"

        # New executable
        if self.config.alert_on_new_executables:
            try:
                if os.access(path, os.X_OK):
                    return "high"
            except OSError:
                pass

        # New SUID binary
        if self.config.alert_on_suid_changes:
            try:
                st = os.stat(path)
                if st.st_mode & stat.S_ISUID:
                    return "critical"
            except OSError:
                pass

        return "info"
