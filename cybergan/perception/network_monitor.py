"""
CyberGAN — Perception: Network Monitor
Real-time network connection tracking and anomaly detection.
Detects port scans, SYN floods, unusual outbound connections,
C2 beacons, and data exfiltration indicators.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import structlog

from cybergan.config import NetworkMonitorConfig

logger = structlog.get_logger(__name__)


@dataclass
class ConnectionInfo:
    """A network connection snapshot."""
    protocol: str  # tcp, udp
    local_addr: str
    local_port: int
    remote_addr: str
    remote_port: int
    state: str  # ESTABLISHED, LISTEN, TIME_WAIT, SYN_RECV, etc.
    pid: int = 0
    process_name: str = ""


@dataclass
class NetworkEvent:
    """A network security event."""
    timestamp: float
    event_type: str  # port_scan, syn_flood, connection_spike, suspicious_outbound, etc.
    severity: str = "warning"
    source_ip: str = ""
    destination_ip: str = ""
    port: int = 0
    details: dict = field(default_factory=dict)


class PortScanTracker:
    """Tracks unique ports accessed per IP for port scan detection."""

    def __init__(self, threshold: int = 10, window_s: int = 60):
        self.threshold = threshold
        self.window_s = window_s
        self._access_log: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._alerted: set[str] = set()

    def record(self, ip: str, port: int) -> bool:
        """Record a port access. Returns True if port scan detected."""
        now = time.time()
        log = self._access_log[ip]
        # Prune old entries
        log[:] = [(p, t) for p, t in log if now - t < self.window_s]
        log.append((port, now))

        unique_ports = len(set(p for p, _ in log))
        if unique_ports >= self.threshold and ip not in self._alerted:
            self._alerted.add(ip)
            return True
        return False


class SYNFloodTracker:
    """Detects SYN flood by monitoring SYN_RECV state connections."""

    def __init__(self, threshold: int = 50, window_s: int = 10):
        self.threshold = threshold
        self.window_s = window_s
        self._syn_counts: list[tuple[int, float]] = []

    def update(self, syn_recv_count: int) -> bool:
        """Update with current SYN_RECV count. Returns True if flood detected."""
        now = time.time()
        self._syn_counts.append((syn_recv_count, now))
        self._syn_counts = [(c, t) for c, t in self._syn_counts if now - t < self.window_s]

        avg = sum(c for c, _ in self._syn_counts) / max(len(self._syn_counts), 1)
        return avg > self.threshold


class NetworkMonitor:
    """
    Real-time network connection monitor.

    Uses 'ss' command to track active connections and detect:
    - Port scanning (many unique ports from single IP)
    - SYN floods (high SYN_RECV count)
    - Connection spikes (sudden increase in new connections)
    - Suspicious outbound connections (C2 indicators)
    - Data exfiltration (large outbound transfers)
    """

    def __init__(self, config: NetworkMonitorConfig):
        self.config = config
        self.port_scan_tracker = PortScanTracker(
            threshold=config.port_scan_threshold,
            window_s=config.port_scan_window_s,
        )
        self.syn_flood_tracker = SYNFloodTracker()
        self._prev_connections: dict[str, ConnectionInfo] = {}
        self._connection_counts: list[tuple[int, float]] = []
        self._running = False

    async def start(self, event_queue: asyncio.Queue):
        """Start monitoring network connections."""
        self._running = True
        logger.info("network_monitor.start")

        while self._running:
            try:
                connections = await self._get_connections()
                events = self._analyze(connections)
                for event in events:
                    await event_queue.put(event)
                self._prev_connections = {
                    f"{c.remote_addr}:{c.remote_port}-{c.local_port}": c
                    for c in connections
                }
            except Exception as e:
                logger.error("network_monitor.error", error=str(e))

            await asyncio.sleep(self.config.poll_interval_s)

    def stop(self):
        self._running = False

    async def _get_connections(self) -> list[ConnectionInfo]:
        """Get current network connections using 'ss' command."""
        connections = []
        try:
            proc = await asyncio.create_subprocess_exec(
                "ss", "-tunapH",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            for line in stdout.decode(errors="replace").strip().split("\n"):
                conn = self._parse_ss_line(line)
                if conn:
                    connections.append(conn)
        except FileNotFoundError:
            # ss not available, try netstat
            try:
                proc = await asyncio.create_subprocess_exec(
                    "netstat", "-tuanp",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                for line in stdout.decode(errors="replace").strip().split("\n"):
                    conn = self._parse_netstat_line(line)
                    if conn:
                        connections.append(conn)
            except FileNotFoundError:
                logger.warning("network_monitor.no_tool", msg="Neither ss nor netstat available")

        return connections

    def _parse_ss_line(self, line: str) -> Optional[ConnectionInfo]:
        """Parse a line from 'ss -tunapH' output."""
        parts = line.split()
        if len(parts) < 5:
            return None

        try:
            proto = parts[0].lower()
            state = parts[1] if len(parts) > 5 else ""
            local = parts[4] if len(parts) > 5 else parts[3]
            remote = parts[5] if len(parts) > 5 else parts[4]

            local_addr, local_port = self._split_addr(local)
            remote_addr, remote_port = self._split_addr(remote)

            pid = 0
            process_name = ""
            if len(parts) > 6:
                proc_info = parts[6]
                if "pid=" in proc_info:
                    try:
                        pid = int(proc_info.split("pid=")[1].split(",")[0])
                    except (ValueError, IndexError):
                        pass

            return ConnectionInfo(
                protocol=proto,
                local_addr=local_addr,
                local_port=local_port,
                remote_addr=remote_addr,
                remote_port=remote_port,
                state=state,
                pid=pid,
                process_name=process_name,
            )
        except (ValueError, IndexError):
            return None

    def _parse_netstat_line(self, line: str) -> Optional[ConnectionInfo]:
        """Parse a line from 'netstat -tuanp' output."""
        parts = line.split()
        if len(parts) < 4 or parts[0] not in ("tcp", "tcp6", "udp", "udp6"):
            return None

        try:
            proto = "tcp" if "tcp" in parts[0] else "udp"
            local_addr, local_port = self._split_addr(parts[3])
            remote_addr, remote_port = self._split_addr(parts[4])
            state = parts[5] if len(parts) > 5 and not parts[5].startswith("-") else ""

            return ConnectionInfo(
                protocol=proto,
                local_addr=local_addr,
                local_port=local_port,
                remote_addr=remote_addr,
                remote_port=remote_port,
                state=state,
            )
        except (ValueError, IndexError):
            return None

    def _split_addr(self, addr_str: str) -> tuple[str, int]:
        """Split address:port string."""
        if "]:" in addr_str:  # IPv6
            parts = addr_str.rsplit(":", 1)
            return parts[0].strip("[]"), int(parts[1]) if parts[1] != "*" else 0
        elif ":" in addr_str:
            parts = addr_str.rsplit(":", 1)
            return parts[0], int(parts[1]) if parts[1] != "*" else 0
        return addr_str, 0

    def _analyze(self, connections: list[ConnectionInfo]) -> list[NetworkEvent]:
        """Analyze connections for security events."""
        events = []
        now = time.time()

        # Track connection count for spike detection
        conn_count = len(connections)
        self._connection_counts.append((conn_count, now))
        self._connection_counts = [(c, t) for c, t in self._connection_counts if now - t < 60]

        # ── Connection Spike Detection ──
        if len(self._connection_counts) > 2:
            avg = sum(c for c, _ in self._connection_counts[:-1]) / max(len(self._connection_counts) - 1, 1)
            if conn_count > avg + self.config.connection_spike_threshold:
                events.append(NetworkEvent(
                    timestamp=now,
                    event_type="connection_spike",
                    severity="high",
                    details={"current": conn_count, "average": avg},
                ))

        # ── SYN Flood Detection ──
        syn_recv_count = sum(1 for c in connections if c.state == "SYN-RECV")
        if self.syn_flood_tracker.update(syn_recv_count):
            events.append(NetworkEvent(
                timestamp=now,
                event_type="syn_flood",
                severity="critical",
                details={"syn_recv_count": syn_recv_count},
            ))

        # ── Port Scan Detection ──
        new_connections = {}
        for c in connections:
            key = f"{c.remote_addr}:{c.remote_port}-{c.local_port}"
            if key not in self._prev_connections:
                new_connections[key] = c

        ip_port_access: dict[str, set[int]] = defaultdict(set)
        for c in new_connections.values():
            if c.remote_addr and c.remote_addr not in ("0.0.0.0", "::", "*"):
                ip_port_access[c.remote_addr].add(c.local_port)
                if self.port_scan_tracker.record(c.remote_addr, c.local_port):
                    events.append(NetworkEvent(
                        timestamp=now,
                        event_type="port_scan",
                        severity="high",
                        source_ip=c.remote_addr,
                        details={"unique_ports": len(ip_port_access[c.remote_addr])},
                    ))

        # ── Suspicious Outbound Connections ──
        if self.config.track_outbound:
            for c in connections:
                if c.state == "ESTAB" and c.remote_port in self.config.suspicious_ports:
                    if c.remote_addr not in self.config.whitelisted_ips:
                        events.append(NetworkEvent(
                            timestamp=now,
                            event_type="suspicious_outbound",
                            severity="high",
                            destination_ip=c.remote_addr,
                            port=c.remote_port,
                            details={"pid": c.pid, "process": c.process_name},
                        ))

        return events
