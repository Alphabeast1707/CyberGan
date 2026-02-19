"""
CyberGAN — Perception: System Metrics Monitor
Collects CPU, RAM, disk, and network I/O metrics.
Detects DDoS (traffic spikes), resource exhaustion, and disk fill attacks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil
import structlog

from cybergan.config import SystemMetricsConfig

logger = structlog.get_logger(__name__)


@dataclass
class SystemMetrics:
    """Snapshot of system metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_bytes: int
    network_recv_bytes: int
    network_connections: int
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    load_avg_15m: float = 0.0
    process_count: int = 0


@dataclass
class MetricsEvent:
    """A metrics-related security event."""
    timestamp: float
    event_type: str  # cpu_alert, memory_alert, disk_alert, network_spike, ddos_indicator
    severity: str = "warning"
    details: dict = field(default_factory=dict)


class SystemMetricsMonitor:
    """
    System metrics collector and anomaly detector.

    Periodically collects system resource metrics and detects:
    - CPU exhaustion (DDoS, cryptojacking)
    - Memory exhaustion (memory bombs, leaks)
    - Disk exhaustion (log bombs, data dumps)
    - Network traffic spikes (DDoS, data exfiltration)
    """

    def __init__(self, config: SystemMetricsConfig):
        self.config = config
        self._prev_net_io: Optional[tuple[int, int]] = None
        self._prev_time: float = 0
        self._metrics_history: list[SystemMetrics] = []
        self._running = False

    async def start(self, event_queue: asyncio.Queue):
        """Start collecting system metrics."""
        self._running = True
        logger.info("system_metrics.start")

        # Initialize network baseline
        net_io = psutil.net_io_counters()
        self._prev_net_io = (net_io.bytes_sent, net_io.bytes_recv)
        self._prev_time = time.time()

        while self._running:
            try:
                metrics = self._collect()
                events = self._analyze(metrics)

                self._metrics_history.append(metrics)
                # Keep last 100 samples
                if len(self._metrics_history) > 100:
                    self._metrics_history = self._metrics_history[-100:]

                for event in events:
                    await event_queue.put(event)

                # Also put metrics snapshot on queue for the brain
                await event_queue.put(metrics)

            except Exception as e:
                logger.error("system_metrics.error", error=str(e))

            await asyncio.sleep(self.config.poll_interval_s)

    def stop(self):
        self._running = False

    def _collect(self) -> SystemMetrics:
        """Collect current system metrics."""
        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        net_io = psutil.net_io_counters()
        try:
            connections = len(psutil.net_connections(kind="inet"))
        except (psutil.AccessDenied, PermissionError, OSError):
            connections = 0  # macOS requires root for net_connections()

        try:
            load_avg = os.getloadavg()
        except (OSError, AttributeError):
            load_avg = (0.0, 0.0, 0.0)

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu,
            memory_percent=mem.percent,
            memory_used_mb=mem.used / 1024 / 1024,
            memory_total_mb=mem.total / 1024 / 1024,
            disk_percent=disk.percent,
            disk_used_gb=disk.used / 1024 / 1024 / 1024,
            disk_total_gb=disk.total / 1024 / 1024 / 1024,
            network_sent_bytes=net_io.bytes_sent,
            network_recv_bytes=net_io.bytes_recv,
            network_connections=connections,
            load_avg_1m=load_avg[0],
            load_avg_5m=load_avg[1],
            load_avg_15m=load_avg[2],
            process_count=len(psutil.pids()),
        )

    def _analyze(self, metrics: SystemMetrics) -> list[MetricsEvent]:
        """Analyze metrics for anomalies."""
        events = []
        now = metrics.timestamp

        # ── CPU Alert ──
        if metrics.cpu_percent > self.config.cpu_alert_threshold:
            events.append(MetricsEvent(
                timestamp=now,
                event_type="cpu_alert",
                severity="high",
                details={
                    "cpu_percent": metrics.cpu_percent,
                    "threshold": self.config.cpu_alert_threshold,
                    "load_avg": metrics.load_avg_1m,
                },
            ))

        # ── Memory Alert ──
        if metrics.memory_percent > self.config.memory_alert_threshold:
            events.append(MetricsEvent(
                timestamp=now,
                event_type="memory_alert",
                severity="high",
                details={
                    "memory_percent": metrics.memory_percent,
                    "used_mb": round(metrics.memory_used_mb),
                    "threshold": self.config.memory_alert_threshold,
                },
            ))

        # ── Disk Alert ──
        if metrics.disk_percent > self.config.disk_alert_threshold:
            events.append(MetricsEvent(
                timestamp=now,
                event_type="disk_alert",
                severity="high" if metrics.disk_percent > 95 else "warning",
                details={
                    "disk_percent": metrics.disk_percent,
                    "used_gb": round(metrics.disk_used_gb, 1),
                    "threshold": self.config.disk_alert_threshold,
                },
            ))

        # ── Network Spike Detection ──
        if self._prev_net_io:
            elapsed = now - self._prev_time
            if elapsed > 0:
                sent_delta = metrics.network_sent_bytes - self._prev_net_io[0]
                recv_delta = metrics.network_recv_bytes - self._prev_net_io[1]
                sent_mbps = (sent_delta * 8 / 1024 / 1024) / elapsed
                recv_mbps = (recv_delta * 8 / 1024 / 1024) / elapsed

                total_mbps = sent_mbps + recv_mbps
                if total_mbps > self.config.network_spike_threshold_mbps:
                    events.append(MetricsEvent(
                        timestamp=now,
                        event_type="network_spike",
                        severity="critical",
                        details={
                            "total_mbps": round(total_mbps, 1),
                            "sent_mbps": round(sent_mbps, 1),
                            "recv_mbps": round(recv_mbps, 1),
                            "threshold": self.config.network_spike_threshold_mbps,
                        },
                    ))

                    # DDoS indicator: high recv + many connections
                    if recv_mbps > sent_mbps * 5 and metrics.network_connections > 500:
                        events.append(MetricsEvent(
                            timestamp=now,
                            event_type="ddos_indicator",
                            severity="critical",
                            details={
                                "recv_mbps": round(recv_mbps, 1),
                                "connections": metrics.network_connections,
                                "ratio": round(recv_mbps / max(sent_mbps, 0.1), 1),
                            },
                        ))

        self._prev_net_io = (metrics.network_sent_bytes, metrics.network_recv_bytes)
        self._prev_time = now

        return events

    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics snapshot."""
        return self._metrics_history[-1] if self._metrics_history else None

    def get_metrics_history(self, count: int = 10) -> list[SystemMetrics]:
        """Get recent metrics history."""
        return self._metrics_history[-count:]


import os  # noqa: E402 — needed for getloadavg
