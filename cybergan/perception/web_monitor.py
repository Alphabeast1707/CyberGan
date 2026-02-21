"""
CyberGAN — Perception: Web Request Monitor
HTTP request analysis layer for WAF-style protection.
Detects SQLi, XSS, CSRF, RCE, LFI, RFI, XXE, SSRF,
command injection, directory traversal in web traffic.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import structlog

from cybergan.config import WebMonitorConfig
from cybergan.analysis.attack_signatures import scan_request, AttackType, Severity

logger = structlog.get_logger(__name__)


@dataclass
class WebEvent:
    """A web security event."""
    timestamp: float
    event_type: str  # sqli, xss, rce, rate_limit, suspicious_ua, etc.
    severity: str = "warning"
    source_ip: str = ""
    method: str = ""
    path: str = ""
    status_code: int = 0
    user_agent: str = ""
    details: dict = field(default_factory=dict)


class RateLimiter:
    """Sliding window rate limiter per IP."""

    def __init__(self, global_limit: int = 1000, per_ip_limit: int = 100, window_s: int = 60):
        self.global_limit = global_limit
        self.per_ip_limit = per_ip_limit
        self.window_s = window_s
        self._global_requests: list[float] = []
        self._ip_requests: dict[str, list[float]] = defaultdict(list)

    def check(self, ip: str) -> Optional[str]:
        """Check if request is rate-limited. Returns reason or None."""
        now = time.time()

        # Global rate limit
        self._global_requests = [t for t in self._global_requests if now - t < self.window_s]
        self._global_requests.append(now)
        if len(self._global_requests) > self.global_limit:
            return "global_limit_exceeded"

        # Per-IP rate limit
        ip_reqs = self._ip_requests[ip]
        ip_reqs[:] = [t for t in ip_reqs if now - t < self.window_s]
        ip_reqs.append(now)
        if len(ip_reqs) > self.per_ip_limit:
            return f"ip_limit_exceeded:{ip}"

        return None


class WebMonitor:
    """
    Web traffic monitor and WAF layer.

    Operates in log_parser mode: tails web server access logs
    and analyzes requests for attack patterns.

    Detects:
    - SQL Injection (UNION, blind, time-based, error-based)
    - Cross-Site Scripting (reflected, stored, DOM)
    - Command Injection (shell commands, reverse shells)
    - Directory Traversal / LFI / RFI
    - XXE, SSRF
    - Rate limit violations
    - Suspicious user agents (scanners, bots)
    - API abuse patterns
    """

    # Access log pattern (combined format)
    ACCESS_LOG_PATTERN = re.compile(
        r'(\S+)\s+\S+\s+\S+\s+\[([^\]]+)\]\s+"(\S+)\s+(\S+)\s+\S+"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"'
    )

    def __init__(self, config: WebMonitorConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            global_limit=config.rate_limit_requests_per_minute,
            per_ip_limit=config.rate_limit_per_ip,
        )
        self._file_position: int = 0
        self._running = False

    async def start(self, event_queue: asyncio.Queue):
        """Start monitoring web access logs."""
        self._running = True
        logger.info("web_monitor.start", mode=self.config.mode)

        # Initialize to end of file
        if os.path.exists(self.config.access_log_path):
            self._file_position = os.path.getsize(self.config.access_log_path)

        while self._running:
            try:
                events = self._read_and_analyze()
                for event in events:
                    await event_queue.put(event)
            except Exception as e:
                logger.error("web_monitor.error", error=str(e))

            await asyncio.sleep(self.config.poll_interval_ms / 1000.0)

    def stop(self):
        self._running = False

    def _read_and_analyze(self) -> list[WebEvent]:
        """Read new log lines and analyze for attacks."""
        events = []
        log_path = self.config.access_log_path

        if not os.path.exists(log_path):
            return events

        current_size = os.path.getsize(log_path)
        if current_size < self._file_position:
            self._file_position = 0  # Log rotated

        if current_size <= self._file_position:
            return events

        try:
            with open(log_path, "r", errors="replace") as f:
                f.seek(self._file_position)
                lines = f.readlines()
                self._file_position = f.tell()

                for line in lines:
                    parsed = self._parse_access_log(line.strip())
                    if parsed:
                        line_events = self._analyze_request(parsed)
                        events.extend(line_events)
        except (PermissionError, OSError) as e:
            logger.warning("web_monitor.read_error", error=str(e))

        return events

    def _parse_access_log(self, line: str) -> Optional[dict]:
        """Parse an access log line."""
        match = self.ACCESS_LOG_PATTERN.match(line)
        if not match:
            return None

        return {
            "ip": match.group(1),
            "timestamp": match.group(2),
            "method": match.group(3),
            "path": match.group(4),
            "status": int(match.group(5)),
            "size": int(match.group(6)),
            "referer": match.group(7),
            "user_agent": match.group(8),
        }

    def _analyze_request(self, request: dict) -> list[WebEvent]:
        """Analyze a parsed request for security threats."""
        events = []
        now = time.time()
        ip = request["ip"]
        path = request["path"]
        method = request["method"]
        user_agent = request["user_agent"]

        # Split path and query
        query = ""
        if "?" in path:
            path, query = path.split("?", 1)

        # ── Rate Limiting ──
        rate_result = self.rate_limiter.check(ip)
        if rate_result:
            events.append(WebEvent(
                timestamp=now,
                event_type="rate_limit",
                severity="high",
                source_ip=ip,
                method=method,
                path=path,
                details={"reason": rate_result},
            ))

        # ── Attack Signature Scanning ──
        matches = scan_request(
            method=method,
            path=path,
            query=query,
            user_agent=user_agent,
            headers={"referer": request.get("referer", "")},
        )

        for sig, component, match in matches:
            severity_map = {
                Severity.CRITICAL: "critical",
                Severity.HIGH: "high",
                Severity.MEDIUM: "medium",
                Severity.LOW: "low",
                Severity.INFO: "info",
            }
            events.append(WebEvent(
                timestamp=now,
                event_type=sig.attack_type.value,
                severity=severity_map.get(sig.severity, "warning"),
                source_ip=ip,
                method=method,
                path=path,
                user_agent=user_agent,
                status_code=request["status"],
                details={
                    "signature": sig.name,
                    "component": component,
                    "matched_text": match.group(0)[:200],
                    "mitre_technique": sig.mitre_technique,
                },
            ))

        return events

    def analyze_raw_request(
        self,
        method: str,
        path: str,
        body: str = "",
        headers: dict[str, str] | None = None,
        source_ip: str = "",
    ) -> list[WebEvent]:
        """
        Analyze a raw HTTP request (for middleware mode).
        Can be called directly by ASGI/WSGI middleware.
        """
        events = []
        now = time.time()
        query = ""
        if "?" in path:
            path, query = path.split("?", 1)

        user_agent = (headers or {}).get("user-agent", "")

        matches = scan_request(
            method=method,
            path=path,
            query=query,
            body=body,
            headers=headers,
            user_agent=user_agent,
        )

        for sig, component, match in matches:
            events.append(WebEvent(
                timestamp=now,
                event_type=sig.attack_type.value,
                severity=sig.severity.value,
                source_ip=source_ip,
                method=method,
                path=path,
                user_agent=user_agent,
                details={
                    "signature": sig.name,
                    "component": component,
                    "matched_text": match.group(0)[:200],
                },
            ))

        return events
