"""
CyberGAN WAF — ASGI Middleware
Drop-in security middleware for FastAPI, Starlette, Django, and any ASGI app.

Usage:
    from cybergan_waf import CyberGANMiddleware

    app.add_middleware(CyberGANMiddleware,
        mode="block",                          # "block" | "log" | "alert"
        dashboard_url="ws://127.0.0.1:8443/ws",
        slack_webhook="https://hooks.slack.com/...",
        rate_limit_per_minute=100,
        whitelist_ips=["127.0.0.1"],
    )
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
import urllib.parse
from collections import defaultdict
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from cybergan_waf.patterns import AttackMatch, scan, scan_all
from cybergan_waf.reporter import WAFReporter


class RateLimiter:
    """Token-bucket per-IP rate limiter."""

    def __init__(self, requests_per_minute: int = 100, burst: int = 20):
        self.rpm = requests_per_minute
        self.burst = burst
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def is_limited(self, ip: str) -> bool:
        now = time.time()
        window = self._buckets[ip]
        # Remove requests older than 60s
        self._buckets[ip] = [t for t in window if now - t < 60]
        if len(self._buckets[ip]) >= self.rpm:
            return True
        self._buckets[ip].append(now)
        return False

    def get_count(self, ip: str) -> int:
        return len(self._buckets.get(ip, []))


class CyberGANMiddleware(BaseHTTPMiddleware):
    """
    Production-grade WAF middleware.

    Inspects every HTTP request for:
    - SQL injection (10+ variants)
    - XSS (10+ variants)
    - Remote Code Execution
    - Local File Inclusion / Path Traversal
    - SSRF (including cloud metadata endpoints)
    - Command Injection
    - XXE / SSTI / CSRF / CRLF
    - Suspicious scanner user agents
    - Per-IP rate limiting
    - Configurable block/log/alert modes
    """

    def __init__(
        self,
        app: ASGIApp,
        mode: str = "block",                    # "block" | "log" | "alert"
        dashboard_url: Optional[str] = None,    # "ws://127.0.0.1:8443/ws"
        slack_webhook: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        rate_limit_per_minute: int = 200,
        rate_limit_burst: int = 30,
        whitelist_ips: Optional[list[str]] = None,
        whitelist_paths: Optional[list[str]] = None,
        max_body_bytes: int = 1_048_576,        # 1 MB
        block_scanners: bool = True,
        inspect_headers: bool = True,
        inspect_cookies: bool = True,
    ):
        super().__init__(app)
        self.mode = mode
        self.max_body_bytes = max_body_bytes
        self.block_scanners = block_scanners
        self.inspect_headers = inspect_headers
        self.inspect_cookies = inspect_cookies

        self._whitelist_ips: set[str] = set(whitelist_ips or [])
        self._whitelist_paths: set[str] = set(whitelist_paths or ["/health", "/ping-internal", "/favicon.ico"])
        self._blocked_ips: set[str] = set()
        self._rate_limiter = RateLimiter(rate_limit_per_minute, rate_limit_burst)

        self._reporter = WAFReporter(
            dashboard_ws_url=dashboard_url,
            slack_webhook=slack_webhook,
            discord_webhook=discord_webhook,
        )

        # Stats
        self._stats = {
            "requests_total": 0,
            "requests_blocked": 0,
            "requests_passed": 0,
            "attacks_detected": 0,
            "rate_limited": 0,
        }

        print(f"\n  🛡️  CyberGAN WAF active — mode: {mode.upper()}")
        if dashboard_url:
            print(f"       Dashboard: {dashboard_url.replace('ws://', 'http://').replace('/ws', '')}")
        if slack_webhook:
            print(f"       Slack alerts: enabled")
        print(f"       Rate limit: {rate_limit_per_minute} req/min per IP\n")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        self._stats["requests_total"] += 1

        client_ip = self._get_client_ip(request)
        path = request.url.path

        # ── 1. Skip whitelisted paths ─────────────────────────
        if path in self._whitelist_paths:
            return await call_next(request)

        # ── 2. Skip whitelisted IPs ───────────────────────────
        if client_ip in self._whitelist_ips:
            self._stats["requests_passed"] += 1
            return await call_next(request)

        # ── 3. Check if IP is permanently blocked ─────────────
        if client_ip in self._blocked_ips:
            self._stats["requests_blocked"] += 1
            return self._block_response("IP address blocked by CyberGAN", client_ip)

        # ── 4. Rate limiting ──────────────────────────────────
        if self._rate_limiter.is_limited(client_ip):
            self._stats["rate_limited"] += 1
            count = self._rate_limiter.get_count(client_ip)
            match = AttackMatch(
                attack_type="Rate Limit Exceeded",
                category="rate_limit",
                severity="medium",
                matched_value=f"{count} req/min",
                matched_pattern="rate_limiter",
                mitre_technique="T1498",
                description=f"IP {client_ip} exceeded rate limit: {count} req/min",
            )
            latency_ms = int((time.time() - start) * 1000)
            asyncio.create_task(self._reporter.report_attack(
                match, client_ip, request.method, path,
                blocked=(self.mode == "block"), latency_ms=latency_ms
            ))
            if self.mode == "block":
                self._stats["requests_blocked"] += 1
                return self._block_response("Rate limit exceeded", client_ip)

        # ── 5. Build inspection corpus ────────────────────────
        values_to_scan: list[tuple[str, str]] = []

        # URL path + query string
        values_to_scan.append((urllib.parse.unquote(str(request.url)), "URL"))

        # Query parameters
        for key, val in request.query_params.multi_items():
            values_to_scan.append((urllib.parse.unquote(key), f"query[{key}]"))
            values_to_scan.append((urllib.parse.unquote(val), f"query[{key}]"))

        # Headers (selective)
        if self.inspect_headers:
            dangerous_headers = ["user-agent", "referer", "x-forwarded-for",
                                  "x-forwarded-host", "origin", "content-disposition"]
            for hdr in dangerous_headers:
                val = request.headers.get(hdr, "")
                if val:
                    values_to_scan.append((val, f"header[{hdr}]"))

        # Cookies
        if self.inspect_cookies:
            for name, val in request.cookies.items():
                values_to_scan.append((val, f"cookie[{name}]"))

        # Request body (POST/PUT/PATCH)
        body_text = ""
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body_bytes = await request.body()
                if len(body_bytes) <= self.max_body_bytes:
                    body_text = body_bytes.decode("utf-8", errors="replace")
                    values_to_scan.append((body_text, "body"))

                    # Try to parse form data / JSON keys too
                    content_type = request.headers.get("content-type", "")
                    if "application/x-www-form-urlencoded" in content_type:
                        for key, val in urllib.parse.parse_qsl(body_text):
                            values_to_scan.append((urllib.parse.unquote(key), f"form[{key}]"))
                            values_to_scan.append((urllib.parse.unquote(val), f"form[{key}]"))
            except Exception:
                pass

        # ── 6. Scan everything ────────────────────────────────
        matches = scan_all(values_to_scan)

        if matches:
            # Take the highest-severity match
            severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            match = max(matches, key=lambda m: severity_order.get(m.severity, 0))

            self._stats["attacks_detected"] += 1
            latency_ms = int((time.time() - start) * 1000)
            blocked = (self.mode == "block")

            # Permanently block critical/high attackers
            if blocked and match.severity in ("critical", "high"):
                self._blocked_ips.add(client_ip)

            asyncio.create_task(self._reporter.report_attack(
                match, client_ip, request.method, path,
                blocked=blocked, latency_ms=latency_ms
            ))

            if blocked:
                self._stats["requests_blocked"] += 1
                return self._block_response(
                    f"Request blocked: {match.attack_type}",
                    client_ip,
                    attack_type=match.category,
                    severity=match.severity,
                )

        # ── 7. Pass through ───────────────────────────────────
        self._stats["requests_passed"] += 1
        response = await call_next(request)
        latency_ms = int((time.time() - start) * 1000)
        response.headers["X-CyberGAN-Protected"] = "true"
        response.headers["X-CyberGAN-Latency"] = f"{latency_ms}ms"
        return response

    def _block_response(
        self,
        message: str,
        client_ip: str = "",
        attack_type: str = "",
        severity: str = "high",
    ) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={
                "error": "Forbidden",
                "message": message,
                "blocked_by": "CyberGAN WAF",
                "severity": severity,
                "request_id": f"cgwaf-{int(time.time())}",
            },
            headers={
                "X-CyberGAN-Protected": "true",
                "X-CyberGAN-Block-Reason": attack_type or "policy",
            },
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP, respecting proxy headers."""
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
            try:
                ipaddress.ip_address(ip)
                return ip
            except ValueError:
                pass
        real_ip = request.headers.get("x-real-ip", "")
        if real_ip:
            return real_ip.strip()
        if request.client:
            return request.client.host
        return "unknown"

    def get_stats(self) -> dict:
        """Get WAF runtime statistics."""
        reporter_stats = self._reporter.get_stats()
        return {
            **self._stats,
            **reporter_stats,
            "blocked_ips_count": len(self._blocked_ips),
            "mode": self.mode,
        }

    def unblock_ip(self, ip: str):
        """Manually unblock an IP."""
        self._blocked_ips.discard(ip)
