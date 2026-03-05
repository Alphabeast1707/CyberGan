"""
CyberGAN WAF — Real-time Reporter
Sends attack events to the CyberGAN dashboard and Slack/Discord webhooks.
Non-blocking — uses fire-and-forget async tasks.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional
import aiohttp

from cybergan_waf.patterns import AttackMatch


class WAFReporter:
    """
    Sends real attack events to:
    1. CyberGAN dashboard (WebSocket)
    2. Slack webhook
    3. Discord webhook
    4. Console (always)
    """

    def __init__(
        self,
        dashboard_ws_url: Optional[str] = None,
        slack_webhook: Optional[str] = None,
        discord_webhook: Optional[str] = None,
    ):
        self.dashboard_ws_url = dashboard_ws_url  # e.g. "ws://127.0.0.1:8443/ws"
        self.slack_webhook = slack_webhook
        self.discord_webhook = discord_webhook
        self._ws = None
        self._ws_lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
        self._total_blocked = 0
        self._total_detected = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def report_attack(
        self,
        match: AttackMatch,
        client_ip: str,
        method: str,
        path: str,
        blocked: bool,
        latency_ms: int = 0,
    ):
        """Fire-and-forget attack report to all configured channels."""
        self._total_detected += 1
        if blocked:
            self._total_blocked += 1

        event = {
            "type": "threat",
            "timestamp": time.time(),
            "event_type": match.category,
            "title": match.attack_type,
            "severity": match.severity,
            "source_ip": client_ip,
            "description": f"{method} {path} — {match.description[:120]}",
            "action_taken": f"WAF {'block (executed)' if blocked else 'log (advisory)'}",
            "mitre_techniques": [match.mitre_technique],
            "kill_chain_stage": _mitre_to_kill_chain(match.category),
            "details": {
                "method": method,
                "path": path,
                "pattern": match.matched_pattern,
                "blocked": blocked,
                "latency_ms": latency_ms,
            },
        }

        # Console always
        severity_emoji = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🔵"}.get(match.severity, "⚠️")
        action = "BLOCKED" if blocked else "LOGGED"
        print(f"  {severity_emoji} WAF [{action}] {match.attack_type} from {client_ip} → {path}")

        # Fire-and-forget tasks
        tasks = []
        if self.dashboard_ws_url:
            tasks.append(asyncio.create_task(self._send_dashboard(event)))
        if self.slack_webhook:
            tasks.append(asyncio.create_task(self._send_slack(event, blocked)))
        if self.discord_webhook:
            tasks.append(asyncio.create_task(self._send_discord(event, blocked)))

        # Don't await — let them run in background
        if tasks:
            asyncio.gather(*tasks, return_exceptions=True)

    async def _send_dashboard(self, event: dict):
        """Send event to CyberGAN dashboard via WebSocket."""
        if not self.dashboard_ws_url:
            return
        try:
            import websockets
            async with websockets.connect(self.dashboard_ws_url, open_timeout=2) as ws:
                await ws.send(json.dumps(event))
        except Exception:
            pass  # Dashboard offline — don't crash the WAF

    async def _send_slack(self, event: dict, blocked: bool):
        """Send Slack notification."""
        if not self.slack_webhook:
            return
        color = "#e53e3e" if event["severity"] in ("critical", "high") else "#f59e0b"
        action = "🛑 BLOCKED" if blocked else "👁️ LOGGED"
        payload = {
            "attachments": [{
                "color": color,
                "title": f"{action}: {event['title']}",
                "fields": [
                    {"title": "IP", "value": event["source_ip"], "short": True},
                    {"title": "Severity", "value": event["severity"].upper(), "short": True},
                    {"title": "Request", "value": event["description"][:200], "short": False},
                    {"title": "MITRE", "value": ", ".join(event["mitre_techniques"]), "short": True},
                ],
                "footer": "CyberGAN WAF",
                "ts": int(event["timestamp"]),
            }]
        }
        try:
            session = await self._get_session()
            async with session.post(self.slack_webhook, json=payload, timeout=aiohttp.ClientTimeout(total=5)):
                pass
        except Exception:
            pass

    async def _send_discord(self, event: dict, blocked: bool):
        """Send Discord notification."""
        if not self.discord_webhook:
            return
        color = 0xe53e3e if event["severity"] in ("critical", "high") else 0xf59e0b
        action = "🛑 BLOCKED" if blocked else "👁️ LOGGED"
        payload = {
            "embeds": [{
                "title": f"{action}: {event['title']}",
                "color": color,
                "fields": [
                    {"name": "IP", "value": event["source_ip"], "inline": True},
                    {"name": "Severity", "value": event["severity"].upper(), "inline": True},
                    {"name": "Path", "value": event["description"][:200], "inline": False},
                    {"name": "MITRE", "value": ", ".join(event["mitre_techniques"]), "inline": True},
                ],
                "footer": {"text": "CyberGAN WAF"},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event["timestamp"])),
            }]
        }
        try:
            session = await self._get_session()
            async with session.post(self.discord_webhook, json=payload, timeout=aiohttp.ClientTimeout(total=5)):
                pass
        except Exception:
            pass

    def get_stats(self) -> dict:
        return {
            "total_detected": self._total_detected,
            "total_blocked": self._total_blocked,
        }


def _mitre_to_kill_chain(category: str) -> str:
    mapping = {
        "sql_injection": "exploitation",
        "xss": "exploitation",
        "rce": "exploitation",
        "lfi": "reconnaissance",
        "ssrf": "reconnaissance",
        "command_injection": "exploitation",
        "xxe": "exploitation",
        "ssti": "exploitation",
        "csrf": "delivery",
        "suspicious_ua": "reconnaissance",
        "crlf_injection": "exploitation",
    }
    return mapping.get(category, "exploitation")
