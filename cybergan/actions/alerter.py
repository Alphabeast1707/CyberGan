"""
CyberGAN — Actions: Alert System
Multi-channel notification system for security alerts.
Supports console, webhook (Slack/Discord), and email.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import structlog

from cybergan.config import AlerterConfig

logger = structlog.get_logger(__name__)


@dataclass
class SecurityAlert:
    """A security alert to be delivered."""
    timestamp: float
    title: str
    severity: str  # critical, high, medium, low, info
    event_type: str
    description: str
    source_ip: str = ""
    action_taken: str = ""
    risk_score: float = 0.0
    mitre_techniques: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "title": self.title,
            "severity": self.severity,
            "event_type": self.event_type,
            "description": self.description,
            "source_ip": self.source_ip,
            "action_taken": self.action_taken,
            "risk_score": self.risk_score,
            "mitre_techniques": self.mitre_techniques,
            "details": self.details,
        }


SEVERITY_COLORS = {
    "critical": "\033[91m",  # Red
    "high": "\033[93m",      # Yellow
    "medium": "\033[94m",    # Blue
    "low": "\033[96m",       # Cyan
    "info": "\033[37m",      # White
}

SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "info": "⚪",
}


class Alerter:
    """
    Multi-channel alert system.

    Sends security alerts to configured channels:
    - Console (rich terminal output)
    - Webhook (Slack/Discord)
    - Email (SMTP)

    Features:
    - Alert cooldown (prevent spam)
    - Alert batching (group related alerts)
    - Severity-based filtering
    """

    def __init__(self, config: AlerterConfig):
        self.config = config
        self._cooldowns: dict[str, float] = {}
        self._alert_history: list[SecurityAlert] = []
        self._batch_buffer: list[SecurityAlert] = []
        self._last_batch_send: float = 0

    async def send(self, alert: SecurityAlert):
        """Send an alert through all enabled channels."""
        # Cooldown check
        cooldown_key = f"{alert.event_type}:{alert.source_ip}"
        now = time.time()
        last_sent = self._cooldowns.get(cooldown_key, 0)
        if now - last_sent < self.config.alert_cooldown_s:
            return

        self._cooldowns[cooldown_key] = now
        self._alert_history.append(alert)

        # Keep history bounded
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]

        # Send to each enabled channel
        for channel in self.config.channels:
            if not channel.enabled:
                continue

            try:
                if channel.type == "console":
                    self._send_console(alert)
                elif channel.type == "webhook":
                    await self._send_webhook(alert, channel.url)
                elif channel.type == "email":
                    await self._send_email(alert, channel)
            except Exception as e:
                logger.error("alerter.send_error", channel=channel.type, error=str(e))

    def _send_console(self, alert: SecurityAlert):
        """Print alert to console with colors."""
        color = SEVERITY_COLORS.get(alert.severity, "\033[37m")
        emoji = SEVERITY_EMOJI.get(alert.severity, "⚪")
        reset = "\033[0m"
        ts = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")

        print(
            f"\n{color}{'━' * 60}{reset}\n"
            f"  {emoji} {color}[{alert.severity.upper()}]{reset} {alert.title}\n"
            f"  {color}Time:{reset} {ts}  {color}Type:{reset} {alert.event_type}\n"
            f"  {color}Description:{reset} {alert.description}"
        )
        if alert.source_ip:
            print(f"  {color}Source IP:{reset} {alert.source_ip}")
        if alert.action_taken:
            print(f"  {color}Action:{reset} {alert.action_taken}")
        if alert.risk_score > 0:
            print(f"  {color}Risk Score:{reset} {alert.risk_score:.0f}/100")
        if alert.mitre_techniques:
            print(f"  {color}MITRE ATT&CK:{reset} {', '.join(alert.mitre_techniques)}")
        print(f"{color}{'━' * 60}{reset}\n")

    async def _send_webhook(self, alert: SecurityAlert, url: str):
        """Send alert to Slack/Discord webhook."""
        if not url:
            return

        try:
            import aiohttp

            # Format for Slack/Discord
            payload = {
                "content": None,
                "embeds": [{
                    "title": f"{SEVERITY_EMOJI.get(alert.severity, '')} {alert.title}",
                    "description": alert.description,
                    "color": {
                        "critical": 0xFF0000,
                        "high": 0xFF8C00,
                        "medium": 0xFFD700,
                        "low": 0x4169E1,
                        "info": 0x808080,
                    }.get(alert.severity, 0x808080),
                    "fields": [
                        {"name": "Severity", "value": alert.severity.upper(), "inline": True},
                        {"name": "Type", "value": alert.event_type, "inline": True},
                        {"name": "Risk Score", "value": f"{alert.risk_score:.0f}", "inline": True},
                    ],
                    "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
                }],
            }
            if alert.source_ip:
                payload["embeds"][0]["fields"].append(
                    {"name": "Source IP", "value": alert.source_ip, "inline": True}
                )
            if alert.action_taken:
                payload["embeds"][0]["fields"].append(
                    {"name": "Action Taken", "value": alert.action_taken, "inline": True}
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status not in (200, 204):
                        logger.warning("alerter.webhook_error", status=resp.status)
        except ImportError:
            logger.warning("alerter.aiohttp_not_installed")
        except Exception as e:
            logger.error("alerter.webhook_error", error=str(e))

    async def _send_email(self, alert: SecurityAlert, channel):
        """Send alert via email."""
        if not channel.smtp_host or not channel.to_addrs:
            return

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["From"] = channel.from_addr
            msg["To"] = ", ".join(channel.to_addrs)
            msg["Subject"] = f"[CyberGAN {alert.severity.upper()}] {alert.title}"

            body = (
                f"Security Alert from CyberGAN\n\n"
                f"Severity: {alert.severity.upper()}\n"
                f"Type: {alert.event_type}\n"
                f"Description: {alert.description}\n"
                f"Time: {datetime.fromtimestamp(alert.timestamp).isoformat()}\n"
            )
            if alert.source_ip:
                body += f"Source IP: {alert.source_ip}\n"
            if alert.action_taken:
                body += f"Action Taken: {alert.action_taken}\n"
            if alert.risk_score > 0:
                body += f"Risk Score: {alert.risk_score:.0f}/100\n"

            msg.attach(MIMEText(body, "plain"))

            # Send async in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._smtp_send(
                channel.smtp_host, channel.smtp_port,
                channel.from_addr, channel.to_addrs, msg.as_string(),
            ))
        except Exception as e:
            logger.error("alerter.email_error", error=str(e))

    def _smtp_send(self, host, port, from_addr, to_addrs, msg_str):
        """Synchronous SMTP send (run in executor)."""
        import smtplib
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.sendmail(from_addr, to_addrs, msg_str)

    def get_history(self, count: int = 50) -> list[dict]:
        """Get recent alert history."""
        return [a.to_dict() for a in self._alert_history[-count:]]
