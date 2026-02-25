"""
CyberGAN WAF — Python Package
Drop-in ASGI security middleware for FastAPI, Starlette, Django.

Quick start:
    from cybergan_waf import CyberGANMiddleware

    app.add_middleware(CyberGANMiddleware,
        mode="block",
        dashboard_url="ws://127.0.0.1:8443/ws",
        slack_webhook="https://hooks.slack.com/services/...",
    )
"""

from cybergan_waf.middleware import CyberGANMiddleware
from cybergan_waf.patterns import scan, scan_all, AttackMatch, ALL_PATTERNS

__version__ = "0.1.0"
__all__ = ["CyberGANMiddleware", "scan", "scan_all", "AttackMatch", "ALL_PATTERNS"]
