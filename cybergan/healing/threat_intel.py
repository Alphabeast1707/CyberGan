"""
CyberGAN — Healing: Threat Intelligence
Pulls from public CVE feeds (NVD, CISA KEV) and converts new CVEs
into exploit templates for the arena.
"""

from __future__ import annotations

import structlog

from cybergan.config import ThreatIntelConfig

logger = structlog.get_logger(__name__)


class ThreatIntelFeedManager:
    """Manage threat intelligence feeds."""

    def __init__(self, config: ThreatIntelConfig):
        self.config = config
        self._known_cves: set[str] = set()

    async def fetch_updates(self) -> list[dict]:
        """Fetch new CVEs from configured feeds."""
        if not self.config.enabled:
            return []

        new_cves = []
        for feed in self.config.feeds:
            try:
                cves = await self._fetch_feed(feed.name, feed.url)
                for cve in cves:
                    cve_id = cve.get("id", "")
                    if cve_id not in self._known_cves:
                        self._known_cves.add(cve_id)
                        new_cves.append(cve)
            except Exception as e:
                logger.error("threat_intel.fetch_error", feed=feed.name, error=str(e))

        return new_cves

    async def _fetch_feed(self, name: str, url: str) -> list[dict]:
        """Fetch CVEs from a specific feed."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_feed(name, data)
        except ImportError:
            logger.warning("threat_intel.aiohttp_not_installed")
        except Exception as e:
            logger.error("threat_intel.error", feed=name, error=str(e))
        return []

    def _parse_feed(self, name: str, data: dict) -> list[dict]:
        """Parse feed-specific JSON format."""
        cves = []
        if name == "NVD":
            for vuln in data.get("vulnerabilities", []):
                cve = vuln.get("cve", {})
                cves.append({
                    "id": cve.get("id", ""),
                    "description": (cve.get("descriptions", [{}])[0].get("value", "") if cve.get("descriptions") else ""),
                    "severity": cve.get("metrics", {}).get("cvssMetricV31", [{}])[0].get("cvssData", {}).get("baseScore", 0) if cve.get("metrics", {}).get("cvssMetricV31") else 0,
                })
        elif name == "CISA KEV":
            for vuln in data.get("vulnerabilities", []):
                cves.append({
                    "id": vuln.get("cveID", ""),
                    "description": vuln.get("shortDescription", ""),
                    "severity": 8.0,  # KEV = high severity by definition
                })
        return cves

    def get_stats(self) -> dict:
        return {
            "known_cves": len(self._known_cves),
            "feeds": len(self.config.feeds),
        }
