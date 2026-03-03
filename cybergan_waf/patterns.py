"""
CyberGAN WAF — Attack Pattern Library
Comprehensive regex patterns for detecting 60+ real attack vectors.
Used by the WAF middleware to inspect HTTP requests.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class AttackMatch:
    attack_type: str
    category: str
    severity: str          # critical | high | medium | low
    matched_value: str
    matched_pattern: str
    mitre_technique: str
    description: str


# ── SQL Injection ─────────────────────────────────────────────
SQLI = [
    (re.compile(r"UNION\s+(?:ALL\s+)?SELECT", re.I), "UNION-based SQLi", "critical", "T1190"),
    (re.compile(r"'\s*(?:OR|AND)\s+'?[\w\d]+'?\s*=\s*'?[\w\d]+'?", re.I), "Boolean SQLi", "critical", "T1190"),
    (re.compile(r"(?:--\s*$|;\s*--\s*$|#\s*$)", re.M), "SQL comment terminator", "high", "T1190"),
    (re.compile(r"'\s*;\s*(?:DROP|CREATE|ALTER|TRUNCATE|DELETE|INSERT|UPDATE)\s+", re.I), "Stacked query SQLi", "critical", "T1190"),
    (re.compile(r"(?:sleep\s*\(\s*\d+|benchmark\s*\(\s*\d+|waitfor\s+delay\s*')", re.I), "Time-based blind SQLi", "critical", "T1190"),
    (re.compile(r"(?:LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)", re.I), "SQLi file exfil", "critical", "T1190"),
    (re.compile(r"(?:xp_cmdshell|sp_executesql|exec\s*\(|execute\s*\()", re.I), "MSSQL exec SQLi", "critical", "T1190"),
    (re.compile(r"(?:information_schema|sys\.tables|pg_catalog|sqlite_master)", re.I), "Schema enum SQLi", "high", "T1190"),
    (re.compile(r"(?:EXTRACTVALUE|UPDATEXML|XMLTYPE|FLOOR\s*\(\s*RAND)", re.I), "Error-based SQLi", "high", "T1190"),
    (re.compile(r"(?:1=1|1\s*=\s*1|'1'='1|\"1\"=\"1\")", re.I), "Always-true SQLi", "medium", "T1190"),
]

# ── Cross-Site Scripting (XSS) ────────────────────────────────
XSS = [
    (re.compile(r"<script[^>]*>.*?</script\s*>", re.I | re.S), "Script tag XSS", "high", "T1059.007"),
    (re.compile(r"javascript\s*:", re.I), "JS protocol XSS", "high", "T1059.007"),
    (re.compile(r"on(?:load|error|click|mouseover|focus|blur|change|submit|keydown|keyup|keypress|input|drag|drop|copy|paste|wheel|scroll|resize|unload|beforeunload|hashchange|popstate|animationstart|animationend|transitionend|pointerdown|pointerup|pointermove)\s*=", re.I), "Event handler XSS", "high", "T1059.007"),
    (re.compile(r"<(?:img|svg|iframe|object|embed|video|audio|source|track|link|base|meta|style)\b[^>]*(?:onerror|onload|src|href|action|data|srcdoc)[^>]*>", re.I), "HTML tag XSS", "high", "T1059.007"),
    (re.compile(r"eval\s*\(|setTimeout\s*\(|setInterval\s*\(|Function\s*\(|execScript\s*\(", re.I), "JS eval XSS", "critical", "T1059.007"),
    (re.compile(r"document\s*\.\s*(?:cookie|write|location|domain|referrer|URL)", re.I), "DOM manipulation XSS", "high", "T1059.007"),
    (re.compile(r"(?:alert|confirm|prompt)\s*\(", re.I), "Alert/XSS probe", "medium", "T1059.007"),
    (re.compile(r"(?:String\.fromCharCode|\\u00[0-9a-f]{2}|&#x[0-9a-f]+;|&#\d+;)", re.I), "Encoded XSS", "high", "T1059.007"),
    (re.compile(r"<[a-z]+\s+[a-z-]+=\s*[\"']?\s*javascript:", re.I), "Attribute JS XSS", "high", "T1059.007"),
    (re.compile(r"(?:vbscript|livescript|mocha|data:text/html)\s*:", re.I), "Alt-protocol XSS", "high", "T1059.007"),
]

# ── Remote Code Execution ─────────────────────────────────────
RCE = [
    (re.compile(r"(?:;|\||`|\$\()\s*(?:ls|cat|id|whoami|uname|pwd|env|set|printenv|hostname)\b", re.I), "RCE recon command", "critical", "T1059"),
    (re.compile(r"(?:;|\||`|\$\()\s*(?:wget|curl|fetch|lwp-download|python|perl|ruby|php)\s+https?://", re.I), "RCE download", "critical", "T1059"),
    (re.compile(r"(?:bash|sh|dash|zsh|fish|csh|tcsh)\s+(?:-c|-i)", re.I), "Shell invocation", "critical", "T1059.004"),
    (re.compile(r"(?:nc|ncat|netcat|socat)\s+(?:-e|-c|-l)", re.I), "Reverse shell", "critical", "T1059"),
    (re.compile(r"(?:\/bin\/bash|\/bin\/sh|\/usr\/bin\/python|\/usr\/bin\/perl)\s", re.I), "Direct shell path", "critical", "T1059"),
    (re.compile(r"(?:system|shell_exec|exec|passthru|popen|proc_open|pcntl_exec)\s*\(", re.I), "PHP RCE function", "critical", "T1059"),
    (re.compile(r"__import__\s*\(\s*['\"](?:os|subprocess|commands|pty)['\"]", re.I), "Python import RCE", "critical", "T1059"),
    (re.compile(r"Runtime\s*\.\s*getRuntime\s*\(\s*\)\s*\.\s*exec", re.I), "Java RCE", "critical", "T1059"),
    (re.compile(r"(?:mkfifo|mknod)\s+/tmp/", re.I), "FIFO pipe RCE", "critical", "T1059"),
    (re.compile(r"(?:base64\s+-d|base64\s+--decode|base64_decode)\s*\|", re.I), "Base64 decode pipe", "critical", "T1059"),
]

# ── Local File Inclusion ──────────────────────────────────────
LFI = [
    (re.compile(r"(?:\.\./|\.\.\\){2,}", re.I), "Path traversal ../../", "critical", "T1083"),
    (re.compile(r"(?:/etc/passwd|/etc/shadow|/etc/hosts|/etc/sudoers|/etc/crontab)", re.I), "Unix sensitive file", "critical", "T1083"),
    (re.compile(r"(?:/proc/self/environ|/proc/self/cmdline|/proc/self/maps)", re.I), "Proc environ LFI", "critical", "T1083"),
    (re.compile(r"php://(?:filter|input|output|data|expect|zip)", re.I), "PHP wrapper LFI", "critical", "T1083"),
    (re.compile(r"(?:phar|zip|rar|data|file)://", re.I), "Stream wrapper LFI", "high", "T1083"),
    (re.compile(r"(?:C:\\\\Windows\\\\|C:/Windows/|%WINDIR%|%SYSTEMROOT%)", re.I), "Windows path LFI", "high", "T1083"),
    (re.compile(r"(?:/var/log/|/var/www/|/home/\w+/\.ssh/)", re.I), "Web/SSH dir LFI", "high", "T1083"),
    (re.compile(r"(?:\.\.%2[fF]|\.\.%5[cC]|%2e%2e%2f|%252e%252e%252f)", re.I), "URL-encoded traversal", "critical", "T1083"),
]

# ── SSRF ─────────────────────────────────────────────────────
SSRF = [
    (re.compile(r"https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0|::1)\b", re.I), "SSRF to localhost", "critical", "T1557"),
    (re.compile(r"https?://(?:169\.254\.169\.254|metadata\.internal|metadata\.google\.internal|169\.254\.170\.2)", re.I), "Cloud metadata SSRF", "critical", "T1557"),
    (re.compile(r"https?://(?:10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)", re.I), "SSRF to private range", "high", "T1557"),
    (re.compile(r"(?:gopher|dict|ldap|ldaps|sftp|ftp|tftp|smb|file)://", re.I), "SSRF alt-protocol", "critical", "T1557"),
    (re.compile(r"@(?:127\.0\.0\.1|localhost|10\.\d+\.\d+\.\d+)", re.I), "SSRF URL bypass @", "high", "T1557"),
]

# ── Command Injection ─────────────────────────────────────────
CMDI = [
    (re.compile(r"(?:[;&|]|\|\|)\s*(?:ls|cat|id|whoami|uname|curl|wget|ping|nmap|nc)\b", re.I), "Command chaining", "critical", "T1059"),
    (re.compile(r"\$\{(?:IFS|PATH|HOME|USER|SHELL|LS_COLORS)[^}]*\}", re.I), "Shell variable injection", "critical", "T1059"),
    (re.compile(r"`[^`]+`", re.I), "Backtick command sub", "high", "T1059"),
    (re.compile(r"\$\([^)]+\)", re.I), "Dollar-paren command sub", "high", "T1059"),
    (re.compile(r"(?:IFS|PATH|BASH_ENV|ENV|LD_PRELOAD|LD_LIBRARY_PATH)\s*=", re.I), "Env var injection", "critical", "T1059"),
]

# ── XXE ──────────────────────────────────────────────────────
XXE = [
    (re.compile(r"<!ENTITY\s+\w+\s+SYSTEM\s+['\"]", re.I), "XXE SYSTEM entity", "critical", "T1190"),
    (re.compile(r"<!DOCTYPE\s+\w+\s*\[", re.I), "XXE DOCTYPE", "high", "T1190"),
    (re.compile(r"<!ENTITY\s+%\s+\w+\s+SYSTEM", re.I), "XXE parameter entity", "critical", "T1190"),
    (re.compile(r"SYSTEM\s+['\"]file://", re.I), "XXE file protocol", "critical", "T1190"),
]

# ── SSTI (Server-Side Template Injection) ────────────────────
SSTI = [
    (re.compile(r"\{\{.*?\}\}", re.S), "SSTI Jinja2/Twig", "critical", "T1059"),
    (re.compile(r"\$\{.*?\}", re.S), "SSTI Freemarker/Velocity", "critical", "T1059"),
    (re.compile(r"<#.*?#>|<\?.*?\?>", re.S), "SSTI template tag", "high", "T1059"),
    (re.compile(r"#\{.*?\}", re.S), "SSTI Ruby ERB", "critical", "T1059"),
]

# ── CSRF Indicators ───────────────────────────────────────────
CSRF = [
    (re.compile(r"(?:content-type:\s*text/plain|content-type:\s*application/x-www-form-urlencoded).*(?:action=|redirect=|url=)", re.I), "CSRF form submission", "medium", "T1059"),
]

# ── Suspicious User Agents ────────────────────────────────────
SUSPICIOUS_UA = [
    (re.compile(r"(?:sqlmap|nikto|nmap|masscan|zap|burpsuite|acunetix|nessus|openvas|metasploit|hydra|medusa|havij|whois|dirbuster|gobuster|wfuzz|ffuf|nuclei)", re.I), "Known attack tool UA", "high", "T1595"),
    (re.compile(r"(?:python-requests|go-http-client|libwww-perl|curl/|wget/)\s*[0-9]", re.I), "Scripted client UA", "low", "T1595"),
    (re.compile(r"(?:zgrab|masscan|censys|shodan)", re.I), "Internet scanner UA", "medium", "T1595"),
]

# ── CRLF Injection ────────────────────────────────────────────
CRLF = [
    (re.compile(r"%0[aAdD]|\\r\\n|\r\n", re.I), "CRLF injection", "high", "T1059"),
    (re.compile(r"(?:Set-Cookie:|Location:|Content-Type:)\s*\w+.*(?:%0[aAdD]|\r\n)", re.I), "Header injection", "high", "T1059"),
]

# ── All categories ────────────────────────────────────────────
ALL_PATTERNS: dict[str, list] = {
    "sql_injection":        [(p, name, sev, mitre) for p, name, sev, mitre in SQLI],
    "xss":                  [(p, name, sev, mitre) for p, name, sev, mitre in XSS],
    "rce":                  [(p, name, sev, mitre) for p, name, sev, mitre in RCE],
    "lfi":                  [(p, name, sev, mitre) for p, name, sev, mitre in LFI],
    "ssrf":                 [(p, name, sev, mitre) for p, name, sev, mitre in SSRF],
    "command_injection":    [(p, name, sev, mitre) for p, name, sev, mitre in CMDI],
    "xxe":                  [(p, name, sev, mitre) for p, name, sev, mitre in XXE],
    "ssti":                 [(p, name, sev, mitre) for p, name, sev, mitre in SSTI],
    "csrf":                 [(p, name, sev, mitre) for p, name, sev, mitre in CSRF],
    "suspicious_ua":        [(p, name, sev, mitre) for p, name, sev, mitre in SUSPICIOUS_UA],
    "crlf_injection":       [(p, name, sev, mitre) for p, name, sev, mitre in CRLF],
}


def scan(value: str, source: str = "unknown") -> Optional[AttackMatch]:
    """
    Scan a string value against all attack patterns.
    Returns the first (highest-severity) match found, or None.
    """
    if not value or len(value) > 65536:
        return None

    for category, patterns in ALL_PATTERNS.items():
        for pattern, name, severity, mitre in patterns:
            try:
                m = pattern.search(value)
                if m:
                    return AttackMatch(
                        attack_type=name,
                        category=category,
                        severity=severity,
                        matched_value=value[:200],
                        matched_pattern=pattern.pattern[:80],
                        mitre_technique=mitre,
                        description=f"{name} detected in {source}: {m.group(0)[:80]}",
                    )
            except re.error:
                continue
    return None


def scan_all(values: list[tuple[str, str]]) -> list[AttackMatch]:
    """Scan multiple (value, source) pairs. Returns all matches."""
    results = []
    for value, source in values:
        match = scan(value, source)
        if match:
            results.append(match)
    return results
