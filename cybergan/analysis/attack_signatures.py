"""
CyberGAN — Attack Signature Database
Pattern-based detection rules for common attack vectors.
Used by the web monitor and analysis engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttackType(Enum):
    """Classification of detected attacks."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    RCE = "rce"
    LFI = "lfi"
    RFI = "rfi"
    COMMAND_INJECTION = "command_injection"
    DIRECTORY_TRAVERSAL = "directory_traversal"
    XXE = "xxe"
    SSRF = "ssrf"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    BRUTE_FORCE = "brute_force"
    SESSION_HIJACK = "session_hijack"
    COOKIE_POISONING = "cookie_poisoning"
    API_ABUSE = "api_abuse"
    PORT_SCAN = "port_scan"
    SYN_FLOOD = "syn_flood"
    UDP_FLOOD = "udp_flood"
    DNS_POISONING = "dns_poisoning"
    ARP_SPOOFING = "arp_spoofing"
    CRYPTOJACKING = "cryptojacking"
    WEB_SHELL = "web_shell"
    BACKDOOR = "backdoor"
    REVERSE_SHELL = "reverse_shell"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    LOG_TAMPERING = "log_tampering"
    SSH_KEY_INJECTION = "ssh_key_injection"
    RANSOMWARE = "ransomware"
    ROOTKIT = "rootkit"
    BANNER_GRAB = "banner_grab"
    SERVICE_ENUM = "service_enumeration"
    DEFAULT_CREDENTIALS = "default_credentials"
    WEAK_PASSWORD = "weak_password"
    MISCONFIGURATION = "misconfiguration"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Attack severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AttackSignature:
    """A pattern-based detection rule."""
    name: str
    attack_type: AttackType
    severity: Severity
    patterns: list[str]  # Regex patterns
    description: str = ""
    mitre_technique: str = ""  # e.g., "T1190"
    compiled_patterns: list[re.Pattern] = field(default_factory=list, repr=False)
    false_positive_rate: float = 0.01  # Expected false positive rate

    def __post_init__(self):
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]

    def match(self, text: str) -> Optional[re.Match]:
        """Check if text matches any pattern in this signature."""
        for pattern in self.compiled_patterns:
            m = pattern.search(text)
            if m:
                return m
        return None


# ============================================================
# Signature Database
# ============================================================

SIGNATURES: list[AttackSignature] = [

    # ── SQL Injection ────────────────────────────────────────

    AttackSignature(
        name="SQLi — UNION SELECT",
        attack_type=AttackType.SQL_INJECTION,
        severity=Severity.CRITICAL,
        mitre_technique="T1190",
        patterns=[
            r"(?:UNION\s+(?:ALL\s+)?SELECT)",
            r"(?:SELECT\s+.*\s+FROM\s+information_schema)",
            r"(?:SELECT\s+.*\s+FROM\s+mysql\.user)",
        ],
        description="UNION-based SQL injection attempt",
    ),
    AttackSignature(
        name="SQLi — Boolean Blind",
        attack_type=AttackType.SQL_INJECTION,
        severity=Severity.HIGH,
        mitre_technique="T1190",
        patterns=[
            r"(?:\b(?:AND|OR)\s+\d+=\d+)",
            r"(?:\b(?:AND|OR)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)",
            r"(?:(?:AND|OR)\s+(?:TRUE|FALSE)\b)",
            r"(?:\bHAVING\s+\d+=\d+)",
        ],
        description="Boolean-based blind SQL injection",
    ),
    AttackSignature(
        name="SQLi — Time-based Blind",
        attack_type=AttackType.SQL_INJECTION,
        severity=Severity.HIGH,
        mitre_technique="T1190",
        patterns=[
            r"(?:SLEEP\s*\(\s*\d+\s*\))",
            r"(?:WAITFOR\s+DELAY\s+['\"])",
            r"(?:BENCHMARK\s*\(\s*\d+)",
            r"(?:pg_sleep\s*\(\s*\d+\s*\))",
        ],
        description="Time-based blind SQL injection",
    ),
    AttackSignature(
        name="SQLi — Error-based",
        attack_type=AttackType.SQL_INJECTION,
        severity=Severity.HIGH,
        mitre_technique="T1190",
        patterns=[
            r"(?:EXTRACTVALUE\s*\()",
            r"(?:UPDATEXML\s*\()",
            r"(?:(?:GROUP_CONCAT|CONCAT_WS)\s*\()",
            r"(?:CONVERT\s*\(\s*INT\s*,)",
        ],
        description="Error-based SQL injection",
    ),
    AttackSignature(
        name="SQLi — Common Payloads",
        attack_type=AttackType.SQL_INJECTION,
        severity=Severity.CRITICAL,
        mitre_technique="T1190",
        patterns=[
            r"(?:['\"];\s*DROP\s+TABLE)",
            r"(?:['\"];\s*DELETE\s+FROM)",
            r"(?:['\"];\s*INSERT\s+INTO)",
            r"(?:['\"];\s*UPDATE\s+\w+\s+SET)",
            r"(?:--\s*$|#\s*$|/\*.*\*/)",
            r"(?:['\"];\s*EXEC\s)",
            r"(?:xp_cmdshell)",
        ],
        description="Common SQL injection payloads",
    ),

    # ── Cross-Site Scripting (XSS) ───────────────────────────

    AttackSignature(
        name="XSS — Script Tag",
        attack_type=AttackType.XSS,
        severity=Severity.HIGH,
        mitre_technique="T1059.007",
        patterns=[
            r"(?:<script[^>]*>.*?</script>)",
            r"(?:<script[^>]*>)",
            r"(?:javascript\s*:)",
            r"(?:vbscript\s*:)",
        ],
        description="Script tag injection attempt",
    ),
    AttackSignature(
        name="XSS — Event Handler",
        attack_type=AttackType.XSS,
        severity=Severity.HIGH,
        mitre_technique="T1059.007",
        patterns=[
            r"(?:\bon\w+\s*=\s*['\"]?[^'\"]*(?:alert|confirm|prompt|eval|document)\b)",
            r"(?:\bon(?:load|error|click|mouseover|focus|blur|submit)\s*=)",
        ],
        description="XSS via HTML event handler attributes",
    ),
    AttackSignature(
        name="XSS — Encoded Payloads",
        attack_type=AttackType.XSS,
        severity=Severity.MEDIUM,
        mitre_technique="T1059.007",
        patterns=[
            r"(?:%3Cscript%3E)",
            r"(?:&#x3C;script&#x3E;)",
            r"(?:\\x3Cscript\\x3E)",
            r"(?:data\s*:\s*text/html)",
        ],
        description="Encoded XSS payloads attempting to bypass filters",
    ),
    AttackSignature(
        name="XSS — DOM Manipulation",
        attack_type=AttackType.XSS,
        severity=Severity.HIGH,
        mitre_technique="T1059.007",
        patterns=[
            r"(?:document\.(?:cookie|domain|write|location))",
            r"(?:window\.(?:location|open)\s*=)",
            r"(?:\.innerHTML\s*=)",
            r"(?:eval\s*\()",
        ],
        description="DOM-based XSS via JavaScript manipulation",
    ),

    # ── Command Injection ────────────────────────────────────

    AttackSignature(
        name="Command Injection — Shell Commands",
        attack_type=AttackType.COMMAND_INJECTION,
        severity=Severity.CRITICAL,
        mitre_technique="T1059",
        patterns=[
            r"(?:;\s*(?:cat|ls|whoami|id|uname|pwd|wget|curl)\b)",
            r"(?:\|\s*(?:cat|ls|whoami|id|uname|pwd|wget|curl)\b)",
            r"(?:`[^`]*(?:cat|ls|whoami|id|uname)\b[^`]*`)",
            r"(?:\$\([^)]*(?:cat|ls|whoami|id|uname)\b[^)]*\))",
        ],
        description="OS command injection via shell metacharacters",
    ),
    AttackSignature(
        name="Command Injection — Reverse Shell",
        attack_type=AttackType.REVERSE_SHELL,
        severity=Severity.CRITICAL,
        mitre_technique="T1059",
        patterns=[
            r"(?:bash\s+-i\s+>[\s&]*\/dev\/tcp)",
            r"(?:nc\s+-[^\s]*e\s+(?:/bin/(?:ba)?sh|cmd\.exe))",
            r"(?:python[23]?\s+-c\s+['\"]import\s+(?:socket|subprocess))",
            r"(?:perl\s+-e\s+['\"].*(?:socket|exec))",
            r"(?:ruby\s+-r\s*socket\s+-e)",
            r"(?:php\s+-r\s+['\"].*(?:fsockopen|exec|system))",
            r"(?:mkfifo\s+.*\s*nc\b)",
        ],
        description="Reverse shell execution attempt",
    ),

    # ── Directory Traversal / LFI ────────────────────────────

    AttackSignature(
        name="Directory Traversal",
        attack_type=AttackType.DIRECTORY_TRAVERSAL,
        severity=Severity.HIGH,
        mitre_technique="T1083",
        patterns=[
            r"(?:\.\.[\\/])",
            r"(?:%2e%2e[\\/])",
            r"(?:%252e%252e[\\/])",
            r"(?:\.\.%2f)",
            r"(?:\.\.%5c)",
        ],
        description="Path traversal via ../ sequences",
    ),
    AttackSignature(
        name="LFI — Sensitive File Access",
        attack_type=AttackType.LFI,
        severity=Severity.CRITICAL,
        mitre_technique="T1005",
        patterns=[
            r"(?:(?:\.\./){2,}(?:etc/(?:passwd|shadow|hosts)))",
            r"(?:/etc/(?:passwd|shadow|group|sudoers))",
            r"(?:/proc/self/(?:environ|cmdline|fd))",
            r"(?:(?:c|C):\\\\(?:windows|Windows)\\\\(?:system32|System32))",
            r"(?:php://(?:input|filter|data))",
        ],
        description="Local file inclusion targeting sensitive files",
    ),

    # ── Remote File Inclusion ────────────────────────────────

    AttackSignature(
        name="RFI — Remote Include",
        attack_type=AttackType.RFI,
        severity=Severity.CRITICAL,
        mitre_technique="T1505.003",
        patterns=[
            r"(?:(?:include|require)(?:_once)?\s*\(\s*['\"]?https?://)",
            r"(?:\?(?:file|page|path|include|dir)=https?://)",
            r"(?:=https?://.*\.(?:php|asp|jsp|txt))",
        ],
        description="Remote file inclusion via URL parameter",
    ),

    # ── XXE ──────────────────────────────────────────────────

    AttackSignature(
        name="XXE — External Entity",
        attack_type=AttackType.XXE,
        severity=Severity.CRITICAL,
        mitre_technique="T1059",
        patterns=[
            r"(?:<!ENTITY\s+\w+\s+SYSTEM)",
            r"(?:<!DOCTYPE\s+\w+\s+\[)",
            r"(?:<!ENTITY\s+%\s+\w+\s+SYSTEM)",
            r"(?:file:///)",
            r"(?:expect://)",
        ],
        description="XML External Entity injection",
    ),

    # ── SSRF ─────────────────────────────────────────────────

    AttackSignature(
        name="SSRF — Internal Network Access",
        attack_type=AttackType.SSRF,
        severity=Severity.HIGH,
        mitre_technique="T1090",
        patterns=[
            r"(?:(?:url|target|dest|redirect|uri|path|link)=(?:https?://)?(?:127\.0\.0\.1|localhost|0\.0\.0\.0|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+))",
            r"(?:(?:url|target)=(?:https?://)?169\.254\.169\.254)",  # Cloud metadata
            r"(?:@(?:127\.0\.0\.1|localhost)\b)",
        ],
        description="Server-side request forgery targeting internal services",
    ),

    # ── Web Shell Detection ──────────────────────────────────

    AttackSignature(
        name="Web Shell — PHP",
        attack_type=AttackType.WEB_SHELL,
        severity=Severity.CRITICAL,
        mitre_technique="T1505.003",
        patterns=[
            r"(?:(?:system|exec|passthru|shell_exec|popen|proc_open)\s*\(\s*\$_(?:GET|POST|REQUEST))",
            r"(?:eval\s*\(\s*(?:base64_decode|gzinflate|str_rot13)\s*\()",
            r"(?:assert\s*\(\s*\$_(?:GET|POST|REQUEST))",
            r"(?:file_put_contents\s*\(.*\$_(?:GET|POST|REQUEST))",
        ],
        description="PHP web shell indicators",
    ),

    # ── Suspicious Log Patterns ──────────────────────────────

    AttackSignature(
        name="Brute Force — SSH",
        attack_type=AttackType.BRUTE_FORCE,
        severity=Severity.HIGH,
        mitre_technique="T1110",
        patterns=[
            r"(?:Failed password for (?:invalid user )?(\S+) from (\S+))",
            r"(?:authentication failure.*rhost=(\S+))",
            r"(?:Invalid user \S+ from (\S+))",
        ],
        description="SSH brute force login attempts",
    ),
    AttackSignature(
        name="Privilege Escalation — sudo",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        severity=Severity.CRITICAL,
        mitre_technique="T1548",
        patterns=[
            r"(?:sudo:.*COMMAND=.*/bin/(?:bash|sh|dash|zsh))",
            r"(?:su\[\d+\]:.*FAILED)",
            r"(?:pkexec.*executed)",
            r"(?:FAILED su for \w+ by \w+)",
        ],
        description="Privilege escalation via sudo/su",
    ),
    AttackSignature(
        name="Cryptominer Detection",
        attack_type=AttackType.CRYPTOJACKING,
        severity=Severity.HIGH,
        mitre_technique="T1496",
        patterns=[
            r"(?:stratum\+tcp://)",
            r"(?:xmrig|minerd|cpuminer|cryptonight|randomx)",
            r"(?:pool\.(?:minexmr|supportxmr|hashvault|nanopool)\.com)",
        ],
        description="Cryptocurrency mining indicators",
    ),
    AttackSignature(
        name="Data Exfiltration Indicators",
        attack_type=AttackType.DATA_EXFILTRATION,
        severity=Severity.CRITICAL,
        mitre_technique="T1048",
        patterns=[
            r"(?:curl\s+.*-d\s+@|wget\s+.*--post-file)",
            r"(?:scp\s+.*\S+@\S+:)",
            r"(?:rsync\s+.*\S+@\S+:)",
            r"(?:base64\s+-w\s*0\s+\S+\s*\|\s*(?:curl|wget|nc))",
        ],
        description="Signs of unauthorized data transfer",
    ),
    AttackSignature(
        name="SSH Key Injection",
        attack_type=AttackType.SSH_KEY_INJECTION,
        severity=Severity.CRITICAL,
        mitre_technique="T1098.004",
        patterns=[
            r"(?:>>?\s*~?/?\.ssh/authorized_keys)",
            r"(?:echo\s+.*ssh-(?:rsa|ed25519|ecdsa)\b)",
        ],
        description="Unauthorized SSH key addition for persistence",
    ),
    AttackSignature(
        name="Log Tampering",
        attack_type=AttackType.LOG_TAMPERING,
        severity=Severity.HIGH,
        mitre_technique="T1070",
        patterns=[
            r"(?:>\s*/var/log/\w+)",
            r"(?:shred\s+.*(?:/var/log|\.log))",
            r"(?:rm\s+.*(?:/var/log|\.bash_history|\.log))",
            r"(?:history\s+-c)",
            r"(?:unset\s+HISTFILE)",
        ],
        description="Attempts to erase or modify log files",
    ),

    # ── Suspicious User Agents ───────────────────────────────

    AttackSignature(
        name="Scanner / Bot User Agent",
        attack_type=AttackType.SERVICE_ENUM,
        severity=Severity.LOW,
        mitre_technique="T1595",
        patterns=[
            r"(?:(?:sqlmap|nikto|nmap|masscan|gobuster|dirbuster|ffuf|wfuzz|burp\s*suite|acunetix|nessus|openvas|nuclei|whatweb))",
        ],
        description="Known security scanner/bot user agent detected",
    ),
]


def get_signatures_by_type(attack_type: AttackType) -> list[AttackSignature]:
    """Get all signatures for a given attack type."""
    return [s for s in SIGNATURES if s.attack_type == attack_type]


def scan_text(text: str) -> list[tuple[AttackSignature, re.Match]]:
    """Scan text against all signatures. Returns matching signatures with match objects."""
    matches = []
    for sig in SIGNATURES:
        m = sig.match(text)
        if m:
            matches.append((sig, m))
    return matches


def scan_request(
    method: str = "",
    path: str = "",
    query: str = "",
    body: str = "",
    headers: dict[str, str] | None = None,
    user_agent: str = "",
) -> list[tuple[AttackSignature, str, re.Match]]:
    """
    Scan an HTTP request against all signatures.
    Returns list of (signature, component, match) tuples.
    """
    results = []
    components = {
        "path": path,
        "query": query,
        "body": body,
        "user_agent": user_agent,
    }
    if headers:
        for k, v in headers.items():
            components[f"header_{k}"] = v

    for name, text in components.items():
        if not text:
            continue
        for sig in SIGNATURES:
            m = sig.match(text)
            if m:
                results.append((sig, name, m))

    return results
