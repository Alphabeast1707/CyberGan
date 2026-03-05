"""
CyberGAN — Arena: Vulnerability & Exploit Catalog (Expanded)
MITRE ATT&CK technique mappings and exploit templates for simulation.
Covers 90+ attack vectors across all categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AttackTechnique(Enum):
    """MITRE ATT&CK technique categories (full kill chain)."""
    RECONNAISSANCE = "Reconnaissance"
    RESOURCE_DEVELOPMENT = "Resource Development"
    INITIAL_ACCESS = "Initial Access"
    EXECUTION = "Execution"
    PERSISTENCE = "Persistence"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    DEFENSE_EVASION = "Defense Evasion"
    CREDENTIAL_ACCESS = "Credential Access"
    DISCOVERY = "Discovery"
    LATERAL_MOVEMENT = "Lateral Movement"
    COLLECTION = "Collection"
    COMMAND_AND_CONTROL = "Command and Control"
    EXFILTRATION = "Exfiltration"
    IMPACT = "Impact"


# Kill chain ordering (lower index = earlier stage)
KILL_CHAIN_ORDER = list(AttackTechnique)


class AttackCategory(Enum):
    """High-level attack classification."""
    WEB_APPLICATION = "Web Application"
    NETWORK = "Network"
    MALWARE = "Malware"
    AUTHENTICATION = "Authentication"
    PRIVILEGE = "Privilege"
    PERSISTENCE = "Persistence"
    RECONNAISSANCE = "Reconnaissance"
    CLOUD_CONTAINER = "Cloud/Container"
    SOCIAL_ENGINEERING = "Social Engineering"
    DATA = "Data"
    MEMORY = "Memory Corruption"
    WIRELESS = "Wireless"


@dataclass
class ExploitTemplate:
    """
    A predefined exploit mapped to a CVE or attack pattern.
    Used by the Red Agent when exploiting a vulnerability.
    """
    name: str
    cve: str
    technique: AttackTechnique
    category: AttackCategory = AttackCategory.WEB_APPLICATION
    base_success_rate: float = 0.5
    required_access: str = "remote"  # remote | local | physical
    yields_credentials: bool = False
    yields_root: bool = False
    is_stealthy: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cve": self.cve,
            "technique": self.technique.value,
            "category": self.category.value,
            "success_rate": self.base_success_rate,
            "required_access": self.required_access,
            "stealthy": self.is_stealthy,
        }


# ============================================================
# Expanded Exploit Catalog — 90+ Attack Vectors
# ============================================================

EXPLOIT_CATALOG: dict[str, ExploitTemplate] = {

    # ── Web Application Attacks ──────────────────────────────

    "CVE-2024-6387": ExploitTemplate(
        name="regreSSHion Race Condition",
        cve="CVE-2024-6387",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.MEMORY,
        base_success_rate=0.55,
        required_access="remote",
        yields_root=True,
        description="Race condition in OpenSSH signal handler — remote root RCE",
    ),
    "CVE-2021-41773": ExploitTemplate(
        name="Apache Path Traversal",
        cve="CVE-2021-41773",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.75,
        required_access="remote",
        yields_credentials=True,
        description="Path traversal reads /etc/passwd via crafted URL",
    ),
    "CVE-2021-42013": ExploitTemplate(
        name="Apache RCE via CGI Bypass",
        cve="CVE-2021-42013",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.65,
        required_access="remote",
        yields_root=True,
        description="RCE via path traversal + CGI bypass in Apache 2.4.49/50",
    ),
    "CVE-2023-22078": ExploitTemplate(
        name="MySQL Privilege Escalation",
        cve="CVE-2023-22078",
        technique=AttackTechnique.PRIVILEGE_ESCALATION,
        category=AttackCategory.PRIVILEGE,
        base_success_rate=0.50,
        required_access="local",
        yields_root=True,
        description="MySQL privilege escalation to DBA root access",
    ),
    "CVE-2021-44142": ExploitTemplate(
        name="Samba Heap Overflow RCE",
        cve="CVE-2021-44142",
        technique=AttackTechnique.LATERAL_MOVEMENT,
        category=AttackCategory.MEMORY,
        base_success_rate=0.60,
        required_access="remote",
        yields_root=True,
        description="Out-of-bounds heap r/w in Samba vfs_fruit module",
    ),
    "CVE-2022-23613": ExploitTemplate(
        name="xrdp Buffer Overflow",
        cve="CVE-2022-23613",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.MEMORY,
        base_success_rate=0.70,
        required_access="remote",
        yields_credentials=True,
        description="Buffer overflow in xrdp login screen — credential capture",
    ),
    "CVE-2023-32002": ExploitTemplate(
        name="Node.js Module Policy Bypass",
        cve="CVE-2023-32002",
        technique=AttackTechnique.DEFENSE_EVASION,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.60,
        required_access="remote",
        yields_root=True,
        description="Module._load bypass allows arbitrary code execution",
    ),

    # ── SQL Injection Variants ───────────────────────────────

    "SQLI-UNION": ExploitTemplate(
        name="SQL Injection (UNION-based)",
        cve="SQLI-UNION",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.70,
        yields_credentials=True,
        description="UNION SELECT injection to extract database contents",
    ),
    "SQLI-BLIND": ExploitTemplate(
        name="SQL Injection (Blind/Boolean)",
        cve="SQLI-BLIND",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.55,
        is_stealthy=True,
        yields_credentials=True,
        description="Blind SQL injection via boolean-based inference",
    ),
    "SQLI-TIME": ExploitTemplate(
        name="SQL Injection (Time-based)",
        cve="SQLI-TIME",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.50,
        is_stealthy=True,
        yields_credentials=True,
        description="Time-based blind SQL injection using SLEEP/WAITFOR",
    ),

    # ── XSS Variants ────────────────────────────────────────

    "XSS-REFLECTED": ExploitTemplate(
        name="Reflected XSS",
        cve="XSS-REFLECTED",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.65,
        yields_credentials=True,
        description="Reflected cross-site scripting via URL parameters",
    ),
    "XSS-STORED": ExploitTemplate(
        name="Stored XSS",
        cve="XSS-STORED",
        technique=AttackTechnique.PERSISTENCE,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.60,
        yields_credentials=True,
        description="Persistent XSS stored in database, executes for all viewers",
    ),
    "XSS-DOM": ExploitTemplate(
        name="DOM-based XSS",
        cve="XSS-DOM",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.55,
        is_stealthy=True,
        description="Client-side DOM manipulation XSS",
    ),

    # ── Other Web Application Attacks ────────────────────────

    "CSRF-EXPLOIT": ExploitTemplate(
        name="Cross-Site Request Forgery",
        cve="CSRF-EXPLOIT",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.60,
        description="Forged cross-site request to perform unauthorized actions",
    ),
    "RCE-EVAL": ExploitTemplate(
        name="Remote Code Execution (eval injection)",
        cve="RCE-EVAL",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.55,
        yields_root=True,
        description="Code injection via eval/exec in server-side code",
    ),
    "LFI-EXPLOIT": ExploitTemplate(
        name="Local File Inclusion",
        cve="LFI-EXPLOIT",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.70,
        yields_credentials=True,
        description="Include local files like /etc/passwd via path manipulation",
    ),
    "RFI-EXPLOIT": ExploitTemplate(
        name="Remote File Inclusion",
        cve="RFI-EXPLOIT",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.50,
        yields_root=True,
        description="Include remote malicious files into server-side processing",
    ),
    "CMD-INJECT": ExploitTemplate(
        name="Command Injection",
        cve="CMD-INJECT",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.60,
        yields_root=True,
        description="OS command injection via unsanitized input to system calls",
    ),
    "DIR-TRAVERSAL": ExploitTemplate(
        name="Directory Traversal",
        cve="DIR-TRAVERSAL",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.65,
        yields_credentials=True,
        description="Path traversal via ../ sequences to access sensitive files",
    ),
    "XXE-EXPLOIT": ExploitTemplate(
        name="XML External Entity Injection",
        cve="XXE-EXPLOIT",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.50,
        yields_credentials=True,
        description="XXE to read files, SSRF, or remote code execution",
    ),
    "SSRF-EXPLOIT": ExploitTemplate(
        name="Server-Side Request Forgery",
        cve="SSRF-EXPLOIT",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.55,
        description="Force server to make requests to internal/external resources",
    ),
    "INSECURE-DESER": ExploitTemplate(
        name="Insecure Deserialization",
        cve="INSECURE-DESER",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.45,
        yields_root=True,
        description="Arbitrary code execution via malicious serialized objects",
    ),

    # ── Authentication Attacks ───────────────────────────────

    "BRUTE-FORCE": ExploitTemplate(
        name="Brute Force Login",
        cve="BRUTE-FORCE",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.AUTHENTICATION,
        base_success_rate=0.30,
        yields_credentials=True,
        description="Automated password guessing against login endpoints",
    ),
    "CREDENTIAL-STUFF": ExploitTemplate(
        name="Credential Stuffing",
        cve="CREDENTIAL-STUFF",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.AUTHENTICATION,
        base_success_rate=0.25,
        yields_credentials=True,
        is_stealthy=True,
        description="Using leaked credential databases for login attempts",
    ),
    "SESSION-HIJACK": ExploitTemplate(
        name="Session Hijacking",
        cve="SESSION-HIJACK",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.AUTHENTICATION,
        base_success_rate=0.40,
        yields_credentials=True,
        description="Stealing session tokens to impersonate authenticated users",
    ),
    "COOKIE-POISON": ExploitTemplate(
        name="Cookie Poisoning",
        cve="COOKIE-POISON",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.AUTHENTICATION,
        base_success_rate=0.45,
        yields_credentials=True,
        description="Manipulating cookie values to bypass authentication",
    ),
    "RDP-BRUTE": ExploitTemplate(
        name="RDP Brute Force",
        cve="RDP-BRUTE",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.AUTHENTICATION,
        base_success_rate=0.25,
        yields_credentials=True,
        description="Brute forcing Remote Desktop Protocol credentials",
    ),
    "DEFAULT-CREDS": ExploitTemplate(
        name="Default Credential Exploitation",
        cve="DEFAULT-CREDS",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.AUTHENTICATION,
        base_success_rate=0.80,
        yields_credentials=True,
        description="Using default/factory credentials on services",
    ),
    "WEAK-PASS": ExploitTemplate(
        name="Weak Password Exploitation",
        cve="WEAK-PASS",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.AUTHENTICATION,
        base_success_rate=0.50,
        yields_credentials=True,
        description="Dictionary attack against weak password policies",
    ),

    # ── Network Attacks ──────────────────────────────────────

    "SYN-FLOOD": ExploitTemplate(
        name="SYN Flood Attack",
        cve="SYN-FLOOD",
        technique=AttackTechnique.IMPACT,
        category=AttackCategory.NETWORK,
        base_success_rate=0.75,
        description="TCP SYN flood to exhaust connection table",
    ),
    "UDP-FLOOD": ExploitTemplate(
        name="UDP Flood Attack",
        cve="UDP-FLOOD",
        technique=AttackTechnique.IMPACT,
        category=AttackCategory.NETWORK,
        base_success_rate=0.70,
        description="UDP packet flood for bandwidth exhaustion",
    ),
    "AMP-ATTACK": ExploitTemplate(
        name="DNS Amplification Attack",
        cve="AMP-ATTACK",
        technique=AttackTechnique.IMPACT,
        category=AttackCategory.NETWORK,
        base_success_rate=0.65,
        description="DNS amplification for volumetric DDoS",
    ),
    "PORT-SCAN": ExploitTemplate(
        name="Port Scanning",
        cve="PORT-SCAN",
        technique=AttackTechnique.RECONNAISSANCE,
        category=AttackCategory.RECONNAISSANCE,
        base_success_rate=0.90,
        is_stealthy=True,
        description="Scanning open ports to identify running services",
    ),
    "IP-SPOOF": ExploitTemplate(
        name="IP Spoofing",
        cve="IP-SPOOF",
        technique=AttackTechnique.DEFENSE_EVASION,
        category=AttackCategory.NETWORK,
        base_success_rate=0.50,
        is_stealthy=True,
        description="Forging source IP addresses to evade detection",
    ),
    "DNS-POISON": ExploitTemplate(
        name="DNS Spoofing / Poisoning",
        cve="DNS-POISON",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.NETWORK,
        base_success_rate=0.40,
        description="Corrupting DNS cache to redirect traffic",
    ),
    "ARP-SPOOF": ExploitTemplate(
        name="ARP Spoofing",
        cve="ARP-SPOOF",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.NETWORK,
        base_success_rate=0.60,
        is_stealthy=True,
        description="ARP cache poisoning for MITM attacks on LAN",
    ),
    "MITM-ATTACK": ExploitTemplate(
        name="Man-in-the-Middle Attack",
        cve="MITM-ATTACK",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.NETWORK,
        base_success_rate=0.45,
        yields_credentials=True,
        description="Intercepting network traffic between two parties",
    ),
    "PACKET-SNIFF": ExploitTemplate(
        name="Packet Sniffing",
        cve="PACKET-SNIFF",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.NETWORK,
        base_success_rate=0.55,
        yields_credentials=True,
        is_stealthy=True,
        description="Passive network traffic capture for credential theft",
    ),

    # ── Malware ──────────────────────────────────────────────

    "RANSOMWARE": ExploitTemplate(
        name="Ransomware Deployment",
        cve="RANSOMWARE",
        technique=AttackTechnique.IMPACT,
        category=AttackCategory.MALWARE,
        base_success_rate=0.45,
        yields_root=True,
        description="Encrypt files and demand ransom for decryption key",
    ),
    "TROJAN": ExploitTemplate(
        name="Trojan Horse",
        cve="TROJAN",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.MALWARE,
        base_success_rate=0.50,
        is_stealthy=True,
        description="Disguised malicious software granting remote access",
    ),
    "ROOTKIT": ExploitTemplate(
        name="Rootkit Installation",
        cve="ROOTKIT",
        technique=AttackTechnique.DEFENSE_EVASION,
        category=AttackCategory.MALWARE,
        base_success_rate=0.35,
        yields_root=True,
        is_stealthy=True,
        description="Kernel-level rootkit to hide malicious activity",
    ),
    "SPYWARE": ExploitTemplate(
        name="Spyware Installation",
        cve="SPYWARE",
        technique=AttackTechnique.COLLECTION,
        category=AttackCategory.MALWARE,
        base_success_rate=0.50,
        is_stealthy=True,
        yields_credentials=True,
        description="Covert data collection and exfiltration software",
    ),
    "BOTNET-RECRUIT": ExploitTemplate(
        name="Botnet Recruitment",
        cve="BOTNET-RECRUIT",
        technique=AttackTechnique.COMMAND_AND_CONTROL,
        category=AttackCategory.MALWARE,
        base_success_rate=0.45,
        description="Enrolling compromised machine into botnet C2 infrastructure",
    ),
    "WORM-SPREAD": ExploitTemplate(
        name="Worm Propagation",
        cve="WORM-SPREAD",
        technique=AttackTechnique.LATERAL_MOVEMENT,
        category=AttackCategory.MALWARE,
        base_success_rate=0.55,
        description="Self-replicating malware spreading across network",
    ),
    "CRYPTOJACK": ExploitTemplate(
        name="Cryptojacking",
        cve="CRYPTOJACK",
        technique=AttackTechnique.IMPACT,
        category=AttackCategory.MALWARE,
        base_success_rate=0.60,
        is_stealthy=True,
        description="Unauthorized cryptocurrency mining on server resources",
    ),
    "WEB-SHELL": ExploitTemplate(
        name="Web Shell Upload",
        cve="WEB-SHELL",
        technique=AttackTechnique.PERSISTENCE,
        category=AttackCategory.MALWARE,
        base_success_rate=0.55,
        yields_root=True,
        description="Upload PHP/JSP/ASPX shell for persistent remote access",
    ),
    "BACKDOOR": ExploitTemplate(
        name="Backdoor Installation",
        cve="BACKDOOR",
        technique=AttackTechnique.PERSISTENCE,
        category=AttackCategory.MALWARE,
        base_success_rate=0.50,
        yields_root=True,
        is_stealthy=True,
        description="Hidden access mechanism bypassing authentication",
    ),

    # ── Privilege Escalation & Memory Attacks ────────────────

    "BUFFER-OVERFLOW": ExploitTemplate(
        name="Buffer Overflow",
        cve="BUFFER-OVERFLOW",
        technique=AttackTechnique.PRIVILEGE_ESCALATION,
        category=AttackCategory.MEMORY,
        base_success_rate=0.40,
        required_access="local",
        yields_root=True,
        description="Stack/heap buffer overflow for code execution",
    ),
    "HEAP-SPRAY": ExploitTemplate(
        name="Heap Spraying",
        cve="HEAP-SPRAY",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.MEMORY,
        base_success_rate=0.35,
        required_access="local",
        yields_root=True,
        description="Heap spray to facilitate exploit reliability",
    ),
    "RACE-CONDITION": ExploitTemplate(
        name="Race Condition Exploit",
        cve="RACE-CONDITION",
        technique=AttackTechnique.PRIVILEGE_ESCALATION,
        category=AttackCategory.MEMORY,
        base_success_rate=0.30,
        required_access="local",
        yields_root=True,
        description="TOCTOU race condition for privilege escalation",
    ),
    "USE-AFTER-FREE": ExploitTemplate(
        name="Use-After-Free",
        cve="USE-AFTER-FREE",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.MEMORY,
        base_success_rate=0.35,
        required_access="local",
        yields_root=True,
        description="Exploiting dangling pointer after memory deallocation",
    ),
    "KERNEL-EXPLOIT": ExploitTemplate(
        name="Kernel Exploit",
        cve="KERNEL-EXPLOIT",
        technique=AttackTechnique.PRIVILEGE_ESCALATION,
        category=AttackCategory.PRIVILEGE,
        base_success_rate=0.30,
        required_access="local",
        yields_root=True,
        description="Linux kernel vulnerability for root privilege escalation",
    ),
    "PRIV-ESC-SUID": ExploitTemplate(
        name="SUID Binary Escalation",
        cve="PRIV-ESC-SUID",
        technique=AttackTechnique.PRIVILEGE_ESCALATION,
        category=AttackCategory.PRIVILEGE,
        base_success_rate=0.55,
        required_access="local",
        yields_root=True,
        description="Exploiting misconfigured SUID binaries for root access",
    ),

    # ── Persistence Mechanisms ───────────────────────────────

    "CRON-BACKDOOR": ExploitTemplate(
        name="Scheduled Task Backdoor",
        cve="CRON-BACKDOOR",
        technique=AttackTechnique.PERSISTENCE,
        category=AttackCategory.PERSISTENCE,
        base_success_rate=0.65,
        is_stealthy=True,
        description="Malicious crontab entry for persistent access",
    ),
    "SSH-KEY-INJECT": ExploitTemplate(
        name="SSH Key Injection",
        cve="SSH-KEY-INJECT",
        technique=AttackTechnique.PERSISTENCE,
        category=AttackCategory.PERSISTENCE,
        base_success_rate=0.70,
        is_stealthy=True,
        yields_credentials=True,
        description="Adding attacker's SSH key to authorized_keys",
    ),
    "LOG-TAMPER": ExploitTemplate(
        name="Log Tampering",
        cve="LOG-TAMPER",
        technique=AttackTechnique.DEFENSE_EVASION,
        category=AttackCategory.DATA,
        base_success_rate=0.60,
        is_stealthy=True,
        description="Modifying or deleting log entries to cover tracks",
    ),

    # ── Data Attacks ─────────────────────────────────────────

    "DATA-EXFIL": ExploitTemplate(
        name="Data Exfiltration",
        cve="DATA-EXFIL",
        technique=AttackTechnique.EXFILTRATION,
        category=AttackCategory.DATA,
        base_success_rate=0.55,
        is_stealthy=True,
        description="Covert extraction of sensitive data from server",
    ),

    # ── Reconnaissance ───────────────────────────────────────

    "BANNER-GRAB": ExploitTemplate(
        name="Banner Grabbing",
        cve="BANNER-GRAB",
        technique=AttackTechnique.RECONNAISSANCE,
        category=AttackCategory.RECONNAISSANCE,
        base_success_rate=0.95,
        is_stealthy=True,
        description="Connecting to services to identify software versions",
    ),
    "SERVICE-ENUM": ExploitTemplate(
        name="Service Enumeration",
        cve="SERVICE-ENUM",
        technique=AttackTechnique.DISCOVERY,
        category=AttackCategory.RECONNAISSANCE,
        base_success_rate=0.85,
        is_stealthy=True,
        description="Enumerating running services and their configurations",
    ),
    "SUBDOMAIN-ENUM": ExploitTemplate(
        name="Subdomain Enumeration",
        cve="SUBDOMAIN-ENUM",
        technique=AttackTechnique.RECONNAISSANCE,
        category=AttackCategory.RECONNAISSANCE,
        base_success_rate=0.90,
        is_stealthy=True,
        description="Discovering subdomains for expanded attack surface",
    ),

    # ── Cloud & Container ────────────────────────────────────

    "CONTAINER-ESCAPE": ExploitTemplate(
        name="Container Escape",
        cve="CONTAINER-ESCAPE",
        technique=AttackTechnique.PRIVILEGE_ESCALATION,
        category=AttackCategory.CLOUD_CONTAINER,
        base_success_rate=0.30,
        required_access="local",
        yields_root=True,
        description="Escaping container sandbox to access host system",
    ),
    "K8S-EXPLOIT": ExploitTemplate(
        name="Kubernetes Exploit",
        cve="K8S-EXPLOIT",
        technique=AttackTechnique.LATERAL_MOVEMENT,
        category=AttackCategory.CLOUD_CONTAINER,
        base_success_rate=0.35,
        yields_root=True,
        description="Exploiting Kubernetes misconfigurations for cluster access",
    ),
    "SUPPLY-CHAIN": ExploitTemplate(
        name="Supply Chain Attack",
        cve="SUPPLY-CHAIN",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.CLOUD_CONTAINER,
        base_success_rate=0.40,
        is_stealthy=True,
        description="Compromising dependencies or build pipeline",
    ),
    "CICD-ATTACK": ExploitTemplate(
        name="CI/CD Pipeline Attack",
        cve="CICD-ATTACK",
        technique=AttackTechnique.EXECUTION,
        category=AttackCategory.CLOUD_CONTAINER,
        base_success_rate=0.45,
        yields_root=True,
        description="Injecting malicious code via CI/CD pipeline compromise",
    ),
    "CLOUD-CRED-THEFT": ExploitTemplate(
        name="Cloud Credential Theft",
        cve="CLOUD-CRED-THEFT",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.CLOUD_CONTAINER,
        base_success_rate=0.50,
        yields_credentials=True,
        description="Stealing cloud IAM credentials from metadata services",
    ),
    "S3-MISCONFIG": ExploitTemplate(
        name="Misconfigured S3 Bucket",
        cve="S3-MISCONFIG",
        technique=AttackTechnique.COLLECTION,
        category=AttackCategory.CLOUD_CONTAINER,
        base_success_rate=0.80,
        description="Accessing publicly exposed cloud storage buckets",
    ),

    # ── Wireless / Remote Access ─────────────────────────────

    "VPN-EXPLOIT": ExploitTemplate(
        name="VPN Exploit",
        cve="VPN-EXPLOIT",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WIRELESS,
        base_success_rate=0.40,
        yields_credentials=True,
        description="Exploiting VPN vulnerabilities for network access",
    ),
    "WIFI-CRACK": ExploitTemplate(
        name="Wi-Fi Cracking",
        cve="WIFI-CRACK",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.WIRELESS,
        base_success_rate=0.35,
        required_access="physical",
        yields_credentials=True,
        description="Cracking wireless network encryption (WPA/WPA2)",
    ),
    "EVIL-TWIN": ExploitTemplate(
        name="Evil Twin Access Point",
        cve="EVIL-TWIN",
        technique=AttackTechnique.CREDENTIAL_ACCESS,
        category=AttackCategory.WIRELESS,
        base_success_rate=0.55,
        required_access="physical",
        yields_credentials=True,
        description="Rogue access point mimicking legitimate network",
    ),

    # ── Misconfiguration Exploits ────────────────────────────

    "OPEN-PORTS": ExploitTemplate(
        name="Open Port Exploitation",
        cve="OPEN-PORTS",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.NETWORK,
        base_success_rate=0.70,
        description="Exploiting unnecessarily exposed network ports",
    ),
    "MISCONFIG": ExploitTemplate(
        name="Service Misconfiguration",
        cve="MISCONFIG",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.NETWORK,
        base_success_rate=0.65,
        description="Exploiting insecure default configurations",
    ),
    "EXPOSED-DB": ExploitTemplate(
        name="Exposed Database",
        cve="EXPOSED-DB",
        technique=AttackTechnique.COLLECTION,
        category=AttackCategory.DATA,
        base_success_rate=0.85,
        yields_credentials=True,
        description="Accessing database exposed without authentication",
    ),
    "VULN-PLUGIN": ExploitTemplate(
        name="Vulnerable Plugin/Module",
        cve="VULN-PLUGIN",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.60,
        description="Exploiting known vulnerable third-party plugins",
    ),
    "POOR-FIREWALL": ExploitTemplate(
        name="Poor Firewall Rules",
        cve="POOR-FIREWALL",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.NETWORK,
        base_success_rate=0.70,
        description="Bypassing misconfigured firewall rulesets",
    ),
    "INSECURE-API": ExploitTemplate(
        name="Insecure API Exploitation",
        cve="INSECURE-API",
        technique=AttackTechnique.INITIAL_ACCESS,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.60,
        yields_credentials=True,
        description="Exploiting APIs without proper authentication/authorization",
    ),
    "API-ABUSE": ExploitTemplate(
        name="API Abuse / Rate Abuse",
        cve="API-ABUSE",
        technique=AttackTechnique.IMPACT,
        category=AttackCategory.WEB_APPLICATION,
        base_success_rate=0.70,
        description="Abusing API endpoints through excessive/malicious requests",
    ),
}


def get_exploit(cve: str) -> ExploitTemplate | None:
    """Look up an exploit template by CVE ID."""
    return EXPLOIT_CATALOG.get(cve)


def get_exploits_by_category(category: AttackCategory) -> list[ExploitTemplate]:
    """Get all exploits in a given category."""
    return [e for e in EXPLOIT_CATALOG.values() if e.category == category]


def get_exploits_by_technique(technique: AttackTechnique) -> list[ExploitTemplate]:
    """Get all exploits using a given ATT&CK technique."""
    return [e for e in EXPLOIT_CATALOG.values() if e.technique == technique]


def get_kill_chain_stage(technique: AttackTechnique) -> int:
    """Get the 0-indexed position in the kill chain."""
    try:
        return KILL_CHAIN_ORDER.index(technique)
    except ValueError:
        return -1
