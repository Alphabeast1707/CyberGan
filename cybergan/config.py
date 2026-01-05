"""
CyberGAN — Configuration Management
Pydantic models for type-safe, validated configuration loading from YAML.
Platform-aware defaults: macOS uses system.log + pf; Linux uses syslog + iptables.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

_IS_MACOS = platform.system() == "Darwin"
_IS_LINUX = platform.system() == "Linux"


def _default_log_files() -> list[str]:
    """Platform-aware default log file paths."""
    if _IS_MACOS:
        paths = ["/var/log/system.log"]
        # macOS also has install.log, wifi logs etc.
        optional = ["/var/log/install.log", "/private/var/log/system.log"]
        for p in optional:
            if Path(p).exists() and p not in paths:
                paths.append(p)
        return paths
    else:  # Linux
        candidates = ["/var/log/syslog", "/var/log/auth.log",
                      "/var/log/messages", "/var/log/secure",
                      "/var/log/kern.log"]
        return [p for p in candidates if Path(p).exists()] or ["/var/log/syslog"]


def _default_watch_paths() -> list[str]:
    """Platform-aware file system watch paths."""
    if _IS_MACOS:
        home = str(Path.home())
        return [
            f"{home}/.ssh/",
            "/usr/local/bin/",
            "/usr/local/sbin/",
            "/Library/LaunchDaemons/",
            "/Library/LaunchAgents/",
            "/etc/",
        ]
    else:  # Linux
        return ["/etc/", "/usr/bin/", "/usr/sbin/", "/var/www/", "/root/.ssh/"]


def _default_firewall_backend() -> str:
    if _IS_MACOS:
        return "pf"      # macOS Packet Filter
    return "iptables"    # Linux default


# ─────────────────────────────────────────────
# Perception Config
# ─────────────────────────────────────────────

class LogMonitorConfig(BaseModel):
    enabled: bool = True
    watch_files: list[str] = Field(default_factory=_default_log_files)
    use_macos_log_stream: bool = _IS_MACOS   # Use `log stream` on macOS
    custom_logs: list[str] = []
    poll_interval_ms: int = 500
    max_lines_per_batch: int = 1000
    brute_force_threshold: int = 5
    brute_force_window_s: int = 300


class NetworkMonitorConfig(BaseModel):
    enabled: bool = True
    poll_interval_s: int = 5
    connection_spike_threshold: int = 100
    port_scan_threshold: int = 10
    port_scan_window_s: int = 60
    track_outbound: bool = True
    suspicious_ports: list[int] = [4444, 5555, 6666, 1337, 31337, 8888, 9999]
    whitelisted_ips: list[str] = []


class FileMonitorConfig(BaseModel):
    enabled: bool = True
    watch_paths: list[str] = Field(default_factory=_default_watch_paths)
    exclude_patterns: list[str] = ["*.log", "*.tmp", "*.swp"]
    alert_on_new_executables: bool = True
    alert_on_suid_changes: bool = not _IS_MACOS   # SUID less relevant on macOS
    alert_on_crontab_changes: bool = True


class ProcessMonitorConfig(BaseModel):
    enabled: bool = True
    poll_interval_s: int = 10
    cpu_threshold_percent: float = 90.0
    memory_threshold_mb: float = 2048.0
    suspicious_process_names: list[str] = [
        "nc", "ncat", "netcat", "socat",
        "xmrig", "minerd", "cpuminer", "cryptonight",
    ]
    detect_reverse_shells: bool = True
    detect_cryptominers: bool = True


class WebMonitorConfig(BaseModel):
    enabled: bool = True
    mode: str = "log_parser"
    access_log_path: str = "/var/log/nginx/access.log"
    error_log_path: str = "/var/log/nginx/error.log"
    poll_interval_ms: int = 500
    rate_limit_requests_per_minute: int = 1000
    rate_limit_per_ip: int = 100
    block_suspicious_user_agents: bool = True


class SystemMetricsConfig(BaseModel):
    enabled: bool = True
    poll_interval_s: int = 15
    cpu_alert_threshold: float = 95.0
    memory_alert_threshold: float = 90.0
    disk_alert_threshold: float = 90.0
    network_spike_threshold_mbps: float = 500.0


class PerceptionConfig(BaseModel):
    log_monitor: LogMonitorConfig = LogMonitorConfig()
    network_monitor: NetworkMonitorConfig = NetworkMonitorConfig()
    file_monitor: FileMonitorConfig = FileMonitorConfig()
    process_monitor: ProcessMonitorConfig = ProcessMonitorConfig()
    web_monitor: WebMonitorConfig = WebMonitorConfig()
    system_metrics: SystemMetricsConfig = SystemMetricsConfig()


# ─────────────────────────────────────────────
# Analysis Config
# ─────────────────────────────────────────────

class AnomalyDetectorConfig(BaseModel):
    baseline_window_hours: int = 24
    deviation_sigma: float = 3.0
    min_samples_for_baseline: int = 100
    adaptive_learning_rate: float = 0.01


class RiskScorerConfig(BaseModel):
    severity_weight: float = 0.4
    confidence_weight: float = 0.3
    impact_weight: float = 0.3
    critical_threshold: float = 80.0
    high_threshold: float = 60.0
    medium_threshold: float = 40.0


class ThreatClassifierConfig(BaseModel):
    mitre_mapping: bool = True
    kill_chain_tracking: bool = True


class AnalysisConfig(BaseModel):
    anomaly_detector: AnomalyDetectorConfig = AnomalyDetectorConfig()
    risk_scorer: RiskScorerConfig = RiskScorerConfig()
    threat_classifier: ThreatClassifierConfig = ThreatClassifierConfig()


# ─────────────────────────────────────────────
# Brain Config
# ─────────────────────────────────────────────

class BrainConfig(BaseModel):
    model_path: str = "checkpoints/blue_production.pt"
    fallback_to_heuristic: bool = True
    observation_window_s: int = 60
    action_selection: str = "stochastic"
    temperature: float = 0.5
    device: str = "cpu"


# ─────────────────────────────────────────────
# Actions Config
# ─────────────────────────────────────────────

class FirewallConfig(BaseModel):
    backend: str = Field(default_factory=_default_firewall_backend)
    chain: str = "CYBERGAN"
    auto_cleanup_hours: int = 24
    max_blocked_ips: int = 10000


class IPBlockerConfig(BaseModel):
    enabled: bool = True
    auto_block_duration_s: int = 3600
    permanent_block_threshold: int = 5
    whitelisted_ips: list[str] = ["127.0.0.1", "::1"]
    whitelisted_cidrs: list[str] = []


class WAFConfig(BaseModel):
    enabled: bool = True
    sqli_protection: bool = True
    xss_protection: bool = True
    csrf_protection: bool = True
    rce_protection: bool = True
    lfi_protection: bool = True
    rfi_protection: bool = True
    xxe_protection: bool = True
    ssrf_protection: bool = True
    command_injection_protection: bool = True
    directory_traversal_protection: bool = True


class ProcessControlConfig(BaseModel):
    enabled: bool = True
    auto_kill_cryptominers: bool = True
    auto_kill_reverse_shells: bool = True
    require_approval_for_kill: bool = False


class ServiceControlConfig(BaseModel):
    enabled: bool = True
    backend: str = "systemd"
    auto_restart_on_crash: bool = True
    max_restarts_per_hour: int = 5


class AlertChannelConfig(BaseModel):
    type: str = "console"
    enabled: bool = True
    url: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    from_addr: str = ""
    to_addrs: list[str] = []


class AlerterConfig(BaseModel):
    channels: list[AlertChannelConfig] = [AlertChannelConfig(type="console", enabled=True)]
    alert_cooldown_s: int = 60
    batch_alerts: bool = True
    batch_window_s: int = 30


class HoneypotConfig(BaseModel):
    enabled: bool = False
    ports: list[int] = [2222, 8080, 8443, 9090]
    log_interactions: bool = True


class ActionsConfig(BaseModel):
    firewall: FirewallConfig = FirewallConfig()
    ip_blocker: IPBlockerConfig = IPBlockerConfig()
    waf: WAFConfig = WAFConfig()
    process_control: ProcessControlConfig = ProcessControlConfig()
    service_control: ServiceControlConfig = ServiceControlConfig()
    alerter: AlerterConfig = AlerterConfig()
    honeypot: HoneypotConfig = HoneypotConfig()


# ─────────────────────────────────────────────
# Healing Config
# ─────────────────────────────────────────────

class OnlineLearningConfig(BaseModel):
    enabled: bool = True
    learning_rate: float = 1e-4
    update_interval_episodes: int = 10
    min_experience_for_update: int = 50
    reward_for_blocked_attack: float = 5.0
    reward_for_missed_attack: float = -10.0
    reward_for_false_positive: float = -3.0


class ArenaTrainingConfig(BaseModel):
    enabled: bool = True
    background_epochs_per_hour: int = 10
    sync_to_production_interval: int = 100


class ThreatIntelFeed(BaseModel):
    name: str
    url: str
    poll_interval_hours: int = 6


class ThreatIntelConfig(BaseModel):
    enabled: bool = True
    feeds: list[ThreatIntelFeed] = []


class CheckpointManagerConfig(BaseModel):
    max_checkpoints: int = 20
    save_interval_epochs: int = 50
    auto_rollback_on_regression: bool = True
    regression_threshold: float = 0.15


class HealingConfig(BaseModel):
    online_learning: OnlineLearningConfig = OnlineLearningConfig()
    arena_training: ArenaTrainingConfig = ArenaTrainingConfig()
    threat_intel: ThreatIntelConfig = ThreatIntelConfig()
    checkpoint_manager: CheckpointManagerConfig = CheckpointManagerConfig()


# ─────────────────────────────────────────────
# Dashboard Config
# ─────────────────────────────────────────────

class DashboardConfig(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8443
    auth_required: bool = True
    api_key: str = ""


# ─────────────────────────────────────────────
# Training Config (Arena)
# ─────────────────────────────────────────────

class TrainingConfig(BaseModel):
    epochs: int = 200
    steps_per_epoch: int = 64
    episodes_per_epoch: int = 8
    eval_episodes: int = 4
    batch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    device: str = "auto"


class LeagueConfig(BaseModel):
    enabled: bool = True
    save_threshold: float = 0.6
    pool_size: int = 10
    initial_elo: float = 1000.0


# ─────────────────────────────────────────────
# Agent Core Config
# ─────────────────────────────────────────────

class AgentCoreConfig(BaseModel):
    mode: str = "hybrid"  # advisory | autonomous | hybrid
    name: str = "CyberGAN-Agent"
    log_level: str = "INFO"
    data_dir: str = "/var/lib/cybergan"
    pid_file: str = "/var/run/cybergan.pid"
    confidence_threshold: float = 0.7
    advisory_threshold: float = 0.4
    max_actions_per_minute: int = 30
    cooldown_seconds: int = 10


# ─────────────────────────────────────────────
# Root Configuration
# ─────────────────────────────────────────────

class CyberGANConfig(BaseModel):
    """Root configuration model for the CyberGAN agent."""
    agent: AgentCoreConfig = AgentCoreConfig()
    perception: PerceptionConfig = PerceptionConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    brain: BrainConfig = BrainConfig()
    actions: ActionsConfig = ActionsConfig()
    healing: HealingConfig = HealingConfig()
    dashboard: DashboardConfig = DashboardConfig()
    training: TrainingConfig = TrainingConfig()
    league: LeagueConfig = LeagueConfig()
    network: dict = {}  # Raw network topology dict for arena

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CyberGANConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        return cls(**raw)

    @classmethod
    def default(cls) -> "CyberGANConfig":
        """Create a configuration with all defaults."""
        return cls()

    def to_yaml(self, path: str | Path):
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
