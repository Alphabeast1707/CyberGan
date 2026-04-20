"""
CyberGAN — Main Agent Daemon
Orchestrates all perception monitors, analysis pipeline, RL brain,
and defensive action engine. The core runtime of the security agent.
"""

from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from cybergan.config import CyberGANConfig
from cybergan.perception.log_monitor import LogMonitor, LogEvent
from cybergan.perception.network_monitor import NetworkMonitor, NetworkEvent
from cybergan.perception.file_monitor import FileMonitor, FileEvent
from cybergan.perception.process_monitor import ProcessMonitor, ProcessEvent
from cybergan.perception.web_monitor import WebMonitor, WebEvent
from cybergan.perception.system_metrics import SystemMetricsMonitor, SystemMetrics, MetricsEvent
from cybergan.analysis.feature_extractor import FeatureExtractor
from cybergan.analysis.anomaly_detector import AnomalyDetector
from cybergan.analysis.threat_classifier import ThreatClassifier
from cybergan.analysis.risk_scorer import RiskScorer
from cybergan.brain.policy import Brain
from cybergan.brain.action_space import DEFENSE_ACTIONS, ActionRisk
from cybergan.actions.firewall import FirewallManager
from cybergan.actions.alerter import Alerter, SecurityAlert

logger = structlog.get_logger(__name__)


@dataclass
class AgentStats:
    """Runtime statistics for the agent."""
    started_at: float = 0.0
    events_processed: int = 0
    threats_detected: int = 0
    actions_taken: int = 0
    ips_blocked: int = 0
    alerts_sent: int = 0
    false_positives: int = 0
    last_event_at: float = 0.0


class CyberGANAgent:
    """
    Main CyberGAN security agent daemon.

    Architecture:
    1. Perception monitors run in async tasks, pushing events to a shared queue
    2. The event processor pulls events, runs them through analysis
    3. The RL brain makes defense decisions
    4. The action engine executes the chosen defense
    5. Results feed back into the experience buffer for online learning

    Modes:
    - advisory: Detect and alert only, no autonomous actions
    - autonomous: Fully automated detection and response
    - hybrid: Autonomous for low-risk actions, approval for high-risk
    """

    def __init__(self, config: CyberGANConfig):
        self.config = config
        self.stats = AgentStats()
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._tasks: list[asyncio.Task] = []

        # ── Perception Layer ──
        self.log_monitor = LogMonitor(config.perception.log_monitor)
        self.network_monitor = NetworkMonitor(config.perception.network_monitor)
        self.file_monitor = FileMonitor(config.perception.file_monitor)
        self.process_monitor = ProcessMonitor(config.perception.process_monitor)
        self.web_monitor = WebMonitor(config.perception.web_monitor)
        self.system_metrics = SystemMetricsMonitor(config.perception.system_metrics)

        # ── Analysis Engine ──
        self.feature_extractor = FeatureExtractor(
            window_s=config.brain.observation_window_s
        )
        self.anomaly_detector = AnomalyDetector(config.analysis.anomaly_detector)
        self.threat_classifier = ThreatClassifier(config.analysis.threat_classifier)
        self.risk_scorer = RiskScorer(config.analysis.risk_scorer)

        # ── RL Brain ──
        obs_dim = self.feature_extractor.get_observation_dim()
        self.brain = Brain(config.brain, obs_dim)

        # ── Action Engine ──
        self.firewall = FirewallManager(config.actions.firewall)
        self.alerter = Alerter(config.actions.alerter)

        # ── Action cooldowns ──
        self._action_cooldowns: dict[str, float] = {}
        self._actions_this_minute: list[float] = []

        # ── Dashboard broadcaster (set via set_dashboard) ──
        self._dashboard_broadcast = None   # Async callable: (dict) -> None

    def set_dashboard(self, broadcast_fn):
        """Register a dashboard broadcast function for real-time updates."""
        self._dashboard_broadcast = broadcast_fn
        logger.info("agent.dashboard_wired", msg="Real events will stream to dashboard")

    async def _broadcast(self, payload: dict):
        """Fire-and-forget broadcast to dashboard."""
        if self._dashboard_broadcast:
            try:
                await self._dashboard_broadcast(payload)
            except Exception:
                pass

    async def start(self):
        """Start the CyberGAN agent."""
        self._running = True
        self.stats.started_at = time.time()

        # ASCII art banner
        print(r"""
   ██████╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗  █████╗ ███╗   ██╗
  ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝ ██╔══██╗████╗  ██║
  ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝██║  ███╗███████║██╔██╗ ██║
  ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗██║   ██║██╔══██║██║╚██╗██║
  ╚██████╗   ██║   ██████╔╝███████╗██║  ██║╚██████╔╝██║  ██║██║ ╚████║
   ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝
         AI-Powered Server Security Agent v0.1.0
        """)

        logger.info(
            "agent.starting",
            mode=self.config.agent.mode,
            name=self.config.agent.name,
            brain="RL" if self.brain.model_loaded else "heuristic",
        )

        # Initialize firewall
        await self.firewall.initialize()

        # Start perception monitors
        monitors = []
        if self.config.perception.log_monitor.enabled:
            monitors.append(("log_monitor", self.log_monitor.start(self._event_queue)))
        if self.config.perception.network_monitor.enabled:
            monitors.append(("network_monitor", self.network_monitor.start(self._event_queue)))
        if self.config.perception.file_monitor.enabled:
            monitors.append(("file_monitor", self.file_monitor.start(self._event_queue)))
        if self.config.perception.process_monitor.enabled:
            monitors.append(("process_monitor", self.process_monitor.start(self._event_queue)))
        if self.config.perception.web_monitor.enabled:
            monitors.append(("web_monitor", self.web_monitor.start(self._event_queue)))
        if self.config.perception.system_metrics.enabled:
            monitors.append(("system_metrics", self.system_metrics.start(self._event_queue)))

        for name, coro in monitors:
            task = asyncio.create_task(coro)
            task.set_name(name)
            self._tasks.append(task)

        # Start event processor
        processor_task = asyncio.create_task(self._process_events())
        processor_task.set_name("event_processor")
        self._tasks.append(processor_task)

        # Start periodic cleanup
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        cleanup_task.set_name("cleanup")
        self._tasks.append(cleanup_task)

        logger.info("agent.started", monitors=len(monitors))

        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("agent.shutting_down")

    async def stop(self):
        """Gracefully stop the agent."""
        self._running = False
        logger.info("agent.stopping")

        # Stop all monitors
        self.log_monitor.stop()
        self.network_monitor.stop()
        self.file_monitor.stop()
        self.process_monitor.stop()
        self.web_monitor.stop()
        self.system_metrics.stop()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        self._print_stats()

    async def _process_events(self):
        """Main event processing loop."""
        while self._running:
            try:
                # Get event from queue (with timeout to check running state)
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Handle SystemMetrics separately (update feature extractor)
                if isinstance(event, SystemMetrics):
                    self.feature_extractor.update_metrics(event)
                    # Feed metrics to anomaly detector
                    self.anomaly_detector.observe_many({
                        "cpu": event.cpu_percent,
                        "memory": event.memory_percent,
                        "connections": event.network_connections,
                        "processes": event.process_count,
                    })
                    # ── Broadcast REAL system health to dashboard ──────────
                    await self._broadcast({
                        "type": "status",
                        "system": {
                            "cpu": event.cpu_percent,
                            "memory": event.memory_percent,
                            "disk": event.disk_percent,
                            "network_mbps": round(getattr(event, "network_sent_bytes", 0) / 1_000_000, 2),
                        },
                        "stats": {
                            "events_processed": self.stats.events_processed,
                            "threats_detected": self.stats.threats_detected,
                            "actions_taken": self.stats.actions_taken,
                            "ips_blocked": self.stats.ips_blocked,
                            "alerts_sent": self.stats.alerts_sent,
                        },
                        "mode": self.config.agent.mode,
                        "brain": "rl" if self.brain.model_loaded else "heuristic",
                        "uptime_s": time.time() - self.stats.started_at,
                    })
                    continue

                if isinstance(event, MetricsEvent):
                    # Metrics events are already high-level alerts
                    pass

                self.stats.events_processed += 1
                self.stats.last_event_at = time.time()

                # Add to feature extractor
                self.feature_extractor.add_event(event)

                # Classify the threat
                classification = self.threat_classifier.classify(event)

                if not classification.is_attack:
                    # Broadcast benign events too (so dashboard is alive with real activity)
                    await self._broadcast({
                        "type": "status",
                        "stats": {
                            "events_processed": self.stats.events_processed,
                            "threats_detected": self.stats.threats_detected,
                            "actions_taken": self.stats.actions_taken,
                            "ips_blocked": self.stats.ips_blocked,
                            "alerts_sent": self.stats.alerts_sent,
                        },
                    })
                    continue

                self.stats.threats_detected += 1

                # Score the risk
                severity = getattr(event, "severity", "medium")
                risk = self.risk_scorer.score(
                    event_type=classification.event_type,
                    severity=severity,
                    confidence=classification.confidence,
                    context=classification.details.get("progression"),
                )

                # Get security state observation
                state = self.feature_extractor.extract()
                state.risk_score = risk.score

                # Ask the brain for a decision
                action, confidence, metadata = self.brain.decide(
                    observation=state.observation,
                    risk_score=risk.score,
                    event_type=classification.event_type,
                )

                # Check mode permissions
                should_execute = self._should_execute_action(action, confidence, risk)

                if should_execute:
                    # Execute the defensive action
                    await self._execute_action(action, event, risk, classification)
                    self.stats.actions_taken += 1

                # Always alert if threshold met
                if risk.should_alert:
                    await self._send_alert(event, classification, risk, action, should_execute)
                    self.stats.alerts_sent += 1

                # ── Broadcast REAL threat to dashboard ────────────────────
                source_ip = getattr(event, "source_ip", "")
                await self._broadcast({
                    "type": "threat",
                    "timestamp": time.time(),
                    "event_type": classification.event_type,
                    "title": classification.event_type.replace("_", " ").title(),
                    "severity": risk.level,
                    "source_ip": source_ip or "internal",
                    "description": (
                        getattr(event, "raw_line", "") or
                        getattr(event, "command_line", "") or
                        getattr(event, "process_name", "") or
                        str(event)[:200]
                    ),
                    "action_taken": f"{action.name} ({'executed' if should_execute else 'advisory only'})",
                    "mitre_techniques": classification.mitre_techniques,
                    "kill_chain_stage": classification.kill_chain_stage,
                    "risk_score": risk.score,
                    "stats": {
                        "events_processed": self.stats.events_processed,
                        "threats_detected": self.stats.threats_detected,
                        "actions_taken": self.stats.actions_taken,
                        "ips_blocked": self.stats.ips_blocked,
                        "alerts_sent": self.stats.alerts_sent,
                    },
                    "blocked_ips": self.firewall.get_blocked_ips()[:20],
                })

            except Exception as e:
                logger.error("agent.process_error", error=str(e), exc_info=True)

    def _should_execute_action(self, action, confidence: float, risk) -> bool:
        """Determine if an action should be executed based on mode and risk."""
        mode = self.config.agent.mode

        if mode == "advisory":
            return False  # Never auto-execute in advisory mode

        if mode == "autonomous":
            return confidence >= self.config.agent.advisory_threshold

        # Hybrid mode
        if action.risk <= ActionRisk.LOW:
            return True  # Always execute low-risk actions
        elif action.risk <= ActionRisk.MEDIUM:
            return confidence >= self.config.agent.confidence_threshold
        else:
            # High/critical risk: only if very confident
            return confidence >= 0.9 and not action.requires_approval

    async def _execute_action(self, action, event, risk, classification):
        """Execute a defensive action."""
        # Rate limiting
        now = time.time()
        self._actions_this_minute = [t for t in self._actions_this_minute if now - t < 60]
        if len(self._actions_this_minute) >= self.config.agent.max_actions_per_minute:
            logger.warning("agent.rate_limited", action=action.name)
            return

        # Cooldown check
        last_use = self._action_cooldowns.get(action.name, 0)
        if now - last_use < action.cooldown_s:
            return

        self._action_cooldowns[action.name] = now
        self._actions_this_minute.append(now)

        source_ip = getattr(event, "source_ip", "")

        try:
            if action.name == "block_ip" and source_ip:
                await self.firewall.block_ip(
                    source_ip,
                    reason=f"{classification.event_type} (risk: {risk.score:.0f})",
                    duration_s=self.config.actions.ip_blocker.auto_block_duration_s,
                )
                self.stats.ips_blocked += 1

            elif action.name == "rate_limit" and source_ip:
                await self.firewall.rate_limit_ip(source_ip)

            elif action.name == "firewall_block":
                if source_ip:
                    await self.firewall.block_ip(source_ip, reason=classification.event_type)

            elif action.name == "kill_process":
                pid = getattr(event, "pid", 0)
                if pid:
                    await self._kill_process(pid)

            elif action.name == "monitor":
                pass  # Do nothing, just continue monitoring

            elif action.name == "alert":
                pass  # Alert is sent separately

            logger.info(
                "agent.action_executed",
                action=action.name,
                event_type=classification.event_type,
                source_ip=source_ip,
                risk_score=risk.score,
            )

        except Exception as e:
            logger.error("agent.action_error", action=action.name, error=str(e))

    async def _kill_process(self, pid: int):
        """Kill a suspicious process."""
        import psutil
        try:
            proc = psutil.Process(pid)
            proc_name = proc.name()
            proc.kill()
            logger.info("agent.process_killed", pid=pid, name=proc_name)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning("agent.kill_failed", pid=pid, error=str(e))

    async def _send_alert(self, event, classification, risk, action, action_executed: bool):
        """Send a security alert."""
        source_ip = getattr(event, "source_ip", "")
        event_type = classification.event_type

        alert = SecurityAlert(
            timestamp=time.time(),
            title=f"Threat Detected: {event_type.replace('_', ' ').title()}",
            severity=risk.level,
            event_type=event_type,
            description=getattr(event, "raw_line", "") or getattr(event, "command_line", "") or str(event),
            source_ip=source_ip,
            action_taken=f"{action.name} ({'executed' if action_executed else 'advisory only'})",
            risk_score=risk.score,
            mitre_techniques=classification.mitre_techniques,
            details={
                "kill_chain_stage": classification.kill_chain_stage,
                "confidence": classification.confidence,
            },
        )

        await self.alerter.send(alert)

    async def _periodic_cleanup(self):
        """Periodic maintenance tasks."""
        while self._running:
            try:
                await self.firewall.cleanup_expired()
            except Exception as e:
                logger.error("agent.cleanup_error", error=str(e))
            await asyncio.sleep(300)  # Every 5 minutes

    def _print_stats(self):
        """Print runtime statistics."""
        elapsed = time.time() - self.stats.started_at
        hours = elapsed / 3600

        print(f"\n{'═' * 60}")
        print(f"  CyberGAN Agent — Session Summary")
        print(f"{'═' * 60}")
        print(f"  Runtime:           {hours:.1f} hours")
        print(f"  Events Processed:  {self.stats.events_processed:,}")
        print(f"  Threats Detected:  {self.stats.threats_detected:,}")
        print(f"  Actions Taken:     {self.stats.actions_taken:,}")
        print(f"  IPs Blocked:       {self.stats.ips_blocked:,}")
        print(f"  Alerts Sent:       {self.stats.alerts_sent:,}")
        print(f"{'═' * 60}\n")

    def get_status(self) -> dict:
        """Get current agent status for dashboard/API."""
        return {
            "running": self._running,
            "mode": self.config.agent.mode,
            "brain": "rl" if self.brain.model_loaded else "heuristic",
            "uptime_s": time.time() - self.stats.started_at if self.stats.started_at else 0,
            "stats": {
                "events_processed": self.stats.events_processed,
                "threats_detected": self.stats.threats_detected,
                "actions_taken": self.stats.actions_taken,
                "ips_blocked": self.stats.ips_blocked,
                "alerts_sent": self.stats.alerts_sent,
            },
            "blocked_ips": self.firewall.get_blocked_ips(),
            "active_threats": self.threat_classifier.get_active_threats(),
            "baselines": self.anomaly_detector.get_all_baselines(),
        }
