"""
CyberGAN — Analysis: Anomaly Detector
Statistical baseline + deviation detection for server behavior.
Uses exponential moving averages to detect anomalies beyond
configurable sigma thresholds.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import structlog

from cybergan.config import AnomalyDetectorConfig

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyAlert:
    """An anomaly detection alert."""
    timestamp: float
    metric_name: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    deviation_sigma: float
    severity: str = "warning"
    details: dict = field(default_factory=dict)


class ExponentialBaseline:
    """Exponential moving average baseline for a single metric."""

    def __init__(self, alpha: float = 0.01, min_samples: int = 100):
        self.alpha = alpha
        self.min_samples = min_samples
        self.mean: float = 0.0
        self.var: float = 0.0
        self.samples: int = 0

    def update(self, value: float):
        """Update the baseline with a new observation."""
        self.samples += 1
        if self.samples == 1:
            self.mean = value
            self.var = 0.0
        else:
            delta = value - self.mean
            self.mean += self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)

    @property
    def std(self) -> float:
        return max(np.sqrt(self.var), 1e-6)

    @property
    def is_ready(self) -> bool:
        return self.samples >= self.min_samples

    def deviation(self, value: float) -> float:
        """Compute how many standard deviations away from the mean."""
        if not self.is_ready:
            return 0.0
        return abs(value - self.mean) / self.std


class AnomalyDetector:
    """
    Statistical anomaly detector using exponential moving averages.

    Maintains baselines for multiple metrics and flags values that
    deviate beyond a configurable number of standard deviations.

    Tracked metrics:
    - Login attempts per minute
    - Active connections
    - CPU usage
    - Network traffic
    - File changes per minute
    - Process count
    - Request rate
    """

    def __init__(self, config: AnomalyDetectorConfig):
        self.config = config
        self._baselines: dict[str, ExponentialBaseline] = defaultdict(
            lambda: ExponentialBaseline(
                alpha=config.adaptive_learning_rate,
                min_samples=config.min_samples_for_baseline,
            )
        )
        self._alert_cooldowns: dict[str, float] = {}

    def observe(self, metric_name: str, value: float) -> AnomalyAlert | None:
        """
        Record a metric observation and check for anomaly.

        Args:
            metric_name: Name of the metric (e.g., "login_attempts", "cpu")
            value: Current value

        Returns:
            AnomalyAlert if anomaly detected, None otherwise.
        """
        baseline = self._baselines[metric_name]
        deviation = baseline.deviation(value)

        # Update baseline (even if anomalous — EMA adapts)
        baseline.update(value)

        # Check for anomaly
        if baseline.is_ready and deviation > self.config.deviation_sigma:
            # Cooldown check
            now = time.time()
            last_alert = self._alert_cooldowns.get(metric_name, 0)
            if now - last_alert < 60:  # 1 minute cooldown per metric
                return None

            self._alert_cooldowns[metric_name] = now

            severity = "critical" if deviation > self.config.deviation_sigma * 2 else "warning"

            return AnomalyAlert(
                timestamp=now,
                metric_name=metric_name,
                current_value=value,
                baseline_mean=baseline.mean,
                baseline_std=baseline.std,
                deviation_sigma=deviation,
                severity=severity,
                details={
                    "samples": baseline.samples,
                    "threshold": self.config.deviation_sigma,
                },
            )

        return None

    def observe_many(self, metrics: dict[str, float]) -> list[AnomalyAlert]:
        """Observe multiple metrics at once."""
        alerts = []
        for name, value in metrics.items():
            alert = self.observe(name, value)
            if alert:
                alerts.append(alert)
        return alerts

    def get_baseline(self, metric_name: str) -> dict:
        """Get current baseline stats for a metric."""
        b = self._baselines.get(metric_name)
        if not b:
            return {}
        return {
            "mean": b.mean,
            "std": b.std,
            "samples": b.samples,
            "ready": b.is_ready,
        }

    def get_all_baselines(self) -> dict[str, dict]:
        """Get all baseline stats."""
        return {name: self.get_baseline(name) for name in self._baselines}
