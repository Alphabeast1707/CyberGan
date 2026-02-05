"""
CyberGAN — League: ELO Rating System
Tracks relative skill of Red and Blue agents across training epochs.
Uses standard ELO with configurable K-factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ELORating:
    """ELO rating for a single agent."""
    rating: float = 1000.0
    history: list[float] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    def record(self):
        """Snapshot current rating to history."""
        self.history.append(self.rating)

    def to_dict(self) -> dict:
        return {
            "rating": round(self.rating, 1),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "history": [round(r, 1) for r in self.history],
        }


class ELOSystem:
    """
    Manages ELO ratings for Red and Blue agents.

    After each episode:
      - If Red scored higher → Red wins
      - If Blue scored higher → Blue wins
      - If scores are within margin → draw
    """

    def __init__(self, k_factor: float = 32.0, draw_margin: float = 2.0, initial_elo: float = 1000.0):
        self.k_factor = k_factor
        self.draw_margin = draw_margin
        self.red = ELORating(rating=initial_elo)
        self.blue = ELORating(rating=initial_elo)

    def update(self, red_score: float, blue_score: float):
        """
        Update ELO ratings based on episode scores.

        Args:
            red_score: Red agent's total reward for the episode
            blue_score: Blue agent's total reward for the episode
        """
        # Determine outcome
        diff = red_score - blue_score
        if abs(diff) < self.draw_margin:
            red_result = 0.5
            blue_result = 0.5
            self.red.draws += 1
            self.blue.draws += 1
        elif diff > 0:
            red_result = 1.0
            blue_result = 0.0
            self.red.wins += 1
            self.blue.losses += 1
        else:
            red_result = 0.0
            blue_result = 1.0
            self.red.losses += 1
            self.blue.wins += 1

        # Expected scores
        exp_red = 1.0 / (1.0 + 10 ** ((self.blue.rating - self.red.rating) / 400))
        exp_blue = 1.0 - exp_red

        # Update ratings
        self.red.rating += self.k_factor * (red_result - exp_red)
        self.blue.rating += self.k_factor * (blue_result - exp_blue)

        # Record history
        self.red.record()
        self.blue.record()

    def get_ratings(self) -> dict:
        return {
            "red": self.red.to_dict(),
            "blue": self.blue.to_dict(),
        }

    def get_summary(self) -> str:
        return (
            f"ELO — Red: {self.red.rating:.0f} ({self.red.wins}W/{self.red.losses}L/{self.red.draws}D) | "
            f"Blue: {self.blue.rating:.0f} ({self.blue.wins}W/{self.blue.losses}L/{self.blue.draws}D)"
        )
