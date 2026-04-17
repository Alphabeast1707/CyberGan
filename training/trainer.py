"""
CyberGAN — Training: Main Trainer
Orchestrates the co-evolutionary training loop:
  1. Self-play rollout (Red vs Blue in the arena)
  2. PPO update for both agents
  3. Evaluation + ELO update
  4. League checkpoint management
"""

from __future__ import annotations

import os
import json
import time
from typing import Optional, Callable

import torch
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from arena.env import CyberGANEnv
from agents.red.policy import RedPolicy
from agents.red.observer import flatten_red_obs
from agents.red.actions import compute_red_action_mask
from agents.blue.policy import BluePolicy
from agents.blue.observer import flatten_blue_obs
from agents.blue.actions import compute_blue_action_mask
from agents.league.elo import ELOSystem
from agents.league.opponent_pool import OpponentPool
from training.buffer import RolloutBuffer
from training.ppo import PPO

console = Console()


class CyberGANTrainer:
    """
    Main training orchestrator for the CyberGAN system.

    Manages:
      - Red and Blue PPO policies
      - Self-play rollout collection
      - PPO optimization for both agents
      - ELO tracking and league updates
      - Logging and dashboard callbacks
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        train_cfg = self.config.get("training", {})
        league_cfg = self.config.get("league", {})

        self.config_path = config_path
        self.epochs = train_cfg.get("epochs", 200)
        self.steps_per_epoch = train_cfg.get("steps_per_epoch", 64)
        self.episodes_per_epoch = train_cfg.get("episodes_per_epoch", 8)
        self.eval_episodes = train_cfg.get("eval_episodes", 4)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.lr = train_cfg.get("lr", 3e-4)
        self.gamma = train_cfg.get("gamma", 0.99)
        self.gae_lambda = train_cfg.get("gae_lambda", 0.95)
        self.clip_epsilon = train_cfg.get("clip_epsilon", 0.2)
        self.entropy_coef = train_cfg.get("entropy_coef", 0.01)
        self.value_coef = train_cfg.get("value_coef", 0.5)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 0.5)
        self.ppo_epochs = train_cfg.get("ppo_epochs", 4)

        # Device
        device_str = train_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_str

        # Environment
        self.env = CyberGANEnv(config_path, max_steps=self.steps_per_epoch)
        N = self.env.network.num_nodes
        V = max(self.env.network.num_vulns, 1)

        # Policies
        self.red_policy = RedPolicy(N, V)
        self.blue_policy = BluePolicy(N, V)

        # PPO optimizers
        self.red_ppo = PPO(
            self.red_policy, lr=self.lr, clip_epsilon=self.clip_epsilon,
            value_coef=self.value_coef, entropy_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm, ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size, device=self.device,
        )
        self.blue_ppo = PPO(
            self.blue_policy, lr=self.lr, clip_epsilon=self.clip_epsilon,
            value_coef=self.value_coef, entropy_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm, ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size, device=self.device,
        )

        # Rollout buffers
        red_obs_dim = self.red_policy.obs_dim
        blue_obs_dim = self.blue_policy.obs_dim
        buf_size = self.steps_per_epoch * self.episodes_per_epoch

        self.red_buffer = RolloutBuffer(buf_size, red_obs_dim, 3, self.gamma, self.gae_lambda)
        self.blue_buffer = RolloutBuffer(buf_size, blue_obs_dim, 2, self.gamma, self.gae_lambda)

        # League system
        initial_elo = league_cfg.get("initial_elo", 1000)
        self.elo = ELOSystem(initial_elo=initial_elo)
        self.opponent_pool = OpponentPool(
            max_size=league_cfg.get("pool_size", 10),
            save_threshold=league_cfg.get("save_threshold", 0.6),
        )
        self.league_enabled = league_cfg.get("enabled", True)

        # Logging
        self.log_dir = os.path.join(os.path.dirname(config_path), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.epoch_logs: list[dict] = []

        # Dashboard callback
        self._ws_callback: Optional[Callable] = None

    def set_ws_callback(self, callback: Callable):
        """Set WebSocket callback for real-time dashboard updates."""
        self._ws_callback = callback

    def train(self):
        """Run the full co-evolution training loop."""
        console.print(Panel(
            "[bold cyan]CyberGAN[/] — Adversarial RL Co-Evolution\n"
            "[dim]Red Agent (attacker) vs Blue Agent (defender)[/]\n"
            f"[dim]Device: {self.device} | Epochs: {self.epochs} | "
            f"Steps/epoch: {self.steps_per_epoch}[/]",
            border_style="cyan",
        ))

        for epoch in range(self.epochs):
            t0 = time.time()

            # 1. Collect self-play rollouts
            rollout_stats = self._collect_rollouts()

            # 2. PPO update for both agents
            red_train_stats = self.red_ppo.update(self.red_buffer)
            blue_train_stats = self.blue_ppo.update(self.blue_buffer)

            # 3. Evaluate
            eval_stats = self._evaluate()

            # 4. Update ELO
            self.elo.update(eval_stats["red_avg_score"], eval_stats["blue_avg_score"])

            # 5. League update
            if self.league_enabled:
                red_wr = eval_stats["red_win_rate"]
                blue_wr = eval_stats["blue_win_rate"]
                self.opponent_pool.maybe_save(
                    self.red_policy, "red", epoch,
                    self.elo.red.rating, red_wr,
                )
                self.opponent_pool.maybe_save(
                    self.blue_policy, "blue", epoch,
                    self.elo.blue.rating, blue_wr,
                )

            elapsed = time.time() - t0

            # Log
            epoch_log = {
                "epoch": epoch,
                "elapsed_s": round(elapsed, 2),
                "rollout": rollout_stats,
                "red_train": red_train_stats,
                "blue_train": blue_train_stats,
                "eval": eval_stats,
                "elo": self.elo.get_ratings(),
                "league": self.opponent_pool.get_pool_info(),
            }
            self.epoch_logs.append(epoch_log)

            # Print summary
            self._print_epoch(epoch, epoch_log)

            # Dashboard callback — broadcast real training data
            if self._ws_callback:
                ev = epoch_log["eval"]
                elo = epoch_log["elo"]
                ro = epoch_log["rollout"]
                self._ws_callback({
                    "type": "training",
                    "epoch": epoch,
                    "total_epochs": self.epochs,
                    "progress_pct": round(epoch / max(self.epochs, 1) * 100, 1),
                    "red_reward":   round(ro["avg_red_reward"], 2),
                    "blue_reward":  round(ro["avg_blue_reward"], 2),
                    "red_elo":      round(elo["red"]["rating"], 0),
                    "blue_elo":     round(elo["blue"]["rating"], 0),
                    "red_win_rate": round(ev["red_win_rate"] * 100, 1),
                    "blue_win_rate":round(ev["blue_win_rate"] * 100, 1),
                    "draw_rate":    round(ev["draw_rate"] * 100, 1),
                    "elapsed_s":    epoch_log["elapsed_s"],
                    "device":       self.device,
                    "title":        f"Arena Training — Epoch {epoch+1}/{self.epochs}",
                    "description":  (
                        f"Red ELO {elo['red']['rating']:.0f} vs "
                        f"Blue ELO {elo['blue']['rating']:.0f} | "
                        f"Blue win rate: {ev['blue_win_rate']*100:.0f}%"
                    ),
                    "severity": "low",
                })

            # Save logs periodically
            if (epoch + 1) % 10 == 0:
                self._save_logs()

            # Save checkpoint every 25 epochs
            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(epoch + 1)
                console.print(f"  [dim]  ↳ Checkpoint saved at epoch {epoch+1}[/dim]")

        # Final save
        self._save_logs()
        self._save_models()
        self._print_final_summary()


    def _collect_rollouts(self) -> dict:
        """Run self-play episodes and fill rollout buffers."""
        self.red_buffer.reset()
        self.blue_buffer.reset()

        total_red_reward = 0.0
        total_blue_reward = 0.0
        total_steps = 0

        for _ in range(self.episodes_per_epoch):
            obs_red, info = self.env.reset()
            obs_blue = info.get("blue_obs", obs_red)

            ep_red_reward = 0.0
            ep_blue_reward = 0.0

            for step in range(self.steps_per_epoch):
                # Red action
                red_mask = compute_red_action_mask(
                    self.env.network, self.env._red_scanned, self.env._red_credentials
                )
                red_action, red_log_prob, red_value = self.red_policy.get_action(obs_red, red_mask)

                # Execute Red
                obs_blue_new, red_reward, _, _, red_info = self.env.step_red(red_action)

                # Blue action
                blue_mask = compute_blue_action_mask(self.env.network)
                blue_action, blue_log_prob, blue_value = self.blue_policy.get_action(obs_blue, blue_mask)

                # Execute Blue
                obs_red_new, obs_blue_final, red_penalty, blue_reward, terminated, truncated, blue_info = \
                    self.env.step_blue(blue_action)

                red_reward += red_penalty  # Apply any penalties from Blue's actions

                # Store transitions
                self.red_buffer.add(
                    flatten_red_obs(obs_red), red_action,
                    red_reward, red_value, red_log_prob,
                    terminated or truncated,
                )
                self.blue_buffer.add(
                    flatten_blue_obs(obs_blue), blue_action,
                    blue_reward, blue_value, blue_log_prob,
                    terminated or truncated,
                )

                obs_red = obs_red_new
                obs_blue = obs_blue_final
                ep_red_reward += red_reward
                ep_blue_reward += blue_reward
                total_steps += 1

                if terminated or truncated:
                    break

            total_red_reward += ep_red_reward
            total_blue_reward += ep_blue_reward

        # Compute GAE advantages
        self.red_buffer.compute_gae(last_value=0.0)
        self.blue_buffer.compute_gae(last_value=0.0)

        return {
            "episodes": self.episodes_per_epoch,
            "total_steps": total_steps,
            "avg_red_reward": total_red_reward / self.episodes_per_epoch,
            "avg_blue_reward": total_blue_reward / self.episodes_per_epoch,
        }

    def _evaluate(self) -> dict:
        """Run evaluation episodes (no gradient) and compute win rates."""
        red_wins = 0
        blue_wins = 0
        draws = 0
        total_red = 0.0
        total_blue = 0.0

        for _ in range(self.eval_episodes):
            obs_red, info = self.env.reset()
            obs_blue = info.get("blue_obs", obs_red)
            ep_red = 0.0
            ep_blue = 0.0

            for step in range(self.steps_per_epoch):
                red_action, _, _ = self.red_policy.get_action(obs_red, deterministic=True)
                obs_blue_new, red_reward, _, _, _ = self.env.step_red(red_action)

                blue_action, _, _ = self.blue_policy.get_action(obs_blue, deterministic=True)
                obs_red_new, obs_blue_final, red_penalty, blue_reward, terminated, truncated, _ = \
                    self.env.step_blue(blue_action)

                ep_red += red_reward + red_penalty
                ep_blue += blue_reward
                obs_red = obs_red_new
                obs_blue = obs_blue_final

                if terminated or truncated:
                    break

            total_red += ep_red
            total_blue += ep_blue

            diff = ep_red - ep_blue
            if abs(diff) < 2.0:
                draws += 1
            elif diff > 0:
                red_wins += 1
            else:
                blue_wins += 1

        n = max(self.eval_episodes, 1)
        return {
            "red_avg_score": total_red / n,
            "blue_avg_score": total_blue / n,
            "red_win_rate": red_wins / n,
            "blue_win_rate": blue_wins / n,
            "draw_rate": draws / n,
            "red_wins": red_wins,
            "blue_wins": blue_wins,
            "draws": draws,
        }

    def _print_epoch(self, epoch: int, log: dict):
        """Print a concise epoch summary."""
        elo = log["elo"]
        ev = log["eval"]
        ro = log["rollout"]

        red_elo = elo["red"]["rating"]
        blue_elo = elo["blue"]["rating"]
        red_bar = "█" * max(int(red_elo / 100), 1)
        blue_bar = "█" * max(int(blue_elo / 100), 1)

        # Determine winner color
        if ev["red_win_rate"] > ev["blue_win_rate"]:
            winner = "[red]Red[/red]"
        elif ev["blue_win_rate"] > ev["red_win_rate"]:
            winner = "[cyan]Blue[/cyan]"
        else:
            winner = "[yellow]Draw[/yellow]"

        console.print(
            f"  [dim]Epoch {epoch:4d}[/] │ "
            f"[red]R[/]={ro['avg_red_reward']:+6.1f} "
            f"[cyan]B[/]={ro['avg_blue_reward']:+6.1f} │ "
            f"ELO [red]{red_elo:5.0f}[/] [cyan]{blue_elo:5.0f}[/] │ "
            f"Win: {winner} │ "
            f"⏱ {log['elapsed_s']:.1f}s"
        )

    def _print_final_summary(self):
        """Print final training summary."""
        elo = self.elo.get_ratings()
        pool = self.opponent_pool.get_pool_info()

        console.print("\n")
        console.print(Panel(
            f"[bold]Training Complete[/bold]\n\n"
            f"Epochs: {self.epochs}\n"
            f"Red ELO: {elo['red']['rating']} "
            f"({elo['red']['wins']}W/{elo['red']['losses']}L/{elo['red']['draws']}D)\n"
            f"Blue ELO: {elo['blue']['rating']} "
            f"({elo['blue']['wins']}W/{elo['blue']['losses']}L/{elo['blue']['draws']}D)\n"
            f"League: {pool['red_pool_size']} Red + {pool['blue_pool_size']} Blue checkpoints\n"
            f"Logs: {self.log_dir}/",
            title="CyberGAN Results",
            border_style="cyan",
        ))

    def _save_logs(self):
        """Save epoch logs to JSON."""
        path = os.path.join(self.log_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(self.epoch_logs, f, indent=2, default=str)

    def _get_ckpt_dir(self) -> str:
        """Always returns repo-root checkpoints/ regardless of config location."""
        # Walk up from config path to find repo root (contains main.py)
        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        # If config is inside config/, go up one level
        candidate = config_dir
        if os.path.basename(candidate) == "config":
            candidate = os.path.dirname(candidate)
        ckpt_dir = os.path.join(candidate, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def _save_models(self):
        """Save final model weights — including production checkpoint for the agent."""
        ckpt_dir = self._get_ckpt_dir()

        # Standard final checkpoints
        torch.save(self.red_policy.state_dict(), os.path.join(ckpt_dir, "red_final.pt"))
        torch.save(self.blue_policy.state_dict(), os.path.join(ckpt_dir, "blue_final.pt"))

        # Production checkpoints — these are what main agent loads
        torch.save(self.blue_policy.state_dict(), os.path.join(ckpt_dir, "blue_production.pt"))
        torch.save(self.red_policy.state_dict(),  os.path.join(ckpt_dir, "red_production.pt"))

        # Save training metadata alongside
        meta = {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epochs_completed": len(self.epoch_logs),
            "final_elo": self.elo.get_ratings(),
            "device": self.device,
            "obs_dim_blue": self.blue_policy.obs_dim,
            "obs_dim_red":  self.red_policy.obs_dim,
        }
        with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.opponent_pool.save_to_disk(ckpt_dir)
        console.print(f"  [green]✓ Models saved to {ckpt_dir}/[/green]")
        console.print(f"  [green]✓ blue_production.pt ready — agent will use RL policy on next run[/green]")

    def save_checkpoint(self, epoch: int):
        """Save a mid-training checkpoint (called periodically)."""
        ckpt_dir = self._get_ckpt_dir()
        torch.save(self.blue_policy.state_dict(), os.path.join(ckpt_dir, f"blue_epoch_{epoch:04d}.pt"))
        torch.save(self.red_policy.state_dict(),  os.path.join(ckpt_dir, f"red_epoch_{epoch:04d}.pt"))
        # Also update production checkpoint in-place so agent can hot-reload
        torch.save(self.blue_policy.state_dict(), os.path.join(ckpt_dir, "blue_production.pt"))

    def get_state(self) -> dict:
        """Get current training state for dashboard."""
        return {
            "epoch": len(self.epoch_logs),
            "total_epochs": self.epochs,
            "elo": self.elo.get_ratings(),
            "league": self.opponent_pool.get_pool_info(),
            "network": self.env.network.to_dict(),
            "recent_logs": self.epoch_logs[-10:] if self.epoch_logs else [],
        }
