"""
CyberGAN — Brain: Experience Buffer
Stores (state, action, reward, next_state) for online RL updates.
Re-exported from the healing module for convenience.
"""

from cybergan.healing.online_learner import Experience, ExperienceBuffer

__all__ = ["Experience", "ExperienceBuffer"]
