"""
PredatorManager
===============
Loads the pretrained T-rex Stage 3 PPO policy from mesozoic-labs and
provides a step() interface to get T-rex actions given its observation.

The T-rex policy expects:
  - obs dim: 83 (joints, pelvis, prey tracking) [from mesozoic-labs trex_env.py]
  - action dim: 21 (3 neck/head + 7 per leg + 4 tail)

In our setup:
  - T-rex's "prey" is the ankylosaurus torso position
  - T-rex policy runs at same control frequency as ankylosaur (50 Hz)
  - T-rex speed is scaled by predator_speed_scale (curriculum: start slow)

If the Stage 3 T-rex checkpoint is not found, falls back to a simple
heuristic pursuit that still poses a meaningful threat.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# T-rex results directory (relative to mesozoic-labs repo root)
_TREX_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "trex"


class PredatorManager:
    """
    Manages the pretrained T-rex predator policy.

    Usage:
        manager = PredatorManager(speed_scale=0.5)  # 50% speed for curriculum
        manager.load()
        action = manager.get_action(trex_obs, prey_pos)
    """

    def __init__(self, speed_scale: float = 1.0):
        self.speed_scale = speed_scale
        self.policy = None          # Loaded PPO policy (or None → heuristic)
        self._pos = np.zeros(3)     # Current tracked world position of T-rex
        self._vel = np.zeros(3)     # Current T-rex velocity estimate

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load pretrained T-rex PPO policy.
        Searches results/trex/ppo/ for the most recent Stage 3 checkpoint.

        Returns True if loaded successfully, False if not found (uses heuristic).
        """
        try:
            from stable_baselines3 import PPO
        except ImportError:
            print("[PredatorManager] stable-baselines3 not installed. Using heuristic.")
            return False

        if model_path is None:
            candidates: list[Path] = []
            for search_dir in [_TREX_RESULTS_DIR, _TREX_RESULTS_DIR / "ppo"]:
                if search_dir.exists():
                    for f in search_dir.rglob("*.zip"):
                        if "stage3" in f.name.lower() or "stage_3" in f.name.lower():
                            candidates.append(f)
            if not candidates:
                print("[PredatorManager] WARNING: No Stage 3 T-rex checkpoint found.")
                print("  Train T-rex first:")
                print("    python environments/trex/scripts/train_sb3.py curriculum --algorithm ppo")
                print("  Falling back to heuristic pursuit predator.")
                self.policy = None
                return False

            model_path = str(sorted(candidates)[-1])  # Most recently named
            print(f"[PredatorManager] Loaded T-rex policy: {model_path}")

        try:
            from stable_baselines3 import PPO
            self.policy = PPO.load(model_path)
            print(f"[PredatorManager] T-rex PPO policy loaded successfully.")
            return True
        except Exception as exc:
            print(f"[PredatorManager] Failed to load policy ({exc}). Using heuristic.")
            self.policy = None
            return False

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def get_action(
        self,
        trex_obs: np.ndarray,
        prey_pos: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Get T-rex action given its 83-dim observation.

        If PPO policy is loaded: use neural network inference.
        If policy is None: use heuristic pursuit.
        speed_scale modulates action magnitude (curriculum difficulty).
        """
        if self.policy is not None:
            action, _ = self.policy.predict(trex_obs, deterministic=deterministic)
        else:
            action = self._heuristic_pursuit(prey_pos)

        return np.clip(action * self.speed_scale, -1.0, 1.0)

    def _heuristic_pursuit(self, prey_pos: np.ndarray) -> np.ndarray:
        """
        Fallback heuristic: move T-rex toward prey using approximate
        joint mapping. Returns 21-dim action vector.

        Joint mapping approximation (TRexEnv action order):
          [0-2]:   neck_pitch, neck_yaw, head_pitch
          [3-9]:   right leg (hip_pitch, hip_roll, knee, ankle, toe×3)
          [10-16]: left leg
          [17-20]: tail

        This is intentionally approximate — the heuristic just drives
        locomotion toward the prey; it won't perfectly match the trained
        T-rex gait but will still pose a meaningful threat.
        """
        action = np.zeros(21, dtype=np.float32)

        delta = prey_pos[:2] - self._pos[:2]
        dist = float(np.linalg.norm(delta))

        if dist > 1.5:
            # Drive forward locomotion: hip flex and knee extension
            action[3] = 0.8    # right hip pitch (flex forward)
            action[5] = 0.6    # right knee
            action[10] = 0.8   # left hip pitch
            action[12] = 0.6   # left knee
            # Head toward prey
            action[0] = 0.3    # neck pitch down (hunting posture)
        else:
            # Within strike range: bite attempt
            action[0] = 1.0    # neck pitch (lunge forward)
            action[1] = 0.2    # neck yaw (aim)
            action[2] = 1.0    # head pitch (bite)

        return action

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def update_position(self, new_pos: np.ndarray) -> None:
        """Update tracked world position of T-rex (called by AnkylosaurusEnv)."""
        self._vel = new_pos - self._pos
        self._pos = new_pos.copy()

    @property
    def position(self) -> np.ndarray:
        return self._pos.copy()
