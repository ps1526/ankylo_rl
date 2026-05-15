"""
AnkylosaurusEnv
===============
MuJoCo Gymnasium environment for Ankylosaurus magniventris survival task.
Extends BaseDinoEnv from environments/shared/base_env.py.

Observation space: 83 dims
  [0:28]   joint positions (rad, normalized to [-1,1] by joint range)
  [28:56]  joint velocities (rad/s, clipped to [-10,10])
  [56:60]  torso orientation quaternion (w,x,y,z) from sensordata
  [60:63]  torso linear velocity (m/s) from qvel[0:3]
  [63:66]  torso angular velocity (rad/s) from gyro sensordata
  [66:70]  foot contact forces (FL, FR, RL, RR) from touch sensors
  [70]     club contact force (binary) from touch sensor
  [71]     agent energy level (0=starving, 1=full)
  [72]     agent health (0=dead, 1=full)
  [73]     predator distance (normalized by TREX_DETECTION_RANGE_M)
  [74]     predator bearing (rad, relative to agent heading)
  [75:83]  food radar: 8-directional food distances (normalized)

Action space: 28 dims, continuous [-1, 1]
  Maps directly to the 28 actuators in ankylosaurus.xml (same order as
  <actuator> block). Scaled to actuator ctrlrange by BaseDinoEnv._scale_action().

Episode termination:
  - energy <= 0 (starvation)
  - health <= 0 (killed by T-rex bites)
  - torso z < FALL_HEIGHT_THRESHOLD_M (fallen, cannot rise)
  - step >= EPISODE_MAX_STEPS (survival achieved — truncation)

Curriculum stages (managed by AnkylosaurCurriculum):
  Stage 1: Balance   — no predator, no food. Learn to stand and walk.
  Stage 2: Forage    — no predator, food present. Learn energy management.
  Stage 3: Survive   — predator active. Learn avoidance + club use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.ankylosaurus.paleo_constants import (
    ARMOR_BITE_ABSORPTION_FRACTION,
    CONTROL_TIMESTEP_S,
    EPISODE_MAX_STEPS,
    FALL_HEIGHT_THRESHOLD_M,
    MAX_SPEED_MS,
    SIM_TIMESTEP_S,
    TREX_ATTACK_COOLDOWN_S,
    TREX_BITE_RANGE_M,
    TREX_DETECTION_RANGE_M,
)

_ASSET_PATH = str(Path(__file__).parent.parent / "assets" / "ankylosaurus.xml")

# 4 sim steps per control step: 0.02s / 0.005s = 4
_FRAME_SKIP = int(CONTROL_TIMESTEP_S / SIM_TIMESTEP_S)

# Sensor indices (must match MJCF <sensor> block order)
_SENSOR_TOUCH_FL   = 10   # sensordata index for foot_FL touch
_SENSOR_TOUCH_FR   = 11
_SENSOR_TOUCH_RL   = 12
_SENSOR_TOUCH_RR   = 13
_SENSOR_TOUCH_CLUB = 14   # club strike touch sensor


class AnkylosaurusEnv(BaseDinoEnv):
    """
    Ankylosaurus magniventris survival environment.

    Properly extends BaseDinoEnv — all five abstract methods are implemented:
      _cache_ids(), _get_obs(), _get_reward_info(), _is_terminated(), _spawn_target()

    All instance state is initialized BEFORE super().__init__() because the
    base constructor calls _cache_ids() then _get_obs() immediately to infer
    the observation space shape.
    """

    _camera_distance = 8.0
    _camera_azimuth = 135
    _camera_elevation = -20
    _camera_track_body = "torso"

    def __init__(
        self,
        stage: int = 1,
        render_mode: str | None = None,
        predator_policy=None,           # PredatorManager instance or None
        n_food_items: int = 8,
        predator_speed_scale: float = 1.0,
        frame_skip: int = _FRAME_SKIP,
        max_episode_steps: int = EPISODE_MAX_STEPS,
    ):
        # ---- Stage / predator / food config ----
        self.stage = stage
        self.predator_policy = predator_policy
        self.n_food_items = n_food_items
        self.predator_speed_scale = predator_speed_scale

        # ---- Internal survival state (initialized before super().__init__) ----
        # BaseDinoEnv.__init__ calls _cache_ids() then _get_obs() immediately,
        # so these must exist with valid defaults.
        self._energy: float = 1.0
        self._health: float = 1.0
        self._prev_energy: float = 1.0
        self._attack_cooldown: float = 0.0
        self._food_positions: list[np.ndarray] = []
        self._predator_pos: np.ndarray = np.array([999.0, 999.0, 0.0])

        # ---- Reward weights (defaults; overridden per-stage via TOML) ----
        self._reward_weights: dict[str, float] = {
            "w_survival":      0.10,
            "w_energy":        2.00,
            "w_loco_cost":     0.05,
            "w_club_hit":     10.00,
            "w_pred_prox":     0.50,
            "w_bite_range":    2.00,
            "w_fall":          1.00,
            "w_joint_limits":  0.10,
            "w_terrain_cover": 0.05,
            "w_tail_ready":    0.20,
        }

        # Required by BaseDinoEnv._reward_action_smoothness
        self.smoothness_weight: float = 0.0
        self._prev_action: np.ndarray | None = None

        # ---- BaseDinoEnv init (loads model, infers obs/act spaces) ----
        super().__init__(
            model_path=_ASSET_PATH,
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            forward_vel_weight=0.0,     # we use custom reward, not forward-vel
            alive_bonus=0.0,
            energy_penalty_weight=0.0,
            fall_penalty=0.0,           # fall handled in _is_terminated + reward
            healthy_z_range=(FALL_HEIGHT_THRESHOLD_M, 3.0),
            max_tilt_angle=1.4,
            reset_noise_scale=0.03,
        )

    # ------------------------------------------------------------------
    # ABSTRACT METHOD 1: _cache_ids
    # ------------------------------------------------------------------

    def _cache_ids(self) -> None:
        """Cache MuJoCo body/geom/site/sensor IDs for fast per-step access."""
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.torso_main_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_main"
        )
        self.club_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "club_site"
        )
        self.imu_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )

        # Foot sites (used for gait metrics)
        self.foot_site_ids = {
            "FL": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "foot_FL_site"),
            "FR": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "foot_FR_site"),
            "RL": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "foot_RL_site"),
            "RR": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "foot_RR_site"),
        }

        # Joint qpos addresses (for reward/analysis helpers)
        self._joint_qpos_addr: dict[str, int] = {}
        for name in [
            "neck_pitch", "neck_yaw", "head_pitch",
            "hip_FL_abduct", "hip_FL_flex", "hip_FL_rotate", "knee_FL", "ankle_FL",
            "hip_FR_abduct", "hip_FR_flex", "hip_FR_rotate", "knee_FR", "ankle_FR",
            "hip_RL_abduct", "hip_RL_flex", "hip_RL_rotate", "knee_RL", "ankle_RL",
            "hip_RR_abduct", "hip_RR_flex", "hip_RR_rotate", "knee_RR", "ankle_RR",
            "tail_base_lat", "tail_base_dv", "tail_mid_lat", "tail_mid_dv",
            "tail_handle_lat",
        ]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qpos_addr[name] = int(self.model.jnt_qposadr[jid])

    # ------------------------------------------------------------------
    # ABSTRACT METHOD 2: _get_obs  (83 dims)
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Assemble 83-dimensional observation vector."""

        # [0:28] Joint positions — normalized to [-1,1] by joint range
        # Skip freejoint (qpos[0:7]); actuated joints start at qpos[7]
        qpos = self.data.qpos[7:35].copy()
        jnt_ranges = self.model.jnt_range[1:]   # skip freejoint row
        jnt_span = jnt_ranges[:, 1] - jnt_ranges[:, 0] + 1e-8
        qpos_norm = 2.0 * (qpos - jnt_ranges[:, 0]) / jnt_span - 1.0

        # [28:56] Joint velocities — clip to [-10, 10] rad/s
        qvel = np.clip(self.data.qvel[6:34].copy(), -10.0, 10.0)

        # [56:60] Torso orientation quaternion from IMU sensor (w,x,y,z)
        torso_quat = self.data.sensordata[
            self._sensor_quat_start : self._sensor_quat_start + 4
        ].copy()

        # [60:63] Torso linear velocity from root freejoint
        torso_linvel = self.data.qvel[0:3].copy()

        # [63:66] Torso angular velocity from gyro sensor
        torso_angvel = self.data.sensordata[
            self._sensor_gyro_start : self._sensor_gyro_start + 3
        ].copy()

        # [66:70] Foot contact forces (binary: > 0.1 N = in contact)
        foot_contacts = np.array([
            float(self.data.sensordata[_SENSOR_TOUCH_FL] > 0.1),
            float(self.data.sensordata[_SENSOR_TOUCH_FR] > 0.1),
            float(self.data.sensordata[_SENSOR_TOUCH_RL] > 0.1),
            float(self.data.sensordata[_SENSOR_TOUCH_RR] > 0.1),
        ], dtype=np.float32)

        # [70] Club contact (binary)
        club_contact = float(self.data.sensordata[_SENSOR_TOUCH_CLUB] > 0.1)

        # [71] Energy level
        energy = np.array([self._energy], dtype=np.float32)

        # [72] Health
        health = np.array([self._health], dtype=np.float32)

        # [73-74] Predator distance and bearing
        pred_dist_norm, pred_bearing = self._get_predator_obs()

        # [75:83] 8-directional food radar
        food_radar = self._get_food_radar()

        obs = np.concatenate([
            qpos_norm.astype(np.float32),
            qvel.astype(np.float32),
            torso_quat.astype(np.float32),
            torso_linvel.astype(np.float32),
            torso_angvel.astype(np.float32),
            foot_contacts,
            [club_contact],
            energy,
            health,
            [pred_dist_norm],
            [pred_bearing],
            food_radar.astype(np.float32),
        ]).astype(np.float32)

        return obs

    # ------------------------------------------------------------------
    # ABSTRACT METHOD 3: _get_reward_info
    # ------------------------------------------------------------------

    def _get_reward_info(
        self, action: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """
        Update internal state then compute shaped survival reward.
        Energy/health are updated here (called once per control step by base).
        """
        # Update survival state before computing reward
        self._update_energy()
        self._update_health()
        self._update_predator_pos()

        from environments.ankylosaurus.envs.reward import compute_reward
        total, components = compute_reward(self, action)

        # Flatten info for W&B / SB3 logging
        info: dict[str, float] = {k: float(v) for k, v in components.items()}
        info["reward_total"] = float(total)
        info["energy"] = self._energy
        info["health"] = self._health
        info["pred_dist_m"] = float(np.linalg.norm(
            self.data.xpos[self.torso_id] - self._predator_pos
        ))
        return float(total), info

    # ------------------------------------------------------------------
    # ABSTRACT METHOD 4: _is_terminated
    # ------------------------------------------------------------------

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        """Episode terminates on starvation, death, or fall."""
        info: dict[str, Any] = {}

        torso_z = float(self.data.xpos[self.torso_id, 2])
        info["torso_height"] = torso_z

        if self._energy <= 0.0:
            info["termination_reason"] = "starvation"
            return True, info

        if self._health <= 0.0:
            info["termination_reason"] = "killed"
            return True, info

        if torso_z < FALL_HEIGHT_THRESHOLD_M:
            info["termination_reason"] = "fallen"
            return True, info

        return False, info

    # ------------------------------------------------------------------
    # ABSTRACT METHOD 5: _spawn_target
    # ------------------------------------------------------------------

    def _spawn_target(self) -> None:
        """
        Reset food positions and predator spawn for a new episode.
        Called by BaseDinoEnv.reset() after qpos noise is applied but
        before mj_forward() — do NOT use data.xpos here.
        """
        # Reset survival state
        self._energy = 1.0
        self._health = 1.0
        self._prev_energy = 1.0
        self._attack_cooldown = 0.0
        self._prev_action = None

        # Spawn food items using Hell Creek vegetation distribution
        self._food_positions = self._spawn_food_positions()

        # Spawn predator (Stage 3 only): use root qpos for agent position
        if self.stage >= 3:
            agent_xy = self.data.qpos[0:2].copy()
            angle = self.np_random.uniform(0, 2 * np.pi)
            dist = self.np_random.uniform(
                TREX_DETECTION_RANGE_M, TREX_DETECTION_RANGE_M * 1.5
            )
            px = agent_xy[0] + dist * np.cos(angle)
            py = agent_xy[1] + dist * np.sin(angle)
            self._predator_pos = np.array([px, py, 0.0])
        else:
            self._predator_pos = np.array([999.0, 999.0, 0.0])

    # ------------------------------------------------------------------
    # PUBLIC HELPERS (used by reward.py and analyze_behavior.py)
    # ------------------------------------------------------------------

    def get_joint_qpos(self, joint_name: str) -> float:
        """Return current qpos value for a named joint."""
        return float(self.data.qpos[self._joint_qpos_addr[joint_name]])

    def get_torso_linvel_2d(self) -> np.ndarray:
        """Return 2D horizontal velocity of the torso (m/s)."""
        return self.data.qvel[0:2].copy()

    def get_torso_speed(self) -> float:
        """Return horizontal speed of the torso (m/s)."""
        return float(np.linalg.norm(self.data.qvel[0:2]))

    def get_club_contact_force(self) -> float:
        """Return club touch sensor reading (N)."""
        return float(self.data.sensordata[_SENSOR_TOUCH_CLUB])

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _update_energy(self) -> None:
        """
        Metabolic energy drain + foraging replenishment.
        Drain rate: 1.0 energy over EPISODE_MAX_STEPS steps (starvation baseline).
        Additional speed-proportional drain (cost of locomotion).
        Food contact replenishes +0.05 energy per item eaten.
        """
        self._prev_energy = self._energy

        # Baseline metabolic drain
        drain = 1.0 / self.max_episode_steps
        self._energy -= drain

        # Speed-proportional locomotion cost (energetically realistic)
        speed = self.get_torso_speed()
        speed_drain = 0.5 * (speed / MAX_SPEED_MS) * drain
        self._energy -= speed_drain

        # Check food proximity (eating radius = 2.0 m)
        torso_xy = self.data.xpos[self.torso_id, :2]
        for i, food_pos in enumerate(self._food_positions):
            if np.linalg.norm(torso_xy - food_pos) < 2.0:
                self._energy = min(1.0, self._energy + 0.05)
                # Respawn consumed food at new random location
                self._food_positions[i] = self._random_food_pos()
                break

        self._energy = max(0.0, self._energy)

    def _update_health(self) -> None:
        """
        Apply T-rex bite damage when predator is within bite range.
        Osteoderm armor absorbs ARMOR_BITE_ABSORPTION_FRACTION of damage.
        Bite damage only applies after attack cooldown resets.
        """
        if self.stage < 3:
            return

        torso_pos = self.data.xpos[self.torso_id]
        pred_dist = float(np.linalg.norm(torso_pos - self._predator_pos))

        if pred_dist < TREX_BITE_RANGE_M and self._attack_cooldown <= 0.0:
            raw_damage = 0.15
            actual_damage = raw_damage * (1.0 - ARMOR_BITE_ABSORPTION_FRACTION)
            self._health -= actual_damage
            self._attack_cooldown = TREX_ATTACK_COOLDOWN_S

        if self._attack_cooldown > 0.0:
            self._attack_cooldown -= CONTROL_TIMESTEP_S

        self._health = max(0.0, self._health)

    def _update_predator_pos(self) -> None:
        """
        Kinematic T-rex movement: moves toward agent at predator speed.
        In Stage 3 with a loaded PPO policy, the PredatorManager is
        consulted; otherwise uses simple pursuit kinematics.
        """
        if self.stage < 3:
            return

        torso_pos = self.data.xpos[self.torso_id, :2]
        delta = torso_pos - self._predator_pos[:2]
        dist = float(np.linalg.norm(delta))

        if dist < 0.5:
            return  # Already at agent position

        from environments.ankylosaurus.paleo_constants import TREX_MAX_SPEED_MS
        speed = TREX_MAX_SPEED_MS * self.predator_speed_scale
        step_dist = min(speed * CONTROL_TIMESTEP_S, dist)
        direction = delta / (dist + 1e-8)
        self._predator_pos[:2] += direction * step_dist

        # Notify PredatorManager of updated position (for PPO inference)
        if self.predator_policy is not None:
            self.predator_policy.update_position(self._predator_pos)

    def _get_predator_obs(self) -> tuple[float, float]:
        """
        Returns (normalized_distance, bearing_rad) to predator.
        bearing = signed angle in agent's local frame (0 = directly ahead).
        """
        torso_pos = self.data.xpos[self.torso_id, :2]
        delta = self._predator_pos[:2] - torso_pos
        dist = float(np.linalg.norm(delta))
        dist_norm = float(min(dist / TREX_DETECTION_RANGE_M, 1.0))

        if dist < 1e-6:
            return dist_norm, 0.0

        # Agent heading from torso quaternion (yaw around Z)
        quat = self.data.sensordata[
            self._sensor_quat_start : self._sensor_quat_start + 4
        ]
        agent_forward = self._quat_to_forward_2d(quat)
        pred_dir = delta / (dist + 1e-8)
        # Signed angle: positive = predator to agent's left
        bearing = float(np.arctan2(
            agent_forward[0] * pred_dir[1] - agent_forward[1] * pred_dir[0],
            agent_forward[0] * pred_dir[0] + agent_forward[1] * pred_dir[1],
        ))
        return dist_norm, bearing

    def _get_food_radar(self) -> np.ndarray:
        """
        8-directional food distance radar (45° bins).
        Returns array of 8 values in [0, 1]; 1.0 = no food detected.
        Direction 0 = agent forward, bins increment counter-clockwise.
        """
        radar = np.ones(8, dtype=np.float32)
        if not self._food_positions:
            return radar

        radar_range = 20.0
        torso_xy = self.data.xpos[self.torso_id, :2]
        quat = self.data.sensordata[
            self._sensor_quat_start : self._sensor_quat_start + 4
        ]
        yaw = float(np.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        ))

        for food_pos in self._food_positions:
            delta = food_pos - torso_xy
            dist = float(np.linalg.norm(delta))
            if dist > radar_range:
                continue
            angle = np.arctan2(delta[1], delta[0]) - yaw
            # Wrap to [-π, π]
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            bin_idx = int(round(4 * angle / np.pi)) % 8
            radar[bin_idx] = min(radar[bin_idx], dist / radar_range)

        return radar

    def _spawn_food_positions(self) -> list[np.ndarray]:
        """
        Spawn food items with Hell Creek vegetation distribution.
        60% clustered near water zones (r = 3–8 m), 40% open ground.
        """
        positions = []
        for _ in range(self.n_food_items):
            positions.append(self._random_food_pos())
        return positions

    def _random_food_pos(self) -> np.ndarray:
        """Generate one random food position."""
        if self.np_random.random() < 0.6:
            r = self.np_random.uniform(3.0, 8.0)
            ang = self.np_random.uniform(0.0, 2 * np.pi)
            return np.array([r * np.cos(ang), r * np.sin(ang)])
        return self.np_random.uniform(-14.0, 14.0, size=2).astype(np.float64)


# Gymnasium registration (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/Ankylosaurus-v0",
    entry_point="environments.ankylosaurus.envs.ankylosaurus_env:AnkylosaurusEnv",
    max_episode_steps=EPISODE_MAX_STEPS,
)
