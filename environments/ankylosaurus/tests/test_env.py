"""
Environment sanity tests
========================
Run these before starting any training — they verify the env is
correctly wired and the MJCF is paleobiologically valid.

Usage::

    pytest environments/ankylosaurus/tests/ -v

All tests should pass before running train_sb3.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _asset_path() -> str:
    return str(Path(__file__).parent.parent / "assets" / "ankylosaurus.xml")


def _make_stage1_env():
    from environments.ankylosaurus.envs.ankylosaurus_env import AnkylosaurusEnv
    return AnkylosaurusEnv(stage=1, n_food_items=0)


def _make_stage2_env():
    from environments.ankylosaurus.envs.ankylosaurus_env import AnkylosaurusEnv
    return AnkylosaurusEnv(stage=2, n_food_items=8)


# ---------------------------------------------------------------------------
# MJCF model tests
# ---------------------------------------------------------------------------

class TestModelLoading:

    def test_mjcf_file_exists(self):
        """MJCF asset file is present."""
        assert Path(_asset_path()).exists(), f"MJCF not found: {_asset_path()}"

    def test_mjcf_parses(self):
        """MJCF parses without errors."""
        import mujoco
        model = mujoco.MjModel.from_xml_path(_asset_path())
        assert model is not None

    def test_actuator_count(self):
        """Model has exactly 28 actuators (3 neck + 20 legs + 5 tail)."""
        import mujoco
        model = mujoco.MjModel.from_xml_path(_asset_path())
        assert model.nu == 28, f"Expected 28 actuators, got {model.nu}"

    def test_joint_count(self):
        """Model has 29 joints (1 freejoint + 28 actuated)."""
        import mujoco
        model = mujoco.MjModel.from_xml_path(_asset_path())
        assert model.njnt == 29, (
            f"Expected 29 joints (1 free + 28 actuated), got {model.njnt}"
        )

    def test_total_mass(self):
        """Total model mass is within paleobiological range (4,000–7,000 kg)."""
        import mujoco
        model = mujoco.MjModel.from_xml_path(_asset_path())
        total_mass = float(np.sum(model.body_mass))
        assert 4000 < total_mass < 7000, (
            f"Total mass {total_mass:.0f} kg outside range [4000, 7000]. "
            f"Expected ~4,800–6,000 kg [Arbour & Currie 2013]."
        )

    def test_tail_handle_single_dof(self):
        """
        Tail handle must have only lateral DOF (no dorsoventral) — Arbour 2009.
        Verify by checking joint name existence and absence of a dv joint.
        """
        import mujoco
        model = mujoco.MjModel.from_xml_path(_asset_path())
        lat_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "tail_handle_lat"
        )
        assert lat_id >= 0, "tail_handle_lat joint not found in MJCF"
        # Should NOT have a dorsoventral handle joint
        dv_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "tail_handle_dv"
        )
        assert dv_id == -1, (
            "tail_handle_dv joint found but should not exist "
            "(Arbour 2009: handle is laterally-only ossified)"
        )

    def test_sensor_layout(self):
        """
        Sensor layout matches expected indices for BaseDinoEnv compatibility.
        gyro=[0-2], accel=[3-5], quat=[6-9], touch_FL=10, touch_FR=11,
        touch_RL=12, touch_RR=13, touch_club=14.
        """
        import mujoco
        model = mujoco.MjModel.from_xml_path(_asset_path())
        expected = {
            "torso_gyro":  0,
            "torso_accel": 1,
            "torso_quat":  2,
            "touch_FL":    3,
            "touch_FR":    4,
            "touch_RL":    5,
            "touch_RR":    6,
            "touch_club":  7,
        }
        for name, expected_idx in expected.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            assert sid == expected_idx, (
                f"Sensor '{name}' expected at index {expected_idx}, "
                f"found at {sid}"
            )


# ---------------------------------------------------------------------------
# Environment construction tests
# ---------------------------------------------------------------------------

class TestEnvironmentConstruction:

    def test_env_creates_stage1(self):
        """Stage 1 environment initializes without errors."""
        env = _make_stage1_env()
        assert env is not None
        env.close()

    def test_obs_space_shape(self):
        """Observation space is (83,) float32."""
        from environments.ankylosaurus.envs.ankylosaurus_env import AnkylosaurusEnv
        env = _make_stage1_env()
        assert env.observation_space.shape == (83,), (
            f"Expected obs shape (83,), got {env.observation_space.shape}"
        )
        assert env.observation_space.dtype == np.float32
        env.close()

    def test_action_space_shape(self):
        """Action space is (28,) continuous in [-1, 1]."""
        env = _make_stage1_env()
        assert env.action_space.shape == (28,)
        assert float(env.action_space.low.min())  == -1.0
        assert float(env.action_space.high.max()) ==  1.0
        env.close()

    def test_gym_registration(self):
        """Gym ID 'MesozoicLabs/Ankylosaurus-v0' is registered."""
        import gymnasium as gym
        assert "MesozoicLabs/Ankylosaurus-v0" in gym.envs.registry


# ---------------------------------------------------------------------------
# Reset / step tests
# ---------------------------------------------------------------------------

class TestEpisodeMechanics:

    def test_reset_returns_valid_obs(self):
        """reset() returns (83,) observation with no NaN."""
        env = _make_stage1_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == (83,)
        assert not np.any(np.isnan(obs)), "Observation contains NaN after reset"
        assert isinstance(info, dict)
        env.close()

    def test_reset_initialises_energy_and_health(self):
        """Energy and health start at 1.0 after reset."""
        env = _make_stage1_env()
        env.reset(seed=0)
        assert env._energy == 1.0
        assert env._health == 1.0
        env.close()

    def test_step_returns_valid_obs(self):
        """Zero-action step returns valid (83,) observation."""
        env = _make_stage1_env()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(28))
        assert obs.shape == (83,)
        assert not np.any(np.isnan(obs)), "NaN in obs after step"
        assert np.isfinite(reward), f"Non-finite reward: {reward}"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_random_steps_no_crash(self):
        """100 random steps complete without error."""
        env = _make_stage1_env()
        env.reset(seed=1)
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.any(np.isnan(obs))
            assert np.isfinite(reward)
            if terminated or truncated:
                env.reset()
        env.close()

    def test_energy_drains_over_time(self):
        """Energy decreases monotonically with zero action (no food)."""
        env = _make_stage1_env()
        env.reset(seed=2)
        initial_energy = env._energy
        for _ in range(200):
            env.step(np.zeros(28))
        assert env._energy < initial_energy, (
            "Energy should drain over time even with zero action"
        )
        env.close()

    def test_fall_terminates_episode(self):
        """Forcing torso below fall threshold triggers termination."""
        import mujoco
        from environments.ankylosaurus.paleo_constants import FALL_HEIGHT_THRESHOLD_M
        env = _make_stage1_env()
        env.reset(seed=3)
        # Force torso below threshold
        env.data.qpos[2] = FALL_HEIGHT_THRESHOLD_M - 0.05
        mujoco.mj_forward(env.model, env.data)
        terminated, _ = env._is_terminated()
        assert terminated, (
            f"Expected termination when torso_z < {FALL_HEIGHT_THRESHOLD_M}"
        )
        env.close()

    def test_reward_always_finite(self):
        """Reward is always finite (no inf/nan) across 100 steps."""
        env = _make_stage1_env()
        env.reset(seed=4)
        for _ in range(100):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward), f"Non-finite reward at step: {reward}"
            if terminated or truncated:
                break
        env.close()

    def test_reward_components_logged(self):
        """Reward info dict contains all expected component keys."""
        env = _make_stage1_env()
        env.reset(seed=5)
        _, _, _, _, info = env.step(np.zeros(28))
        expected_keys = [
            "survival", "energy", "locomotion_cost", "club_hit",
            "pred_proximity", "fall", "joint_limits",
            "terrain_cover", "tail_readiness", "reward_total",
        ]
        for key in expected_keys:
            assert key in info, f"Missing reward component key: '{key}'"
        env.close()

    def test_stage2_food_spawns(self):
        """Stage 2 env spawns exactly 8 food items after reset."""
        env = _make_stage2_env()
        env.reset(seed=6)
        assert len(env._food_positions) == 8, (
            f"Expected 8 food items, got {len(env._food_positions)}"
        )
        env.close()

    def test_curriculum_makes_env(self):
        """AnkylosaurCurriculum.make_env() creates a working environment."""
        from environments.ankylosaurus.envs.curriculum import AnkylosaurCurriculum
        curriculum = AnkylosaurCurriculum(start_stage=1)
        env = curriculum.make_env()
        obs, _ = env.reset(seed=7)
        assert obs.shape == (83,)
        env.close()
