"""
Ankylosaur Curriculum
=====================
3-stage curriculum, same structure as velociraptor / T-rex in mesozoic-labs.

Stage 1: BALANCE
  - No predator, no food.
  - Reward: stay upright, don't fall, move forward.
  - Success: avg episode length > 200 steps without falling.
  - Why: 28 DOF with wide-gauge stance — needs to learn to walk first.

Stage 2: FORAGE
  - No predator, food present (n=8 items).
  - Reward: full reward minus predator terms.
  - Success: avg energy > 0.4 at episode end over 10 episodes.
  - Why: must learn energy management before adding threat.

Stage 3: SURVIVE
  - Pretrained T-rex predator active.
  - Full reward function.
  - Success: avg survival time > 60s (3000 steps) over 10 episodes.
  - This is the scientific contribution stage.

Training time estimates (A100 / L4 GPU):
  Stage 1 (Balance):  ~3–4 hours  (~5M steps, 8 envs)
  Stage 2 (Forage):   ~4–5 hours  (~7M steps, 8 envs)
  Stage 3 (Survive):  ~5–7 hours  (~10M steps, 8 envs)
  Total:              ~12–16 hours single GPU

Recommendation: use MJX (JAX GPU-batched) with 64 parallel envs for
~4–5× speedup → ~3–4 hours total on A100.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from environments.ankylosaurus.envs.ankylosaurus_env import AnkylosaurusEnv
    from environments.ankylosaurus.envs.predator_manager import PredatorManager


STAGE_CONFIGS: dict[int, dict] = {
    1: {
        "name": "balance",
        "description": "Learn stable wide-gauge quadruped gait",
        "predator_active": False,
        "food_active": False,
        "predator_speed_scale": 0.0,
        "success_metric": "episode_length",
        "success_threshold": 200,       # Steps without falling
        "min_episodes": 20,
        "timesteps": 5_000_000,
        "reward_weights": {
            "w_survival":      0.20,
            "w_energy":        0.00,    # No food in stage 1
            "w_loco_cost":     0.02,
            "w_club_hit":      0.00,
            "w_pred_prox":     0.00,
            "w_bite_range":    0.00,
            "w_fall":          2.00,    # High: falling is the failure mode
            "w_joint_limits":  0.20,
            "w_terrain_cover": 0.00,
            "w_tail_ready":    0.00,
        },
    },
    2: {
        "name": "forage",
        "description": "Learn energy management and foraging",
        "predator_active": False,
        "food_active": True,
        "predator_speed_scale": 0.0,
        "success_metric": "final_energy",
        "success_threshold": 0.40,      # Energy > 40% at episode end
        "min_episodes": 10,
        "timesteps": 7_000_000,
        "reward_weights": {
            "w_survival":      0.10,
            "w_energy":        3.00,    # Primary driver: learn to eat
            "w_loco_cost":     0.05,
            "w_club_hit":      0.00,
            "w_pred_prox":     0.00,
            "w_bite_range":    0.00,
            "w_fall":          1.00,
            "w_joint_limits":  0.10,
            "w_terrain_cover": 0.05,    # Mild: vegetation near food clusters
            "w_tail_ready":    0.00,
        },
    },
    3: {
        "name": "survive",
        "description": "Survive against trained T-rex predator",
        "predator_active": True,
        "food_active": True,
        "predator_speed_scale": 1.0,    # Full T-rex speed
        "success_metric": "survival_time",
        "success_threshold": 3000,      # Steps = 60s at 50 Hz
        "min_episodes": 10,
        "timesteps": 10_000_000,
        "reward_weights": {
            "w_survival":      0.10,
            "w_energy":        2.00,
            "w_loco_cost":     0.05,
            "w_club_hit":     10.00,    # Big reward: successful deterrence
            "w_pred_prox":     0.50,
            "w_bite_range":    2.00,
            "w_fall":          1.00,
            "w_joint_limits":  0.10,
            "w_terrain_cover": 0.10,    # Use terrain: prey-animal behavior
            "w_tail_ready":    0.20,
        },
    },
}


class AnkylosaurCurriculum:
    """
    Manages stage transitions for the ankylosaur training curriculum.

    Usage::

        curriculum = AnkylosaurCurriculum()
        env = curriculum.make_env()

        # ... train on env ...

        if curriculum.should_advance(episode_stats):
            curriculum.advance()
            env = curriculum.make_env()  # rebuild with new stage config
    """

    def __init__(self, start_stage: int = 1):
        self.current_stage = start_stage
        self.episode_history: list[dict] = []

    # ------------------------------------------------------------------
    # Stage queries
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return STAGE_CONFIGS[self.current_stage]

    def get_reward_weights(self) -> dict[str, float]:
        return STAGE_CONFIGS[self.current_stage]["reward_weights"].copy()

    def should_advance(self, recent_stats: dict) -> bool:
        """
        Check whether performance metrics warrant advancing to next stage.

        Args:
            recent_stats: Averaged stats over recent N episodes, e.g.::
                {
                    "episode_length": float,   # mean steps before termination
                    "final_energy":   float,   # mean energy at episode end
                    "survival_time":  float,   # mean steps alive (= episode_length)
                }

        Returns:
            True if the success metric exceeds the threshold for this stage.
        """
        if self.current_stage >= 3:
            return False

        config = self.get_config()
        metric = config["success_metric"]
        threshold = config["success_threshold"]

        if metric not in recent_stats:
            return False

        return float(recent_stats[metric]) >= float(threshold)

    def advance(self) -> None:
        """Advance to the next curriculum stage."""
        if self.current_stage < 3:
            self.current_stage += 1
            cfg = STAGE_CONFIGS[self.current_stage]
            print(
                f"\n[Curriculum] Advancing to Stage {self.current_stage}: "
                f"{cfg['name'].upper()}"
            )
            print(f"  {cfg['description']}\n")

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------

    def make_env(
        self,
        predator_policy: Optional["PredatorManager"] = None,
    ) -> "AnkylosaurusEnv":
        """
        Create an AnkylosaurusEnv configured for the current stage.
        Applies stage-specific reward weights from STAGE_CONFIGS.
        """
        from environments.ankylosaurus.envs.ankylosaurus_env import AnkylosaurusEnv

        config = self.get_config()
        env = AnkylosaurusEnv(
            stage=self.current_stage,
            predator_policy=predator_policy if config["predator_active"] else None,
            n_food_items=8 if config["food_active"] else 0,
            predator_speed_scale=config["predator_speed_scale"],
        )
        # Apply stage-specific reward weights
        env._reward_weights = self.get_reward_weights()
        return env
