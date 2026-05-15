"""
Survival Reward Function
========================
All reward terms are grounded in Late Cretaceous Ankylosaurus paleobiology.

Term                    | Scientific justification
------------------------|--------------------------------------------------
survival_bonus          | Dense reward for staying alive each step
energy_delta            | Drives foraging behavior (herbivore feeding)
locomotion_cost         | Cost of transport penalty (energetic realism)
predator_deterrence     | Club contact with T-rex → stagger reward
predator_proximity      | Penalty for entering bite range
fall_penalty            | Falling = catastrophic for 5-ton armored animal
joint_limit_violation   | Anatomical realism constraint
terrain_cover_bonus     | Reward for using obstacles (prey behavior)
tail_club_readiness     | Small bonus for orienting tail toward threat

Reward shaping rationale:
  The goal is emergent behavior, not reward hacking. All terms are
  physically motivated. If the agent learns to stand near food and swing
  its tail — that is consistent with paleobiological evidence. If it
  learns to flee — compare against trackway evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environments.ankylosaurus.envs.ankylosaurus_env import AnkylosaurusEnv

from environments.ankylosaurus.paleo_constants import (
    FALL_HEIGHT_THRESHOLD_M,
    MAX_SPEED_MS,
    TREX_BITE_RANGE_M,
    TREX_DETECTION_RANGE_M,
)

# Sensor index constants (must match MJCF sensor order)
_SENSOR_TOUCH_CLUB = 14


def compute_reward(
    env: "AnkylosaurusEnv",
    action: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Compute shaped survival reward.

    Args:
        env:    AnkylosaurusEnv instance (post physics step, state updated).
        action: 28-dim action applied this step (unused directly but
                passed for future action-smoothness extension).

    Returns:
        (total_reward, components_dict) where components_dict has one
        float per named term for W&B logging.
    """
    w = env._reward_weights
    components: dict[str, float] = {}

    # ----------------------------------------------------------------
    # 1. SURVIVAL BONUS (dense)
    #    +w per step survived. Primary learning signal — without this
    #    the agent has no incentive to stay alive.
    # ----------------------------------------------------------------
    components["survival"] = w["w_survival"]

    # ----------------------------------------------------------------
    # 2. ENERGY DELTA
    #    Reward positive energy gain (successful foraging).
    #    Only reward gains — no penalty for metabolic baseline drain.
    # ----------------------------------------------------------------
    energy_delta = env._energy - env._prev_energy
    components["energy"] = w["w_energy"] * max(energy_delta, 0.0)

    # ----------------------------------------------------------------
    # 3. LOCOMOTION COST (penalty)
    #    Energy-conservative movement is paleobiologically realistic.
    #    5-ton animal has very high cost of transport.
    #    Penalty ∝ speed² (kinetic energy proxy).
    # ----------------------------------------------------------------
    speed = env.get_torso_speed()
    speed_norm = speed / (MAX_SPEED_MS + 1e-8)
    components["locomotion_cost"] = -w["w_loco_cost"] * (speed_norm ** 2)

    # ----------------------------------------------------------------
    # 4. PREDATOR DETERRENCE (sparse positive)
    #    Club touch contact + predator within detection range → large reward.
    #    Grounded in: FEA shows club delivers bone-fracture-level forces
    #    (7,281–14,360 N, Arbour 2009) — a hit genuinely deters a T-rex.
    # ----------------------------------------------------------------
    club_force = env.get_club_contact_force()
    torso_pos = env.data.xpos[env.torso_id]
    pred_dist_raw = float(np.linalg.norm(torso_pos - env._predator_pos))

    club_hit = 0.0
    if club_force > 0.1 and pred_dist_raw < TREX_DETECTION_RANGE_M:
        club_hit = w["w_club_hit"]
    components["club_hit"] = club_hit

    # ----------------------------------------------------------------
    # 5. PREDATOR PROXIMITY PENALTY (graduated)
    #    No penalty outside detection range.
    #    Linear penalty as predator closes.
    #    Extra spike penalty if within actual bite range.
    # ----------------------------------------------------------------
    pred_dist_norm = min(pred_dist_raw / TREX_DETECTION_RANGE_M, 1.0)
    proximity_penalty = 0.0
    if pred_dist_norm < 1.0:
        proximity_penalty = -w["w_pred_prox"] * (1.0 - pred_dist_norm)
    if pred_dist_raw < TREX_BITE_RANGE_M:
        proximity_penalty -= w["w_bite_range"]
    components["pred_proximity"] = proximity_penalty

    # ----------------------------------------------------------------
    # 6. FALL PENALTY (continuous gradient approaching fall threshold)
    #    Ankylosaur falling is realistically catastrophic:
    #    exposed unarmored ventrum + 5-ton weight = cannot right itself.
    # ----------------------------------------------------------------
    torso_z = float(env.data.xpos[env.torso_id, 2])
    fall_penalty = 0.0
    if torso_z < 1.0:
        fall_penalty = -w["w_fall"] * max(0.0, 1.0 - torso_z)
    components["fall"] = fall_penalty

    # ----------------------------------------------------------------
    # 7. JOINT LIMIT VIOLATION PENALTY
    #    Keeps motion within paleobiologically-grounded ranges.
    #    Penalizes any joint that exceeds 95% of its range limit.
    # ----------------------------------------------------------------
    jnt_pos = env.data.qpos[7:35]
    jnt_ranges = env.model.jnt_range[1:]  # skip freejoint row
    jnt_span = jnt_ranges[:, 1] - jnt_ranges[:, 0]
    jnt_center = (jnt_ranges[:, 1] + jnt_ranges[:, 0]) / 2.0
    jnt_deviation = np.abs(jnt_pos - jnt_center) / (jnt_span / 2.0 + 1e-8)
    limit_violations = float(np.sum(np.maximum(0.0, jnt_deviation - 0.95)))
    components["joint_limits"] = -w["w_joint_limits"] * limit_violations

    # ----------------------------------------------------------------
    # 8. TERRAIN COVER BONUS
    #    Reward for moving into vegetation belt (5–12 m from arena centre).
    #    Prey animals use terrain defensively — this drives that behavior.
    #    Grounded in: Hell Creek vegetation clustered near water [HELLCREEK].
    # ----------------------------------------------------------------
    torso_xy = env.data.xpos[env.torso_id, :2]
    dist_center = float(np.linalg.norm(torso_xy))
    cover_bonus = w["w_terrain_cover"] if 5.0 < dist_center < 12.0 else 0.0
    components["terrain_cover"] = cover_bonus

    # ----------------------------------------------------------------
    # 9. TAIL CLUB READINESS
    #    Small bonus for sweeping tail toward detected predator.
    #    Grounded in: if club is the defense strategy, agent should
    #    position tail toward threat before contact is possible.
    # ----------------------------------------------------------------
    readiness = 0.0
    if pred_dist_raw < TREX_DETECTION_RANGE_M and env.stage >= 3:
        tail_angle = env.get_joint_qpos("tail_handle_lat")
        _, pred_bearing = env._get_predator_obs()
        alignment = float(np.cos(tail_angle - pred_bearing))
        readiness = w["w_tail_ready"] * max(0.0, alignment)
    components["tail_readiness"] = readiness

    total = float(sum(components.values()))
    return total, components
