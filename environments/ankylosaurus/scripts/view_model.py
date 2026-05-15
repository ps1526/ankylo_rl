"""
view_model.py
=============
Visualize the Ankylosaurus MJCF model in the MuJoCo interactive viewer.

Usage::

    # Passive simulation (gravity only):
    python environments/ankylosaurus/scripts/view_model.py

    # Random sinusoidal actions to see all 28 DOF in motion:
    python environments/ankylosaurus/scripts/view_model.py --random-actions

    # Hold default pose (no gravity, no control):
    python environments/ankylosaurus/scripts/view_model.py --reset-pose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_ASSET_PATH = Path(__file__).parent.parent / "assets" / "ankylosaurus.xml"

_ACTUATOR_NAMES = [
    "act_neck_pitch", "act_neck_yaw", "act_head_pitch",
    "act_hip_FL_abduct", "act_hip_FL_flex", "act_hip_FL_rotate",
    "act_knee_FL", "act_ankle_FL",
    "act_hip_FR_abduct", "act_hip_FR_flex", "act_hip_FR_rotate",
    "act_knee_FR", "act_ankle_FR",
    "act_hip_RL_abduct", "act_hip_RL_flex", "act_hip_RL_rotate",
    "act_knee_RL", "act_ankle_RL",
    "act_hip_RR_abduct", "act_hip_RR_flex", "act_hip_RR_rotate",
    "act_knee_RR", "act_ankle_RR",
    "act_tail_base_lat", "act_tail_base_dv",
    "act_tail_mid_lat",  "act_tail_mid_dv",
    "act_tail_handle_lat",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="View Ankylosaurus MJCF model.")
    parser.add_argument("--random-actions", action="store_true",
                        help="Apply random sinusoidal control to all 28 DOF.")
    parser.add_argument("--reset-pose", action="store_true",
                        help="Hold zero-control pose (passive gravity).")
    args = parser.parse_args()

    print(f"Loading: {_ASSET_PATH}")
    model = mujoco.MjModel.from_xml_path(str(_ASSET_PATH))
    data  = mujoco.MjData(model)

    total_mass = float(np.sum(model.body_mass))
    print(f"\nModel summary:")
    print(f"  Bodies:    {model.nbody}")
    print(f"  Joints:    {model.njnt - 1} actuated  + 1 freejoint")
    print(f"  Actuators: {model.nu}")
    print(f"  Sensors:   {model.nsensor}")
    print(f"  Total DOF: {model.nv}")
    print(f"\n  Total mass: {total_mass:.1f} kg")
    print(f"  Expected:   4,800–6,000 kg  [Arbour & Currie 2013]")
    if not (4000 < total_mass < 7000):
        print(f"  WARNING: Mass outside expected range — check geom masses.")

    print("\nPress Esc or close the viewer window to exit.\n")

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.30  # Initial torso height

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            if args.random_actions:
                t = step * model.opt.timestep
                for i in range(model.nu):
                    freq = 0.5 + 0.3 * (i % 4)
                    data.ctrl[i] = 0.4 * np.sin(2 * np.pi * freq * t + i)
            else:
                data.ctrl[:] = 0.0  # Passive (gravity drives everything)

            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
