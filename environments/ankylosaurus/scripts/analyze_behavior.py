"""
Behavioral Analysis
===================
Runs the trained Stage 3 policy and logs behavioral metrics for
paleobiological analysis.

Metrics logged
--------------
- Average locomotion speed  (compare to trackway evidence: ~1.4 m/s preferred)
- Tail club swing frequency per predator encounter
- Terrain use patterns     (how often agent seeks vegetation cover)
- Distance maintained from predator (threat assessment behavior)
- Energy management patterns (forage/rest cycles)

These metrics constitute falsifiable behavioral hypotheses::

    "Under paleobiologically-grounded constraints, RL-optimized
     Ankylosaurus behavior is consistent with / inconsistent with
     trace fossil evidence X."

Usage::

    python environments/ankylosaurus/scripts/analyze_behavior.py \\
        --model results/.../final_stage3.zip --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on path
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False
    print("[analyze_behavior] WARNING: pandas/matplotlib not installed. "
          "Figures will not be generated. Run: pip install pandas matplotlib seaborn")

from stable_baselines3 import PPO

from environments.ankylosaurus.envs.ankylosaurus_env import AnkylosaurusEnv
from environments.ankylosaurus.envs.predator_manager import PredatorManager
from environments.ankylosaurus.paleo_constants import (
    CONTROL_TIMESTEP_S,
    MAX_SPEED_MS,
    PREFERRED_SPEED_MS,
    TREX_DETECTION_RANGE_M,
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def run_analysis(model_path: str, n_episodes: int = 50) -> "pd.DataFrame | list":
    """
    Run trained policy for n_episodes and collect per-step behavioral data.
    Returns a DataFrame (or list of dicts if pandas unavailable).
    """
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    predator_manager = PredatorManager(speed_scale=1.0)
    predator_manager.load()

    env = AnkylosaurusEnv(
        stage=3,
        predator_policy=predator_manager,
        n_food_items=8,
    )

    all_records: list[dict] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        prev_tail_angle = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            torso_pos    = env.data.xpos[env.torso_id].copy()
            torso_linvel = env.data.qvel[0:2]
            speed        = float(np.linalg.norm(torso_linvel))
            pred_dist    = float(np.linalg.norm(torso_pos - env._predator_pos))
            tail_angle   = env.get_joint_qpos("tail_handle_lat")
            club_force   = env.get_club_contact_force()
            dist_center  = float(np.linalg.norm(torso_pos[:2]))

            all_records.append({
                "episode":          ep,
                "step":             step,
                "time_s":           step * CONTROL_TIMESTEP_S,
                "speed_ms":         speed,
                "speed_kmh":        speed * 3.6,
                "energy":           env._energy,
                "health":           env._health,
                "pred_dist_m":      pred_dist,
                "pred_detected":    pred_dist < TREX_DETECTION_RANGE_M,
                "tail_handle_deg":  float(np.degrees(tail_angle)),
                "tail_swing_rate":  float(abs(tail_angle - prev_tail_angle)),
                "club_contact":     float(club_force > 0.1),
                "torso_x":          float(torso_pos[0]),
                "torso_y":          float(torso_pos[1]),
                "torso_z":          float(torso_pos[2]),
                "dist_from_center": dist_center,
                "in_cover":         5.0 < dist_center < 12.0,
                "reward":           float(reward),
                "terminated":       bool(terminated),
                "truncated":        bool(truncated),
            })
            prev_tail_angle = tail_angle
            step += 1

        survival_s = step * CONTROL_TIMESTEP_S
        print(
            f"  Ep {ep+1:3d}/{n_episodes}: "
            f"survival={survival_s:6.1f}s  "
            f"health={env._health:.2f}  "
            f"energy={env._energy:.2f}  "
            f"{'SURVIVED' if truncated else info.get('termination_reason','?')}"
        )

    env.close()

    if _HAS_VIZ:
        return pd.DataFrame(all_records)
    return all_records


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(df, output_dir: str) -> None:
    """Generate analysis figures and print summary statistics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _HAS_VIZ:
        print("[analyze_behavior] Skipping figures (pandas/matplotlib unavailable).")
        _print_basic_stats(df)
        return

    import pandas as pd  # guaranteed available here
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    print("\n" + "=" * 60)
    print("BEHAVIORAL ANALYSIS REPORT")
    print("=" * 60)

    # -- 1. Locomotion speed --
    print(f"\n1. LOCOMOTION SPEED")
    mean_kmh   = df["speed_kmh"].mean()
    median_kmh = df["speed_kmh"].median()
    preferred  = PREFERRED_SPEED_MS * 3.6
    max_speed  = MAX_SPEED_MS * 3.6
    print(f"   Mean speed:      {mean_kmh:.2f} km/h")
    print(f"   Median speed:    {median_kmh:.2f} km/h")
    print(f"   Paleo preferred: ~{preferred:.1f} km/h  [trackway analysis]")
    consistent = "CONSISTENT" if preferred * 0.5 < median_kmh < preferred * 2.0 else "INCONSISTENT"
    print(f"   Assessment:      {consistent} with preferred-speed estimate")

    fig, ax = plt.subplots(figsize=(8, 4))
    df["speed_kmh"].clip(upper=MAX_SPEED_MS * 3.6 * 1.2).hist(bins=60, ax=ax,
        color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(preferred, color="red",    ls="--", label=f"Preferred (~{preferred:.1f} km/h)")
    ax.axvline(max_speed, color="orange", ls="--", label=f"Max estimate ({max_speed:.1f} km/h)")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Frequency (steps)")
    ax.set_title("Emergent Locomotion Speed Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "speed_distribution.png", dpi=150)
    plt.close()

    # -- 2. Tail club usage vs predator proximity --
    print(f"\n2. TAIL CLUB USAGE")
    encounters = df[df["pred_detected"]]
    club_hits  = df[df["club_contact"] > 0]
    if len(encounters) > 0:
        rate = len(club_hits) / len(encounters)
        print(f"   Predator-detected steps: {len(encounters)}")
        print(f"   Club-contact events:     {len(club_hits)}")
        print(f"   Club use rate:           {rate:.4f} per encounter step")
        strategy = "Reactive (last-resort)" if rate < 0.05 else "Preemptive"
        print(f"   Interpretation:          {strategy} defense strategy")
    else:
        print("   No predator encounters (check Stage 3 config).")

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(
        df["pred_dist_m"], df["tail_handle_deg"].abs(),
        c=df["episode"], cmap="viridis", alpha=0.15, s=2,
    )
    ax.axvline(TREX_DETECTION_RANGE_M, color="orange", ls="--",
               label=f"Detection range ({TREX_DETECTION_RANGE_M:.0f}m)")
    ax.axvline(1.5, color="red", ls="--", label="Bite range (1.5m)")
    ax.set_xlabel("Predator Distance (m)")
    ax.set_ylabel("|Tail Handle Angle| (°)")
    ax.set_title("Tail Club Angle vs Predator Distance")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Episode")
    fig.tight_layout()
    fig.savefig(out / "tail_vs_predator.png", dpi=150)
    plt.close()

    # -- 3. Survival statistics --
    ep_stats = df.groupby("episode").agg(
        survival_steps=("step", "max"),
        final_health=("health", "last"),
        final_energy=("energy", "last"),
        mean_pred_dist=("pred_dist_m", "mean"),
        cover_fraction=("in_cover", "mean"),
        club_contacts=("club_contact", "sum"),
    )
    ep_stats["survival_time_s"] = ep_stats["survival_steps"] * CONTROL_TIMESTEP_S
    full_episodes = int((ep_stats["survival_time_s"] >= 299).sum())

    print(f"\n3. SURVIVAL STATISTICS  (n={len(ep_stats)} episodes)")
    print(f"   Mean survival time:    {ep_stats['survival_time_s'].mean():.1f}s")
    print(f"   Median survival time:  {ep_stats['survival_time_s'].median():.1f}s")
    print(f"   Episodes reaching 300s: {full_episodes}")
    print(f"   Mean final health:     {ep_stats['final_health'].mean():.3f}")
    print(f"   Mean final energy:     {ep_stats['final_energy'].mean():.3f}")
    print(f"   Mean cover fraction:   {ep_stats['cover_fraction'].mean():.3f}")
    print(f"   Total club contacts:   {int(ep_stats['club_contacts'].sum())}")

    # -- 4. Spatial trajectories --
    fig, ax = plt.subplots(figsize=(8, 8))
    for ep_id in df["episode"].unique()[:20]:
        ep_df = df[df["episode"] == ep_id]
        ax.plot(ep_df["torso_x"], ep_df["torso_y"], alpha=0.3, linewidth=0.8)
    for r, color, label in [
        (5,  "green",  "Inner veg. zone (5m)"),
        (12, "olive",  "Outer veg. zone (12m)"),
    ]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=color,
                                ls="--", linewidth=1.2, label=label))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Spatial Trajectories — first 20 episodes")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=150)
    plt.close()

    # -- 5. Energy over time (averaged across episodes) --
    mean_energy = df.groupby("step")["energy"].mean()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(mean_energy.index * CONTROL_TIMESTEP_S, mean_energy.values,
            color="forestgreen", linewidth=1.2)
    ax.axhline(0.4, color="red", ls="--", alpha=0.6, label="Stage 2 success threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (mean across episodes)")
    ax.set_title("Mean Energy Trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "energy_trajectory.png", dpi=150)
    plt.close()

    # Save raw data
    df.to_csv(out / "behavioral_data.csv", index=False)
    ep_stats.to_csv(out / "episode_stats.csv")

    print(f"\n[analyze_behavior] Report saved to: {out.resolve()}")
    print("  Files:")
    for f in sorted(out.iterdir()):
        print(f"    {f.name}")


def _print_basic_stats(records: list[dict]) -> None:
    """Minimal console report when pandas/matplotlib are unavailable."""
    speeds    = [r["speed_kmh"] for r in records]
    energies  = [r["energy"] for r in records]
    survivals = {}
    for r in records:
        ep = r["episode"]
        survivals[ep] = max(survivals.get(ep, 0), r["step"])

    print(f"\nMean speed: {np.mean(speeds):.2f} km/h")
    print(f"Mean energy: {np.mean(energies):.3f}")
    print(f"Mean survival steps: {np.mean(list(survivals.values())):.0f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-training behavioral analysis for Ankylosaurus RL."
    )
    parser.add_argument("--model", required=True,
                        help="Path to trained Stage 3 .zip model file.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", default="analysis_output",
                        help="Output directory for figures and CSV files.")
    args = parser.parse_args()

    data = run_analysis(args.model, n_episodes=args.episodes)
    generate_report(data, args.output)
