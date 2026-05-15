"""
train_sb3.py
============
Stable-Baselines3 PPO/SAC training for Ankylosaurus with curriculum learning.
Mirrors the train_sb3.py structure used by mesozoic-labs trex and velociraptor.

Usage
-----
Full 3-stage curriculum (recommended)::

    python environments/ankylosaurus/scripts/train_sb3.py curriculum \\
        --algorithm ppo --n-envs 8

Single stage::

    python environments/ankylosaurus/scripts/train_sb3.py train \\
        --stage 2 --timesteps 5000000

Continue from checkpoint::

    python environments/ankylosaurus/scripts/train_sb3.py train \\
        --stage 3 --checkpoint path/to/model.zip

With Weights & Biases logging::

    python environments/ankylosaurus/scripts/train_sb3.py curriculum \\
        --algorithm ppo --wandb --project ankylosaur-rl
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

from environments.ankylosaurus.envs.curriculum import STAGE_CONFIGS, AnkylosaurCurriculum
from environments.ankylosaurus.envs.predator_manager import PredatorManager

_OUTPUT_DIR = Path(__file__).parent.parent / "results"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameter templates
# ---------------------------------------------------------------------------

PPO_HYPERPARAMS: dict = {
    # Tuned for large-body quadruped locomotion (28 DOF, 83-dim obs).
    # Reference: mesozoic-labs velociraptor PPO tuning notes.
    "learning_rate":  3e-4,
    "n_steps":        2048,
    "batch_size":     256,
    "n_epochs":       10,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "ent_coef":       0.01,
    "vf_coef":        0.5,
    "max_grad_norm":  0.5,
    "policy_kwargs": {
        "net_arch": [512, 512, 256],
        "activation_fn": torch.nn.Tanh,
    },
}

SAC_HYPERPARAMS: dict = {
    # Use SAC if PPO struggles with 28 DOF continuous control.
    # Better sample efficiency but slower wall-clock per step.
    "learning_rate":   3e-4,
    "buffer_size":     1_000_000,
    "learning_starts": 10_000,
    "batch_size":      256,
    "tau":             0.005,
    "gamma":           0.99,
    "train_freq":      1,
    "gradient_steps":  1,
    "policy_kwargs": {
        "net_arch": [512, 512, 256],
    },
}

# Per-stage PPO overrides (discount + entropy tuning)
_STAGE_PPO_OVERRIDES: dict[int, dict] = {
    1: {"ent_coef": 0.02,  "gamma": 0.99,  "clip_range": 0.20},
    2: {"ent_coef": 0.01,  "gamma": 0.995, "clip_range": 0.20,
        "learning_rate": 2e-4},
    3: {"ent_coef": 0.005, "gamma": 0.995, "clip_range": 0.15,
        "learning_rate": 1e-4, "n_steps": 4096, "batch_size": 512},
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env_fn(stage: int, predator_policy, rank: int):
    """SubprocVecEnv factory — each subprocess gets its own env + seed."""
    def _init():
        curriculum = AnkylosaurCurriculum(start_stage=stage)
        env = curriculum.make_env(predator_policy=predator_policy)
        env.reset(seed=rank * 1000 + stage)
        return env
    return _init


# ---------------------------------------------------------------------------
# Single-stage training
# ---------------------------------------------------------------------------

def train_stage(
    stage: int,
    algorithm: str = "ppo",
    n_envs: int = 8,
    timesteps: int | None = None,
    checkpoint: str | None = None,
    use_wandb: bool = False,
    project: str = "ankylosaur-rl",
) -> str:
    """Train a single curriculum stage. Returns path to saved final model."""
    config = STAGE_CONFIGS[stage]
    if timesteps is None:
        timesteps = config["timesteps"]

    print(f"\n{'='*60}")
    print(f"Stage {stage}: {config['name'].upper()}")
    print(f"  {config['description']}")
    print(f"  Algorithm:  {algorithm.upper()}")
    print(f"  Envs:       {n_envs}")
    print(f"  Timesteps:  {timesteps:,}")
    print(f"{'='*60}\n")

    # Load predator if stage 3
    predator_policy = None
    if config["predator_active"]:
        manager = PredatorManager(speed_scale=config["predator_speed_scale"])
        manager.load()
        predator_policy = manager  # heuristic fallback handled inside

    # Vectorised training envs
    env_fns = [_make_env_fn(stage, predator_policy, i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Single eval env (no subprocess)
    eval_env = _make_env_fn(stage, predator_policy, rank=99)()

    # Run naming and output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"stage{stage}_{config['name']}_{algorithm}_{timestamp}"
    checkpoint_dir = _OUTPUT_DIR / run_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if use_wandb:
        import wandb
        hparams = PPO_HYPERPARAMS if algorithm == "ppo" else SAC_HYPERPARAMS
        wandb.init(
            project=project,
            name=run_name,
            config={"stage": stage, "algorithm": algorithm,
                    "n_envs": n_envs, "timesteps": timesteps, **hparams},
        )

    # Build model
    ModelClass = PPO if algorithm == "ppo" else SAC
    if checkpoint:
        print(f"  Resuming from checkpoint: {checkpoint}")
        model = ModelClass.load(checkpoint, env=vec_env)
    else:
        base_params = (PPO_HYPERPARAMS if algorithm == "ppo" else SAC_HYPERPARAMS).copy()
        if algorithm == "ppo":
            base_params.update(_STAGE_PPO_OVERRIDES.get(stage, {}))
        model = ModelClass(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=str(_OUTPUT_DIR / "tb_logs"),
            **base_params,
        )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"ankylo_s{stage}",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(checkpoint_dir / "eval_logs"),
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(checkpoint is None),
    )

    # Save final
    final_path = str(_OUTPUT_DIR / run_name / f"final_stage{stage}.zip")
    model.save(final_path)
    print(f"\n[Train] Saved final model: {final_path}")

    vec_env.close()
    eval_env.close()

    if use_wandb:
        import wandb
        wandb.finish()

    return final_path


# ---------------------------------------------------------------------------
# Full curriculum
# ---------------------------------------------------------------------------

def train_curriculum(
    algorithm: str = "ppo",
    n_envs: int = 8,
    use_wandb: bool = False,
    project: str = "ankylosaur-rl",
    start_stage: int = 1,
    checkpoint: str | None = None,
) -> None:
    """Run all 3 curriculum stages sequentially."""
    prev_checkpoint = checkpoint
    for stage in range(start_stage, 4):
        final_path = train_stage(
            stage=stage,
            algorithm=algorithm,
            n_envs=n_envs,
            checkpoint=prev_checkpoint,
            use_wandb=use_wandb,
            project=project,
        )
        prev_checkpoint = final_path  # chain stages

    print("\n[Curriculum] Complete — all 3 stages trained.")
    print(f"  Final model: {prev_checkpoint}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Ankylosaurus RL agent with curriculum learning."
    )
    sub = parser.add_subparsers(dest="command")

    # -- curriculum --
    c = sub.add_parser("curriculum", help="Run all 3 stages sequentially.")
    c.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    c.add_argument("--n-envs", type=int, default=8)
    c.add_argument("--wandb", action="store_true")
    c.add_argument("--project", default="ankylosaur-rl")
    c.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3])
    c.add_argument("--checkpoint", default=None,
                   help="Resume from this checkpoint (applies to start-stage only).")

    # -- single stage --
    t = sub.add_parser("train", help="Train a single curriculum stage.")
    t.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    t.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    t.add_argument("--n-envs", type=int, default=8)
    t.add_argument("--timesteps", type=int, default=None)
    t.add_argument("--checkpoint", default=None)
    t.add_argument("--wandb", action="store_true")
    t.add_argument("--project", default="ankylosaur-rl")

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.command == "curriculum":
        train_curriculum(
            algorithm=args.algorithm,
            n_envs=args.n_envs,
            use_wandb=args.wandb,
            project=args.project,
            start_stage=args.start_stage,
            checkpoint=args.checkpoint,
        )
    elif args.command == "train":
        train_stage(
            stage=args.stage,
            algorithm=args.algorithm,
            n_envs=args.n_envs,
            timesteps=args.timesteps,
            checkpoint=args.checkpoint,
            use_wandb=args.wandb,
            project=args.project,
        )
    else:
        _build_parser().print_help()
