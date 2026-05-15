# Ankylosaurus Behavioral Paleontology via Reinforcement Learning

A computational paleontology tool that uses reinforcement learning to generate **falsifiable behavioral hypotheses** about how *Ankylosaurus magniventris* survived in the Late Cretaceous Hell Creek formation (~66 Ma).

This is not a game. Every physics parameter traces to a peer-reviewed source. The RL agent discovers behavior; the paleobiology constrains the physics within which that behavior emerges.

---

## What this project answers

- Under realistic biomechanical constraints, what survival strategy does an Ankylosaurus converge on?
- Does the emergent strategy match trace fossil evidence (trackways, bite marks on osteoderms)?
- When does the tail club get used — preemptively, reactively, or as a last resort?
- Does the agent exploit terrain cover, consistent with a prey animal in a predator-rich environment?

---

## The predator is real

The T-rex adversary is the **pretrained Stage 3 PPO policy** from [mesozoic-labs](https://github.com/kuds/mesozoic-labs) (96.7% bite success rate). The ankylosaur is not fighting a scripted dummy — it is fighting a trained hunter. This matters for behavioral validity.

---

## Repository structure

```
ankylo_rl/
├── environments/
│   ├── shared/                    # Base classes from mesozoic-labs (DO NOT MODIFY)
│   │   ├── base_env.py            # BaseDinoEnv — AnkylosaurusEnv extends this
│   │   ├── curriculum.py          # CurriculumManager
│   │   └── ...
│   ├── trex/                      # Pretrained T-rex adversary (DO NOT MODIFY)
│   │   └── assets/trex.xml
│   └── ankylosaurus/              # This project
│       ├── assets/
│       │   └── ankylosaurus.xml   # 28-DOF MJCF body model
│       ├── envs/
│       │   ├── ankylosaurus_env.py  # Gymnasium environment (83-dim obs, 28-dim action)
│       │   ├── reward.py            # 9 paleobiologically grounded reward terms
│       │   ├── predator_manager.py  # T-rex policy loader + heuristic fallback
│       │   └── curriculum.py        # 3-stage AnkylosaurCurriculum
│       ├── scripts/
│       │   ├── train_sb3.py         # PPO/SAC training entry point
│       │   ├── view_model.py        # Interactive MJCF viewer
│       │   └── analyze_behavior.py  # Post-training behavioral analysis
│       ├── tests/
│       │   └── test_env.py          # 21 sanity tests — run before training
│       ├── configs/
│       │   └── ankylosaurus.toml    # Per-stage hyperparameters and reward weights
│       └── paleo_constants.py       # All physics constants with citations
```

---

## Physics grounding

Every constant has a citation. Constants with no direct fossil basis are marked `# ASSUMPTION`.

| Parameter | Value | Source |
|---|---|---|
| Body mass | 4,800–6,000 kg | Arbour & Currie 2013, PLoS ONE 8(5):e62421 |
| Tail club impact force | 7,281–14,360 N | Arbour & Snively 2009, Anat Rec 292:1412 (FEA) |
| Tail handle DOF | Lateral only | Arbour 2009 — ossified caudal vertebra morphology |
| Max locomotion speed | 8–13 km/h | Alexander 1989 + trackway analysis |
| Gait type | Wide-gauge quadruped | Fossil acetabulum geometry |
| T-rex bite force | ~57 kN | Bates & Falkingham 2012 |
| T-rex max speed | 5.5 m/s | Hutchinson & Garcia 2002, Nature 415 |
| Hell Creek vegetation | Cycad, fern, conifer, angiosperm | Fastovsky & Sheehan 2005 |

---

## Model summary

**MJCF:** `environments/ankylosaurus/assets/ankylosaurus.xml`

| Region | DOF | Paleobiological basis |
|---|---|---|
| Neck (pitch + yaw) | 2 | Short stiff neck, nearly horizontal posture |
| Head (pitch) | 1 | Ground-level low browsing |
| Hip × 4 (abduct + flex + rotate) | 12 | Wide-gauge gait from acetabulum orientation |
| Knee × 4 | 4 | Condyle fossil geometry |
| Ankle × 4 | 4 | Simplified hinge |
| Tail base (lat + dv) | 2 | Proximal caudal vertebrae flexibility |
| Tail mid (lat + dv) | 2 | Stiffening toward distal end |
| Tail handle (lat only) | 1 | **Ossified caudals — Arbour 2009** |
| **Total actuated** | **28** | |

Total model mass: ~4,660 kg (within 4,800–6,000 kg range).

---

## Curriculum

Three stages, each building on the last:

| Stage | Name | Predator | Food | Success metric |
|---|---|---|---|---|
| 1 | Balance | No | No | Episode length > 200 steps |
| 2 | Forage | No | Yes (8 items) | Final energy > 40% |
| 3 | Survive | Yes (trained T-rex) | Yes | Survival time > 60s |

Estimated training time on an L4 GPU with 8 parallel envs: ~12–16 hours total.

---

## Quick start

```bash
git clone https://github.com/ps1526/ankylo_rl.git
cd ankylo_rl
pip install -e ".[train]"

# Run sanity tests first
pytest environments/ankylosaurus/tests/ -v

# View the model interactively
python environments/ankylosaurus/scripts/view_model.py --random-actions

# Full 3-stage curriculum training
python environments/ankylosaurus/scripts/train_sb3.py curriculum --algorithm ppo --n-envs 8

# Post-training behavioral analysis
python environments/ankylosaurus/scripts/analyze_behavior.py \
    --model environments/ankylosaurus/results/.../final_stage3.zip \
    --episodes 50
```

---

## Reward function

Nine terms, each with a paleobiological justification:

| Term | Weight (Stage 3) | Justification |
|---|---|---|
| `survival` | 0.10 | Dense: incentive to stay alive |
| `energy` | 2.00 | Herbivore foraging necessity |
| `locomotion_cost` | −0.05 | High cost of transport for 5-ton animal |
| `club_hit` | **10.00** | FEA: club delivers bone-fracture-level force |
| `pred_proximity` | −0.50 | Graduated penalty as T-rex closes |
| `bite_range` | −2.00 | Spike penalty inside 1.5m bite range |
| `fall` | −1.00 | Falling = exposed ventrum, catastrophic |
| `joint_limits` | −0.10 | Anatomical range enforcement |
| `terrain_cover` | 0.10 | Prey animals use vegetation defensively |
| `tail_readiness` | 0.20 | Orient tail toward threat |

---

## Built on mesozoic-labs

This project is built on top of **[mesozoic-labs](https://github.com/kuds/mesozoic-labs)** by [Michael Kudlaty](https://github.com/kuds), an open-source platform for dinosaur biomechanics and robotic locomotion via reinforcement learning.

Specifically, this project:
- Extends `BaseDinoEnv` from `environments/shared/base_env.py`
- Uses the shared MuJoCo + Gymnasium + Stable-Baselines3 training infrastructure
- Loads the **pretrained T-rex Stage 3 PPO policy** (96.7% bite success) as the adversary

The ankylosaurus environment follows the same structural conventions as the existing velociraptor, T-rex, and brachiosaurus environments in that repo.

Please cite mesozoic-labs if you use this work:

```bibtex
@software{mesozoic_labs,
  title   = {Mesozoic Labs: Dinosaur Locomotion via Reinforcement Learning},
  author  = {Michael Kudlaty},
  year    = {2025},
  url     = {https://github.com/kuds/mesozoic-labs},
  license = {MIT}
}
```

---

## Key papers

```
[1] Arbour & Snively 2009 — Tail club FEA (primary damage model source)
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2726940/

[2] Arbour & Currie 2013 — Body mass, morphology
    https://doi.org/10.1371/journal.pone.0062421

[3] Hutchinson & Garcia 2002 — T-rex speed upper bound
    https://doi.org/10.1038/415001a

[4] Fastovsky & Sheehan 2005 — Hell Creek Formation environment
    https://doi.org/10.1130/1052-5173(2005)015<4:TETODI>2.0.CO;2
```

---

## License

MIT
