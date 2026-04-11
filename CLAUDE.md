# RoboTamer4Qmini — Claude Code Context

## What this repo is

RL training + sim2sim validation for the **Unitree Qmini biped robot**.
- **Training**: Isaac Gym (GPU-parallel, 4096 envs)
- **Validation**: MuJoCo sim2sim before real deployment
- **Deployment**: C++ SDK in `~/code/RoboTamerSdk4Qmini`

The robot has 10 actuated joints (5 per leg: hip_yaw, hip_roll, hip_pitch, knee, ankle).

---

## Key files

| File | Purpose |
|------|---------|
| `train.py` | Entry point for training |
| `config/Base.py` | All hyperparameters and ranges |
| `config/BIRL.py` | BIRL-specific overrides (extends Base) |
| `env/legged_robot.py` | Isaac Gym environment, physics, resets |
| `env/tasks/birl_task.py` | Observation, reward, action logic |
| `env/tasks/base_task.py` | Command resampling |
| `env/utils/phase_modulator.py` | Leg phase oscillator |
| `export_pt2onnx.py` | Export `.pt` checkpoint to `.onnx` |
| `deploy/sim2sim/sim2sim.py` | Interactive MuJoCo sim2sim |
| `deploy/sim2sim/evaluate.py` | Batch evaluation + CSV + TensorBoard |
| `deploy/sim2sim/configs/qmini_birl.yaml` | Sim2sim config (must mirror training) |

---

## Training

```bash
# Start fresh
/home/woody/miniconda3/envs/qmini/bin/python train.py \
    --config BIRL --name <run_name> \
    --sim2sim_interval 500 --num_envs 4096

# Resume from checkpoint
/home/woody/miniconda3/envs/qmini/bin/python train.py \
    --config BIRL --name <run_name> --resume <run_name> \
    --max_iterations 10000

# Override max iterations
... --max_iterations 5000
```

Experiments saved to `experiments/<name>/`.
ONNX exported automatically at each sim2sim eval → `experiments/<name>/deploy/policy_<iter>.onnx`.

TensorBoard: `tensorboard --logdir experiments/` (already running on pts/2, PID 239703).

---

## Sim2sim

```bash
# Interactive viewer
/home/woody/miniconda3/envs/qmini/bin/python deploy/sim2sim/sim2sim.py \
    --cmd_vx 0.5 --cmd_vy 0.0 --cmd_yaw 0.0 \
    --policy experiments/<name>/deploy/policy_<iter>.onnx

# Headless
... --headless

# Carpet-like friction
... --floor_friction 3.0
```

---

## Evaluation

```bash
/home/woody/miniconda3/envs/qmini/bin/python deploy/sim2sim/evaluate.py \
    --policy experiments/<name>/deploy/policy_<iter>.onnx \
    --runs 10 --duration 10
```

Quick eval also runs automatically during training every `--sim2sim_interval` iters and logs to TensorBoard as `sim2sim/*` scalars.

---

## Export

```bash
# Export specific iteration
/home/woody/miniconda3/envs/qmini/bin/python export_pt2onnx.py \
    --name <run_name> --iter 2000

# Export latest (policy.pt)
/home/woody/miniconda3/envs/qmini/bin/python export_pt2onnx.py --name <run_name>
```

---

## Observation vector (44 dims per step × 3 history = 132 total)

| Index | Content | Notes |
|-------|---------|-------|
| 0 | cmd_vx | forward velocity command |
| 1 | cmd_vy | lateral velocity command ← **added April 2026** |
| 2 | cmd_yaw | yaw rate command |
| 3–4 | roll, pitch | from imu_in_torso body |
| 5–7 | angular velocity × 0.5 | body frame |
| 8–17 | joint_pos − ref_joint_pos | |
| 18–27 | joint_vel × 0.1 | |
| 28–37 | joint_act − joint_pos | tracking error |
| 38–41 | sin/cos of leg phases × static_flag | phase modulator |
| 42–43 | (freq × 0.3 − 1.0) × static_flag | phase modulator |

**`static_flag`** = 1 if `‖[vx, vy, yaw]‖ ≥ 0.15`, else 0 (zeroes phase signal when standing).

This changed from 43→44 dims when `cmd_vy` was added. **Checkpoints trained before April 2026 are incompatible** with the current obs size.

---

## Action vector (12 dims)

| Index | Content |
|-------|---------|
| 0–1 | leg frequencies [Hz], scaled from [0.5, 3.5] |
| 2–11 | joint position increments [rad/s equivalent], scaled from [-15, 15] |

`use_increment = True` — actions are **deltas** added to current joint targets each policy step.

---

## Commands (config/Base.py)

```python
lin_vel_x_range  = [-0.3, 0.7]   # forward/back
lin_vel_y_range  = [-0.3, 0.3]   # lateral (sideways) — enabled April 2026
ang_vel_yaw_range = [-1, 1]      # yaw rate
```

---

## Key reward terms (env/tasks/birl_task.py)

| Name | Weight | What it does |
|------|--------|--------------|
| `fwd_vel` | 2.3 | tracks cmd_vx |
| `lateral_vel` | 0.7 | tracks cmd_vy (was: penalize all vy) |
| `yaw_rat` | 2.5 | tracks cmd_yaw |
| `balance` | 1.5 | penalizes roll/pitch tilt |
| `base_heit` | 1.0 | keeps body at 0.45m height |
| `foot_heit` | 0.7 | encourages 5cm step height, penalizes >6cm |
| `twist` | 2.5 | penalizes roll+pitch |
| `act_smo` | 1.5 | smooth action (2nd derivative) |

**`lin_vel_x_norm`** is used throughout as a normalization term. When `cmd_vx=0`, it clips to min 0.3+0.2=0.5 to avoid division by zero.

---

## Domain randomization (config/Base.py)

- Friction: `[0.2, 3.0]`
- Mass: ×`[0.5, 1.5]`
- PD gains: ×`[0.8, 1.2]`
- Torque: ×`[0.8, 1.2]`
- Obs delay: joints 10–40 steps, IMU 20–50 steps
- Pushes: every 3s, up to 0.5 m/s

---

## Known issues / things to watch

- **foot_slip_rew** still penalizes foot lateral velocity regardless of cmd_vy — this may partially fight sideways walking. Consider gating it when `|cmd_vy| > 0.1`.
- The `action_constraint_rew` penalizes joints [0,1,5,6] (hip_yaw, hip_roll both legs) deviating from ref — these joints are needed for sideways walking, so this may need tuning.
- Observation delay simulation adds up to ~600ms latency — main sim-to-real gap mitigator.

---

## Best known checkpoints (as of April 2026)

| Run | Iter | Notes |
|-----|------|-------|
| `carpet_v8` | 2000 | Best overall — 100% survival fr≤1.5, good forward tracking |
| `baseline_v10` | ~1 | Suspended/stopped very early, not useful |
| `carpet_v9` | unknown | Was running, status unknown |

All pre-April-2026 checkpoints use 43-dim obs and are **incompatible** with current code (44-dim).

---

## Related repo

**`~/code/RoboTamerSdk4Qmini`** — C++ SDK that runs the policy on the real robot.
The obs vector in `source/user/rl_controller.cpp::get_observation()` must exactly mirror `birl_task.py::pure_observation()`.
