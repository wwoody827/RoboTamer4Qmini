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
| `config/loader.py` | YAML config loader with `_base:` inheritance |
| `configs/base.yaml` | All defaults (reward weights, DR, PD gains, obs slots, phase mode) |
| `configs/birl.yaml` | Phase `output` mode (12-dim action, BIRL baseline) |
| `configs/bdx.yaml` | Phase `input` mode (10-dim action, external clock, BD_X style) |
| `configs/mirl*.yaml` | Phase `none` mode (10-dim action, reference clip imitation) |
| `env/legged_robot.py` | Isaac Gym environment, physics, resets |
| `env/tasks/birl_task.py` | Unified task class — all phase modes via config |
| `env/tasks/null_task.py` | Abstract task interface + registry |
| `env/obs_builder.py` | Config-driven observation construction (named slots) |
| `env/utils/phase_modulator.py` | Internal leg phase oscillator + external phase clock |
| `utils/mirror.py` | `BIRLMirror` (12-dim action) + `BDXMirror` (10-dim) for symmetry aug |
| `export_pt2onnx.py` | Export `.pt` checkpoint to `.onnx` |
| `deploy/sim2sim/sim2sim.py` | Interactive MuJoCo sim2sim |
| `deploy/sim2sim/evaluate.py` | Batch evaluation + CSV + TensorBoard |
| `deploy/manifest.py` | Shared manifest builder for export + sim2sim |
| `docs/rewards.md` | Reward term reference |

---

## Training

```bash
# BIRL baseline (phase.mode=output, 12-dim action)
/home/woody/miniconda3/envs/qmini/bin/python train.py \
    --config configs/birl_fwd.yaml --name <run_name> \
    --sim2sim_interval 500 --num_envs 4096

# BD_X style (phase.mode=input, 10-dim action, external clock)
/home/woody/miniconda3/envs/qmini/bin/python train.py \
    --config configs/bdx.yaml --name bdx_v1 \
    --sim2sim_interval 500 --num_envs 4096

# MIRL forward expert (phase.mode=none, 10-dim action)
/home/woody/miniconda3/envs/qmini/bin/python train.py \
    --config configs/mirl_fwd.yaml --name mirl_fwd_v1 \
    --sim2sim_interval 500 --num_envs 4096

# Resume from checkpoint
/home/woody/miniconda3/envs/qmini/bin/python train.py \
    --config configs/birl_fwd.yaml --name <run_name> --resume <run_name> \
    --max_iterations 10000
```

Sim2sim config is built automatically from the training config via manifest (no separate YAML needed).
Experiments saved to `experiments/<name>/`.
ONNX + manifest exported automatically at each sim2sim eval → `experiments/<name>/deploy/policy_<iter>.onnx` + `policy_<iter>_manifest.yaml`.

TensorBoard: `tensorboard --logdir experiments/`

---

## Sim2sim

```bash
# Interactive viewer (BIRL or MIRL — auto-detected from action dim)
/home/woody/miniconda3/envs/qmini/bin/python deploy/sim2sim/sim2sim.py \
    --cmd_vx 0.5 --cmd_vy 0.0 --cmd_yaw 0.0 \
    --policy experiments/<name>/deploy/policy_<iter>.onnx

# Headless / carpet friction
... --headless --floor_friction 3.0

# Record reference clip for MIRL imitation
... --record data/reference_clips/walk_fwd.npz --record_skill walk --record_loop --headless --duration 10
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

## Phase modes

A single unified task class (`BIRLTask`) handles all variants, selected by `phase.mode` in config:

| `phase.mode` | Action dim | Phase source | Representative config |
|---|---|---|---|
| `output` | 12 (2 leg-freq + 10 joints) | Policy → PhaseModulator integrates | `configs/birl_fwd.yaml` (BIRL baseline) |
| `input`  | 10 (joints only) | External clock, `freq = base + vel_scale·‖cmd‖` | `configs/bdx.yaml` |
| `none`   | 10 (joints only) | No phase signal | `configs/mirl*.yaml` |

Mirror augmentation: `BIRLMirror` (12-dim) or `BDXMirror` (10-dim), selected automatically from `phase.mode`.

---

## BIRL interface

### Observation vector — 44 dims per step × 3 history = 132 total

| Index | Content | Notes |
|-------|---------|-------|
| 0 | cmd_vx | forward velocity command |
| 1 | cmd_vy | lateral velocity command |
| 2 | cmd_yaw | yaw rate command |
| 3–4 | roll, pitch | from imu_in_torso body |
| 5–7 | angular velocity × 0.5 | body frame |
| 8–17 | joint_pos − ref_joint_pos | |
| 18–27 | joint_vel × 0.1 | |
| 28–37 | joint_act − joint_pos | tracking error |
| 38–41 | sin/cos of leg phases × static_flag | phase modulator |
| 42–43 | (freq × 0.3 − 1.0) × static_flag | phase modulator |

### Action vector — 12 dims

| Index | Content |
|-------|---------|
| 0–1 | leg frequencies [Hz], scaled from [0.5, 3.5] |
| 2–11 | joint position increments [rad/s], scaled from [-15, 15] |

Default `action.action_mode = increment` — actions are **deltas** added to current joint targets each policy step. `action.action_mode = absolute` switches to direct position targets with optional low-pass filter (`action_lowpass_alpha`).

**`static_flag`** = 1 if `‖[vx, vy, yaw]‖ ≥ 0.15`, else 0 (zeroes phase signals when standing still).

---

## MIRL interface

### Observation vector — 64 dims per step × 3 history = 192 total

| Index | Content | Notes |
|-------|---------|-------|
| 0–7 | commands[0:8] | [vx, vy, yaw, height(0), 0, 0, 0, 0] — slots 3-7 always 0 |
| 8–9 | roll, pitch | from imu_in_torso body |
| 10–12 | angular velocity × 0.5 | body frame |
| 13–22 | joint_pos − ref_joint_pos | standing pose reference |
| 23–32 | joint_vel × 0.1 | |
| 33–42 | joint_act − joint_pos | tracking error |
| 43–52 | ref_joint_pos[t] − joint_pos | reference clip frame; **zeros if no clip loaded** |
| 53–62 | ref_joint_vel[t] | reference clip frame; **zeros if no clip loaded** |
| 63 | phase_progress (0→1) | position in reference clip; **zero if no clip loaded** |

### Action vector — 10 dims

| Index | Content |
|-------|---------|
| 0–9 | joint position increments [rad/s], scaled from [-15, 15] |

No phase modulator. All 10 outputs are joint deltas: `joint_act += action * dt`.

### Commands (configs/mirl*.yaml)

Each expert config enables only the relevant slot(s):

| Config | vx range | vy range | yaw range |
|--------|----------|----------|-----------|
| `mirl_fwd.yaml` | [-0.3, 0.7] | [0, 0] | [0, 0] |
| `mirl_strafe.yaml` | [0, 0] | [-0.3, 0.3] | [0, 0] |
| `mirl_turn.yaml` | [0, 0] | [0, 0] | [-1, 1] |
| `mirl_combined.yaml` | [-0.3, 0.7] | [-0.3, 0.3] | [-1, 1] |

Slots with range `[0, 0]` are always zero — `_resample_commands` skips them entirely (avoids `random.choice([])` crash on single-direction configs).

### Reference clip format (.npz)

Recorded by `deploy/sim2sim/sim2sim.py --record` at policy rate (67 Hz):

| Key | Shape | Content |
|-----|-------|---------|
| `joint_pos` | [T, 10] | joint positions at each policy step |
| `joint_vel` | [T, 10] | joint velocities at each policy step |
| `base_pos` | [T, 3] | base position |
| `base_quat` | [T, 4] | base orientation |
| `base_lin_vel` | [T, 3] | base linear velocity |
| `base_ang_vel` | [T, 3] | base angular velocity |
| `dt` | float | policy dt (≈ 0.015s) |
| `skill` | str | label, e.g. `"walk"` |
| `loop` | bool | whether clip loops |

### Imitation reward (active only when ref_clip_paths ≠ [])

```
jp_imit = exp(-5.0 * ‖joint_pos - ref_joint_pos[t]‖²)
jv_imit = exp(-0.1 * ‖joint_vel - ref_joint_vel[t]‖²)
imit_rew = (jp_imit + jv_imit) * 0.5
```

Reward scaling via config: `w_imit` scales imitation, `w_task` scales all task rewards.

### MIRL training flow

```
Step 1: train 3 experts (no clips):
  configs/mirl_fwd.yaml    → mirl_fwd_v1     (vx only)
  configs/mirl_strafe.yaml → mirl_strafe_v1  (vy only)
  configs/mirl_turn.yaml   → mirl_turn_v1    (yaw only)

Step 2: record reference clips from each expert:
  python deploy/sim2sim/sim2sim.py \
    --policy experiments/mirl_fwd_v1/deploy/policy_5000.onnx \
    --record data/reference_clips/walk_fwd.npz \
    --record_skill walk --record_loop --headless --duration 10
  (repeat for strafe → walk_strafe.npz, turn → walk_turn.npz)

Step 3: set task.ref_clip_paths in configs/mirl_combined.yaml, then:
  python train.py --config configs/mirl_combined.yaml --name mirl_combined_v1
```

---

## Key reward terms (env/tasks/birl_task.py — see docs/rewards.md for full list)

| Name | Weight | What it does |
|------|--------|--------------|
| `fwd_vel` | 2.3 | tracks cmd_vx |
| `lateral_vel` | 0.7 | tracks cmd_vy |
| `yaw_rat` | 2.5 | tracks cmd_yaw |
| `balance` | 1.5 | penalizes roll/pitch tilt |
| `base_heit` | 1.0 | keeps body at 0.45m height |
| `foot_heit` | 0.7 | encourages 5cm step height |
| `twist` | 2.5 | penalizes roll+pitch |
| `act_smo` | 1.5 | smooth action (2nd derivative) |

MIRL adds `jp_imit` and `jv_imit` when clips are loaded (see above).

**`lin_vel_x_norm`** = `clip(‖[vx, vy]‖, 0.3, 2.0) + 0.2` — normalization denominator, min 0.5.

---

## Domain randomization (configs/base.yaml)

- Friction: `[0.2, 3.0]`
- Mass: ×`[0.5, 1.5]`
- PD gains: ×`[0.8, 1.2]`
- Torque: ×`[0.8, 1.2]`
- Obs delay: joints 10–40 steps, IMU 20–50 steps
- Pushes: every 3s, up to 0.5 m/s

---

## SDK interface (`~/code/RoboTamerSdk4Qmini`)

**IMPORTANT**: The C++ SDK currently implements only the **BIRL interface**.
File: `source/user/rl_controller.cpp::get_observation()` (line ~85)

### Current SDK obs layout (BIRL, 44-dim)

```cpp
obs << target_command,              // [0-2]  vx, vy, yaw  (3-dim)
       base_rpy.segment(0, 2),      // [3-4]  roll, pitch
       base_rpy_rate * 0.5,         // [5-7]  angular vel × 0.5
       joint_pos - _ref_joint_act,  // [8-17] joint pos error
       joint_vel * 0.1,             // [18-27] joint vel
       joint_pos_error,             // [28-37] tracking error
       pm_phase_sin_cos * static_flag,  // [38-41] phase sin/cos
       (pm_f * 0.3 - 1) * static_flag; // [42-43] freq signal
```

### Current SDK action (BIRL, 12-dim)

```cpp
// joint_increment_control():
pm_f = increment.segment(0, NUM_LEGS);        // [0-1] leg frequencies
compute_pm_phase(pm_f);
joint_act += increment.segment(NUM_LEGS, NUM_ACTUAT_JOINTS) * dt;  // [2-11] joint deltas
```

### To deploy a MIRL policy on the real robot

The SDK must be updated to match the MIRL interface. Changes needed in `rl_controller.cpp`:

1. **`get_observation()`** — replace with 64-dim MIRL layout:
   ```cpp
   obs << target_command_8,           // [0-7]  8 cmd slots (slots 3-7 = 0)
          base_rpy.segment(0, 2),     // [8-9]  roll, pitch
          base_rpy_rate * 0.5,        // [10-12] angular vel × 0.5
          joint_pos - _ref_joint_act, // [13-22] joint pos error
          joint_vel * 0.1,            // [23-32] joint vel
          joint_pos_error,            // [33-42] tracking error
          Eigen::VectorXf::Zero(21);  // [43-63] ref slots (zeros at deploy time)
   ```

2. **`joint_increment_control()`** — remove phase modulator, all 10 outputs are joint deltas:
   ```cpp
   // Remove: pm_f = increment.segment(0, NUM_LEGS); compute_pm_phase(pm_f);
   joint_act += increment.segment(0, NUM_ACTUAT_JOINTS) * dt;  // all 10 dims
   ```

3. **Config** (`configParams`):
   - `num_observations = 64`
   - `num_stacks = 3`
   - `num_actions = 10`

---

## Known issues / things to watch

- **foot_slip_rew**: lateral foot velocity is gated by `(1 - vy_walking)` where `vy_walking = |cmd_vy| > 0.1`. When strafing, the lateral penalty is correctly suppressed.
- **action_constraint_rew**: penalizes hip_yaw/hip_roll [0,1,5,6] from ref when not strafing — this is also gated by `(1 - vy_walking)`.
- Observation delay simulation adds up to ~600ms latency — main sim-to-real gap mitigator.

---

## Best known checkpoints (as of April 2026)

| Run | Iter | Task | Notes |
|-----|------|------|-------|
| `carpet_v8` | 2000 | BIRL | Best BIRL — 100% survival fr≤1.5, good forward tracking |
| `mirl_fwd_v1` | in progress | MIRL_Fwd | Forward/backward expert, training from iter 1 |

All pre-April-2026 checkpoints use 43-dim obs and are **incompatible** with current BIRL code (44-dim).
BIRL checkpoints (44-dim) are **incompatible** with MIRL (64-dim) and vice versa.

---

## Related repo

**`~/code/RoboTamerSdk4Qmini`** — C++ SDK that runs the policy on the real robot.
Currently implements BIRL interface only. See SDK interface section above for MIRL migration.
