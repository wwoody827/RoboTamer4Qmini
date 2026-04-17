# RoboTamer4Qmini — Design & Usage

## Overview

RL training framework for the **Unitree Qmini biped robot** (10 actuated joints, 5 per leg). Trains locomotion policies in Isaac Gym (GPU-parallel, 4096 envs), validates in MuJoCo sim2sim, and deploys to real hardware via C++ SDK.

Two task types:

| | BIRL | MIRL |
|---|---|---|
| Purpose | Phase-modulated walking | Reference-clip imitation |
| Action dim | 12 (2 leg freq + 10 joints) | 10 (joints only) |
| Obs dim/step | 44 | 64 |
| Total obs (3-frame history) | 132 | 192 |
| Phase modulator | Yes | No |
| Command slots | 3 (vx, vy, yaw) | 8 (vx, vy, yaw, height, 4 reserved) |

---

## Architecture

```
train.py
  |
  |-- config/loader.py      YAML config with _base inheritance
  |-- configs/*.yaml         All experiment configs
  |
  |-- env/
  |   |-- legged_robot.py    Isaac Gym environment, physics, resets
  |   |-- obs_builder.py     Config-driven observation construction
  |   |-- tasks/
  |   |   |-- base_task.py   Command sampling, episode management
  |   |   |-- birl_task.py   BIRL: obs, reward, action logic
  |   |   |-- mirl_task.py   MIRL: extends BIRLTask with ref clips
  |   |-- utils/
  |       |-- phase_modulator.py   Leg phase oscillator (BIRL)
  |       |-- math.py              Action scaling, transforms
  |       |-- delay_torch_deque.py Observation delay simulation
  |
  |-- rl/
  |   |-- alg/ppo.py         PPO with mirror augmentation
  |   |-- storage/            Rollout buffer with GAE
  |   |-- module/             Actor-critic network modules
  |
  |-- model/
  |   |-- simple_policy.py   MLP policy (512, 256)
  |
  |-- deploy/
      |-- manifest.py        Self-describing export metadata
      |-- sim2sim/
          |-- sim2sim.py      MuJoCo interactive validation
          |-- evaluate.py     Batch eval with metrics + TensorBoard
```

---

## Config System

All hyperparameters live in YAML files under `configs/`. Inheritance via `_base:` key with recursive deep merge.

```
base.yaml          All defaults (reward weights, domain rand, PD gains, obs slots, ...)
  birl.yaml        BIRL: 12-dim action, phase modulator slots
    birl_fwd.yaml  Forward expert: vx only, heading reward enabled
    birl_teacher.yaml  +base_lin_vel privileged obs
  mirl.yaml        MIRL: 10-dim action, ref clip slots, disable phase rewards
    mirl_fwd.yaml    Forward expert
    mirl_strafe.yaml Lateral expert
    mirl_turn.yaml   Turning expert
    mirl_combined.yaml All commands + imitation
```

### Loading

```python
from config.loader import load_config
cfg = load_config('configs/birl_fwd.yaml')
cfg = load_config('configs/birl_fwd.yaml', overrides={'reward.fwd_vel': 3.0})
```

Access with dot notation (`cfg.reward.fwd_vel`) or brackets (`cfg['reward']['fwd_vel']`). Missing keys return `None`.

### Adding a new config

1. Create `configs/my_experiment.yaml`
2. Set `_base: birl.yaml` (or `mirl.yaml`)
3. Override only what differs

```yaml
_base: birl.yaml
command:
  lin_vel_x_range: [0.0, 1.0]
reward:
  fwd_vel: 3.0
  act_smo: 2.0
```

---

## Observation Builder

Observations are assembled from named **slots** listed in config. Each slot is a registered function returning `[num_envs, dim]`.

### Config

```yaml
observation:
  history: 3      # frames stacked
  slots:
    - commands_3
    - base_euler
    - base_ang_vel
    - joint_pos_err
    - joint_vel
    - joint_tracking_err
    - phase_sin_cos
    - phase_freq
```

### Available slots

| Slot | Dim | Description | Used by |
|------|-----|-------------|---------|
| `commands_3` | 3 | vx, vy, yaw | BIRL |
| `commands_8` | 8 | 8 command slots (4 active + 4 reserved) | MIRL |
| `base_euler` | 2 | roll, pitch | Both |
| `base_ang_vel` | 3 | angular velocity x 0.5 | Both |
| `joint_pos_err` | 10 | joint_pos - ref_joint_pos | Both |
| `joint_vel` | 10 | joint velocities x 0.1 | Both |
| `joint_tracking_err` | 10 | joint_act - joint_pos | Both |
| `phase_sin_cos` | 4 | sin/cos of leg phases x static_flag | BIRL |
| `phase_freq` | 2 | (freq x 0.3 - 1) x static_flag | BIRL |
| `base_lin_vel` | 3 | base linear velocity (privileged) | Teacher |
| `ref_joint_pos_err` | 10 | ref_clip - joint_pos (zeros if no clip) | MIRL |
| `ref_joint_vel` | 10 | ref_clip velocity (zeros if no clip) | MIRL |
| `ref_phase_progress` | 1 | position in clip 0->1 (zero if no clip) | MIRL |

### Adding a new slot

1. Add a function in `env/obs_builder.py`:

```python
@obs_slot('my_sensor', dim=4)
def _my_sensor(task):
    return task.some_tensor[:, :4]
```

2. Add `my_sensor` to the config YAML `observation.slots` list.

No other code changes needed.

---

## Reward System

Reward weights are defined in config YAML, not in task code. Each reward term in `birl_task.py` / `mirl_task.py` reads its weight from `self.rew_weights`.

### Config

```yaml
reward:
  fwd_vel: 2.3     # velocity tracking
  balance: 1.5     # roll/pitch penalty
  act_smo: 1.5     # action smoothness
  heading: 0.0     # disabled (weight = 0)
  power: 0.0       # MIRL-specific, disabled for BIRL
```

### Tuning rewards

Override in a child config or via CLI:

```bash
python train.py --config configs/birl_fwd.yaml --name test \
    --set reward.fwd_vel=3.0 --set reward.act_smo=2.0
```

Set any weight to `0.0` to disable that term entirely.

### All reward terms

**Shared (BIRL + MIRL):**
`constant`, `base_heit`, `balance`, `fwd_vel`, `yaw_rat`, `lateral_vel`, `vertical_vel`, `ang_vel`, `twist`, `base_acc`, `foot_clr`, `foot_supt`, `foot_heit`, `leg_width_rew`, `act_const`, `sa_const`, `foot_phase`, `jnt_pos_err`, `act_smo`, `net_smo`, `net_out_val`, `foot_slip`, `foot_vz`, `foot_acc`, `foot_sft`, `jnt_vel`, `feet_py`, `feet_frc`, `joint_tor`, `pmf`, `heading`, `yaw_smooth`

**MIRL-specific:**
`power`, `air_time`

---

## Training

### Quick start

```bash
conda activate qmini
cd ~/code/RoboTamer4Qmini

# BIRL forward expert
python train.py --config configs/birl_fwd.yaml --name birl_fwd_v1 \
    --sim2sim_interval 500 --num_envs 4096

# MIRL forward expert (no reference clips)
python train.py --config configs/mirl_fwd.yaml --name mirl_fwd_v1 \
    --sim2sim_interval 500 --num_envs 4096

# Resume training
python train.py --config configs/birl_fwd.yaml --name birl_fwd_v1 \
    --resume birl_fwd_v1 --max_iterations 10000
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/birl.yaml` | YAML config path |
| `--name` | `test` | Experiment name -> `experiments/<name>/` |
| `--resume` | None | Resume from `experiments/<name>/model/policy.pt` |
| `--num_envs` | 4096 | Number of parallel environments |
| `--max_iterations` | 5000 | Training iterations |
| `--sim2sim_interval` | 0 | Auto sim2sim eval every N iters (0 = disabled) |
| `--render` | False | Show Isaac Gym viewer |
| `--set` | None | Override config values: `--set reward.fwd_vel=3.0` |

### Output structure

```
experiments/<name>/
  model/
    policy.pt             Latest checkpoint
    policy_<iter>.pt      Periodic checkpoints
    cfg.yaml              Saved training config
  deploy/
    policy_<iter>.onnx    ONNX export (auto during sim2sim eval)
    policy_<iter>_manifest.yaml   Self-describing manifest
  runs/                   TensorBoard logs
```

### Monitoring

```bash
tensorboard --logdir experiments/
# Open http://localhost:6006
```

Key metrics: `1:Train/mean_reward`, `1:Train/mean_episode_time`, `4:Rewards/*`, `sim2sim/*`

---

## Export & Deployment

### Export to ONNX

```bash
# Export latest checkpoint
python export_pt2onnx.py --name my_run

# Export specific iteration
python export_pt2onnx.py --name my_run --iter 2000
```

Produces `policy.onnx` + `policy_manifest.yaml` in `experiments/<name>/deploy/`.

### Manifest

Every export generates a `manifest.yaml` alongside the ONNX file. It contains everything needed for inference — no separate config files required.

```yaml
format_version: 2
task_type: BIRL
obs_per_step: 44
obs_history: 3
obs_slots: [commands_3, base_euler, base_ang_vel, ...]
obs_total: 132
action_dim: 12
action_mode: increment
action_scaling:
  low: [0.5, 0.5, -15, ...]
  high: [3.5, 3.5, 15, ...]
ref_joint_pos: [0.4, -0.1, -1.5, ...]
pd_gains:
  kps: [55, 105, 75, 45, 30, 55, 105, 75, 45, 30]
  kds: [0.3, 2.5, 0.3, 0.5, 0.25, 0.3, 2.5, 0.3, 0.5, 0.25]
  decimation: 15
joint_limits: { low: [...], high: [...] }
phase_modulator: { enabled: true, num_legs: 2, ... }
```

Sim2sim, evaluate, and the SDK all read the manifest to auto-configure — no manual sync needed.

---

## Sim2Sim Validation

MuJoCo-based validation before real deployment. Auto-discovers manifest from `--policy` path.

### Interactive

```bash
python deploy/sim2sim/sim2sim.py \
    --policy experiments/my_run/deploy/policy_2000.onnx \
    --cmd_vx 0.5 --cmd_vy 0.0 --cmd_yaw 0.0

# Headless + custom friction
... --headless --floor_friction 3.0

# Record reference clip for MIRL
... --record data/reference_clips/walk_fwd.npz \
    --record_skill walk --record_loop --headless --duration 10
```

### Batch evaluation

```bash
python deploy/sim2sim/evaluate.py \
    --policy experiments/my_run/deploy/policy_2000.onnx \
    --runs 10 --duration 10
```

Auto-detects BIRL vs MIRL from manifest. Reports survival rate, velocity tracking, stability metrics.

---

## MIRL Training Flow

MIRL trains locomotion experts from reference motion clips.

### Step 1 — Train experts (no clips)

```bash
python train.py --config configs/mirl_fwd.yaml --name mirl_fwd_v1
python train.py --config configs/mirl_strafe.yaml --name mirl_strafe_v1
python train.py --config configs/mirl_turn.yaml --name mirl_turn_v1
```

### Step 2 — Record reference clips

```bash
python deploy/sim2sim/sim2sim.py \
    --policy experiments/mirl_fwd_v1/deploy/policy_5000.onnx \
    --record data/reference_clips/walk_fwd.npz \
    --record_skill walk --record_loop --headless --duration 10
# Repeat for strafe, turn
```

### Step 3 — Train combined with imitation

Edit `configs/mirl_combined.yaml` to set `task.ref_clip_paths`, then:

```bash
python train.py --config configs/mirl_combined.yaml --name mirl_combined_v1
```

Imitation reward: `total = w_task * task_reward + w_imit * imitation_reward`

---

## Testing

131 tests in `tests/`, all run without GPU or Isaac Gym.

```bash
/home/woody/miniconda3/envs/qmini/bin/python -m pytest tests/ -v
```

| Test file | What it covers |
|-----------|---------------|
| `test_config.py` | YAML loading, inheritance, merge, CLI overrides, validation |
| `test_obs_builder.py` | Slot registry, dim calc, BIRL/MIRL layouts, config consistency |
| `test_rewards.py` | Reward weights from config, overrides, disabled terms |
| `test_export.py` | Manifest generation, required fields, sim2sim config conversion |
| `test_mirror.py` | Mirror augmentation round-trip, L/R joint permutation |
| `test_ppo.py` | PPO update step, loss finite, gradient norms |
| `test_storage.py` | Rollout buffer add/clear, GAE computation, mini-batch shapes |
| `test_action_scaling.py` | scale_transform round-trip, increment mode |

---

## SDK Deployment

The C++ SDK at `~/code/RoboTamerSdk4Qmini` runs the ONNX policy on the real robot.

**Currently implements BIRL only.** Key files:
- `source/user/rl_controller.cpp` — `get_observation()` (44-dim), `joint_increment_control()` (12-dim action)
- `include/user/rl_controller.h` — config params (`num_observations`, `num_actions`, `num_stacks`)

To deploy MIRL, the SDK needs updates to match the 64-dim obs / 10-dim action layout. See the MIRL interface spec in CLAUDE.md.

---

## Domain Randomization

Applied during Isaac Gym training to improve sim-to-real transfer:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Friction | [0.2, 3.0] | Floor surface variation |
| Mass | x[0.5, 1.5] | Payload uncertainty |
| PD gains | x[0.8, 1.2] | Actuator model error |
| Torque | x[0.8, 1.2] | Motor variance |
| Obs delay (joints) | 10-40 steps | Sensor latency |
| Obs delay (IMU) | 20-50 steps | IMU latency |
| Pushes | up to 0.5 m/s every 3s | External disturbances |

None applied in sim2sim — intentionally different physics tests robustness.

---

## Future TODOs

### High priority

- **SDK MIRL support** — Update `rl_controller.cpp` for 64-dim obs / 10-dim action layout so MIRL policies can deploy on the real robot. Requires: 8-slot commands, no phase modulator, all 10 action outputs as joint deltas, ref clip slots zeroed at deploy time.

- **MIRL combined training** — Train the combined multi-skill policy with reference clips from the 3 experts (forward, strafe, turn). Validate that imitation reward improves gait quality over pure task RL.

- **Stair curriculum** — Add stair terrain to the terrain generator and train with terrain curriculum. Expected to work with BIRL without MIRL references.

### Medium priority

- **train.py decomposition (Phase 5)** — Break monolithic `train()` into `TrainingRunner` + callbacks (sim2sim eval, mirror augmentation, torque logging, checkpointing). Do when train.py needs more experiment hooks.

- **Algorithm interface (Phase 6)** — Abstract `BaseAlgorithm` so PPO isn't hardwired. Enables plugging in SAC/AMP/DARC. Do when a second algorithm is actually needed.

- **Get-up policy** — Generate keyframe reference clip (lying flat -> standing) and train MIRLTask with one-shot RSI curriculum. See MIRL_PLAN.md for design.

- **Squat policy** — Keyframe reference clip for variable-height squatting. Activate `cmd_height` slot in commands.

- **Height command** — Activate command slot 3 (`cmd_height`) with range [0.33, 0.50]. Add height-tracking reward term. No obs dim change needed (slot already reserved as zero).

### Low priority

- **MoE distillation** — Gate network to blend walk/getup/squat experts into a single deployable policy. See MIRL_PLAN.md Phase 5.

- **Recurrent policy** — LSTM/GRU option for history instead of frame stacking. Would require storage changes for hidden states.

- **AMP / adversarial motion prior** — If naturalistic gaits become a goal. Currently MIRL imitation reward is sufficient.

- **CI pipeline** — Add GitHub Actions to run `pytest tests/` on push. All 131 tests are CPU-only, no GPU needed.

- **Config schema validation** — Add pydantic or dataclass validation at load time. Currently typos in config keys are silently ignored (they become unused CfgNode attributes).

- **Terrain curriculum** — Progressive difficulty schedule for rough terrain training (flat -> slope -> stairs -> rubble).
