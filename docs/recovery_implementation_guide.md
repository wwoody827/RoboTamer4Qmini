# Qmini Recovery Policy — Implementation Guide

> **Target audience**: Engineer / coding agent implementing a fall recovery policy for the Unitree Qmini bipedal robot, based on the existing `RoboTamer4Qmini` (Isaac Gym + PPO) codebase.
>
> **Goal**: Train an RL policy that takes the robot from arbitrary fallen states back to a stable standing pose, deployable on real hardware via ONNX.
>
> **Estimated effort**: 3–4 weeks total (feasibility study + training + deployment).

---

## 0. Context & Hardware Constraints

### 0.1 Robot Profile

- **Platform**: Unitree Qmini (small bipedal)
- **DoF**: 11 motors total
  - Each leg: 3 DoF (yaw + 2 unspecified pitch joints, **no ankle roll**)
  - Hip: 2 motors
  - Upper body: 3 yaw motors (head/torso chain)
- **Sensors**: GY-91 (MPU9250 + BMP280 IMU), no contact sensors, no vision
- **Compute (deploy)**: Jetson Orin Nano 8GB
- **Structure**: 3D printed (PLA / PLA-CF) — **fragile**, must avoid violent motions
- **Foot**: TPU, can be slippery

### 0.2 Existing Codebase

- Repo: `RoboTamer4Qmini` (forked from `vsislab/RoboTamer4Qmini`)
- Stack: Isaac Gym + PPO + ONNX export + MuJoCo sim2sim
- Walking policy: works, ~1 minute stable on flat ground (uses phase, `birl_fwd` config)
- Existing files of interest:
  - `env/` — environment definitions
  - `configs/` — YAML configs with `_base:` inheritance
  - `model/` — actor/critic architectures
  - `rl/` — PPO algo + runner
  - `deploy/` — ONNX export + sim2sim
  - `utils/` — helpers (mirror aug, etc.)

### 0.3 Hardware Capability Boundaries (Critical)

Qmini has **no arms** and **no ankle roll**. Some fallen poses may be physically impossible to recover from. The implementation **must** start with a feasibility study (Section 1) before training.

Expected feasibility:
- **Prone (face down)**: likely feasible — push-up-like motion
- **Supine (face up)**: likely infeasible — no arms to push off
- **Side-lying**: marginal — may require body yaw to reorient first
- **Partial fall (one foot still grounded)**: easiest — most common in practice

**Do not attempt to train recovery from infeasible poses.** Accept hardware limits.

---

## 1. Phase 1: Feasibility Study (3–5 days)

**Goal**: Determine which fallen poses Qmini can physically recover from, before committing to RL training.

### 1.1 Build a feasibility test script

Create `tests/recovery_feasibility.py`:

- Load Qmini URDF in Isaac Gym (single env, viewer enabled)
- Initialize the robot in 6 candidate poses:
  1. Prone (chest down, head forward)
  2. Supine (back down)
  3. Left-side lying
  4. Right-side lying
  5. Forward kneeling (knees down, torso forward)
  6. Backward sitting (legs forward, torso back)
- For each pose, allow keyboard control of joints (or scripted motion) to manually attempt recovery
- Record: which poses are recoverable, what motion strategy works

### 1.2 Output

A markdown file `docs/recovery_feasibility.md` documenting:
- Recoverable poses (high / medium / low confidence)
- Observed strategies (e.g. "prone: bend knees → push hips up → straighten legs")
- Infeasible poses (excluded from training)

### 1.3 Decision criteria

If only 1–2 poses are clearly recoverable, **scope down** the training target. Better to have a robust policy for 2 poses than a flaky one trying for 6.

---

## 2. Phase 2: Training Task Setup

### 2.1 New Task File

Create `env/recovery_task.py` based on the existing walking task structure. Inherits the unified `BIRLTask` pattern from the existing codebase.

Key differences from walking task:
- **No phase signal** in observations
- **No velocity command** input
- **Episodic** (5 second episodes, not perpetual)
- **Different initial state distribution**
- **Different termination conditions**

### 2.2 New Config

Create `configs/recovery/recovery.yaml` extending a base config. Key fields:

```yaml
_base: configs/_base/qmini_base.yaml  # adjust path to actual base

env:
  task_class: RecoveryTask
  episode_length_s: 5.0
  num_observations: ~40    # see Section 3.1
  num_actions: 11          # all 11 motors
  num_envs: 4096

control:
  control_freq: 50         # 50 Hz, same as walking
  decimation: 4
  pd_kp: [...]             # match walking config; tune later if needed
  pd_kd: [...]

initial_state:
  mode: simulated_fall     # see Section 2.3
  num_fall_states: 2000
  fall_pose_filter: [prone, side_left, side_right]  # from feasibility study

reward_weights:
  # see Section 4 for details
  height: 1.0
  orientation: 1.5
  pose_when_upright: 3.0
  still_when_upright: 2.0
  joint_acc: 0.0001
  action_rate: 0.01
  torque: 0.0001
  qdot_excess: 0.01
  joint_limit: 0.5
  success_bonus: 20.0
  timeout_penalty: 2.0

termination:
  success_height_ratio: 0.9   # base_z > 0.9 * h_nominal
  success_tilt_deg: 15
  success_hold_s: 1.0
  timeout_s: 5.0

curriculum:
  enabled: true
  stages: 4   # see Section 5

ppo:
  # inherit from walking, override these:
  gamma: 0.95           # shorter horizon than walking
  num_steps_per_env: 24
  learning_rate: 3e-4
```

### 2.3 Initial State Generation

**Do not** randomly sample joint angles — most random poses are non-physical or unrecoverable.

Instead, **generate initial states by simulating real falls**:

```
Algorithm: generate_fall_dataset
Input: num_states (e.g. 2000)
Output: List of robot states (base pose + joint angles + velocities)

For i in 1..num_states:
    1. Reset robot to nominal standing pose
    2. Apply random force/torque impulse:
       - direction: uniform random in horizontal plane
       - magnitude: uniform [10, 30] N or appropriate torque
    3. Step simulation for random duration in [0.5, 1.5] seconds
       (let physics evolve naturally; do not control)
    4. Capture full robot state at end
    5. Filter: classify fall pose
       - if prone / side / kneeling: add to dataset
       - if supine: discard (per feasibility study)
    6. Save to dataset

Save dataset to data/recovery_init_states.npz
```

This dataset is generated **once**, then loaded at training time for sampling initial conditions.

### 2.4 Episode Reset

At each episode reset:
1. Sample one state from the fall dataset (with curriculum-controlled distribution)
2. Set robot's base pose, joint angles, joint velocities to that state
3. Add small randomization (±5° base orientation, ±0.1 rad joint angles) to prevent overfitting
4. Reset PPO buffers, reward accumulators

---

## 3. Phase 3: Observation & Action Design

### 3.1 Observation Space (~40 dims, no history)

Recovery is a strongly Markovian task. **Do not use long observation history.** A single timestep is sufficient.

```
obs = [
    projected_gravity,        # 3D, gravity vector in body frame (key state info)
    base_ang_vel,             # 3D, IMU angular velocity
    joint_pos - q_nominal,    # 11D, joint deviation from nominal
    joint_vel,                # 11D
    last_action,              # 11D, for action smoothness
    episode_progress,         # 1D, t / t_max in [0, 1]
]
Total: ~40D
```

**Rationale**:
- `projected_gravity` encodes orientation directly — most important signal
- `episode_progress` lets policy "know it's running out of time" (useful for episodic tasks)
- No phase, no command, no scandots, no history

**Optional (only if first version fails)**:
- 5-step history of `(joint_pos, joint_vel, action)` as implicit noise filter (not for state inference)

### 3.2 Action Space

Same as walking: 11D joint position targets, fed through the same PD controller. Output range and scaling should match walking config to keep deployment consistent.

### 3.3 Network Architecture

```
Actor: MLP [obs_dim, 256, 256, action_dim]
       Activation: ELU
       LayerNorm after first hidden layer
       Output: tanh-bounded, scaled to action range

Critic: MLP [obs_dim, 256, 256, 1]
        Same activation/norm setup
```

**No RNN, no attention, no special architecture.** This is intentional — recovery doesn't need it.

---

## 4. Phase 4: Reward Function (Most Critical Section)

Reward design is where recovery training most often fails. Implement in **layers** with **non-overlapping magnitudes**.

### 4.1 Layer 1 — Primary Goal (magnitude ~1–2)

Always active. Defines "what success looks like".

```python
# Base height reward
h_target = NOMINAL_STANDING_HEIGHT  # measure from URDF, ~0.30m for Qmini
h_current = base_pos_z
r_height = exp(-10.0 * (h_target - h_current)**2)

# Base orientation reward
# projected_gravity[2] = -1 when perfectly upright
gravity_z_body = projected_gravity[2]
r_orient = exp(-3.0 * (gravity_z_body - (-1.0))**2)

reward_layer1 = 1.0 * r_height + 1.5 * r_orient
```

**Use exponential form, not negative quadratic.** This bounds magnitudes and reduces gradient explosions.

### 4.2 Layer 2 — Stability Bonus (magnitude ~5, conditional)

Only active when robot is approximately upright. Strongly rewards holding the pose.

```python
is_upright = (h_current > 0.9 * h_target) and (gravity_z_body < -0.95)

if is_upright:
    # Joint pose close to nominal
    r_pose = exp(-1.0 * sum((q - q_nominal)**2))
    # Low velocity (standing still)
    r_still = exp(-0.1 * sum(qdot**2))
    reward_layer2 = 3.0 * r_pose + 2.0 * r_still
else:
    reward_layer2 = 0.0
```

**Rationale**: discrete jump in reward when standing creates strong learning signal. Prevents the "lean-but-don't-stand" local optimum.

### 4.3 Layer 3 — Smoothness Penalties (magnitude ~0.05–0.2, negative)

Prevent violent motions that would damage Qmini's 3D printed parts.

```python
r_acc      = -0.0001 * sum(joint_accelerations**2)
r_action   = -0.01   * sum((action_t - action_prev)**2)
r_torque   = -0.0001 * sum(torques**2)

# Joint velocity excess (only penalize beyond threshold)
qdot_excess = clip(abs(qdot) - 5.0, 0, inf)
r_qdot     = -0.01   * sum(qdot_excess**2)

reward_layer3 = r_acc + r_action + r_torque + r_qdot
```

**Calibration rule**: Layer 3 total magnitude should be **5–15% of Layer 1** under normal motion. If trained policy is too timid, reduce Layer 3 weights. If too violent, increase them.

**Qmini-specific**: Lean toward higher Layer 3 weights (top of the 5–15% range). Fragile hardware — prefer conservative motion.

### 4.4 Layer 4 — Safety Penalties (magnitude ~0.5, sparse)

```python
# Joint limit excess
q_mid = (q_min + q_max) / 2
q_range = (q_max - q_min) / 2
joint_excess = clip(abs(q - q_mid) - 0.95 * q_range, 0, inf)
r_limit = -0.5 * sum(joint_excess**2)

# Self-collision penalty (if collision detection available)
r_collision = -1.0 if any_self_collision else 0.0

reward_layer4 = r_limit + r_collision
```

**Layer 4 should rarely trigger** in a well-trained policy. If it triggers often, the policy is reaching for unsafe states — investigate before tuning weights up.

### 4.5 Terminal Reward (sparse, end-of-episode)

```python
if episode_succeeded:    # is_upright held for success_hold_s
    terminal_reward = +20.0
elif episode_timeout:
    terminal_reward = -2.0
elif episode_failed_catastrophically:  # e.g. extreme angle, joint limit violation
    terminal_reward = -5.0
else:
    terminal_reward = 0.0
```

**Important**: Don't make `timeout_penalty` too large, or policy will resort to desperate violent actions when running out of time.

### 4.6 Total Reward Per Step

```python
reward = (
    reward_layer1 +
    reward_layer2 +
    reward_layer3 +
    reward_layer4
)
# terminal_reward applied separately at episode end
```

In a converged training run, expected per-step reward range: **2.0 – 5.0** (when upright). During recovery process: **0.3 – 1.5**.

---

## 5. Phase 5: Curriculum

Recovery benefits significantly from curriculum learning. Implement in 4 stages.

### 5.1 Stage Definitions

| Stage | Initial Pose Distribution | Layer 3 Weight Multiplier | Notes |
|-------|---------------------------|---------------------------|-------|
| 1 | Near-nominal: ±20° tilt only | 1.0× | Easy, learn basic balance recovery |
| 2 | Filtered fall states: prone-dominant | 1.0× | Real fall dynamics, easy fall types |
| 3 | All recoverable fall states | 0.8× | Add side-lying, harder poses |
| 4 | Stage 3 + initial joint velocities + extra domain randomization | 0.7× | Robustness; just-fallen states with motion |

### 5.2 Stage Transition

Use **moving average success rate** over the last N episodes (e.g. N=200) as the upgrade trigger:

```
if avg_success_rate > 0.80 for 200 consecutive iterations:
    advance_stage()
```

**Critical: do not erase old stages.** When in Stage 3, sample 80% from Stage 3 distribution and 20% from Stages 1+2 to prevent catastrophic forgetting.

### 5.3 Logging

Log per stage:
- Current stage index
- Success rate (rolling window)
- Mean episode length
- Mean reward per layer

This makes it easy to debug "why is training stuck" — usually a stage transition issue.

---

## 6. Phase 6: Domain Randomization

Apply during training to support sim-to-real transfer.

### 6.1 Per-Episode Randomization

Resample once per episode reset:

- Friction coefficient: uniform [0.4, 1.5] (wider than walking — large surface contact)
- Base mass: ±25% of nominal
- Base CoM offset: ±2 cm in xy
- Motor PD gains: ±15% of nominal kp/kd
- Motor strength scale: uniform [0.85, 1.15]

### 6.2 Per-Step Randomization

- IMU noise: gaussian σ=0.05 rad/s on angular velocity
- IMU bias: ±0.02 rad/s (constant per episode, drift if needed)
- Joint position noise: gaussian σ=0.01 rad
- Joint velocity noise: gaussian σ=0.5 rad/s
- Action latency: 5–25 ms (uniform per episode)

### 6.3 Recovery-Specific

Lying poses have different contact characteristics than standing. Specifically randomize:
- Ground stiffness/damping (if simulator supports it)
- Friction asymmetry (different friction at different contact points)

---

## 7. Phase 7: Training Procedure

### 7.1 Phased Training Build-up

**Don't enable everything at once.** Add reward layers incrementally to debug.

```
Phase A (500 iter): Layer 1 only
    Goal: see policy *attempt* to stand up
    Pass criterion: episode length increasing, layer 1 reward rising

Phase B (500 iter): Add Layer 3 (smoothness)
    Goal: motions become less violent
    Pass criterion: action_rate metric drops, no reward regression

Phase C (1000 iter): Add Layer 2 (stability bonus)
    Goal: visible jump in reward when standing achieved
    Pass criterion: success rate > 30%

Phase D (1500 iter): Add Layer 4 + terminal reward + full curriculum
    Goal: robust recovery across pose distribution
    Pass criterion: success rate > 80% on Stage 3
```

Total: **~3500–5000 iterations**, ~2–4 hours on a single 4090.

### 7.2 Training Command

```bash
# Phase A - D iteratively, each phase resumes from previous
python train.py --config configs/recovery/recovery.yaml --name recovery_v1 \
    --max_iterations 500 \
    --reward_phase A

# resume:
python train.py --config configs/recovery/recovery.yaml --name recovery_v1 \
    --resume --max_iterations 1000 \
    --reward_phase B
# ... etc
```

(`--reward_phase` is a new flag to add to `train.py` that gates which reward layers are active.)

### 7.3 What to Log (Tensorboard)

Standard PPO metrics plus:

- `reward/layer1`, `reward/layer2`, `reward/layer3`, `reward/layer4`
- `reward/layer*_variance` (per-episode variance of each layer — see Section 8.2)
- `success_rate` (rolling window)
- `episode_length_mean`
- `terminal/success_count`, `terminal/timeout_count`, `terminal/failure_count`
- `action_norm_mean`, `action_rate_mean`
- `curriculum/stage`, `curriculum/stage_transition_count`

### 7.4 Periodic Sim2Sim Validation

Use the existing MuJoCo sim2sim hook every 200 iter:

```bash
--sim2sim_interval 200
```

This catches Isaac-Gym-specific overfitting early. If success rate is high in Isaac Gym but low in MuJoCo, there's a physics gap to investigate.

---

## 8. Phase 8: Debugging Failures

### 8.1 Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Reward stuck at medium value, robot wobbles in place | Layer 2 not strong enough; "lean-but-don't-stand" local optimum | Increase Layer 2 weights (3.0→5.0); check `is_upright` threshold |
| Violent motions, joints at limits | Layer 3 too weak | Increase smoothness penalties (esp. `action_rate`) |
| "Successful" but immediately falls again | Success criterion too lenient | Tighten `success_height_ratio` to 0.92, add `success_hold_s` enforcement |
| Training doesn't converge at all | Reward magnitudes wrong; one layer dominates | Log per-layer means + variance, check ratios |
| Stage 3 fails after Stage 2 succeeds | Curriculum jump too large; catastrophic forgetting | Reduce Stage 3 difficulty; mix more Stage 2 samples |
| Sim works, real robot fails | Insufficient domain randomization; control freq jitter | Widen randomization ranges; check real-robot control loop timing |

### 8.2 Variance Contribution Analysis

Don't only look at reward **means** — look at **variance**.

Within an episode, log:
```
var_layer1 = variance of layer1_reward over the episode
var_layer2 = variance of layer2_reward over the episode
...
```

Layer 1 should contribute **most** of the variance. Layer 3 (penalties) should contribute **least**. If Layer 3 variance dominates, smoothness penalties are too strong and creating noisy learning signals.

### 8.3 Video Recording

Record videos every 200 iter from a few representative initial states. Visual debugging is far faster than numerical debugging for locomotion-style tasks.

---

## 9. Phase 9: ONNX Export & Deployment

### 9.1 Export

Recovery has a different observation dimension than walking. Update `export_pt2onnx.py` if needed, or create `export_recovery_onnx.py` to handle the 40D input.

```bash
python export_pt2onnx.py --name recovery_v1 \
    --output deploy/recovery_policy.onnx
```

### 9.2 State Machine on the Robot

Implement on the deployment side (Orin Nano):

```python
class HighLevelController:
    STATE_WALKING = "walking"
    STATE_FALLEN_WAIT = "fallen_wait"   # let physics settle
    STATE_RECOVERING = "recovering"
    STATE_GIVE_UP = "give_up"

    def __init__(self):
        self.walking = load_onnx("walking.onnx")
        self.recovery = load_onnx("recovery.onnx")
        self.state = self.STATE_WALKING
        self.fall_timer = 0.0
        self.recovery_attempts = 0

    def step(self, dt, obs):
        base_z = compute_base_height(obs)
        tilt = compute_tilt_angle(obs)

        if self.state == self.STATE_WALKING:
            if base_z < 0.5 * H_NOMINAL or tilt > 60.0:
                self.fall_timer += dt
                if self.fall_timer > 0.3:  # confirmed fall
                    self.state = self.STATE_FALLEN_WAIT
                    self.fall_timer = 0.0
            else:
                self.fall_timer = 0.0
            return self.walking(self.build_walking_obs(obs))

        elif self.state == self.STATE_FALLEN_WAIT:
            self.fall_timer += dt
            if self.fall_timer > 0.5:
                self.state = self.STATE_RECOVERING
                self.fall_timer = 0.0
            return ZERO_ACTION

        elif self.state == self.STATE_RECOVERING:
            self.fall_timer += dt
            if base_z > 0.9 * H_NOMINAL and tilt < 15.0 and self.fall_timer > 1.0:
                # success
                self.state = self.STATE_WALKING
                self.recovery_attempts = 0
                return self.walking(self.build_walking_obs(obs))
            elif self.fall_timer > 8.0:
                self.recovery_attempts += 1
                if self.recovery_attempts >= 3:
                    self.state = self.STATE_GIVE_UP
                else:
                    self.state = self.STATE_FALLEN_WAIT
                self.fall_timer = 0.0
            return self.recovery(self.build_recovery_obs(obs))

        elif self.state == self.STATE_GIVE_UP:
            return ZERO_ACTION  # wait for human
```

### 9.3 Deployment Safety Checklist

Before first real-robot test:
- [ ] Robot on soft mat / foam
- [ ] Torque output limited to **70%** of training value (gradually increase)
- [ ] Emergency stop within reach
- [ ] Test from **easiest** pose first (slight forward lean)
- [ ] Have replacement 3D printed parts ready (small leg, lateral connector)
- [ ] Record IMU + joint data for sim-to-real gap analysis

### 9.4 Fall Detection Robustness

Real IMU is noisy. Use:
- `base_z` from forward kinematics (using joint angles), **not** IMU height integration
- `tilt` from IMU gravity vector projection
- Low-pass filter both signals (single-pole, cutoff ~5 Hz)
- Confirmation period of 0.3s before triggering state transition

---

## 10. Validation & Acceptance Criteria

### 10.1 Simulation Acceptance

- [ ] Stage 4 (final curriculum) success rate ≥ 85%
- [ ] Mean recovery time ≤ 3.5 seconds
- [ ] Action rate metric within smoothness budget
- [ ] No joint limit violations in 95th percentile of episodes
- [ ] MuJoCo sim2sim success rate within 10% of Isaac Gym

### 10.2 Real Robot Acceptance

- [ ] Recover from prone pose: 5/5 attempts
- [ ] Recover from side pose: 3/5 attempts
- [ ] Recover from partial fall: 5/5 attempts
- [ ] State machine integration: walking → fall → recovery → walking, end-to-end, 3 cycles
- [ ] No hardware damage in 20+ recovery attempts

### 10.3 Documentation Deliverables

- [ ] `docs/recovery_feasibility.md` — feasibility study results
- [ ] `docs/recovery_training.md` — training procedure, hyperparameters used, lessons learned
- [ ] `docs/recovery_deployment.md` — deployment notes, safety procedures
- [ ] Reward weight version history in git (each major change as separate commit)

---

## 11. Suggested File Structure

```
RoboTamer4Qmini/
├── env/
│   ├── recovery_task.py              # NEW: recovery RL environment
│   └── recovery_init_generator.py    # NEW: simulated fall dataset generator
├── configs/
│   └── recovery/recovery.yaml        # NEW: recovery training config
├── data/
│   └── recovery_init_states.npz      # GENERATED: fall state dataset
├── deploy/
│   ├── recovery_policy.onnx          # GENERATED: exported policy
│   └── high_level_controller.py      # NEW: state machine for deployment
├── tests/
│   └── recovery_feasibility.py       # NEW: feasibility study script
├── docs/
│   ├── recovery_feasibility.md       # NEW: feasibility findings
│   ├── recovery_training.md          # NEW: training notes
│   └── recovery_deployment.md        # NEW: deployment notes
└── train.py                          # MODIFIED: support --reward_phase flag
```

---

## 12. Implementation Order (Recommended Sequence)

For the implementing agent — execute in this order, do not skip:

1. **Feasibility study** (Section 1) — 3–5 days
   - Output: `docs/recovery_feasibility.md` with concrete findings
   - **Stop and review with human before proceeding** if findings are significantly different from expectations
2. **Initial state generator** (Section 2.3) — 1–2 days
   - Generate and save fall dataset
   - Visually verify samples in viewer
3. **Recovery task + config + minimal reward (Layer 1 only)** — 2–3 days
   - Goal: get Phase A training running, see basic learning
4. **Add Layers 2, 3, 4 incrementally** (Section 7.1) — 1 week
   - One layer at a time, validate each
5. **Add curriculum** (Section 5) — 3–5 days
   - Verify stage transitions work, no forgetting
6. **Full training run + sim2sim validation** — 2–3 days
   - Iterate on reward weights as needed
7. **ONNX export + state machine** (Section 9) — 2–3 days
8. **Real robot deployment** (Section 9.3) — 1 week
   - Conservative ramp-up; expect iteration

---

## 13. Things to Explicitly NOT Do

- Do **not** use a recurrent network or long observation history. Recovery is Markovian.
- Do **not** use phase signals or any periodic clock. This is an episodic task.
- Do **not** use velocity command tracking rewards. There is no target velocity.
- Do **not** smoothly blend walking and recovery policies on the deployment side. Switch discretely with a stabilization pause.
- Do **not** add Layer 5/6/etc. reward terms to patch failure modes. Fix the root cause (observation, action, termination, or initial state).
- Do **not** delete the `mirl_fwd` (no-phase walking) config. It will be useful as future reference and for ablation studies.
- Do **not** tune reward weights without recording the change in git. Version history is essential.
- Do **not** train recovery from supine pose unless feasibility study contradicts this guide.
- Do **not** test on real robot without the safety checklist (Section 9.3) completed.

---

## 14. References for Implementing Agent

Recommended reading (in priority order):

1. **Lee et al. 2019** — "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning". Original recovery work; reward design template.
2. **Hwangbo et al. 2019** — "Learning agile and dynamic motor skills for legged robots". General sim-to-real approach for ANYmal.
3. **Rudin et al. 2022** — "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning". Modern Isaac Gym training methodology.
4. **Kumar et al. 2021** — "RMA: Rapid Motor Adaptation for Legged Robots". Privileged-information distillation; relevant if extending recovery with environment adaptation later.
5. The existing `RoboTamer4Qmini` repo's `docs/rewards.md` for the project's reward conventions.

---

**End of guide.**
