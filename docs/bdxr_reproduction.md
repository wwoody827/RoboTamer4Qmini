# BDX-R-MjLab Faithful Reproduction Spec

Source: <https://github.com/BDX-R/BDX-R-MjLab>
Target: Qmini biped on Isaac Gym (RoboTamer4Qmini).
Started: 2026-05-11. **Concluded: 2026-05-12.**

This document is the **authoritative spec** for our reproduction. The code
and YAML follow it line-by-line; deviations are documented in §10 with
hardware justification.

---

## ⛔ OUTCOME: BDX-R recipe does not port cleanly to Qmini

Three attempts (walk_bdxr_full → walk_bdxr_qmini → walk_bdxr_repro) all
failed to learn a deployable gait. Best result: sim2sim survival ~0.5-1 s
across 1000+ training iterations.

### Root cause

BDX-R's reward set assumes their **kd=5.0** hip/knee actuator damping
(`bdxr_constants_legs.py:KD_ROBSTRIDE_03`). This provides strong passive
joint damping → policy doesn't need to actively rotate the body to
maintain balance → their `body_ang_vel: -0.2` and `angular_momentum: -0.04`
penalties work because the robot is naturally stable.

Qmini's PD has **kd=0.3** for hip_pitch (16× lower damping). Active body
rotation (e.g. counter-rotation to recover from a sway) is the primary
balance mechanism. The same penalties that work on BDX-R **suppress the
very corrections Qmini needs**. Training discovers "minimise body motion
+ collect upright/pose_speed bell rewards" → 134-step survival in
IG-with-DR-pushes (DR resets it before fall-mode accumulates) → 0.5 s
survival in deterministic MuJoCo (small initial sway compounds).

### Evidence
| Trial | Setup | sim2sim @final | Training l_n |
|---|---|---|---|
| walk_bdxr_full | first attempt, hardware-aware action | 10.5 s @ iter 1000 | 96 |
| walk_bdxr_qmini | + BDX-R hyperparams (entropy 0.01, 3-layer ELU, init_noise 1.0) | **0.5 s stuck** | 60 |
| walk_bdxr_repro | + L1 foot_clearance + new penalties + faithful weights | **0.5 s stuck** | 90 |
| walk_bdxr_repro (entropy 0.003) | rollback entropy to 0.003 | **0.5 s stuck** | 105 |
| walk_bdxr_repro (entropy 0.003 + stage-0 cmds) | narrow cmd ranges | **0.5 s stuck** | 134 |

Across all rollback dimensions (entropy, cmd range, network depth),
sim2sim survival plateaued at the same value while training reward
climbed. This is the signature of "policy learns to maximise reward
shape, not the underlying task" failure mode.

### What this tells us
1. Pure BDX-R reward set is **hardware-coupled** — works on their robot,
   doesn't on ours. Not a generic biped recipe.
2. Removing body-rotation penalties (option B from the kill-decision)
   would restore active balance, but at that point we've deviated from
   "faithful BDX-R reproduction" and it's just another hand-tuned recipe.
3. **walk_rl_v4_iter4800 (72/72 s sim2sim)** remains the deploy
   candidate. This research finding archives what we tried and why.

### Items implemented (kept in code, reusable)
- `BDXRTask` class (`env/tasks/bdxr_task.py`)
- New obs slot `projected_gravity` (3-dim, IMU-style) in `env/obs_builder.py`
- New rewards in `env/tasks/birl_task.py`:
  `pose_speed` (speed-conditional std, per-joint support), `upright` (exp bell),
  `feet_clearance_l1` (L1 + cmd-gated), `feet_swing_height_peak` (peak-tracking
  + first_contact reset), `feet_slip_l2`, `soft_landing`, `action_rate_l2`,
  `body_ang_vel`, `angular_momentum` (proxy), `dof_pos_limits`,
  `foot_clearance_l2` (G1-style velocity-gated).
- Configurable `task.tilt_termination_angle` (default 0.7 rad; BDX-R uses 1.22).
- Configurable `policy.init_noise_std` (default 0.8; BDX-R uses 1.0).
- Sim2sim dispatch supports 39-dim BDXR obs (`is_bdxr` branch in evaluate.py
  and sim2sim.py).
- Sim2sim cmd test grid narrowed (vx ±0.3, vy ±0.1, yaw ±0.3) — universally
  in training distribution for our policies.

---

---

## 1. Observation space

### 1.1 Actor observation (with corruption, noise applied)

| Slot | Source | Shape | Scale | Noise (Uniform) |
|---|---|---|---|---|
| `base_ang_vel` | IMU sensor (`robot/imu_ang_vel`) | (3,) | identity | ±0.2 |
| `projected_gravity` | `mdp.projected_gravity` | (3,) | identity | ±0.05 |
| `joint_pos` | `mdp.joint_pos_rel` = `q − default_q` | (J,) | identity | ±0.01 |
| `joint_vel` | `mdp.joint_vel_rel` = `dq − default_dq` | (J,) | identity | ±1.5 |
| `actions` | `mdp.last_action` = previous raw network output | (J,) | identity | none |
| `command` | `mdp.generated_commands` from "twist" | (3,) | identity | none |

Concatenated. For BDX-R legs (J=10): total dim = 3+3+10+10+10+3 = **39**.

### 1.2 Critic observation (NO corruption, privileged)

All actor terms plus:

| Slot | Source | Shape | Clip |
|---|---|---|---|
| `base_lin_vel` | `robot/imu_lin_vel` | (3,) | — |
| `height_scan` | `envs_mdp.height_scan` with `terrain_scan` sensor | (N_rays,) | (-1.0, 1.0) — **legs variant removes this** |
| `foot_height` | site z-coordinate at feet sites | (N_feet,) | — |
| `foot_air_time` | contact-sensor `current_air_time` | (N_feet,) | — |
| `foot_contact` | `(found > 0).float()` | (N_feet,) | — |
| `foot_contact_forces` | `sign(F)·log1p(|F|)` flattened | (N_feet·3,) | — |

Legs variant: `height_scan` is removed; otherwise identical.

---

## 2. Action

```python
JointPositionActionCfg(
    actuator_names=(".*",),
    scale=0.5,                   # base; overridden by per-actuator scale
    use_default_offset=True,
)
```

Effective applied target: `q_target = default_joint_pos + scale[i] × net_out[i]`,
where `scale[i] = 0.25 × effort_limit[i] / stiffness[i]` (hardware-derived).

For BDX-R legs:
- RobStride 03 (Hip Yaw/Roll/Pitch, Knee): `effort=42`, `kp=78.957` → `scale ≈ 0.133`
- RobStride 02 (Ankle): `effort=10.9`, `kp=16.581` → `scale ≈ 0.164`

There is **no explicit LP filter** on actions. Smoothness comes from:
1. small per-step scale (~0.13–0.16 rad/step)
2. high motor damping (`kd=5.027` for hip/knee, `kd=1.056` for ankle)
3. `action_rate_l2` reward (weight −0.1)
4. delayed actuator (`delay_min/max_lag=3` steps)

---

## 3. Commands

```python
"twist": UniformVelocityCommandCfg(
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.1,          # 10 % of envs commanded to stand still
    rel_heading_envs=0.3,           # 30 % of envs use heading control
    heading_command=True,           # PD on yaw error replaces yaw rate cmd
    heading_control_stiffness=0.5,
    ranges=Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-0.4, 0.4),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
    ),
)
```

Legs variant overrides at training:
```python
ranges.lin_vel_x = (0.0, 0.0)
ranges.lin_vel_y = (-1, -1)
ranges.ang_vel_z = (0.0, 0.0)
```
(Effectively a single-direction strafe task at training — odd but explicit.)

At play time: `lin_vel_x = (0.4, 1.0)`, `lin_vel_y = 0`, `ang_vel_z = 0`.

---

## 4. Events (domain randomization)

| Name | Mode | Schedule | Params |
|---|---|---|---|
| `reset_base` | reset | episode start | pose: x±0.5, y±0.5, z=(0.01,0.05), yaw±π; velocity: empty |
| `reset_robot_joints` | reset | episode start | position offset 0, velocity 0 (exact home pose) |
| `push_robot` | interval | every 1.0–3.0 s | velocity: x±0.5, y±0.5, z±0.4, roll±0.52, pitch±0.52, yaw±0.78 |
| `foot_friction` | startup | once per env | `geom_friction = (0.3, 1.2)`, abs, shared across foot geoms |
| `encoder_bias` | startup | once per env | per-joint position offset ±0.015 rad |
| `base_com` | startup | once per env | `body_ipos[base_link]` += (x±0.03, y±0.05, z±0.07) |
| `body_mass` | startup | once per env | `body_mass *= U(0.8, 1.3)` |
| `pd_gains` | startup | once per env | kp ×U(0.7,1.3), kd ×U(0.7,1.3) |

---

## 5. Rewards (legs variant, after overrides)

| Term | Weight | Function | Exact formula |
|---|---|---|---|
| `track_linear_velocity` | **+2.0** | `mdp.track_linear_velocity` | `exp(−(\|\|cmd_xy − v_xy\|\|² + v_z²) / std²)`, std=0.25 |
| `track_angular_velocity` | **+2.0** | `mdp.track_angular_velocity` | `exp(−((cmd_yaw − ω_z)² + \|\|ω_xy\|\|²) / std²)`, std=0.5 |
| `upright` | **+1.5** | `mdp.flat_orientation` | `exp(−\|\|g_xy(body)\|\|² / std²)`, std=√0.1 |
| `pose` | **+1.0** | `mdp.variable_posture` (see §5.1) | `exp(−mean((q − default_q)² / σ_joint(speed)²))` |
| `body_ang_vel` | **−0.2** | `mdp.body_angular_velocity_penalty` | `\|\|ω_xy(base_link)\|\|²` |
| `angular_momentum` | **−0.04** | `mdp.angular_momentum_penalty` | `\|\|L\|\|²` (whole-body angular momentum) |
| `dof_pos_limits` | **−1.0** | `mdp.joint_pos_limits` | sum of soft-limit overshoot per joint |
| `action_rate_l2` | **−0.1** | `mdp.action_rate_l2` | `\|\|a_t − a_{t−1}\|\|²` |
| `air_time` | **+1.5** | `mdp.feet_air_time` (legs override) | `Σ I[0.1 ≤ current_air_time[i] ≤ 0.6] × (cmd_total > 0.05)` |
| `foot_clearance` | **−4.0** | `mdp.feet_clearance` (legs override) | `Σ \|foot_z[i] − 0.06\| × \|\|v_xy(foot[i])\|\| × (cmd_total > 0.05)` |
| `foot_swing_height` | **−3.0** | `mdp.feet_swing_height` (legs override) | `Σ (peak_z[i]/0.06 − 1)² × first_contact[i] × (cmd_total > 0.05)` — peak tracked since last touchdown |
| `foot_slip` | **−0.2** | `mdp.feet_slip` | `Σ \|\|v_xy(foot[i])\|\|² × in_contact[i] × (cmd_total > 0.01)` |
| `soft_landing` | **−2e-5** | `mdp.soft_landing` | `Σ \|\|F_contact[i]\|\| × first_contact[i] × (cmd_total > 0.05)` |
| `self_collisions` | **−1.0** | `mdp.self_collision_cost` | count of self-collision matches from contact sensor |

`cmd_total = \|\|cmd_xy\|\| + \|cmd_yaw\|`. Many penalties are command-gated:
inactive when commanded to stand.

### 5.1 `variable_posture` (the "pose" reward)

```python
total_speed = ||cmd_xy|| + |cmd_yaw|
mask_stand = (total_speed < walking_threshold)        # 0.05
mask_walk  = (walking_threshold ≤ total_speed < running_threshold)  # 0.1
mask_run   = (total_speed ≥ running_threshold)
σ_joint    = std_standing × mask_stand + std_walking × mask_walk + std_running × mask_run

err_sq = (q − default_q) ** 2                   # [B, J]
pose_rew = exp(−mean(err_sq / σ_joint ** 2))    # [B, 1] in (0, 1]
```

Per-joint std values for legs variant:

| Joint pattern | std_standing | std_walking | std_running |
|---|---|---|---|
| `.*_Hip_Pitch.*` | 0.05 | 0.5 | 0.8 |
| `.*_Hip_Roll.*` | 0.05 | 0.04 | 0.04 |
| `.*_Hip_Yaw.*` | 0.05 | 0.15 | 0.2 |
| `.*_Knee.*` | 0.05 | 0.5 | 0.8 |
| `.*_Ankle.*` | 0.05 | 0.1 | 0.5 |

Note: thresholds are very low (0.05 / 0.1). Any non-trivial cmd → walking mode.

### 5.2 `feet_swing_height` (the peak-tracking reward)

Class-based (holds `peak_heights[B, N_feet]` state across steps):
1. `in_air = (contact_sensor.found == 0)` per foot
2. `peak_heights = max(peak_heights, foot_z) where in_air else peak_heights`
3. `first_contact = contact_sensor.compute_first_contact(dt)`
4. `cost = Σ (peak/target − 1)² × first_contact × cmd_active`
5. After computing cost: `peak_heights[first_contact] = 0` (reset for next swing)

---

## 6. Terminations

```python
"time_out": time_out=True (episode reached max length)
"fell_over": bad_orientation, limit_angle = math.radians(70.0)  # > 70° from upright
```

The legs variant also has `self_collision_cost` reward but no termination on it
(it's a penalty, not an episode-ending event).

---

## 7. Curriculum

```python
"terrain_levels": progress robots through difficulty levels based on distance walked.
  (Disabled for flat terrain variant.)

"command_vel": expand command ranges over training steps.
  Stage 0 (step 0):       lin_x ±0.4, lin_y ±0.1, ang_z ±0.3
  Stage 1 (5000 × 24):    lin_x ±0.7, lin_y ±0.25, ang_z ±0.6
  Stage 2 (10000 × 24):   lin_x ±1.0, lin_y ±0.4,  ang_z ±1.0
```

`5000 × 24` = 5000 PPO iterations × 24 steps/env = 120 000 env-steps.
This is **iteration-based**, scaled by `num_steps_per_env` so it matches PPO update count.

---

## 8. RL hyperparameters

```python
RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    num_steps_per_env=24,
    max_iterations=30_000,
    save_interval=50,
)
```

---

## 9. Simulation & misc

```python
sim.timestep = 0.005
sim.iterations = 10        # MuJoCo solver iterations
decimation = 4             # → policy rate = 1/(0.005·4) = 50 Hz
episode_length_s = 20.0    # 400 policy steps per episode
nconmax = 35 (35 for legs variant 45)
njmax = 1500
```

Initial state: `KNEES_BENT_KEYFRAME` (which is identical to `HOME_KEYFRAME` with
all joints at 0 — straight legs, base z=0.33).

Soft joint pos limit factor: 0.9.

Actuator delay: `delay_min_lag = delay_max_lag = 3` physics steps for both
RobStride 03 and RobStride 02.

---

## 10. Hardware-driven deviations (explicit list)

These cannot be matched 1:1 because they encode Qmini's physical setup.
Everything else MUST match BDX-R exactly.

### 10.1 Policy frequency
- BDX-R: `dt=0.005, decimation=4` → **50 Hz**
- Qmini: `dt=0.001, decimation=15` → **66.67 Hz**

User-confirmed: keep Qmini's existing rate.

### 10.2 Action scale (effective per-step Δq)
- BDX-R formula: `0.25 × effort_limit / stiffness` (per actuator) ≈ 0.13–0.16 rad/step
- Qmini: motor effort and PD gains are different; we use URDF joint limits
  + LP filter (`action_lowpass_alpha=0.75`) as the analogous smoothness cap

The intent is identical (max per-step joint target change is bounded for
hardware safety), but the mechanism differs (BDX-R uses scaled offset; we
use range clip + temporal LP).

### 10.3 Default joint pose
- BDX-R legs `KNEES_BENT_KEYFRAME` = ALL zeros (straight-leg stance).
- Qmini `ref_joint_pos = [0.4, -0.1, -1.5, 1.0, -1.3, …]` (squat stance).

This is determined by the URDF / mechanical design. Cannot match.

### 10.4 Actuator damping
- BDX-R `kd = 5.027` (hip/knee), 1.056 (ankle) — high damping.
- Qmini `kd = 0.3–2.5` (per joint, much lower).

Hardware property of the motors. We compensate by stronger `action_rate_l2`
weight when reproducing.

### 10.5 Body count
- BDX-R has 18 joints (full body with neck/head/arms) — but the **legs**
  variant operates only on 10 leg joints.
- Qmini has 10 joints (legs only).

We match the legs variant, so this is aligned.

### 10.6 Sensors / terrain features that depend on infrastructure
- `height_scan` (terrain raycaster): legs variant removes it, so we skip.
- `feet_ground_contact` (ContactSensor with track_air_time=True): mjlab feature.
  We use `self._td_event` and existing foot_frc tracking as equivalent.
- `self_collision` ContactSensor: we can approximate via existing
  `termination_contact_indices` machinery (knee/base/hip contact).

### 10.7 Curriculum
- BDX-R command_vel curriculum is in physical env-step counts.
- We must convert to our PPO iteration count (BDX-R `5000 × 24` = 120 k
  env-steps; we have `num_steps_per_env=24` so same iter count fits).
  → Stage 0: iter 0, Stage 1: iter 5000, Stage 2: iter 10000.

---

## 11. Implementation checklist

Status after first pass implementation (2026-05-11):

**LANDED** ✅ / **DEFERRED** ⏳ / **NOT APPLICABLE** ➖ (hardware-driven, §10)

### 11.1 Obs (actor)
- ➖ `base_ang_vel` raw — we use × 0.5 conventional scaling (small deviation, deferred)
- ✅ `projected_gravity` (3-dim) — new slot added
- ✅ `joint_pos_rel` (q − ref_joint_action, equivalent to BDX-R's q − default_q for Qmini)
- ➖ `joint_vel_rel` — we use × 0.1 conventional scaling
- ➖ `last_action` — we use `joint_tracking_err` (similar info via different parametrisation)
- ✅ `commands_3`
- ✅ Total dim = 39 (3+3+3+10+10+10), history 5 × skip 2 = 195 total
- ⏳ Per-term Unoise on actor obs (we use obs delay DR instead, similar effect)

### 11.2 Obs (critic, privileged)
- ✅ Inherited from BIRLTask's `critic_observation()` which already includes base_lin_vel, foot_frc, foot_height, foot_vel — covers most BDX-R privileged slots
- ⏳ `foot_contact` binary (we have foot_frc which is richer)
- ⏳ `foot_air_time` slot in obs (we have it internally as state)

### 11.3 Action
- ➖ joint_pos mode with BDX-R `scale × default offset` not exactly matched.
  We use absolute mode + LP filter 0.75 + URDF limits (hardware-equivalent,
  see §10.2). Same per-step Δq bound.

### 11.4 Commands
- ✅ ranges lin_x ±1.0, lin_y ±0.4, ang_z ±1.0
- ⏳ rel_standing_envs 0.1 (10 % envs standing)
- ⏳ rel_heading_envs 0.3
- ⏳ heading_command (would replace yaw_rate cmd for subset of envs)
- ⏳ resampling 3-8 s per-env range (we use fixed 5 s)
- ⏳ Curriculum (3-stage cmd_vel)

### 11.5 Events / DR
- ✅ `reset_base`: handled by env reset
- ✅ `reset_robot_joints`: handled by env reset
- ✅ `push_robot`: interval 2 s, vel ±0.5, push_rate 0.78
- ✅ `foot_friction`: 0.3-1.2
- ⏳ `encoder_bias`: ±0.015 rad — NOT IMPLEMENTED
- ⏳ `base_com`: offsets — NOT IMPLEMENTED
- ✅ `body_mass`: scale 0.8-1.3
- ✅ `pd_gains`: 0.7-1.3

### 11.6 Rewards
- ✅ `fwd_vel: 1.0` + `lateral_vel: 1.0` (= BDX-R track_lin_vel w=2.0)
  — shape is linear `1 - clip(α|err|)` (ours) vs `exp(-err²/std²)` (BDX-R). Deviation.
- ✅ `yaw_rat: 2.0`
- ✅ `upright: 1.5`, `upright_std: 0.316` (= √0.1)
- ✅ `pose_speed: 1.0` with per-joint std (BDX-R legs values), thresholds 0.05/0.10
- ✅ `body_ang_vel: 0.2` (× −1 in code)
- ✅ `angular_momentum: 0.04` (proxy: ||ω_base||²; BDX-R uses MuJoCo `root_angmom` sensor)
- ✅ `dof_pos_limits: 1.0`
- ✅ `action_rate_l2: 0.1`
- ➖ `air_time: 1.5` — uses our held-value implementation, not BDX-R's
  per-step in-range count. Same intent, different shape. Deferred TODO.
- ✅ `feet_clearance_l1: 4.0` — NEW L1 form: `Σ |foot_z − target| × ||v_xy|| × cmd_active`
- ✅ `feet_swing_height_peak: 3.0` — peak-tracking with `_td_event` reset
- ✅ `feet_slip_l2: 0.2` — `Σ ||v_xy||² × in_contact × cmd_active`
- ✅ `soft_landing: 2e-5`
- ⏳ `self_collisions: 1.0` — covered by existing `terminate_after_contacts_on: [base, knee, hip]` (episode terminates instead of reward penalty)
- ✅ All BDX-R-absent rewards (foot_phase, foot_stand, base_heit, balance, twist, cmd_track_lp_alpha, etc.) set to 0

### 11.7 Terminations
- ✅ `time_out` (existing)
- ✅ `fell_over`: tilt > 70° via `task.tilt_termination_angle: 1.22` rad

### 11.8 RL hyperparameters
- ✅ hidden_layers (512, 256, 128) — 3-layer
- ✅ activation `elu`
- ✅ init_noise_std 1.0 — Actor module + simple_policy.py now accept it via config
- ✅ obs normalization — handled by existing PPO (return_rms + running normalizer)
- ✅ entropy_coef 0.01
- ✅ num_learning_epochs 5
- ✅ num_mini_batches 4
- ✅ gamma 0.99
- ✅ lam 0.95
- ✅ desired_kl 0.01
- ✅ clip_param (eps_clip) 0.2
- ✅ max_grad_norm 1.0
- ✅ num_steps_per_env 24 (existing)

---

## 12. What needs new code in our repo

Items in §11 marked "NEEDS NEW IMPLEMENTATION":

1. **`encoder_bias` DR event**: per-env, sample once at episode start, add
   ±0.015 rad bias to all joints' observation readings.
2. **`base_com` DR event**: per-env, perturb base_link inertia position
   (`body_ipos[base_link]`) by sampled offsets.
3. **`angular_momentum_penalty` reward**: compute whole-body angular momentum
   magnitude squared. (Approximation: use base angular velocity if true
   angmom isn't computed elsewhere — note this as deviation.)
4. **L1 `foot_clearance` reward (not L2)**: replace our `foot_clearance_l2`
   with proper `sum(|foot_z - target| × ||v_xy||) × cmd_active`.
5. **`init_noise_std` config param**: currently `self.std = ones * 0.8`
   hardcoded; needs to read from config (BDX-R uses 1.0).
6. **3-stage command velocity curriculum**: at iter 5000 / 10000, expand
   command ranges.
7. **70° tilt termination**: terminate if projected_gravity_z > cos(70°).
8. **`heading_command` mode**: yaw cmd computed from heading PD (only 30 %
   of envs); existing yaw_rate cmd path stays for the other 70 %.

Items already implemented (just need correct config):
- ELU activation ✓ (rl/module/common.py supports it)
- 3-layer MLP ✓ (just `hidden_layers: [512, 256, 128]`)
- `pose_speed` with per-joint std ✓
- `upright` bell ✓
- `feet_swing_height_peak` ✓ (called slightly differently, check formula)
- `soft_landing` ✓
- `action_rate_l2` ✓
- `body_ang_vel` ✓
- `dof_pos_limits` ✓
