# V2 Stand Task — Experiment Report

**Goal**: train a policy that holds the Qmini standing still under cmd=(0,0,0)
in both Isaac Gym (training) and MuJoCo (deployment), with minimal sim2sim gap.

**Result**: `v2_run16 iter 4000` — robot lands at slight forward lean
(pitch −6°, roll +1.3°) and locks in place. Over 10 s in MuJoCo, yaw drifts
only 5° (settles into a steady state), xy drift 7 cm total, velocity ≈ 0.

| Sim    | Pitch (RMS)   | Yaw drift / 10s | XY drift / 10s | Falls in 10s |
|--------|---------------|-----------------|----------------|--------------|
| Isaac  | 6.5°          | n/a (training)  | n/a            | 0 / 10s      |
| MuJoCo | constant −6°  | **−5° locked**  | **7 cm**       | 0 / 10s      |

Code: `env/tasks/v2_task.py`, `v2_train.py`, `configs/walk_clean_v2.yaml`.

---

## 1. Setup

- **Task**: V2Task (registered as `cfg: V2`) — fresh implementation, no reuse
  of BIRL/MIRL reward terms.
- **Trainer**: `v2_train.py` — PPO with inline stand metrics computed from the
  training rollout itself (no separate eval rollout, no post-eval env reset).
- **Config**: `configs/walk_clean_v2.yaml` (cmd always 0; 14 reward terms;
  residual action mode; minimal DR — only friction randomized).
- **PD**: standard `τ = kp·err − kd·dq`. The legacy formula with kd-as-bias +
  joint_tor_offset was removed (committed earlier — caused net non-zero
  equilibrium torque, biased Isaac toward soft-contact tolerance that MuJoCo
  didn't share).

## 2. Reward design (14 terms)

| Group           | Term         | Form                                       | Weight | σ_bell |
|-----------------|--------------|--------------------------------------------|--------|--------|
| Tracking bells  | height       | `robust_bell((z-0.45)²)`                   | 2.0    | 0.04   |
|                 | upright      | `robust_bell(\|\|g_xy\|\|²)`                 | 2.0    | 0.30   |
|                 | lin_vel      | `robust_bell(\|\|v-v_cmd\|\|²)`              | 2.0    | 0.05   |
|                 | ang_vel      | `robust_bell(yaw_err²+0.3·rp_avel²)`       | 1.0    | 0.20   |
| Foot geometry   | foot_geom    | `robust_bell((foot_xy_hd-base_xy_hd-ref)²)`| 2.0    | 0.10   |
|                 | foot_slip    | `robust_bell(slip_sq)·contact_avg`         | 1.5    | 0.05   |
|                 | foot_rot     | `robust_bell(rot_sq)·contact_avg`          | 1.5    | 0.30   |
| Drift           | drift        | `robust_bell(\|\|xy-init_xy\|\|²)`           | 2.0    | 0.20   |
| L2 locks        | yaw_lock     | `-clip(yaw_rate²/σ², 0, 1)`                | 2.0    | 1.0    |
|                 | body_lock    | `-clip(rp_avel²/σ², 0, 1)`                 | 3.0    | 1.0    |
| Small reg       | joint_vel    | `-\|\|joint_vel\|\|²/1000`                   | 3.0    | —      |
|                 | torque       | `-\|\|torques\|\|²/100000`                   | 1.0    | —      |
|                 | smooth       | `-\|\|action_diff\|\|²/10`                   | 1.0    | —      |
| Termination     | term         | `-1 on fall (not timeout)`                 | 50.0   | —      |

Max per-step positive ≈ +14, fall step ≈ −50.

### 2.1 Key design idea: `robust_bell`

```
robust_bell(err², σ, ratio=4, l2_w=0.1)
  = exp(-err²/σ²) − l2_w · clip(err²/(ratio·σ)², 0, 1)
```

Hybrid of bell and clipped L2 (RL analog of Huber loss):

- |err| ≤ σ: bell dominates, reward ≈ +1.
- |err| ≈ 4σ: bell ≈ 0, L2 clips at 1, reward = −l2_w.
- |err| → ∞: reward saturates at −l2_w; L2 still has constant inward gradient.

This solves the "bell σ-too-strict" failure mode where pure bells saturate
beyond ~3σ and give the policy zero learning signal. The L2 floor keeps a
constant pull even when policy is far from target.

`l2_w` defaults to 0.1 (set higher per-term if a bell shows plateau
behaviour). 0.5 was tried (v2_run15) and triggered suicide: 8 stacked bells
× 0.5 × avg weight 2 = up to −8 / step floor → PPO chose early termination.

### 2.2 Contact-gated foot rewards

```python
in_contact = (foot_frc > 5N)              # [N, 2]
contact_avg = (in_contact_L + in_contact_R) * 0.5
r_foot_slip = robust_bell(slip_sq · in_contact) * contact_avg
r_foot_rot  = robust_bell(rot_sq  · in_contact) * contact_avg
```

The multiplier by `contact_avg` was the fix for the v2_run6 hopping bug.
Without it, airborne foot has `in_contact=0` → slip/rot = 0 → bell maxes
at +1. Policy learned to LIFT BOTH FEET to maximise these rewards.
Multiplying by `contact_avg` makes airborne = 0 reward; planted+stationary
= max reward.

### 2.3 Base-relative `foot_geom`

```python
base_xy_hd = env.base_pos_hd[:, 0:2]
foot_state = concat([
    env.foot_pos_hd[:, 0:2] - base_xy_hd,   # L foot xy rel to base
    env.foot_pos_hd[:, 3:5] - base_xy_hd,   # R foot xy rel to base
])  # 4 dim
```

`foot_pos_hd` is world position rotated to heading frame — but **still
contains the env's world XY**. Isaac spawns envs on a grid, so naive
cross-env `foot_state.mean()` is meaningless. Subtracting `base_pos_hd`
gives foot-relative-to-base in a body-yaw-aligned frame; invariant to
where each env lives.

Initial ref `foot_state` is captured on the first reward() call from the
4096-env mean (averages out the ±0.1 rad joint reset noise).

## 3. Training trajectory (v2_run6 → v2_run16)

| Run        | Key change                              | Pitch_rms best | MuJoCo behaviour       |
|------------|-----------------------------------------|----------------|------------------------|
| v2_run6    | bounded bells + L2 yaw_lock (no foot)   | 0.15           | falls @ 2 s            |
| v2_run7    | foot_slip/rot contact-gated, body_lock  | 0.13           | unstable (10 s)        |
| v2_run9    | + pose reward (joint-level, σ=0.20)     | **0.091**      | n/a                    |
| v2_run10/11| foot_geom in **world** frame (bug)      | 0.40 (broken)  | n/a                    |
| **v2_run12**| foot_geom base-relative, σ=0.10        | 0.093 sustained| **stable 10 s**, drift 35 cm |
| v2_run13   | + target_z 0.45→0.40                    | 0.18-0.24      | worse (target off equilibrium) |
| v2_run14   | + foot_geom σ=0.05, w_yaw_lock 2→4      | 0.15-0.21      | worse (σ too tight)    |
| v2_run15   | replace all bells with robust_bell, l2_w=0.5 | 0.40 (suicide) | n/a                    |
| **v2_run16**| robust_bell l2_w=0.1                   | **0.099-0.116**| **stable, yaw locked −5°, drift 7 cm** |

### 3.1 v2_run12 (previous winner, pure bells)

- Best ckpt iter 800: pitch 0.093 (5.3°), foot_geom 1.69, ep_len 347.
- MuJoCo trace: z=0.384, pitch=±1.5°, yaw drift 30°/10s, xy drift 35 cm.
- Late training (iter 1400–2000) showed plateau / regression: pitch rebound
  to 0.19. Cause: pure exp bells lose gradient when policy drifts beyond
  ~3σ — the deeper L2 trick was missing on most bells.

### 3.2 v2_run15 failure (l2_weight 0.5)

- ep_len collapsed from 350 → 10 within 200 iters.
- Per-step reward ~ −3 (8 bells × 0.5 × avg weight 2 = −8 floor when far,
  plus existing yaw_lock/body_lock penalties).
- PPO chose to terminate early to escape the negative-reward grind. Same
  failure mode as the very first v2 attempts before bounded positives.
- Fix: drop l2_weight to 0.1. Per-step floor drops to ~ −1.6, net stays
  positive in healthy states.

### 3.3 v2_run16 success (robust_bell, l2_weight=0.1)

Pitch trajectory diverges from v2_run12 in late training:

```
iter:   200  400  600  800  1000  1200  1400  1600  1800  2000  2200  2400
v12:    .12  .09  .10  .09  .11   .11   .18   .16   .18   .19   .15   .15
v16:    .13  .10  .10  .10  .11   .14   .12   .12   .14   .13   .11   .12
                                          ^ pure bell rebound ^
```

v2_run12 saturates and rebounds; v2_run16 stays flat — robust_bell's L2
floor keeps a constant gradient that re-pulls the policy back as soon as
it drifts.

Final iter 4000 in MuJoCo (see `/tmp/v2_videos/v2_run16_iter4000_stand_mujoco.mp4`):

```
 t      x      y      z   roll  pitch    yaw   vx   vy
 1.0  -0.026  -0.025  0.398   3.3°  -4.0°  -25°   .03  .02
 2.0  -0.040  -0.002  0.401   1.4°  -6.3°   -2°   .03  .00
 3.0  -0.038  -0.003  0.400   1.5°  -6.0°   -5°   .01  .00
 ...
10.0  -0.037  -0.002  0.400   1.3°  -6.0°   -5°   .00  .00
```

After t=2 s the robot is essentially frozen in pose (pitch −6° constant
lean, yaw locked at −5°, zero velocity). No fall, no drift.

z is the IMU body world position to match training (env.base_pos in
Isaac reads IMU, not base_link). Isaac reports base_z ≈ 0.420 m; MuJoCo
0.400 m — the remaining ~2 cm is contact stiffness (PhysX softer, MuJoCo
harder, mesh penetration differs).

## 4. Bugs and lessons

### Bug 1: hop-exploit in contact-gated foot rewards (v2_run6)
**Symptom**: foot_slip/foot_rot saturate at 1.0 while robot bounces in air.
**Cause**: airborne foot has in_contact=0 → slip_sq=0 → exp(0)=1 (max).
**Fix**: multiply final bell by `contact_avg ∈ {0, 0.5, 1.0}`. Airborne =
0 reward; only planted-and-still gets max.

### Bug 2: foot_geom world-frame mean (v2_run10/v2_run11)
**Symptom**: foot_geom reward identically 0 forever; pitch worse than baseline.
**Cause**: `foot_pos_hd` is rotated by heading-inverse but still contains
the env's world XY. Isaac spawns envs on a grid → cross-env mean of
`foot_pos_hd` is meaningless. Adding `foot_yaw` (also world-frame) made
it worse — base RPY randomisation puts each env at a different world yaw.
**Fix**: subtract `base_pos_hd` to get foot-relative-to-base. Drop foot_yaw
entirely (constrained implicitly by hip_yaw via leg kinematics).

### Bug 3: target_z below physical equilibrium (v2_run13)
**Symptom**: pitch_rms ~2× higher than v2_run12.
**Cause**: the robot's natural standing height with this PD/mass/leg config
is ≈ 0.42. Setting `target_z=0.40` puts the height bell's peak BELOW
where the robot physically wants to settle → height reward is always
suboptimal → policy compromises elsewhere.
**Fix**: target_z = 0.45 (the bell σ=0.04 happily covers the physical
equilibrium at 0.42 with reward ≈ 0.4).

### Bug 4: σ too tight (v2_run10, v2_run14)
**Symptom**: reward always near 0; no learning signal.
**Cause**: pure exp bells have vanishing gradient beyond ~3σ. Policy
starts far from target and cannot find the gradient back.
**Fix**: `robust_bell` (Section 2.1) — adds a clipped L2 companion that
gives a constant inward pull regardless of err magnitude.

### Bug 5: l2_weight too high (v2_run15)
**Symptom**: ep_len 10 (suicide pattern).
**Cause**: 8 bells × 0.5 × avg weight 2 = up to −8/step floor. PPO
prefers immediate termination (−50) over enduring −8/step ≈ −800 over
100 steps.
**Fix**: l2_weight = 0.1 default. Per-step floor stays modest (~ −1.6),
positives still dominate when policy is near target.

### Pattern: design when adding a new bell
1. Start σ loose enough that initial random policy gets ≥ 0.3 reward (so
   gradient exists from step 1).
2. Use `robust_bell` not pure `exp` — costs nothing when within bell,
   prevents late-training plateau.
3. If quantity is world-frame and varies across envs, transform to a
   body-relative frame first; otherwise cross-env reference is noise.
4. Multiply by mask (`contact_avg`, `static_flag`, etc.) when the term
   should only count in specific states.

## 5. Files

- `env/tasks/v2_task.py` — task class + `robust_bell` helper.
- `v2_train.py` — trainer (inline stand metrics, no separate eval).
- `configs/walk_clean_v2.yaml` — current best config (v2_run16 baseline).
- `experiments/20260516_0832_v2_run16/model/all/policy_4000.pt` — best ckpt.
- `experiments/20260516_0832_v2_run16/deploy/policy_4000.onnx` — exported.
- Videos:
  - `/tmp/v2_videos/v2_run16_iter4000_stand_mujoco.mp4` (MuJoCo, winner)
  - `experiments/20260516_0832_v2_run16/debug/...iter4000_*.mp4` (Isaac)
