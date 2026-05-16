# Qmini Walking Physics Reference

Use this when tuning reward weights or interpreting sim2sim gait metrics.
All numbers derived from the URDF, the LIPM (linear inverted-pendulum) walking
model, and empirical biped-walking gait literature. Computed once and frozen.

---

## 1. Robot specs (from `assets/q1/urdf/q1.urdf`)

| Quantity | Value |
|---|---|
| Total mass | 11.15 kg |
| Hip width (full) | 0.108 m |
| Hip yaw vertical drop → hip roll | 0.107 m |
| Hip pitch → knee link length | ~0.115 m |
| Knee → ankle link length | ~0.155 m |
| Fully extended leg length L_max | ~0.38 m |
| Walking effective leg length L | ~0.40 m (knees slightly bent, base 0.45 m) |
| Base default height (z_target) | 0.45 m |
| Policy frequency | 67 Hz (dt = 15 ms) |

---

## 2. Nominal walking at cmd_vx = 0.3 m/s

LIPM with Froude number Fr = v²/(gL):

| Symbol | Formula | Value at v=0.3 | Notes |
|---|---|---|---|
| Fr | v²/(gL) | **0.023** | very slow walk (humans cruise at Fr ≈ 0.16) |
| Preferred speed v* | √(0.16·g·L) | 0.79 m/s | what the geometry "wants" |
| stride_length | L · (Fr/0.16)^0.6 | **0.12 m** | per step, per leg |
| step_freq (per leg) | v / stride | **2.5 Hz** | each leg cycle = 0.40 s |
| swing time | empirical 0.4·period | **0.16 s** | 0.16 s in the air |
| stance time | period − swing | **0.24 s** | 0.24 s on ground |
| duty_factor (per leg) | stance/period | **0.60** | walking range 0.55–0.65 |
| CoM-z amplitude (peak-to-peak) | empirical biped at v=0.3 | **3–6 cm** | minimum LIPM value 0.5 cm but real robots add push-off + heel-toe |
| CoM-z **true** oscillation RMS | ≈ 0.35 × peak-peak | **1.5–2.5 cm** | around walking *mean* height, NOT around target |
| CoM-z **walking mean** | below stand target | **0.32–0.40 m** | bent-knee gait — empirical from walk_v34 (h≈0.32), v5 (h≈0.38) |
| ⇒ `com_z_rms` metric (in eval CSV) | √(bias² + osc²) | **5–15 cm** | dominated by bias from target_h=0.45 |
| CoM-y oscillation (peak-to-peak) | ≈ hip half-width | **3–5 cm** | weight-shift |
| Pitch peak | empirical 0.3·Fr·rad → small angle | **3–5°** | RMS ≈ 0.6·peak ≈ 2–3° |
| Roll peak | empirical | 2–4° | RMS ≈ 1.5° |
| Foot peak swing height | stride/2 + clearance margin | **5–8 cm** | matches target 0.06–0.08 m |
| Foot horizontal velocity peak (body frame) | stride / swing | **0.75–1.0 m/s** | forward sweep |
| Hip pitch peak angular vel | ~0.5 rad / 0.16 s | **3–4 rad/s** | swing initiation |
| Knee peak angular vel | ~0.7 rad / 0.16 s | **5–8 rad/s** | swing initiation |
| Peak joint accel (any joint) | empirical | **30–50 rad/s²** | very brief, swing onset |
| Mechanical power (CoT ≈ 0.25) | m·g·v·CoT | **~8 W** | (~40 W peak instantaneous) |
| Foot contact force peak | empirical 1.2 × m·g | **~130 N** | per foot at midstance |

At other commanded speeds: stride scales ≈ v^0.83, step freq scales ≈ v^0.17,
CoM-z amplitude scales ≈ stride² ≈ v^1.66. So at v=0.5 m/s expect stride
~18 cm, step freq ~2.7 Hz, CoM-z RMS ~3 cm.

---

## 3. Phase clock — currently mis-scaled

`configs/walk_noclock_v5.yaml` inherits `phase.base_freq=1.0`, `vel_scale=1.0`.
At cmd_vx=0.3 → internal phase clock runs at **1.3 Hz**, but physics wants
**2.5 Hz** per-leg cadence. The policy crams ~2 mini-steps inside each phase
half-cycle to keep moving → shuffle.

Recommended: `phase.base_freq: 2.0`, `phase.vel_scale: 1.0`. cmd 0.3 → 2.3 Hz,
cmd 0.7 → 2.7 Hz. (Or keep base_freq=2.0 and reduce vel_scale to 0.3 for less
freq variation with command.)

---

## 4. Per-reward magnitude budget (good walk vs v6 shuffle)

At cmd_vx=0.3 m/s, the **per-step** reward contributions if the robot were
truly walking nominally vs what v6 actually produces:

| Reward term | Weight (v6) | Form | Good-walk value | v6 shuffle value | Notes |
|---|---|---|---|---|---|
| `fwd_vel` | 2.5 | exp bell on \|v−cmd\| | ~2.4 (vx tracks) | ~1.8 (bias 0.12) | OK |
| `lateral_vel` | 1.5 | exp bell | ~1.4 | ~1.4 | OK |
| `yaw_rat` | 3.0 | exp bell | ~2.7 | ~2.0 | OK |
| `base_heit` | 2.0 | exp(−70·Δh²) | ~1.7 (h≈0.45) | ~1.5 | OK |
| `balance` | 1.5 | base_heit × tilt-bell | ~1.2 | ~1.0 | OK |
| `upright` | 1.5 | exp(−\|g_xy\|²/0.1) | ~1.4 | ~1.4 | OK |
| `pose_speed` | 1.5 | per-joint bell, σ walking | ~1.0 | ~1.1 (legs barely move) | **rewards shuffle** |
| `twist` | 2.5 | −\|roll,pitch\| | −0.18 (4°) | −0.12 (3°) | **rewards shuffle** |
| `vertical_vel` | 0.6 | exp(−5·\|v_z\|) | ~0.28 (vz≈0.15) | ~0.50 (vz≈0.05) | **rewards shuffle** |
| `base_acc` | 0.01 | −\|acc−g\| · 0.1 | −0.005 | −0.003 | OK |
| `foot_phase` | 8.0 | phase-contact match | ~7.2 | ~6.0 | OK |
| `foot_stand` | 1.0 | only at static | 0 | 0 | OK |
| `foot_clr/heit/supt` | 1.0/0.7/0.7 | clip form, soft | ~1.5 total | ~0.5 total | OK |
| `air_time` | 2.0 | held_air_delta mean | ~0.05/step | ~0.02/step | OK |
| `feet_clearance_l1` | 4.0 | −\|h−0.06\| · v_xy · cmd_gate | ~−0.16 | ~−0.04 (low foot vel) | **rewards shuffle** |
| `feet_swing_height_peak` | 3.0 | −(peak/0.06−1)² at TD | ~−0.05 (peak 0.06) | ~−0.4 (peak 0.03) | reward gap exists but weak |
| `soft_landing` | 1e-5 | −F_contact at TD | ~−0.013 | ~−0.008 | OK |
| `act_smo` | 0.30 | 2nd-deriv L1 | −0.04 (sharp swing) | −0.02 (smooth shuffle) | **rewards shuffle** |
| `jnt_vel` | 0.001 | −\|q̇\|² | −0.025 (peak 5 rad/s²) | −0.005 | **rewards shuffle** |
| `joint_tor` | 1e-4 | −\|τ\|² | −0.005 | −0.002 | **rewards shuffle** |
| `power` | 0.1 | −\|τ·v\|/100 | −0.08 (8 W avg) | −0.03 (3 W) | **rewards shuffle** |

**Sum of "rewards shuffle" terms:**
- Good walk pays: 0.18 + 0.28 + 0.04 + 0.025 + 0.005 + 0.08 = **0.61** per step
- Shuffle pays: 0.12 + 0.50 + 0.02 + 0.005 + 0.002 + 0.03 = **0.68** per step
- → Shuffle is **0.07 reward/step CHEAPER** than walking in regulator/tilt
  budget. Over a 4 s episode that's ~17 reward units of "free saving".

The strong foot rewards must overcome this gap. Currently `foot_phase` gives
~7.2 either way (phase clock is easy to follow), and `feet_clearance_l1`
gives only 0.12 reward gap because foot_xy_vel is also low in shuffle.

---

## 5. What this implies for v8 / v9 design

Five terms reward shuffle gait. Ranked by impact:

1. **`vertical_vel` (0.6 weight)** — currently `exp(−5·\|v_z\|)` rewards
   v_z = 0 (pogo-free). But good walking REQUIRES v_z RMS ≈ 0.15 m/s.
   Confirmed empirically: v5 (no-clock SHUFFLE) has true CoM-z osc 0.5 cm
   (over-compressed), while walk_v34 (CLOCK baseline that walks) has ~2 cm.
   Replace with a bell centered at target ≈ 0.15:
   `exp(−((|v_z| − 0.15) / 0.10)²)`. Saturates at the right amplitude
   instead of pushing it to zero.

2. **`twist` (2.5 weight)** — minimizes pitch+roll, but good walking has
   pitch peak 4–5°. Replace with bell at target tilt:
   `exp(−((|pitch| − 0.07) / 0.05)²)` for pitch alone.
   Roll can stay minimize.

3. **`feet_clearance_l1` (4.0 weight)** — already cmd-gated, but its
   foot_xy_vel weighting silently *reduces* the penalty when feet are slow
   (shuffle). Remove the v_xy multiplier, keep only the cmd-gate:
   `−|h_foot − target| · cmd_active`. Stronger signal in shuffle.

4. **`pose_speed` per-joint σ walking is too loose** for hip_pitch/knee
   (0.50). Good walk hip_pitch RMS ≈ 0.3 rad, knee 0.4 rad — fits inside
   σ=0.50 so the term rewards static poses (RMS=0). Tighten to σ=0.30 for
   hip_pitch/knee, or make σ depend on cmd magnitude.

5. **Regulators (`jnt_vel` 0.001, `joint_tor` 1e-4, `power` 0.1)** — these
   directly reward minimum-effort motion. Shuffle uses less power than walk.
   For these, consider DIVIDING by expected nominal value (i.e., normalize
   so the penalty is 0 at nominal walking, ≤0 only above nominal). E.g.
   `−clip(power − 8 W, min=0) · 0.1`.

---

## 6. Mapping reward magnitudes to TB metrics

After applying these calibrations, expect at convergence:

| sim2sim metric | Target value | Acceptable range |
|---|---|---|
| `walk_quality` | ≥ 0.75 | ≥ 0.60 |
| `shuffle_flag` | 0 | 0 |
| `stride_length` | 0.12 m | ≥ 0.10 m |
| `duty_factor` | 0.60 | 0.50–0.65 |
| `measured_freq` | 2.5 Hz | 2.0–3.0 Hz |
| `com_z_rms` (metric is mostly bias from 0.45 m) | 0.08 m | 0.05–0.15 m — high values DO NOT mean pogo, they mean bent-knee crouch (normal). True osc must be inferred from `com_z_mean`. |
| `vx_bias_fwd` | 0 | ±0.05 m/s |
| `pitch_rms_fwd` | 3° | 2–6° |
| `survive_time_fr1.0` | 15 s | ≥ 14 s |

v6 @ iter 2000: walk_quality 0.39, stride 5 cm, duty 0.79, com_z_rms 7 cm,
freq 3.7 Hz, pitch 4.2°. Three of nine metrics out of target range.
