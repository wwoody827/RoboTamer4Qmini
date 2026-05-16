# walk_noclock_v5_iter1800 — no-clock-in-obs deploy candidate

Trained from `configs/walk_noclock_v5.yaml`. **First successful no-clock-in-obs
policy on Qmini** — achieves perfect 40/40 s sim2sim survival on the in-
distribution cmd grid (5 cmds × 3 seeds × 8 s).

## Why this checkpoint

iter 1800 is the sweet spot. Training continued past this point overfits —
iter 2400 dropped to 39.3 / 40 s (one strafe fall), iter 3000 to 33.8 / 40 s
(85 %).

## Recipe — Cassie + BDX-R hybrid

This is the first run that successfully reproduces "no clock in observation"
on Qmini. Three reasons it works where v1-v4 + walk_bdxr_repro failed:

1. **Phase clock still ticks during training** (`phase.mode=input`) — used by
   `foot_phase` reward (weight 4.0) to enforce a proper swing/stance pattern.
   But `phase_clock` and `phase_freq_cmd` are NOT in the obs slot list, so
   the deployed policy never sees the clock. This is the Cassie-style design
   (`docs/walk_noclock_research.md`).

2. **BDX-R-MjLab reward innovations** are imported but with conservative
   weights to suit Qmini's lower-damped hardware:
   - `pose_speed` (weight 1.5) with **per-joint speed-conditional std** —
     tight at standstill (forces upright ref pose), loose during walking
     (allows leg swing). Per-joint values from BDX-R legs config.
   - `upright` (weight 1.5, std=√0.1) — exp bell on projected gravity XY.
   - `feet_clearance_l1` (weight 1.5) — L1 cmd-gated foot clearance.
   - `feet_swing_height_peak` (weight 1.0) — peak-tracking foot height
     evaluated at touchdown.
   - `soft_landing` (weight 1e-5) — landing-force penalty.
   - `projected_gravity` obs slot (3-dim, IMU-style) replaces `base_euler`.

3. **walk_v34's Qmini-tuned core kept** — not removed like in
   walk_bdxr_repro which failed:
   - `cmd_track_lp_alpha=0.95` (anti-Jensen-gaming gait-sway filter).
   - Linear `1 - clip(α|err|)` tracking shape (Qmini torque-scaled).
   - Standard regulators (act_smo 0.30, jnt_vel 1e-3, joint_tor 1e-4).
   - `foot_stand: 1.0` standstill double-support reward.
   - 2-layer 512-256 ReLU + `entropy_coef=0.0005` + 3 epochs +
     `discount_factor=0.995` (BDX-R hyperparams stalled training; ours work).
   - `body_ang_vel` / `angular_momentum` penalties from BDX-R DISABLED
     because Qmini's `kd=0.3` damping needs ACTIVE body rotation for
     balance — see `docs/bdxr_reproduction.md` outcome section.

## Observation (NO CLOCK, ~~deployment can be clock-free~~)

5 frames × skip 2 × 39 dim/frame = 195-dim total obs. Per-frame layout:

| Slot | Idx | Dim | Content |
|---|---|---|---|
| `commands_3` | 0-2 | 3 | `[cmd_vx, cmd_vy, cmd_yaw]` |
| `base_ang_vel` | 3-5 | 3 | body-frame ω × 0.5 |
| `projected_gravity` | 6-8 | 3 | body-frame gravity vector (IMU-style) |
| `joint_pos_err` | 9-18 | 10 | `joint_pos - ref_joint_pos` |
| `joint_vel` | 19-28 | 10 | `joint_vel × 0.1` |
| `joint_tracking_err` | 29-38 | 10 | `joint_act_target - joint_pos` |

**No phase_clock / phase_freq_cmd / clock-related signals in obs.** The
trained policy is fully clock-free at deploy.

## Action

10-dim absolute position targets, LP filter α=0.75, scaled to URDF joint
limits. Same as walk_v34 / walk_rl_v4. PD gains identical (`policy_manifest.yaml`).

## Sim2sim numbers (iter 1800, 3 seeds × 8 s, friction 1.0, cmd_freq 2.5)

| cmd | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| stand (0,0,0) | 8.0 | 8.0 | 8.0 |
| fwd 0.3 | 8.0 | 8.0 | 8.0 |
| bwd -0.3 | 8.0 | 8.0 | 8.0 |
| strafe ±0.1 | 8.0 | 8.0 | 8.0 |
| yaw 0.3 | 8.0 | 8.0 | 8.0 |

Total: **40 / 40 s (100 %)**.

## ⚠️ Caveat: this policy SHUFFLES (do not deploy as-is)

Survival was the only metric tracked during this run. Forensic gait-quality
sweep at fr=1.0, cmd_vx=0.3 revealed:

| Metric         | walk_rl_v4 (with clock) | walk_noclock_v5 |
|----------------|-------------------------|-----------------|
| stride_length  | 0.142 m                 | **0.042 m**     |
| duty_factor    | 0.530 (alternating)     | **0.918** (almost always double-support) |
| measured_freq  | 2.50 Hz                 | 1.51 Hz         |
| vx_bias_body   | 0.054 m/s               | **0.232 m/s** (only 0.07 actual when cmd 0.30) |

v5 satisfies the phase-clock reward by twitching its feet without actually
translating the body — a classic shuffle exploit. Survival is real, walking
is not. The `walk_noclock_v6` config bumps foot_phase / feet_swing_height_peak /
feet_clearance_l1 weights to force real swing.

Use this checkpoint as a baseline curve in TB (`sim2sim/walk_quality` shows
the shuffle), not as a deployable artifact.

## Caveats

- Trained with **stage-0 cmd ranges** (lin_x ±0.4, lin_y ±0.1, yaw ±0.3) — the
  policy has only seen these. Driving it at higher commands is out-of-
  distribution.
- Not tested across the full walk_rl_v4 grid (lin_x ±0.5, lin_y ±0.3, yaw ±0.5)
  because most of those values are out of training distribution.
- This is the first no-clock variant that *survives* — the v6 config attempts
  to make it also *walk*.

## Run

```bash
python scripts/release_eval.py --candidate deploy_candidates/walk_noclock_v5_iter1800
```
