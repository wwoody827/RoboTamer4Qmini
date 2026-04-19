# Reward Terms — `env/tasks/birl_task.py::reward()`

All reward components are stored in `rew_dict`, multiplied by their config weight (`configs/base.yaml` under `reward:`), clipped to `[-4, 5]`, then scaled by `env.dt` (0.015s). The episode return is the sum across all terms and all timesteps.

**Common multipliers you'll see in every formula:**

| Name | Meaning |
|------|---------|
| `lin_vel_x_norm` | `clip(‖cmd_xy‖, 0.3, 2.0) + 0.2` — normalization denominator; min 0.5. Higher speed commands widen tolerances. |
| `yaw_rate_norm` | `clip(|cmd_yaw|, 0.3, 1.5) + 0.2` — same idea for yaw. |
| `static_flag` | 1 if commanded to move (`‖[vx,vy,yaw]‖ ≥ 0.15`), else 0. Gates terms that only matter when moving. |
| `vy_walking` | 1 if `|cmd_vy| > 0.1`, else 0. Disables strafe-incompatible constraints. |
| `balance_rew` | Multiplies many terms — suppresses them when body tilted or at wrong height. |
| `foot_swing_mask` / `foot_support_mask` | Desired per-leg phase state (from phase modulator, external clock, or contact mode). |

---

## 1. Tracking (the goal)

| Term | Default weight | Meaning |
|------|----------------|---------|
| `fwd_vel` | **2.3** | `exp(-k·(cmd_vx - base_vx)²)` — track forward velocity command. `k` adapts to command speed. |
| `lateral_vel` | **0.7** | `exp(-k·(cmd_vy - base_vy)²) − 0.6/norm·|err|·static_flag` — track lateral vel, plus a small linear penalty when moving. |
| `yaw_rat` | **2.5** | `exp(-k·(cmd_yaw - base_yaw_rate)²)` — track yaw rate command. |
| `vertical_vel` | 0.6 | `exp(-k·vz²) − 0.2·|vz|·static_flag` — discourage vertical bouncing. |
| `ang_vel` | 0.6 | `exp(-k·‖ω_xy‖²)` — discourage roll/pitch rates (spinning body). |
| `base_acc` | 0.1 | `−0.4·‖(a − g)·0.1‖·static_flag` — penalize IMU-frame acceleration while moving. |

**Interpretation:** The four gaussian-shaped "track" rewards (`fwd_vel`, `lateral_vel`, `yaw_rat`) form the main task signal. These are the *target function* everything else is regularizing around.

---

## 2. Posture (stay upright)

| Term | Default weight | Meaning |
|------|----------------|---------|
| `base_heit` | 1.0 | `exp(-70·(base_z − 0.45)²)` — keep COM at 0.45m. |
| `balance` | **1.5** | `0.5·(base_heit · exp(-k·‖roll,pitch‖) + 1)` — combined height + tilt penalty. Also used as a multiplier on ~17 other terms. |
| `twist` | **2.5** | `−‖roll,pitch‖` — raw tilt penalty. |
| `constant` | 0.3 | `1.0` — survival bonus (rewards staying alive). |

**Interpretation:** `balance_rew` is the workhorse — nearly every joint/foot regularizer is gated by it, so if the robot falls over those terms stop contributing. `twist` is the unconditional tilt penalty.

---

## 3. Foot scheduling (gait timing)

| Term | Default weight | Meaning |
|------|----------------|---------|
| `foot_clr` | 1.0 | Fraction of legs correctly in swing (desired=swing ∧ actual=swing) × static_flag. |
| `foot_supt` | 0.7 | Fraction of legs correctly in support × static_flag. |
| `foot_heit` | 0.7 | +40·clip(h, 0, 0.05) in swing, −20·(h−0.06)⁺ always, −0.2·score in support. Encourages 5cm step height. |
| `foot_phase` | 0.3 | **Phase-mode dependent**: if `output`, penalizes L/R legs being in-phase (should be antiphase). If `input`, rewards matching external clock schedule. If `none`, zero. |
| `air_time` | 0.0 (disabled) | `sum(clip(foot_air_time, 0.3) · is_in_air) · static_flag` — **continuous**: rewards each step a foot is airborne, saturated at 0.3s. |

**Interpretation:** These make the gait look like walking: one foot up, one foot down, ~5cm steps, legs alternating. `air_time` was recently converted from event-driven (fire at landing) to continuous (fire every step in air) so its weight has the same scale as `foot_clr` / `foot_supt`.

---

## 4. Foot mechanics (don't drag, don't slam)

| Term | Default weight | Meaning |
|------|----------------|---------|
| `foot_slip` | 0.5 | Three-part: reward swing-foot moving forward (pushes off), penalize lateral vel when not strafing, penalize horizontal vel during support (slip). |
| `foot_vz` | 0.2 | Penalize downward foot vel near ground (prevents slamming); reward slight downward intent when standing. |
| `foot_acc` | 0.05 | `−0.4·‖foot_z_vel‖` — smooth foot z-motion. |
| `foot_sft` | **2.7** | `−0.1·‖Δfoot_frc‖/100` — penalize large changes in contact force (soft landings). High weight because amplitude is small. |
| `feet_frc` | 0.001 | `−‖frc · swing_mask‖ − ‖(|frc−55| · support_mask)⁺‖` — keep swing feet off ground, support feet near 55N target. |
| `feet_py` | 0.5 | `−0.5·‖foot_pitch‖` — foot sole should point at the ground, not up. |
| `leg_width_rew` | 0.5 | `−‖|foot_y − base_y| − 0.14‖` — maintain ~14cm stance width. |

**Interpretation:** These are contact-quality regularizers. `foot_sft` has high weight because the raw signal is pre-divided by 100.

---

## 5. Joint regularizers (smooth, not at limits)

| Term | Default weight | Meaning |
|------|----------------|---------|
| `act_smo` | **1.5** | `−0.3·‖a[t] − 2a[t-1] + a[t-2]‖` — 2nd derivative (jerk) of joint action. Big weight — smoothness is critical. |
| `net_smo` | 0.001 | Same for raw net_out (skips freq prefix when `phase.mode=output`). Squared. |
| `net_out_val` | 0.0001 | `−0.4·‖net_out‖²` — encourage small output magnitudes (prevents saturation). |
| `act_const` | 0.2 | `−0.1·‖joint_act − ref_pose‖ − 3·‖act[0,1,5,6] − ref‖·(1−vy_walking)` — stay near default pose; heavy penalty on hip yaw/roll when not strafing. |
| `sa_const` | 0.1 | Support-aware constraint: support-foot leg's joints should stay near ref pose. |
| `jnt_pos_err` | 0.2 | `−0.4·‖cmd_joint − actual_joint‖²` — PD tracking error. |
| `jnt_vel` | 0.003 | `−0.4·‖joint_vel‖² − ‖vel[0,1,5,6]‖²·(1−vy_walking)` — low joint speeds. |
| `joint_tor` | 0.001 | `−0.4·sum((|τ| − τ_lim)⁺)` — penalize exceeding torque limits. |
| `pmf` | 0.03 | **Phase-mode dependent**: if `output`, smooth freq outputs + zero freq during support. Otherwise zero. |
| `power` | 0.0 (disabled) | `−sum(|τ · joint_vel|)/100` — mechanical power. |

**Interpretation:** `act_smo` is second in importance only to the tracking rewards — unsmooth actions = robot shakes itself apart. The `[0,1,5,6]` indices are hip_yaw_L/R and hip_roll_L/R; these are punished hard when not strafing because they mostly cause drift.

---

## 6. Optional / conditional

| Term | Default weight | Condition |
|------|----------------|-----------|
| `heading` | 0.0 | Only when command specifies heading (currently off). `exp(-3·heading_err²)` gated on near-zero yaw cmd. |
| `yaw_smooth` | 0.0 | Penalize yaw-rate jerk. Off by default. |
| `jp_imit` / `jv_imit` | 0.5/0.5 | Only when `ref_clip_paths` is non-empty (MIRL imitation mode). Rewards matching reference motion's joint pos/vel. When active, task rewards scaled by `w_task=0.5`. |

---

## Tuning heuristics

- **Low-weight continuous ≠ unimportant.** `foot_sft` 2.7 looks huge, but raw value is tiny after the `/100`. Conversely `fwd_vel` 2.3 has raw value ~1.0. Look at the `Rewards/` TB scalars, not just the config weight.
- **`balance_rew` multiplies many terms.** If robot is falling over, those terms go to zero regardless of their weight. Posture has to work first.
- **`static_flag` gates locomotion-only rewards.** When standing still, foot_clr / foot_phase / air_time / foot_slip / etc. all drop to 0 — this is intentional.
- **`lin_vel_x_norm` stretches tolerances at speed.** A `k=5` decay at 0.5 m/s becomes `k=3` at 1.0 m/s — the robot gets more slack the faster it's going.
- **Event-driven vs continuous.** `air_time` was event-driven (only fired at landing, ~2 times/s per foot). Now continuous (fires every of ~67 steps/s in air). Weight semantics match other per-step rewards.

---

## Phase-mode dependencies

Three terms behave differently based on `phase.mode`:

| Term | `output` (BIRL) | `input` (BD_X) | `none` (MIRL) |
|------|-----------------|-----------------|----------------|
| `foot_phase` | Anti-phase constraint on internal phase | Match external clock schedule | 0 |
| `pmf` | Smooth freq + zero freq in support | 0 | 0 |
| `net_smo` | Skip first 2 dims (freq prefix) | Full net_out | Full net_out |

---

## Evaluation reward

`eval_rew` returned alongside main reward = only `fwd_vel + yaw_rat + ang_vel + lateral_vel + vertical_vel + twist` (×dt). This is the "is it doing the task?" number used by the sim2sim evaluator.
