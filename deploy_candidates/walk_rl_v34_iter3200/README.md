# walk_rl_v34_iter3200 — current deployment candidate

Pure-RL policy from `configs/walk_rl.yaml` at iteration 3200. Trained without
motion imitation. Selected for the deployment slot because its **gait
frequency lock is the cleanest of all checkpoints surveyed** — the policy
follows the external phase clock at cmd_freq = 2.5 Hz across the full
command grid.

| File | What |
|---|---|
| `policy.onnx`            | trained policy weights, exported via `export_pt2onnx.py` |
| `policy_manifest.yaml`   | sim2sim config (phase mode, obs slots, action range, …) |
| `videos/`                | 11 demo videos at common commands (10 s headless MuJoCo) |
| `eval/eval_vx.csv`       | sim2sim omni eval over (friction × cmd_vx) × 3 runs |
| `README.md`              | this file |

## Why this checkpoint

| Metric | Value | Note |
|---|---|---|
| **measured_freq across cmd_vx** | **2.50 – 2.56 Hz** | 0.06 Hz total range — gait freq stays locked to cmd_freq=2.5 |
| stride_asymm | 0.001 – 0.045 | left/right symmetric |
| vx_fwd_err (+0.3, fric 1.0) | 0.043 | clean forward tracking |
| vx_bwd_err (−0.3, fric 1.0) | 0.289 | the residual backward asymmetry inherited from walk_rl |
| disp_y (cmd_vy = +0.3, 10 s) | 1.59 m best | first ckpt with real strafe |
| Survival (friction ∈ {0.5, 1.0, 1.5}) | 100 % except corner cell (1.5, +0.7) | |

## Training recipe (what went into this policy)

- `_base: walk_bdx_base.yaml` — BD_X stack
  (phase.mode=input, action.action_mode=absolute, history 5 × skip 2,
   regime sampling pure+pairs, action low-pass α=0.75).
- `reg_use_norm_scaling: false` — drops the `1/lin_vel_x_norm` amplification
  that used to double regulators in pure_yaw mode and break tracking.
- Regulator weights scaled to legged_gym / Unitree G1 ranges
  (`act_smo 0.30`, `jnt_vel 1e-3`, `joint_tor 1e-4`, …).
- Tracking weights: `fwd_vel 2`, `yaw_rat 3`, `lateral_vel 2`. Slopes 3 / 5 / 3.
- `cmd_track_lp_alpha: 0.95` — 290 ms EMA on body vel/ang for reward computation,
  defeating gait-sway gaming.
- `command.resampling_time: 10` (= episode length) so a single cmd holds the
  whole episode and gait converges. `regime_weights: [1, 2, 1, 1.5, 1.5]`
  biases sampling toward pure_vy.

Full report: `docs/walk_v34_summary.md`.

## Input / Output format

### Input: 215-dim observation vector

Single forward pass = `[obs(t-8), obs(t-6), obs(t-4), obs(t-2), obs(t)]`
concatenated. Each frame is **43-dim**, and the policy receives **5 frames
stacked with skip=2** (so the visible window is the last 8 policy steps,
≈ 120 ms at 67 Hz).

Per-frame layout (43-dim, BD_X-style):

| Slot           | Index range | Dim | Content |
|---|---|---|---|
| `commands_3`         | 0–2   | 3  | `[cmd_vx, cmd_vy, cmd_yaw]` body-frame |
| `base_euler`         | 3–4   | 2  | roll, pitch (from `imu_in_torso` body) |
| `base_ang_vel`       | 5–7   | 3  | body-frame angular velocity × 0.5 |
| `joint_pos_err`      | 8–17  | 10 | `joint_pos − ref_joint_pos` |
| `joint_vel`          | 18–27 | 10 | `joint_vel × 0.1` |
| `joint_tracking_err` | 28–37 | 10 | `joint_act_target − joint_pos` |
| `phase_clock`        | 38–41 | 4  | `[sin(φ_L), sin(φ_R), cos(φ_L), cos(φ_R)] × static_flag` |
| `phase_freq_cmd`     | 42    | 1  | `((cmd_freq − 2.5)/0.5) × static_flag` (normalized) |

`static_flag = 1` when `‖[cmd_vx, cmd_vy, cmd_yaw]‖ ≥ 0.15`, else `0`
(zeroes phase signals when standing still).

**Joint order** (used in slots 8–37): hip_yaw_l, hip_roll_l, hip_pitch_l,
knee_pitch_l, ankle_pitch_l, hip_yaw_r, hip_roll_r, hip_pitch_r,
knee_pitch_r, ankle_pitch_r.

**Observation must be clipped to ±3.0 element-wise** before feeding the policy.

### Output: 10-dim action vector (absolute joint targets + low-pass)

Network output ∈ [-1, 1] is scaled to the joint pos range (URDF limits) per
joint. With `action_mode: absolute` and `action_lowpass_alpha: 0.75`, the
policy step is:

```
raw = scale_transform(net_out, jlim_low, jlim_high)   # [-1,1] → [jlim_low, jlim_high]
lp_target = 0.75 * raw + 0.25 * lp_target_prev        # 60ms time constant at 67Hz
joint_target = clip(lp_target, joint_limits.low, joint_limits.high)
```

`joint_target` then drives a PD controller at 1 kHz physics rate
(`decimation = 15`, so 15 physics steps per 1 policy step):

```
τ = kp · (joint_target − q) − kd · q_dot   (per-joint PD)
```

PD gains (from manifest):

| Joint            | kp   | kd  |
|---|---|---|
| hip_yaw_l/r      | 55   | 0.3 |
| hip_roll_l/r     | 105  | 2.5 |
| hip_pitch_l/r    | 75   | 0.3 |
| knee_pitch_l/r   | 45   | 0.5 |
| ankle_pitch_l/r  | 30   | 0.25 |

Policy rate: **66.67 Hz** (sim dt 1 ms × decimation 15). External phase
clock advances at `cmd_freq ∈ [2.0, 3.0] Hz` (default 2.5); training was
done with `cmd_freq=2.5`.

## How to run

```bash
# Visualize in MuJoCo viewer:
python deploy/sim2sim/sim2sim.py \
    --policy deploy_candidates/walk_rl_v34_iter3200/policy.onnx \
    --cmd_vx 0.3 --cmd_vy 0.0 --cmd_yaw 0.0

# Batch eval:
python deploy/sim2sim/evaluate.py \
    --policy deploy_candidates/walk_rl_v34_iter3200/policy.onnx \
    --grid vx --runs 6 --duration 10
```

## Known weaknesses

- vx_bwd is 5–7× weaker than vx_fwd (systemic asymmetry of this training stack).
- vy strafe is real but tops out at ~70 % efficiency of the issued command.
- (friction = 1.5, vx = ±0.7) corner cases drop survival.

If these matter for your use case, look at the MIRL run (in progress at
`experiments/.../walk_mirl`) for an alternative once it finishes.
