# V2 Stand Policy — SDK Deploy Spec

**Policy**: `policy.onnx` (v2_run33 iter 1600, 10-reward cleanup version, peak quality_score 0.93 with all friction surviving 15s, pitch ~1.4°)
**Manifest**: `manifest.yaml` (source of truth — values below are for documentation; SDK should load from manifest)

Task: **stand still** (commanded velocity = 0). No walking, no turning.

---

## 1. Observation input

### Network input shape: `[1, 195]`
195 = 39 (per-step obs) × 5 (history frames stacked).

### Per-step observation: 39 dims, concatenated in this exact order

| Idx | Slot                     | Dim | Source                          | Scaling |
|-----|--------------------------|-----|---------------------------------|---------|
| 0–2 | `commands`               | 3   | `[cmd_vx, cmd_vy, cmd_yaw]`     | (use [0,0,0] for stand) |
| 3–5 | `base_ang_vel * 0.5`     | 3   | IMU body-frame angular rate     | **× 0.5** |
| 6–8 | `projected_gravity`      | 3   | gravity vector in body frame    | unit-norm gravity rotated to body frame |
| 9–18 | `joint_pos − ref_joint_pos` | 10 | measured joint pos minus reference | no scale |
| 19–28 | `joint_vel * 0.1`       | 10  | measured joint velocity         | **× 0.1** |
| 29–38 | `joint_tracking_err`    | 10  | `current_joint_act − joint_pos` (target minus measured) | no scale |

**After concatenation, clip every element to `[-3.0, +3.0]`.**

### Joint order (10 joints, L then R)

```
[hip_yaw_L, hip_roll_L, hip_pitch_L, knee_L, ankle_L,
 hip_yaw_R, hip_roll_R, hip_pitch_R, knee_R, ankle_R]
```

### `ref_joint_pos` (from manifest, also baked into SDK):
```
[+0.4, -0.1, -1.5, +1.0, -1.3,    # left leg
 -0.4, +0.1, +1.5, -1.0, +1.3]    # right leg
```

### `projected_gravity` — how to compute

Given body orientation quaternion `q = (w, x, y, z)` (from IMU):
```
R_world_to_body = quat_to_R(q).T   # body-to-world transpose
projected_gravity = R_world_to_body @ [0, 0, -1]
```
When robot is upright, this should be roughly `[0, 0, -1]`.

### `commands` — fixed zeros for stand

For stand task, SDK should output `[0.0, 0.0, 0.0]` always.

---

## 2. Observation history (frame stacking)

Network expects **5 frames sampled every 2 steps** (skip=2) from a rolling buffer.

### Buffer requirements
- Length: `(history−1) × skip + 1 = (5−1) × 2 + 1 = 9` frames
- Initialize all 9 slots to **zeros** (39-dim each) at boot.

### Per policy step
1. Compute new 39-dim obs (`obs_now`)
2. Push `obs_now` into buffer (FIFO — drop oldest, append newest)
3. Sample 5 frames at indices `[0, 2, 4, 6, 8]` from buffer (oldest is index 0, newest is index 8)
4. Concatenate to form `[1, 195]` network input

```cpp
// pseudocode
ObsBuffer.push(obs_now);   // FIFO, maxlen=9
auto input = concat(ObsBuffer[0], ObsBuffer[2], ObsBuffer[4], ObsBuffer[6], ObsBuffer[8]);
// shape: [1, 195]
```

**Order matters**: the input is **oldest → newest**, NOT newest → oldest.

---

## 3. Output handling

### Network output: `[1, 10]` (one action per joint, joint order matches obs).

### Action conversion to motor position target

```cpp
// Step 1: scale network output from [-1, 1] to residual range [-0.5, 0.5] rad
clip(net_out, -1.0, 1.0);
offset = (net_out + 1.0) / 2.0 * (residual_high - residual_low) + residual_low;
// residual_low = -0.5, residual_high = +0.5 per joint
// equivalent: offset = net_out * 0.5  (since range is symmetric ±0.5)

// Step 2: low-pass filter (state across policy steps)
//   alpha = 0.75 (from manifest), initialize lp_target = zeros[10] at boot
lp_target = 0.75 * offset + 0.25 * lp_target_prev;

// Step 3: form absolute joint position target
joint_target = ref_joint_pos + lp_target;

// Step 4: clip to joint limits (use URDF limits)
joint_target = clip(joint_target, joint_lower_limit, joint_upper_limit);
```

### State to maintain across policy steps
- `lp_target[10]` — initialize to **zeros** at boot. **Not** to `ref_joint_pos`.
- `obs_buffer[9][39]` — initialize to **zeros**.

---

## 4. Motor PD

After computing `joint_target`, run standard PD on each joint at high rate (1 kHz):

```cpp
tau[i] = kps[i] * (joint_target[i] - q[i]) - kds[i] * dq[i];
```

PD gains (per joint, L then R):

| Joint        | kp  | kd   |
|--------------|-----|------|
| hip_yaw      | 55  | 0.3  |
| hip_roll     | 105 | 2.5  |
| hip_pitch    | 75  | 0.3  |
| knee         | 45  | 0.5  |
| ankle        | 30  | 0.25 |

(L and R share the same gains.)

**No torque bias, no kd-as-bias, no joint_tor_offset.** Standard PD only.

### Rates
- Policy: **66.7 Hz** (every 15 ms — `decimation = 15` × `simulation_dt = 0.001`)
- Motor PD: **1 kHz** (driver-side or SDK-side)
- Between policy steps, the SAME `joint_target` is sent to the PD loop.

---

## 5. Boot / reset sequence

Before policy starts running:
1. Robot's joints should be at or near `ref_joint_pos` (standard stand pose).
2. Initialize `lp_target = zeros[10]`.
3. Initialize `obs_buffer` to **9 frames of zeros[39]**.
4. Set commands = `[0, 0, 0]`.
5. First policy step: compute obs (will be mostly stable around ref since lp_target=0), append to buffer (now buffer has 8 zeros + 1 real frame), sample input (mostly zeros), run network.
6. The first ~9 policy steps (~135 ms) the buffer is partially zero-padded — this is **intentional** and matches training (history starts from zeros).

---

## 6. SDK integration checklist (changes from BIRL SDK)

The current SDK at `RoboTamerSdk4Qmini` implements **BIRL** (44-dim obs, 12-dim action with phase frequencies). Changes needed for V2:

### `get_observation()` — 39-dim layout

```cpp
// Drop BIRL's phase_sin_cos and phase_freq slots.
obs << target_command,                                  // 3  (use [0,0,0])
       base_rpy_rate * 0.5,                             // 3
       projected_gravity,                               // 3
       joint_pos - _ref_joint_pos,                      // 10
       joint_vel * 0.1,                                 // 10
       _current_joint_act - joint_pos;                  // 10
// total = 39, then clip to [-3, 3]
```

**Note**: V2 uses `projected_gravity` (3-dim), **not** `base_rpy` (which is also 3-dim but different values). BIRL used `base_rpy[0:2]` (roll, pitch) — V2 uses full 3-D gravity vector. Must change.

### `joint_increment_control()` → replace with residual mode

```cpp
// Drop phase modulator (no pm_f anymore).
// Drop joint increment accumulation.

// Apply [-0.5, 0.5] scaling
offset = network_output * 0.5;  // since residual range is symmetric ±0.5

// Low-pass filter (maintain _lp_target as member variable)
_lp_target = 0.75 * offset + 0.25 * _lp_target;

// Form absolute target
_current_joint_act = _ref_joint_pos + _lp_target;

// Clip to joint limits
_current_joint_act = clip(_current_joint_act, _joint_lower, _joint_upper);
```

### Obs history buffer

```cpp
constexpr int OBS_DIM = 39;
constexpr int OBS_HIST = 5;
constexpr int OBS_SKIP = 2;
constexpr int BUFFER_LEN = (OBS_HIST - 1) * OBS_SKIP + 1;  // = 9

std::deque<Eigen::VectorXf> _obs_buffer;  // maxlen 9

// init in constructor:
for (int i = 0; i < BUFFER_LEN; ++i)
    _obs_buffer.push_back(Eigen::VectorXf::Zero(OBS_DIM));

// each policy step:
_obs_buffer.pop_front();
_obs_buffer.push_back(obs_now);
Eigen::VectorXf network_input(OBS_HIST * OBS_DIM);
for (int i = 0; i < OBS_HIST; ++i)
    network_input.segment(i * OBS_DIM, OBS_DIM) = _obs_buffer[i * OBS_SKIP];
// network_input is now [195], oldest → newest
```

### `configParams`

```cpp
num_observations = 195;       // 39 × 5
num_stacks = 1;               // stacking done by us, network sees flat 195
num_actions = 10;
ref_joint_act = [+0.4, -0.1, -1.5, +1.0, -1.3, -0.4, +0.1, +1.5, -1.0, +1.3];
action_lowpass_alpha = 0.75;
decimation = 15;
// kps and kds as above
```

---

## 7. Verification

After SDK changes, expected behavior on robot:
- Stand still indefinitely with knees slightly bent at ref pose
- Total xy drift < 5 cm over 30 seconds on flat floor
- Pitch / roll oscillation < ±10° (typical 3-7°)
- Yaw should **not** rotate (locked, not drifting)

If you see any of these failure modes, debug in this order:
1. **Robot tips over within 5s** → obs scaling wrong (verify `*0.5` on ang_vel, `*0.1` on joint_vel)
2. **Robot's stance collapses (legs bend, feet cross)** → `ref_joint_pos` not applied, or residual scale wrong
3. **Robot stands but yaw spins** → `commands` not zeroed, or projected_gravity has wrong sign
4. **Robot jitters violently** → `lp_target` not retained across steps (resets every call), or `decimation` not respected (policy running too fast)
