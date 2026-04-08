# Sim2Sim Design

## Purpose

Sim2Sim validates a policy trained in Isaac Gym inside MuJoCo before deploying to real hardware. If the policy survives a different physics engine (different contact solver, integrator, and numerical properties), it is more likely to transfer to the real robot.

```
Isaac Gym training  →  MuJoCo validation (here)  →  Real robot deployment
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     sim2sim.py                          │
│                                                         │
│  build_mujoco_model()                                   │
│    URDF → MjSpec → patch base_link → freejoint          │
│    → add actuators → compile → disable base collision   │
│                                                         │
│  run()                                                  │
│    ┌─────────────────────────────────────────────────┐  │
│    │  Physics loop (1000 Hz)                         │  │
│    │  ┌───────────────────────────────────────────┐  │  │
│    │  │  Every 15 steps (67 Hz policy rate):       │  │  │
│    │  │    get_obs() → stack 3 frames → ONNX       │  │  │
│    │  │    scale_transform() → phase + joint inc   │  │  │
│    │  └───────────────────────────────────────────┘  │  │
│    │  compute_torques() → mj_step()                  │  │
│    └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Scene Building (`build_mujoco_model`)

The URDF at `assets/q1/urdf/q1.urdf` requires several fixes before MuJoCo can use it correctly.

### 1. base_link mesh patch

`base_link` in the URDF has a `<visual>` element but no `<collision>`. MuJoCo only loads meshes that have a collision geometry, so the torso body would be invisible. The URDF string is patched in memory before parsing to inject a `<collision>` block referencing `base_link.STL`.

After compilation, the collision is immediately **disabled** (`contype=0`, `conaffinity=0`) so the mesh is purely visual and does not interact with the floor.

```python
# Patch URDF string (in memory, not on disk)
urdf_str = urdf_str.replace('</link>\n\n <!-- IMU -->', '<collision>...</collision>\n</link>\n\n <!-- IMU -->', 1)

# After compile, disable its collision
model.geom_contype[base_geom_id]    = 0
model.geom_conaffinity[base_geom_id] = 0
```

### 2. Visual meshes

`spec.discardvisual = False` must be set before `from_string()` to preserve STL visual meshes. The default (`True`) strips all visual geometry on URDF import.

### 3. Freejoint

Isaac Gym gives every robot a floating base automatically. In MuJoCo this must be added explicitly:

```python
base = spec.find_body('base_link')
base.pos = [0, 0, init_height]   # start above floor
fj = base.add_freejoint()
```

This means `qpos` layout is `[x, y, z, qw, qx, qy, qz, j0...j9]` and `qvel` is `[vx, vy, vz, wx, wy, wz, dj0...dj9]`.

### 4. Actuators

The URDF has no actuators. Ten torque actuators are added programmatically, one per revolute joint, with effort limits matching the URDF `<limit effort="...">` values.

### 5. Timestep

`MjSpec.option` is not accessible via Python bindings in MuJoCo 3.x. Timestep and gravity are set directly on the compiled model after `spec.compile()`.

---

## Observation (`get_obs`)

Mirrors `BIRLTask.pure_observation()` in [env/tasks/birl_task.py](../../env/tasks/birl_task.py) exactly.

### IMU body

Training reads orientation and angular velocity from the `imu_in_torso` rigid body (a fixed joint offset inside the torso), not `base_link`. This must match in sim2sim.

```python
# In Isaac Gym training:
self.base_quat = self.rigid_body_param[:, self.imu_in_torso_indice, 3:7]
self.base_avel = self.rigid_body_param[:, self.imu_in_torso_indice, 10:13]

# In MuJoCo sim2sim:
quat         = data.xquat[imu_body_id]      # [w, x, y, z]
world_angvel = data.cvel[imu_body_id][0:3]  # cvel layout: [ang(3), lin(3)]
```

Note: MuJoCo `cvel` packs `[angular(3), linear(3)]` — the opposite of what you might expect.

### Angular velocity frame

Angular velocity is expressed in the **body frame**, matching Isaac Gym's `quat_rotate_inverse(base_quat, base_avel)`. The rotation matrix is constructed from the quaternion and transposed to go from world → body.

### Observation vector (43 dimensions per step)

| Index | Content | Scale | Source |
|---|---|---|---|
| 0–1 | `[cmd_vx, cmd_yaw]` | none | command |
| 2–3 | `[roll, pitch]` | ×1.0 | imu_in_torso euler |
| 4–6 | angular velocity | ×0.5 | imu body frame |
| 7–16 | joint pos − ref_joint_pos | ×1.0 | qpos[7:17] |
| 17–26 | joint velocity | ×0.1 | qvel[6:16] |
| 27–36 | joint_act − joint_pos | ×1.0 | tracking error |
| 37–40 | `[sin(φ_L), sin(φ_R), cos(φ_L), cos(φ_R)]` × static_flag | — | phase modulator |
| 41–42 | `(freq × 0.3 − 1.0)` × static_flag | — | phase modulator |

`static_flag = 1` if `‖[cmd_vx, cmd_yaw]‖ ≥ 0.15`, else `0`. When standing still, the phase signal is zeroed.

### History stacking

Three consecutive observation frames are concatenated → `[1, 129]` policy input. This gives the policy temporal context equivalent to training (`obs_history.maxlen=3`).

---

## Policy Inference

- Format: ONNX (exported from PyTorch via `export_pt2onnx.py`)
- Default model: `experiments/q2/deploy/policy.onnx`
- Input: `float32[1, 129]`
- Output: `float32[1, 12]` — values in `[-1, 1]` (tanh output)
- Rate: every `decimation=15` physics steps → ~67 Hz

---

## Action Pipeline

### 1. Scale

Policy output `[-1, 1]` is scaled to `[inc_low, inc_high]` matching `scale_transform()` in [env/utils/math.py](../../env/utils/math.py):

```python
scaled = (net_out + 1) / 2 * (inc_high - inc_low) + inc_low
```

Ranges: frequency `[0.5, 3.5]` Hz, joint increments `[-15, 15]` deg/s equivalent.

### 2. Phase modulator

First 2 values of scaled output → left/right leg frequencies. The phase modulator integrates:

```
phase = (phase + 2π × freq × dt) mod 2π
```

This is a direct port of [env/utils/phase_modulator.py](../../env/utils/phase_modulator.py).

### 3. Joint targets (increment mode)

Remaining 10 values are added to the current joint target each policy step:

```python
current_joint_act += scaled[2:] * policy_dt
current_joint_act  = clip(current_joint_act, joint_limit_low, joint_limit_high)
```

This matches `cfg.action.use_increment = True` in training.

---

## Torque Computation

Matches `legged_robot.py` line 414 exactly:

```python
torque = kp × error + kd_bias − dq + joint_tor_offset − 3.5 × sign(dq) × vel_sign
```

**Important:** `kd` here is a **constant bias term**, not a velocity-proportional damping gain. This is an unusual PD formulation — the damping comes from the `−dq` term directly (unit gain), while `kd` provides a small constant offset. This must match exactly or the robot behaves differently from training.

`joint_tor_offset` compensates for static friction and gravity biases per joint. `vel_sign` applies additional friction only on joints that tend to back-drive.

PD gains (from `config/Base.py`):

| Joint | kp | kd |
|---|---|---|
| hip_yaw | 55 | 0.3 |
| hip_roll | 105 | 2.5 |
| hip_pitch | 75 | 0.3 |
| knee | 45 | 0.5 |
| ankle | 30 | 0.25 |

---

## Timing

| Loop | Rate | Notes |
|---|---|---|
| Physics (`mj_step`) | 1000 Hz | matches Isaac Gym `sim.dt=0.001` |
| Policy inference | 67 Hz | every `decimation=15` physics steps |
| Torque application | 1000 Hz | PD runs at physics rate |
| Viewer sync | real-time | sleep to match wall clock |

---

## Known Differences vs Isaac Gym

| Aspect | Isaac Gym | MuJoCo |
|---|---|---|
| Contact solver | PhysX TGS | MuJoCo CG |
| Friction model | Coulomb + restitution | Elliptic cone |
| Domain randomization | Yes (mass, gains, delays, pushes) | No |
| Observation delay | Simulated (10–50 steps) | None |
| Noise | Added to all sensors | None |
| Number of envs | 4096 parallel | 1 |

These differences are intentional — if the policy works despite them, it is more likely to transfer to the real robot.

---

## Usage

```bash
cd ~/code/RoboTamer4Qmini
LD_LIBRARY_PATH=~/miniconda3/envs/qmini/lib \
~/miniconda3/envs/qmini/bin/python deploy/sim2sim/sim2sim.py \
    [--config deploy/sim2sim/configs/qmini_birl.yaml] \
    [--cmd_vx 0.5] \
    [--cmd_yaw 0.0] \
    [--duration 30] \
    [--headless]       # no viewer — prints x/y/z/roll/pitch/yaw/vx/vy to stdout
```

To use a different trained model, export it first then point the config at it:

```bash
python export_pt2onnx.py --name my_run
# Edit policy_path in qmini_birl.yaml → experiments/my_run/deploy/policy.onnx
```

---

## File Reference

| File | Purpose |
|---|---|
| [sim2sim.py](sim2sim.py) | Main script |
| [configs/qmini_birl.yaml](configs/qmini_birl.yaml) | Robot + policy config |
| [../../assets/q1/urdf/q1.urdf](../../assets/q1/urdf/q1.urdf) | Robot URDF |
| [../../env/tasks/birl_task.py](../../env/tasks/birl_task.py) | Observation/action reference |
| [../../env/legged_robot.py](../../env/legged_robot.py) | Torque formula reference |
| [../../env/utils/phase_modulator.py](../../env/utils/phase_modulator.py) | Phase modulator reference |
