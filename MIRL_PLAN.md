# Multi-Skill Locomotion Development Plan
## Motion Imitation RL (MIRL) for Qmini

---

## Goal

Train a single deployable policy that handles:
- Forward / backward / strafing / turning (already working)
- Stair climbing and descending
- Squat / variable height command
- Get-up from ground (fallen recovery)

---

## Key Architecture Decision: Two Task Classes

**Keep `birl_task` for walking** — it works, don't break it.

**Build `MIRLTask` for new skills** — different obs layout, no phase modulator, uses reference clips + RSI.

Why not extend birl_task for everything:
- Phase modulator (obs indices 38-43, action outputs 0-1) is meaningless for non-cyclic skills (get-up, squat)
- birl_task has no concept of external reference clips or RSI
- Action space `[freq_L, freq_R, Δjoint×10]` conflicts with MIRL's position-residual approach

Final unified policy uses **Mixture of Experts (MoE)** distillation — train skill experts independently, learn a gate network to select between them.

---

## What is MIRL (vs pure RL, vs SFT→RL)

Standard RL trains from scratch — reward signal only. Sparse for complex skills like get-up.

SFT→RL (like LLM RLHF) does not work well for locomotion because:
- BC/SFT does not produce a physically stable policy — it just minimizes trajectory error offline
- When RL starts, the policy encounters states never seen during SFT → immediate distribution shift
- RL gradient quickly overwrites SFT knowledge (catastrophic forgetting)
- LLM analogy breaks down: an SFT LLM already generates valid text; an SFT locomotion policy still falls over

MIRL keeps the reference motion **during** RL training:
- `total_reward = w_task × task_reward + w_imit × imitation_reward`
- **RSI (Reference State Initialization)**: each episode starts from a random frame of the reference clip — seeds exploration throughout the full motion, not just from the start
- Imitation reward is soft — RL can deviate if task reward is higher
- Reference does not need to be perfectly physically feasible; RL adapts to find feasible solutions nearby

---

## Reference Clip Format

All clips saved to `data/reference_clips/` as `.npz`:

```python
{
  "joint_pos":    float32[T, 10],  # absolute joint positions (rad)
  "joint_vel":    float32[T, 10],  # joint velocities (rad/s)
  "base_pos":     float32[T, 3],   # world-frame XYZ
  "base_quat":    float32[T, 4],   # [w, x, y, z]
  "base_lin_vel": float32[T, 3],   # world-frame linear velocity
  "base_ang_vel": float32[T, 3],   # body-frame angular velocity
  "dt":           float,           # seconds per frame
  "source":       str,             # "rollout" | "keyframe" | "trajopt"
  "skill":        str,             # "walk" | "getup" | "squat"
  "loop":         bool,            # True = cyclic, False = one-shot
}
```

---

## Data Collection Plan

### Walking clips — collect from existing policy (trivial, do first)

Script: `deploy/collect_reference.py`

Run existing policy in MuJoCo sim2sim, record full state at policy frequency (67 Hz).

| Command | Duration | Filename |
|---------|----------|----------|
| vx=0.3, vy=0, yaw=0 | 6 sec | `walk_fwd_slow.npz` |
| vx=0.5, vy=0, yaw=0 | 6 sec | `walk_fwd_fast.npz` |
| vx=0, vy=0.2, yaw=0 | 6 sec | `walk_strafe.npz` |
| vx=0.3, vy=0, yaw=0.5 | 6 sec | `walk_turn.npz` |

Walking is cyclic (`loop=True`). 1-2 full strides is enough — 6 sec is generous.
These clips also validate the MIRL pipeline before touching harder skills.

---

### Squat clip — manual keyframes (~2 hours)

Script: `deploy/make_squat_reference.py`

Define 3 poses in joint space using the Qmini URDF:
- `pose_stand`: current `ref_joint_pos` from config
- `pose_mid`: knee/hip bent to ~0.38m base height
- `pose_deep`: knee/hip bent to ~0.33m base height

Linearly interpolate over ~2 sec per transition. Compute base height from FK.
Save as `loop=True` (stand → squat → stand cycles).

RL does not need to match exactly — the reference just seeds RSI at intermediate heights.

---

### Get-up clip — keyframes or trajectory optimization (1-2 days)

#### Option A: Manual keyframes (try first)

Define 6 poses:
```
pose_0: flat on back     (base height ~0.08m, joints near zero)
pose_1: knees bent up    (feet flat, hips flexed)
pose_2: side roll prone
pose_3: kneeling         (hip height ~0.25m)
pose_4: crouch           (feet placed, ready to stand)
pose_5: stand            (= pose_stand)
```

Interpolate with variable timing. `loop=False`.

#### Option B: Crocoddyl trajectory optimization (if option A fails after 2000 iters)

Use Qmini URDF in Crocoddyl whole-body OCP. Solve minimum-torque trajectory from flat→stand with explicit contact schedule. Output is physically consistent.

Qmini has no arms → get-up is significantly simpler than full humanoid.

---

### Stairs clip — skip, use terrain curriculum instead

Stair climbing works well with pure terrain curriculum in birl_task without MIRL reference.
Only generate a reference if curriculum alone fails to discover correct stepping pattern.

---

## Command Vector

```
obs[0:8] = [cmd_vx, cmd_vy, cmd_yaw, cmd_height, 0, 0, 0, 0]
```

**8 command slots are always present in the obs vector, unused ones padded with zero.**

This is a forward-compatibility design: when a new command is added in the future, the obs dimension does not change — the checkpoint loads cleanly and training resumes. The policy initially ignores the new slot (it was always zero during pretraining), then gradually learns to use it as the new reward term shapes the gradient.

Active now:
| Slot | Command | Range |
|------|---------|-------|
| 0 | `cmd_vx` | [-0.3, 0.7] |
| 1 | `cmd_vy` | [-0.3, 0.3] |
| 2 | `cmd_yaw` | [-1.0, 1.0] |
| 3 | `cmd_height` | [0.33, 0.50] |
| 4-7 | reserved (zero) | — |

To activate a reserved slot in future training: set its range in config, add reward term, resume checkpoint. No architecture change, no obs dimension change.

Considered and rejected for active slots:
- `cmd_step_height`: not needed — stair terrain forces correct foot clearance naturally
- `cmd_step_freq`: not needed — policy learns optimal cadence from velocity; reference clip encodes it implicitly
- `cmd_gait` (discrete): not needed — `cmd_height` + terrain type is sufficient

---

## MIRLTask Design

### Observation vector (64 dims per step × 3 history = 192 total)

| Index | Content | Notes |
|-------|---------|-------|
| 0-7 | `[vx, vy, yaw, height, 0, 0, 0, 0]` | 8 command slots, 4 reserved as zero |
| 8-9 | `[roll, pitch]` | |
| 10-12 | angular velocity × 0.5 | |
| 13-22 | joint_pos − default_joint_pos | |
| 23-32 | joint_vel × 0.1 | |
| 33-42 | joint_act − joint_pos | |
| 43-52 | **ref_joint_pos[t] − joint_pos** | replaces phase modulator |
| 53-62 | **ref_joint_vel[t]** | reference velocity target |
| 63 | **phase_progress** (0→1) | position in the clip |

The phase modulator (sin/cos/freq) is replaced by explicit reference frame info.
This works for both cyclic and one-shot motions.

### Action space (10 dims)

```
output: Δjoint_pos [10]   (position increment, same scale as birl_task)
```

No leg frequency outputs — gait timing is encoded in the reference clip + phase_progress obs.

### RSI — Reference State Initialization

```python
def reset_env(env_id):
    if clip.loop:
        start_frame = random.randint(0, len(clip) - 1)
    else:
        # One-shot curriculum: start early frames first, unlock later frames as training progresses
        max_frame = int(training_progress * len(clip))
        start_frame = random.randint(0, max_frame)

    # Initialize from reference frame + small noise
    data.qpos[7:17] = clip.joint_pos[start_frame] + noise(0.02)
    data.qvel[6:16] = clip.joint_vel[start_frame] + noise(0.1)
    data.base_pos   = clip.base_pos[start_frame]
    data.base_quat  = clip.base_quat[start_frame]
```

For get-up, the one-shot RSI curriculum is critical: policy first learns the initial motion from lying flat, progressively learns to continue from later stages.

### Reward structure

```python
# Task rewards (same as birl_task where applicable)
velocity_reward    # cmd tracking
height_reward      # base height target
balance_reward     # roll/pitch penalty

# Imitation rewards (new)
joint_pos_imit  = exp(-5.0  * ||joint_pos - ref_joint_pos[t]||²)
joint_vel_imit  = exp(-0.1  * ||joint_vel - ref_joint_vel[t]||²)
base_orient_imit = exp(-10.0 * ||quat_diff(base_quat, ref_quat[t])||²)

total = w_task * task_reward + w_imit * imitation_reward
```

Weight schedule:
- Iterations 0-500:   `w_imit=0.8`, `w_task=0.2`  — learn to follow reference first
- Iterations 500-2000: decay `w_imit` → 0.3, raise `w_task` → 0.7
- After 2000: `w_imit=0.2`, `w_task=0.8`  — optimize actual task

---

## Training Phases

### Phase 0 — Stair curriculum on existing birl_task (1-2 weeks)

No new code. Extend terrain generator in existing training setup.

- Add `StairTerrain` to terrain types
- Rise curriculum: 3cm → 18cm, step width 25-30cm
- Resume from `sidewalk_v2` checkpoint when ready
- Evaluate every 500 iters: must not regress on flat terrain

Target: policy handles flat + moderate stairs (≤18cm rise).
Deliverable: `stair_v1` checkpoint.

---

### Phase 1 — Reference collection pipeline (3-5 days)

New file: `deploy/collect_reference.py`

- Reuse sim2sim infrastructure (MuJoCo model, obs builder)
- Run policy for N seconds, record `(qpos, qvel, xquat, xpos)` at each policy step
- Save `.npz` per command configuration

Collect all walking clips from `stair_v1` (or `sidewalk_v2`) policy.

Deliverable: `data/reference_clips/walk_*.npz`

---

### Phase 2 — MIRLTask + walking validation (1 week)

New file: `env/tasks/mirl_task.py`

- Inherit from `BaseTask`
- 64-dim obs vector always (8 command slots, ref slots, phase_progress)
- 10-dim action space (no frequencies — phase modulator dropped)
- **Reference clip is optional**:
  - `ref_clip=None` → ref obs slots (43-63) set to zero, no imitation reward, no RSI — pure task RL
  - `ref_clip=path` → ref slots populated from clip, imitation reward active, RSI enabled

Training sequence exploiting this:

**Step 2a — train without reference (pure RL):**
```
MIRLTask(ref_clip=None) → mirl_base_v1
```
Policy learns walking with new 64-dim obs format using only task reward.
Reference slots are always zero — policy learns to ignore them.

**Step 2b — collect reference from mirl_base_v1:**
```
sim2sim --policy mirl_base_v1 --record data/reference_clips/walk_fwd.npz --record_loop
```

**Step 2c — fine-tune with reference (MIRL proper):**
```
MIRLTask(ref_clip=walk_fwd.npz) → mirl_walk_v1  (resume from mirl_base_v1)
```
Reference slots now populated — policy already knows how to walk, fine-tunes toward reference style.

Also update `deploy/sim2sim/sim2sim.py` for MIRL policy compatibility:
- Auto-detect from ONNX output dim: `num_actions==12` → birl_task mode, `num_actions==10` → MIRL mode
- MIRL mode: skip `pm.compute()`, use all 10 outputs as joint increments
- MIRL mode: replace phase modulator obs terms with ref slots (zeros if no clip provided)
- Optional `--ref_clip` arg for MIRL sim2sim runs

Deliverable: `mirl_base_v1` (no ref), reference clips, `mirl_walk_v1` (with ref), sim2sim updated.

---

### Phase 3 — Get-up policy (2-4 weeks)

1. Generate get-up reference clip (keyframes → Crocoddyl if needed)
2. Train `MIRLTask` with get-up clip + RSI one-shot curriculum
3. Termination logic: only terminate on fall after `phase_progress > 0.8`
4. Validate in sim2sim with `--stand_only` first (confirm reference clip looks correct), then full policy

Tuning knobs:
- RSI curriculum speed (`max_frame` growth rate)
- `w_imit` decay schedule
- Early termination threshold

Deliverable: `getup_v1` policy, sim2sim demo.

---

### Phase 4 — Squat policy (1-2 weeks)

1. Generate squat reference clip (keyframes)
2. Add height command to obs: `cmd_height` replaces or augments existing height reward
3. Train `MIRLTask` with cyclic squat clip
4. Validate height command tracking in sim2sim

Deliverable: `squat_v1` policy with `cmd_height ∈ [0.33, 0.50]`.

---

### Phase 5 — MoE distillation (2-3 weeks)

Gate network:

```python
gate_input = concat([is_fallen,        # bool: base_height < 0.2m
                     base_height,       # current height
                     cmd_norm,          # ||[vx, vy, yaw]||
                     terrain_type])     # flat / stair / ...
gate_output = softmax([w_walk, w_getup, w_squat])
policy_output = sum(w_i * expert_i(obs))
```

Training:
1. Freeze all experts
2. Train gate with RL on combined environment (random fall resets, mixed terrain)
3. Optionally unfreeze and fine-tune end-to-end for 500-1000 iters

If gate learning is unstable: hard-code gate first (`is_fallen → getup`, else → walk/squat based on `cmd_height`), then learn soft weights on top.

Deliverable: `unified_v1` — single policy handles walk + getup + squat + stairs.

---

## File Structure

```
data/
  reference_clips/
    walk_fwd_slow.npz
    walk_fwd_fast.npz
    walk_strafe.npz
    walk_turn.npz
    squat.npz
    getup.npz

deploy/
  collect_reference.py     # record clips from sim2sim rollouts
  make_squat_reference.py  # generate squat clip from keyframes
  make_getup_reference.py  # generate get-up clip (keyframes or Crocoddyl)

env/tasks/
  birl_task.py             # unchanged — walking experts trained here
  mirl_task.py             # new — get-up, squat, MIRL walking trained here

env/utils/
  reference_loader.py      # load .npz clips, handle RSI sampling, frame stepping
  moe_policy.py            # gate network + expert mixing for distillation
```

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Get-up keyframes too crude, policy can't follow | Medium | Generate with Crocoddyl if >2000 iters without progress |
| MIRLTask walking worse than birl_task | Low | Keep birl_task for walking expert; MIRL walking is validation only |
| MoE gate fails to select correct expert | Medium | Hard-code gate logic first, learn soft weights later |
| Stair curriculum degrades flat walking | Low | Evaluate flat terrain every 500 iters, stop if regression >10% |
| RSI causes unstable resets (robot spawns mid-air) | Medium | Add `mj_forward` + collision check after RSI; reject invalid states |

---

## What Not to Build

- Do not retarget human MoCap — keyframes + trajectory optimization are sufficient for Qmini's DOF and avoid retargeting complexity
- Do not use AMP/ASE — naturalness is not a goal; the robot doesn't need human-like motion style
- Do not train all skills jointly from scratch — always train individual experts first, distill second
- Do not modify birl_task for MIRL — keep it stable for the walking experts

---

## Current Status (April 2026)

| Item | Status |
|------|--------|
| Walking (vx/vy/yaw) | Working — `sidewalk_v2` training |
| Sidewalk reward fixes | Done — `sidewalk_v2` in progress |
| Stair curriculum | Not started — Phase 0 |
| Reference collection script | Not started — Phase 1 |
| MIRLTask | Not started — Phase 2 |
| Get-up policy | Not started — Phase 3 |
| Squat policy | Not started — Phase 4 |
| MoE distillation | Not started — Phase 5 |
