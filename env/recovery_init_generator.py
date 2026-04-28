"""
Recovery init-state generator (Phase 2).

Generates a dataset of *settled fallen poses* for the recovery RL policy. Rather
than synthesizing fall poses by hand (joint angles guessed by a human), we let
MuJoCo physics produce realistic ones:

    1. Spawn the robot in nominal standing pose.
    2. Apply a random external impulse (push) at the base for a short window,
       while a PD controller tries to hold the standing pose.
    3. Continue stepping the simulation until the robot has settled
       (base velocities below threshold) or a max-time cap is hit.
    4. Classify the resulting pose (prone / supine / side_L / side_R /
       kneeling / standing) from base orientation and height.
    5. Discard "standing" outcomes (pushes too weak), keep the rest.
    6. Save a single .npz with stacked arrays.

The output file is consumed by RecoveryTask (Phase 3) at episode reset:
    sample_idx ~ Uniform({0..N-1})
    set qpos/qvel from the snapshot, optionally with small jitter.

Usage:
    python env/recovery_init_generator.py
    python env/recovery_init_generator.py --num_states 200 --render
    python env/recovery_init_generator.py --out data/recovery_init_states.npz
"""

import argparse
import os
import sys
import time

import numpy as np
import mujoco

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'deploy', 'sim2sim'))

from sim2sim import build_mujoco_model

URDF_PATH = os.path.join(_REPO, 'assets', 'q1', 'urdf', 'q1.urdf')
DEFAULT_OUT = os.path.join(_REPO, 'data', 'recovery_init_states.npz')

# Joint order: hip_yaw, hip_roll, hip_pitch, knee, ankle (L then R). From configs/base.yaml.
NOMINAL_JOINT_POS = np.array([0.4, -0.1, -1.5, 1.0, -1.3, -0.4, 0.1, 1.5, -1.0, 1.3], dtype=np.float64)
KP = np.array([55., 105., 75., 45., 30., 55., 105., 75., 45., 30.], dtype=np.float64)
KD = np.array([0.3, 2.5, 0.3, 0.5, 0.25, 0.3, 2.5, 0.3, 0.5, 0.25], dtype=np.float64)
NOMINAL_HEIGHT = 0.45

# Classification thresholds.
TILT_UPRIGHT_DEG = 25.0       # below this → standing-ish
HEIGHT_FALLEN_FRAC = 0.6      # base z below this fraction of nominal → fallen
SIDE_TILT_MIN_DEG = 60.0      # min tilt to classify as side / prone / supine

# Settling thresholds (base 6dof velocity).
SETTLE_LIN_VEL = 0.10         # m/s
SETTLE_ANG_VEL = 0.50         # rad/s


def rpy_to_wxyz(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def base_tilt_deg(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    body_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return float(np.degrees(np.arccos(np.clip(body_z_world_z, -1.0, 1.0))))


def world_axis_in_body(quat_wxyz):
    """Express world +Z in body frame. Sign of body-x and body-y components
    reveals which side the robot is rolled toward / which way pitched."""
    qw, qx, qy, qz = quat_wxyz
    bx = 2.0 * (qx * qz - qw * qy)
    by = 2.0 * (qy * qz + qw * qx)
    bz = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.array([bx, by, bz], dtype=np.float64)


def classify_pose(quat_wxyz, base_z):
    """Return one of: 'standing', 'prone', 'supine', 'side_left', 'side_right',
    'kneeling', 'unknown'.

    Uses world-Z direction expressed in body frame:
        body_z > 0  → robot upright-ish (head up)
        body_z < 0  → robot upside down
        body_x dominant → fell forward (prone) or backward (supine on back)
        body_y dominant → fell on a side
    """
    tilt = base_tilt_deg(quat_wxyz)
    if tilt < TILT_UPRIGHT_DEG and base_z > HEIGHT_FALLEN_FRAC * NOMINAL_HEIGHT:
        return 'standing'

    bx, by, bz = world_axis_in_body(quat_wxyz)

    if tilt < SIDE_TILT_MIN_DEG:
        # Tilted but not flat. Likely kneeling / sitting / awkward.
        return 'kneeling'

    # Strongly tilted. Decide by which body axis points up.
    ax, ay = abs(bx), abs(by)
    if ax > ay:
        # forward/backward fall (pitch dominated)
        # bx > 0 → world +Z is along +body-x → robot rotated so back faces up → prone (face down)
        return 'prone' if bx > 0 else 'supine'
    else:
        # roll dominated
        return 'side_right' if by > 0 else 'side_left'


def reset_to_standing(model, data, rng, jitter=True):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:2] = 0.0
    data.qpos[2] = NOMINAL_HEIGHT
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7:17] = NOMINAL_JOINT_POS
    if jitter:
        data.qpos[7:17] += rng.uniform(-0.02, 0.02, size=10)
    mujoco.mj_forward(model, data)


def step_pd(model, data, target, n_steps):
    qpos_idx = slice(7, 17)
    qvel_idx = slice(6, 16)
    for _ in range(n_steps):
        q = data.qpos[qpos_idx]
        qd = data.qvel[qvel_idx]
        data.ctrl[:] = KP * (target - q) - KD * qd
        mujoco.mj_step(model, data)


def step_pd_with_impulse(model, data, target, n_steps, force_xyz, torque_xyz):
    """Same as step_pd, but also applies an external wrench at the base (qfrc_applied)."""
    qpos_idx = slice(7, 17)
    qvel_idx = slice(6, 16)
    for _ in range(n_steps):
        q = data.qpos[qpos_idx]
        qd = data.qvel[qvel_idx]
        data.ctrl[:] = KP * (target - q) - KD * qd
        data.qfrc_applied[0:3] = force_xyz
        data.qfrc_applied[3:6] = torque_xyz
        mujoco.mj_step(model, data)
    data.qfrc_applied[0:6] = 0.0


def base_velocity_norm(data):
    lin = float(np.linalg.norm(data.qvel[0:3]))
    ang = float(np.linalg.norm(data.qvel[3:6]))
    return lin, ang


def is_settled(data):
    lin, ang = base_velocity_norm(data)
    return lin < SETTLE_LIN_VEL and ang < SETTLE_ANG_VEL


def sample_impulse(rng, force_min=20.0, force_max=80.0, torque_min=0.0, torque_max=15.0):
    """Random push: force in horizontal plane + small free torque."""
    theta = rng.uniform(0, 2 * np.pi)
    fmag = rng.uniform(force_min, force_max)
    fxy = np.array([np.cos(theta), np.sin(theta)]) * fmag
    fz = rng.uniform(-10.0, 10.0)  # mild vertical kick
    force = np.array([fxy[0], fxy[1], fz], dtype=np.float64)
    torque = rng.uniform(-torque_max, torque_max, size=3)
    if torque_min > 0:
        # ensure at least some yaw component sometimes
        pass
    return force, torque


def generate_one(model, data, rng, viewer=None,
                 push_dt=0.15, settle_max_t=2.5,
                 force_min=20.0, force_max=80.0):
    """Run one push-and-settle episode. Returns (snapshot_dict, label) or (None, 'standing'/'unstable')."""
    sim_dt = model.opt.timestep
    push_steps = max(1, int(push_dt / sim_dt))
    pre_relax_steps = max(1, int(0.1 / sim_dt))
    chunk = max(1, int(0.05 / sim_dt))
    settle_max_steps = int(settle_max_t / sim_dt)

    reset_to_standing(model, data, rng)

    # let the robot relax onto the ground briefly
    step_pd(model, data, NOMINAL_JOINT_POS, pre_relax_steps)

    force, torque = sample_impulse(rng, force_min=force_min, force_max=force_max)
    step_pd_with_impulse(model, data, NOMINAL_JOINT_POS, push_steps, force, torque)

    # Settle.
    elapsed = 0
    settled = False
    while elapsed < settle_max_steps:
        n = min(chunk, settle_max_steps - elapsed)
        step_pd(model, data, NOMINAL_JOINT_POS, n)
        elapsed += n
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break
        # Only check settling once base z is plausibly low (avoid false "settled" mid-fall).
        if elapsed > int(0.4 / sim_dt) and is_settled(data):
            settled = True
            break

    quat = np.array(data.qpos[3:7], dtype=np.float64)
    base_z = float(data.qpos[2])
    label = classify_pose(quat, base_z)

    if not settled and label == 'standing':
        # likely never tipped; reject.
        return None, 'standing'

    snap = dict(
        base_pos=np.array(data.qpos[0:3], dtype=np.float64),
        base_quat=quat,
        base_lin_vel=np.array(data.qvel[0:3], dtype=np.float64),
        base_ang_vel=np.array(data.qvel[3:6], dtype=np.float64),
        joint_pos=np.array(data.qpos[7:17], dtype=np.float64),
        joint_vel=np.array(data.qvel[6:16], dtype=np.float64),
        impulse=np.concatenate([force, torque]).astype(np.float64),
        settle_time=float(elapsed * sim_dt),
        settled=bool(settled),
    )
    return snap, label


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--num_states', type=int, default=2000,
                    help='target number of valid (non-standing) snapshots to save')
    ap.add_argument('--max_attempts_factor', type=float, default=3.0,
                    help='give up after num_states * factor attempts')
    ap.add_argument('--out', type=str, default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sim_dt', type=float, default=0.002)
    ap.add_argument('--force_min', type=float, default=20.0)
    ap.add_argument('--force_max', type=float, default=80.0)
    ap.add_argument('--push_dt', type=float, default=0.15)
    ap.add_argument('--settle_max_t', type=float, default=2.5)
    ap.add_argument('--render', action='store_true', help='launch passive viewer (slow)')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[recovery_init_gen] URDF={URDF_PATH}")
    print(f"[recovery_init_gen] target={args.num_states}, force=[{args.force_min},{args.force_max}]N, "
          f"push_dt={args.push_dt}s, settle_max={args.settle_max_t}s")
    print(f"[recovery_init_gen] out={args.out}")

    model = build_mujoco_model(URDF_PATH, sim_dt=args.sim_dt, init_height=NOMINAL_HEIGHT, fix_base=False)
    data = mujoco.MjData(model)

    viewer = None
    if args.render:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    keep_labels = ('prone', 'supine', 'side_left', 'side_right', 'kneeling')
    snaps = []
    label_counts = {k: 0 for k in keep_labels}
    label_counts['standing'] = 0
    label_counts['unknown'] = 0

    max_attempts = int(args.num_states * args.max_attempts_factor)
    t0 = time.time()
    attempt = 0
    try:
        while len(snaps) < args.num_states and attempt < max_attempts:
            attempt += 1
            snap, label = generate_one(
                model, data, rng, viewer=viewer,
                push_dt=args.push_dt,
                settle_max_t=args.settle_max_t,
                force_min=args.force_min,
                force_max=args.force_max,
            )
            label_counts[label] = label_counts.get(label, 0) + 1
            if snap is None:
                continue
            if label not in keep_labels:
                continue
            snap['pose_label'] = label
            snaps.append(snap)
            if len(snaps) % 50 == 0 or len(snaps) == 1:
                elapsed = time.time() - t0
                rate = len(snaps) / max(elapsed, 1e-6)
                print(f"  collected {len(snaps)}/{args.num_states} "
                      f"(attempts={attempt}, {rate:.1f}/s) "
                      f"counts={ {k:v for k,v in label_counts.items() if v>0} }")
            if viewer is not None and not viewer.is_running():
                break
    finally:
        if viewer is not None:
            viewer.close()

    if not snaps:
        print("[recovery_init_gen] ERROR: no snapshots collected")
        return 1

    # Stack into arrays.
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    arr = lambda key: np.stack([s[key] for s in snaps], axis=0)
    np.savez_compressed(
        args.out,
        base_pos=arr('base_pos'),
        base_quat=arr('base_quat'),
        base_lin_vel=arr('base_lin_vel'),
        base_ang_vel=arr('base_ang_vel'),
        joint_pos=arr('joint_pos'),
        joint_vel=arr('joint_vel'),
        impulse=arr('impulse'),
        settle_time=np.array([s['settle_time'] for s in snaps], dtype=np.float64),
        settled=np.array([s['settled'] for s in snaps], dtype=bool),
        pose_label=np.array([s['pose_label'] for s in snaps], dtype='U16'),
        # bookkeeping
        meta_force_min=args.force_min,
        meta_force_max=args.force_max,
        meta_push_dt=args.push_dt,
        meta_settle_max_t=args.settle_max_t,
        meta_seed=args.seed,
        meta_attempts=attempt,
    )

    elapsed = time.time() - t0
    print(f"\n[recovery_init_gen] saved {len(snaps)} snapshots to {args.out}  ({elapsed:.1f}s, {attempt} attempts)")
    print(f"[recovery_init_gen] label distribution:")
    for k, v in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"    {k:14s} {v:5d}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
