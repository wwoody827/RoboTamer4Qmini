"""
Verify a recovery init pool is physically consistent:
  1. Initial penetration: after mj_resetData + qpos set + mj_forward, what's
     the deepest contact distance? (negative = penetration; should be ≥0)
  2. PD-to-ref drift: from each sampled state, run PD pulling joints to
     ref_jp for 1 second. Did the state stay near init? (z drift, tilt growth)
  3. Optional viewer: --viewer flag → step through samples in MuJoCo viewer.

Usage:
    python scripts/verify_init_pool.py --pool data/balance_init_states.npz --n 50
    python scripts/verify_init_pool.py --pool data/balance_init_states.npz --viewer --n 5
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'deploy', 'sim2sim'))
from sim2sim import build_mujoco_model


def _kp_kd_ref():
    base = yaml.safe_load(open(os.path.join(_REPO, 'configs', 'base.yaml')))
    pd = base['pd_gains']
    order = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle']
    kp = np.array([pd['stiffness'][k] for k in order] * 2, dtype=np.float32)
    kd = np.array([pd['damping'][k]   for k in order] * 2, dtype=np.float32)
    ref = np.array(base['action']['ref_joint_pos'], dtype=np.float32)
    return kp, kd, ref, int(pd['decimation'])


def _tilt_deg(q_wxyz):
    qx, qy = float(q_wxyz[1]), float(q_wxyz[2])
    cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
    return float(np.degrees(np.arccos(cos_tilt)))


def _set_state(data, s):
    data.qpos[0:3]  = s['base_pos']
    data.qpos[3:7]  = s['base_quat']
    data.qpos[7:17] = s['joint_pos']
    data.qvel[0:3]  = s['base_lin_vel']
    data.qvel[3:6]  = s['base_ang_vel']
    data.qvel[6:16] = s['joint_vel']


def _max_penetration(data):
    """Most-negative contact dist after mj_forward (positive return = penetration depth)."""
    pen = 0.0
    for ci in range(data.ncon):
        d = float(data.contact[ci].dist)
        if d < 0.0:
            pen = max(pen, -d)
    return pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--urdf', default=os.path.join(_REPO, 'assets/q1/urdf/q1.urdf'))
    ap.add_argument('--n', type=int, default=50, help='samples to check')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--settle_s', type=float, default=1.0,
                    help='PD-to-ref settle duration')
    ap.add_argument('--viewer', action='store_true',
                    help='Show samples in MuJoCo viewer (one at a time)')
    ap.add_argument('--viewer_hold_s', type=float, default=2.0)
    args = ap.parse_args()

    pool = np.load(args.pool, allow_pickle=False)
    N = pool['base_pos'].shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=min(args.n, N), replace=False)

    labels = pool['pose_label'] if 'pose_label' in pool.files else np.array(['?'] * N)
    print(f"[verify] pool {args.pool}  {N} states, sampling {len(idx)}")
    unique, counts = np.unique(labels[idx], return_counts=True)
    print(f"[verify] sampled labels: {dict(zip(unique.tolist(), counts.tolist()))}")

    sim_dt = 1.0 / 1000.0
    kp, kd, ref_jp, _ = _kp_kd_ref()
    model = build_mujoco_model(args.urdf, sim_dt, init_height=0.5,
                               floor_friction=1.0, fix_base=False)
    data = mujoco.MjData(model)
    effort = np.abs(model.actuator_forcerange[:10, 1]).astype(np.float32)
    n_settle = int(args.settle_s / sim_dt)

    init_pen   = []
    init_z     = []
    init_tilt  = []
    final_z    = []
    final_tilt = []
    z_drift    = []
    tilt_drift = []
    fell_over  = []  # tilt > 80° after settle

    for k, i in enumerate(idx):
        s = {key: pool[key][i] for key in
             ('base_pos', 'base_quat', 'base_lin_vel', 'base_ang_vel',
              'joint_pos', 'joint_vel')}
        mujoco.mj_resetData(model, data)
        _set_state(data, s)
        mujoco.mj_forward(model, data)

        pen0 = _max_penetration(data)
        z0   = float(data.qpos[2])
        t0   = _tilt_deg(data.qpos[3:7])
        init_pen.append(pen0); init_z.append(z0); init_tilt.append(t0)

        for _ in range(n_settle):
            q  = data.qpos[7:17]
            dq = data.qvel[6:16]
            tau = kp * (ref_jp - q) - kd * dq
            tau = np.clip(tau, -effort, effort)
            data.ctrl[:10] = tau
            mujoco.mj_step(model, data)

        z1 = float(data.qpos[2])
        t1 = _tilt_deg(data.qpos[3:7])
        final_z.append(z1); final_tilt.append(t1)
        z_drift.append(z1 - z0); tilt_drift.append(t1 - t0)
        # Treat as "fell over" only if tilt grew dramatically *during* the
        # window — pools may legitimately contain fallen poses (tilt > 80° at
        # init), and we want to flag instability, not the starting pose.
        fell_over.append((t1 - t0) > 30.0)

    init_pen   = np.array(init_pen)
    init_z     = np.array(init_z)
    init_tilt  = np.array(init_tilt)
    final_z    = np.array(final_z)
    final_tilt = np.array(final_tilt)
    z_drift    = np.array(z_drift)
    tilt_drift = np.array(tilt_drift)

    def stats(name, arr, fmt='.3f'):
        print(f"  {name:>16s}: mean {arr.mean():{fmt}}  "
              f"min {arr.min():{fmt}}  max {arr.max():{fmt}}")

    print("\n[verify] init contact (mj_forward, no step):")
    stats('penetration (m)', init_pen, '.4f')
    n_pen = int((init_pen > 1e-5).sum())
    print(f"    samples with penetration > 0.01mm: {n_pen}/{len(init_pen)}")

    print(f"\n[verify] init pose:")
    stats('z (m)',  init_z)
    stats('tilt (°)', init_tilt, '.1f')

    print(f"\n[verify] after {args.settle_s}s PD-to-ref settle:")
    stats('z (m)',     final_z)
    stats('tilt (°)',  final_tilt, '.1f')
    stats('z drift',   z_drift, '.3f')
    stats('tilt drift', tilt_drift, '.1f')
    n_fell = int(np.sum(fell_over))
    print(f"    samples falling over (tilt > 80° after settle): {n_fell}/{len(idx)}")

    # Verdict
    print("\n[verify] verdict:")
    if init_pen.max() < 1e-5:
        print("  ✓ no initial penetration")
    else:
        print(f"  ✗ {n_pen} samples have initial penetration (max {init_pen.max():.4f} m)")
    if n_fell == 0:
        print(f"  ✓ none fell over within {args.settle_s}s of PD-to-ref")
    else:
        print(f"  ⚠ {n_fell} samples fell over within {args.settle_s}s")

    if args.viewer:
        print(f"\n[verify] launching viewer for {len(idx)} samples "
              f"({args.viewer_hold_s}s each, Ctrl+C to skip)…")
        try:
            from mujoco import viewer as mj_viewer
        except ImportError:
            print("  mujoco.viewer not available; install mujoco>=2.3.4 with viewer support")
            return
        with mj_viewer.launch_passive(model, data) as viewer:
            for k, i in enumerate(idx):
                s = {key: pool[key][i] for key in
                     ('base_pos', 'base_quat', 'base_lin_vel', 'base_ang_vel',
                      'joint_pos', 'joint_vel')}
                mujoco.mj_resetData(model, data)
                _set_state(data, s)
                mujoco.mj_forward(model, data)
                t_end = time.time() + args.viewer_hold_s
                while time.time() < t_end and viewer.is_running():
                    q  = data.qpos[7:17]; dq = data.qvel[6:16]
                    tau = np.clip(kp * (ref_jp - q) - kd * dq, -effort, effort)
                    data.ctrl[:10] = tau
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(sim_dt)
                if not viewer.is_running():
                    break


if __name__ == '__main__':
    main()
