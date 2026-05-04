"""
Generate a balance-init pool: standing reference pose perturbed by small
random offsets, then settled in MuJoCo with PD pulling toward the standing
reference. Output is physically self-consistent "almost-standing-but-tipped"
states for Phase 1 of the two-stage recovery curriculum.

Why: pure-fallen init kills credit assignment for quasi-static stand-up.
ANYmal recovery (Lee 2020) and humanoid work (HumanPlus, OmniH2O) all
train balance-from-perturbation first, then expand toward fallen poses.

Usage (defaults are what Phase 1 needs):
    python scripts/extract_balance_pool.py --n 1500 \
        --out data/balance_init_states.npz

Tilted variant for the mixed pool (Phase 2):
    python scripts/extract_balance_pool.py --n 1500 \
        --out data/tilted_init_states.npz \
        --tilt_max 30 --jp_jitter 0.3 --av_jitter 2.0 \
        --reject_z 0.25 --reject_tilt 60

Output schema matches data/recovery_init_states.npz exactly so the existing
RecoveryTask._load_init_pool needs no change. pose_label is set to
'balance' (small) or 'tilted' (medium) — auto-derived from --tilt_max.
"""
import argparse
import os
import sys

import numpy as np
import mujoco
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'deploy', 'sim2sim'))
from sim2sim import build_mujoco_model


def _kp_kd_ref_from_base():
    """Read PD gains + standing reference from configs/base.yaml so this
    script stays in sync with training."""
    base_path = os.path.join(_REPO, 'configs', 'base.yaml')
    with open(base_path) as f:
        base = yaml.safe_load(f)
    pd = base['pd_gains']
    order = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle']
    kp_one = [pd['stiffness'][k] for k in order]
    kd_one = [pd['damping'][k]   for k in order]
    kps = np.array(kp_one + kp_one, dtype=np.float32)  # L + R
    kds = np.array(kd_one + kd_one, dtype=np.float32)
    ref = np.array(base['action']['ref_joint_pos'], dtype=np.float32)
    decimation = int(pd['decimation'])
    return kps, kds, ref, decimation


def _euler_to_quat_wxyz(roll, pitch, yaw=0.0):
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2),  np.sin(roll / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


def _tilt_deg_from_quat(q_wxyz):
    qx, qy = float(q_wxyz[1]), float(q_wxyz[2])
    cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
    return np.degrees(np.arccos(cos_tilt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=os.path.join(_REPO, 'assets/q1/urdf/q1.urdf'))
    ap.add_argument('--out',  default=os.path.join(_REPO, 'data/balance_init_states.npz'))
    ap.add_argument('--n',    type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label', default=None,
                    help="pose_label for these states; default auto from tilt_max "
                         "(<=15 → 'balance', else 'tilted')")
    ap.add_argument('--settle_steps', type=int, default=50,
                    help="MuJoCo physics steps to settle after randomization")
    # Perturbation magnitudes — small defaults are Phase 1 (balance pool)
    ap.add_argument('--jp_jitter',   type=float, default=0.10, help="rad, joint pos perturb")
    ap.add_argument('--jv_jitter',   type=float, default=0.5,  help="rad/s, joint vel perturb")
    ap.add_argument('--tilt_max',    type=float, default=10.0, help="deg, base roll/pitch perturb")
    ap.add_argument('--av_jitter',   type=float, default=1.0,  help="rad/s, base ang_vel")
    ap.add_argument('--lv_jitter',   type=float, default=0.3,  help="m/s, base lin_vel")
    ap.add_argument('--z_jitter',    type=float, default=0.02, help="m, base height perturb")
    ap.add_argument('--base_z',      type=float, default=0.45, help="nominal standing height")
    ap.add_argument('--clearance',   type=float, default=0.01,
                    help="m, lift base so lowest robot geom is at least this above floor "
                         "before settling — avoids initial interpenetration giving the "
                         "constraint solver an unmodeled correction impulse")
    # Reject filters after settling — drop samples that are too far from upright
    ap.add_argument('--reject_z',    type=float, default=0.40, help="min final z to keep")
    ap.add_argument('--reject_tilt', type=float, default=15.0, help="max final tilt deg to keep")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    kps, kds, ref_jp, decimation = _kp_kd_ref_from_base()
    sim_dt = 1.0 / 1000.0  # base.yaml dt; settle at full sim rate

    label = args.label or ('balance' if args.tilt_max <= 15.0 else 'tilted')
    print(f"[balance] target {args.n} states, label='{label}', "
          f"settle={args.settle_steps} steps")
    print(f"[balance] perturb: jp±{args.jp_jitter} jv±{args.jv_jitter} "
          f"tilt±{args.tilt_max}° av±{args.av_jitter} lv±{args.lv_jitter}")
    print(f"[balance] reject: z<{args.reject_z} or tilt>{args.reject_tilt}°")

    model = build_mujoco_model(args.urdf, sim_dt, init_height=0.5,
                               floor_friction=1.0, fix_base=False)
    data = mujoco.MjData(model)
    effort = np.abs(model.actuator_forcerange[:10, 1]).astype(np.float32)

    saved = {k: [] for k in ('base_pos', 'base_quat', 'base_lin_vel',
                              'base_ang_vel', 'joint_pos', 'joint_vel')}
    saved['pose_label'] = []

    n_attempts = 0
    n_rejected_z = 0
    n_rejected_tilt = 0
    while len(saved['base_pos']) < args.n:
        n_attempts += 1
        mujoco.mj_resetData(model, data)
        # Randomize root and joint state around standing.
        z = args.base_z + rng.uniform(-args.z_jitter, args.z_jitter)
        roll  = np.deg2rad(rng.uniform(-args.tilt_max, args.tilt_max))
        pitch = np.deg2rad(rng.uniform(-args.tilt_max, args.tilt_max))
        data.qpos[0:3] = [0.0, 0.0, z]
        data.qpos[3:7] = _euler_to_quat_wxyz(roll, pitch)
        data.qpos[7:17] = ref_jp + rng.uniform(-args.jp_jitter,
                                               args.jp_jitter, size=10)
        data.qvel[0:3]  = rng.uniform(-args.lv_jitter, args.lv_jitter, size=3)
        data.qvel[3:6]  = rng.uniform(-args.av_jitter, args.av_jitter, size=3)
        data.qvel[6:16] = rng.uniform(-args.jv_jitter, args.jv_jitter, size=10)

        # If randomized qpos interpenetrates the floor (or self), the MuJoCo
        # constraint solver injects a correction impulse on step 1 that gives
        # the body unmodeled velocity. Detect penetration via mj_forward
        # (which populates data.contact with .dist; negative = overlap) and
        # lift the base so the deepest overlap is gone plus --clearance margin.
        mujoco.mj_forward(model, data)
        max_pen = 0.0
        for ci in range(data.ncon):
            d = float(data.contact[ci].dist)
            if d < 0.0:
                max_pen = max(max_pen, -d)
        if max_pen > 0.0:
            data.qpos[2] += max_pen + args.clearance
            mujoco.mj_forward(model, data)

        # Settle physics with PD pulling joints toward ref_jp.
        for _ in range(args.settle_steps):
            q  = data.qpos[7:17]
            dq = data.qvel[6:16]
            tau = kps * (ref_jp - q) - kds * dq
            tau = np.clip(tau, -effort, effort)
            data.ctrl[:10] = tau
            mujoco.mj_step(model, data)

        z_final    = float(data.qpos[2])
        tilt_final = _tilt_deg_from_quat(data.qpos[3:7])
        if z_final < args.reject_z:
            n_rejected_z += 1; continue
        if tilt_final > args.reject_tilt:
            n_rejected_tilt += 1; continue

        saved['base_pos'].append(data.qpos[0:3].astype(np.float32).copy())
        saved['base_quat'].append(data.qpos[3:7].astype(np.float32).copy())
        saved['base_lin_vel'].append(data.qvel[0:3].astype(np.float32).copy())
        saved['base_ang_vel'].append(data.qvel[3:6].astype(np.float32).copy())
        saved['joint_pos'].append(data.qpos[7:17].astype(np.float32).copy())
        saved['joint_vel'].append(data.qvel[6:16].astype(np.float32).copy())
        saved['pose_label'].append(label)

    out = {k: np.stack(v) if k != 'pose_label' else np.array(v)
           for k, v in saved.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)

    accept_rate = len(saved['base_pos']) / max(n_attempts, 1)
    print(f"\n[balance] saved {len(saved['base_pos'])} states → {args.out}")
    print(f"[balance] {n_attempts} attempts ({accept_rate*100:.1f}% accept)")
    print(f"[balance] rejected: {n_rejected_z} (z), {n_rejected_tilt} (tilt)")
    z_arr   = out['base_pos'][:, 2]
    tilt_arr = np.array([_tilt_deg_from_quat(q) for q in out['base_quat']])
    qv_arr  = np.abs(out['joint_vel']).max(axis=1)
    print(f"[balance] z         : mean {z_arr.mean():.3f}  range [{z_arr.min():.3f}, {z_arr.max():.3f}]")
    print(f"[balance] tilt (°)  : mean {tilt_arr.mean():.1f}  range [{tilt_arr.min():.1f}, {tilt_arr.max():.1f}]")
    print(f"[balance] |qv| max  : mean {qv_arr.mean():.2f}  range [{qv_arr.min():.2f}, {qv_arr.max():.2f}]")


if __name__ == '__main__':
    main()
