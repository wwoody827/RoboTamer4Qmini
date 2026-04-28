"""
Phase 1 feasibility study: which fallen poses can Qmini recover from?

For each candidate pose we spawn the robot in MuJoCo with a floating base, set
the base orientation + joint angles, and run a fixed PD policy that simply
holds the nominal standing posture. Whether the robot stands up tells us
whether *passive PD recovery* is feasible from that pose. RL recovery may
expand this set, but anything that succeeds here is "easy" and anything that
clearly fails here is a hard problem (or infeasible) for RL too.

Usage:
    python tests/recovery_feasibility.py                      # all poses, with viewer
    python tests/recovery_feasibility.py --headless           # no viewer, prints metrics
    python tests/recovery_feasibility.py --pose prone --duration 6
    python tests/recovery_feasibility.py --hold-zero          # zero-torque baseline
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import mujoco

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'deploy', 'sim2sim'))

from sim2sim import build_mujoco_model

URDF_PATH = os.path.join(_REPO, 'assets', 'q1', 'urdf', 'q1.urdf')

# From configs/base.yaml — joint order: hip_yaw, hip_roll, hip_pitch, knee, ankle (L then R)
NOMINAL_JOINT_POS = np.array([0.4, -0.1, -1.5, 1.0, -1.3, -0.4, 0.1, 1.5, -1.0, 1.3], dtype=np.float64)
KP = np.array([55., 105., 75., 45., 30., 55., 105., 75., 45., 30.], dtype=np.float64)
KD = np.array([0.3, 2.5, 0.3, 0.5, 0.25, 0.3, 2.5, 0.3, 0.5, 0.25], dtype=np.float64)
NOMINAL_HEIGHT = 0.45

UPRIGHT_HEIGHT_RATIO = 0.85
UPRIGHT_TILT_DEG = 25.0


@dataclass
class Pose:
    name: str
    rpy: tuple              # base roll/pitch/yaw [rad]
    z: float                # base height [m]
    qpos: np.ndarray        # 10 joint angles [rad]


def _knee_tuck(deep=False):
    """Both legs folded: hip flexed forward, knee bent, ankle dorsiflexed.
    Leg sign convention: L hip_pitch < 0 = flex, R hip_pitch > 0 = flex.
    Knee L > 0 = flex, R < 0 = flex.
    """
    f = 1.8 if deep else 1.4
    a = 0.5 if deep else 0.3
    # idx:        0       1       2       3       4       5       6       7       8       9
    return np.array([0.0,    0.0,   -f,      f,     -a,    0.0,    0.0,    f,     -f,      a], dtype=np.float64)


def _legs_extended():
    return np.array([0.0, 0.0, -0.2, 0.1, -0.05, 0.0, 0.0, 0.2, -0.1, 0.05], dtype=np.float64)


POSES = {
    'prone': Pose('prone',
                  rpy=(0.0, np.pi / 2, 0.0), z=0.10,
                  qpos=_legs_extended()),
    'supine': Pose('supine',
                   rpy=(np.pi, 0.0, 0.0), z=0.10,
                   qpos=_legs_extended()),
    'side_left': Pose('side_left',
                      rpy=(np.pi / 2, 0.0, 0.0), z=0.10,
                      qpos=NOMINAL_JOINT_POS.copy()),
    'side_right': Pose('side_right',
                       rpy=(-np.pi / 2, 0.0, 0.0), z=0.10,
                       qpos=NOMINAL_JOINT_POS.copy()),
    'forward_kneel': Pose('forward_kneel',
                          rpy=(0.0, 0.7, 0.0), z=0.18,
                          qpos=_knee_tuck(deep=True)),
    'back_sit': Pose('back_sit',
                     rpy=(0.0, -0.7, 0.0), z=0.18,
                     qpos=_knee_tuck(deep=False)),
}


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


def set_state(model, data, pose, base_xy=(0.0, 0.0)):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:2] = base_xy
    data.qpos[2] = pose.z
    data.qpos[3:7] = rpy_to_wxyz(pose.rpy)
    data.qpos[7:17] = pose.qpos
    mujoco.mj_forward(model, data)


def base_tilt_deg(data):
    """Angle between body +Z and world +Z, in degrees."""
    qw, qx, qy, qz = data.qpos[3:7]
    body_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return float(np.degrees(np.arccos(np.clip(body_z_world_z, -1.0, 1.0))))


def step_with_pd(model, data, target, sim_steps, kp=KP, kd=KD, hold_zero=False):
    qpos_idx = slice(7, 17)
    qvel_idx = slice(6, 16)
    for _ in range(sim_steps):
        if hold_zero:
            data.ctrl[:] = 0.0
        else:
            q = data.qpos[qpos_idx]
            qd = data.qvel[qvel_idx]
            data.ctrl[:] = kp * (target - q) - kd * qd
        mujoco.mj_step(model, data)


def run_one(model, data, pose, duration, render, headless, hold_zero):
    set_state(model, data, pose)
    sim_dt = model.opt.timestep
    total_steps = int(duration / sim_dt)
    chunk = max(1, int(0.05 / sim_dt))  # 50ms chunks for sampling

    metrics = dict(
        max_z=data.qpos[2],
        z_at_end=data.qpos[2],
        max_tilt_drop=0.0,
        tilt_at_end=base_tilt_deg(data),
        recovered=False,
    )
    initial_tilt = metrics['tilt_at_end']
    target = NOMINAL_JOINT_POS

    viewer = None
    if render and not headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)

    try:
        steps_done = 0
        last_render = time.time()
        while steps_done < total_steps:
            n = min(chunk, total_steps - steps_done)
            step_with_pd(model, data, target, n, hold_zero=hold_zero)
            steps_done += n
            z = float(data.qpos[2])
            tilt = base_tilt_deg(data)
            metrics['max_z'] = max(metrics['max_z'], z)
            metrics['max_tilt_drop'] = max(metrics['max_tilt_drop'], initial_tilt - tilt)
            if viewer is not None:
                # Slow down to roughly real-time for visual inspection
                viewer.sync()
                now = time.time()
                target_dt = chunk * sim_dt
                spare = target_dt - (now - last_render)
                if spare > 0:
                    time.sleep(spare)
                last_render = time.time()
                if not viewer.is_running():
                    break
        metrics['z_at_end'] = float(data.qpos[2])
        metrics['tilt_at_end'] = base_tilt_deg(data)
        metrics['recovered'] = (
            metrics['z_at_end'] > UPRIGHT_HEIGHT_RATIO * NOMINAL_HEIGHT
            and metrics['tilt_at_end'] < UPRIGHT_TILT_DEG
        )
    finally:
        if viewer is not None:
            viewer.close()
    return metrics


def fmt(metrics):
    flag = 'PASS' if metrics['recovered'] else 'FAIL'
    return (f"{flag}  z_end={metrics['z_at_end']:.3f}m (max={metrics['max_z']:.3f}m)  "
            f"tilt_end={metrics['tilt_at_end']:5.1f}deg  "
            f"tilt_drop_max={metrics['max_tilt_drop']:5.1f}deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--pose', choices=list(POSES) + ['all'], default='all')
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('--headless', action='store_true', help='no viewer, faster')
    ap.add_argument('--hold-zero', action='store_true',
                    help='zero torque instead of PD-to-nominal — sanity baseline')
    ap.add_argument('--sim-dt', type=float, default=0.002)
    args = ap.parse_args()

    print(f"[feasibility] URDF: {URDF_PATH}")
    print(f"[feasibility] hold_zero={args.hold_zero}, duration={args.duration}s, headless={args.headless}")
    print(f"[feasibility] success criterion: z_end > {UPRIGHT_HEIGHT_RATIO * NOMINAL_HEIGHT:.3f}m  AND  tilt_end < {UPRIGHT_TILT_DEG:.0f}deg")

    model = build_mujoco_model(URDF_PATH, sim_dt=args.sim_dt, init_height=NOMINAL_HEIGHT, fix_base=False)
    data = mujoco.MjData(model)

    pose_names = list(POSES) if args.pose == 'all' else [args.pose]
    results = {}
    for name in pose_names:
        print(f"\n[pose={name}]")
        m = run_one(model, data, POSES[name],
                    duration=args.duration,
                    render=not args.headless,
                    headless=args.headless,
                    hold_zero=args.hold_zero)
        results[name] = m
        print(f"  {fmt(m)}")

    print("\n[summary]")
    for name, m in results.items():
        print(f"  {name:14s}  {fmt(m)}")


if __name__ == '__main__':
    main()
