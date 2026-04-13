"""
Replay a recorded reference clip (.npz) in the MuJoCo viewer.

The robot's joints are driven directly to the recorded positions — no policy,
no PD control. Use this to verify clip quality before using it for MIRL training.

Usage:
    python deploy/sim2sim/replay_clip.py --clip data/reference_clips/walk_fwd.npz
    python deploy/sim2sim/replay_clip.py --clip data/reference_clips/walk_fwd.npz --speed 0.5
    python deploy/sim2sim/replay_clip.py --clip data/reference_clips/walk_fwd.npz --loop
"""

import argparse
import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim import build_mujoco_model


URDF_PATH   = 'assets/q1/urdf/q1.urdf'
INIT_HEIGHT = 0.5
SIM_DT      = 0.001
QPOS_START  = 7   # free joint: 3 pos + 4 quat
NUM_JOINTS  = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip',  required=True, help='Path to .npz reference clip')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (0.5 = half speed)')
    parser.add_argument('--loop',  action='store_true', help='Loop the clip indefinitely')
    parser.add_argument('--urdf',  default=URDF_PATH)
    args = parser.parse_args()

    clip = np.load(args.clip, allow_pickle=True)
    joint_pos  = clip['joint_pos']   # [T, 10]
    base_pos   = clip.get('base_pos',  None)
    base_quat  = clip.get('base_quat', None)
    policy_dt  = float(clip['dt'])
    T          = joint_pos.shape[0]
    duration   = T * policy_dt

    print(f"Clip:     {args.clip}")
    print(f"Frames:   {T}  ({duration:.1f}s @ {1/policy_dt:.0f}Hz)")
    print(f"skill:    {clip.get('skill', '?')}  loop={clip.get('loop', '?')}")
    if base_pos is not None:
        print(f"x travel: {base_pos[:,0].min():.2f} → {base_pos[:,0].max():.2f} m")
    if 'base_lin_vel' in clip:
        vx = clip['base_lin_vel'][:,0]
        print(f"vx:       mean={vx.mean():.3f}  std={vx.std():.3f} m/s")
    print()

    model = build_mujoco_model(args.urdf, SIM_DT, INIT_HEIGHT)
    data  = mujoco.MjData(model)

    # Set initial pose
    mujoco.mj_resetData(model, data)
    data.qpos[QPOS_START:QPOS_START + NUM_JOINTS] = joint_pos[0]
    if base_pos  is not None: data.qpos[0:3] = base_pos[0]
    if base_quat is not None: data.qpos[3:7] = base_quat[0]
    mujoco.mj_forward(model, data)

    frame_period = policy_dt / args.speed   # wall-clock seconds per frame

    print("Opening viewer — close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame   = 0
        t_start = time.perf_counter()

        while viewer.is_running():
            # Anchor to wall clock before doing any work
            target = t_start + frame * frame_period

            # Drive joints directly to recorded positions
            data.qpos[QPOS_START:QPOS_START + NUM_JOINTS] = joint_pos[frame]
            if base_pos  is not None: data.qpos[0:3] = base_pos[frame]
            if base_quat is not None: data.qpos[3:7] = base_quat[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()

            frame += 1
            if frame >= T:
                if args.loop:
                    frame   = 0
                    t_start = time.perf_counter()
                    print("  looping")
                else:
                    print("Clip finished.")
                    break

            # Busy-wait the remainder — more precise than time.sleep for short intervals
            while time.perf_counter() < target:
                time.sleep(0.001)


if __name__ == '__main__':
    main()
