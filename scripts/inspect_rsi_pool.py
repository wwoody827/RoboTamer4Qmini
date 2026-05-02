"""
Render snapshots from a recovery init pool .npz so you can see what
states will actually seed training. Samples N frames per pose-label and
draws them in a grid with metadata (tilt / z / qvel / ang_vel).

Usage:
    python scripts/inspect_rsi_pool.py \
        --pool data/recovery_init_states_rsi.npz \
        --urdf assets/q1/urdf/q1.urdf \
        --per_label 6 \
        --out /tmp/rsi_pool_view.png
"""
import argparse
import os
import sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'deploy', 'sim2sim'))
from sim2sim import build_mujoco_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--urdf', default=os.path.join(_REPO, 'assets/q1/urdf/q1.urdf'))
    ap.add_argument('--per_label', type=int, default=6)
    ap.add_argument('--out', default='/tmp/rsi_pool_view.png')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    pool = np.load(args.pool, allow_pickle=True)
    labels = pool['pose_label']
    uniq = sorted(set(labels.tolist()))
    rng = np.random.default_rng(args.seed)
    print(f'[pool] {len(labels)} states, labels: {uniq}')

    model = build_mujoco_model(args.urdf, 0.001, init_height=0.5,
                               floor_friction=1.0, fix_base=False)
    model.vis.headlight.ambient[:]  = 0.5
    model.vis.headlight.diffuse[:]  = 0.6
    model.vis.headlight.specular[:] = 0.2
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 1.4
    cam.elevation = -15.0
    cam.azimuth   = 135.0

    renderer = mujoco.Renderer(model, height=240, width=320)
    nrows = len(uniq)
    ncols = args.per_label
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.0*nrows))
    if nrows == 1: axes = axes[None, :]
    if ncols == 1: axes = axes[:, None]

    for r, lab in enumerate(uniq):
        idx_pool = np.where(labels == lab)[0]
        rng.shuffle(idx_pool)
        sel = idx_pool[:ncols]
        for c, idx in enumerate(sel):
            mujoco.mj_resetData(model, data)
            data.qpos[0:3]  = pool['base_pos'][idx]
            data.qpos[3:7]  = pool['base_quat'][idx]
            data.qpos[7:17] = pool['joint_pos'][idx]
            data.qvel[0:3]  = pool['base_lin_vel'][idx]
            data.qvel[3:6]  = pool['base_ang_vel'][idx]
            data.qvel[6:16] = pool['joint_vel'][idx]
            mujoco.mj_forward(model, data)
            cam.lookat[0] = data.qpos[0]
            cam.lookat[1] = data.qpos[1]
            cam.lookat[2] = max(0.15, data.qpos[2])
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            qx, qy = pool['base_quat'][idx][1], pool['base_quat'][idx][2]
            tilt = np.degrees(np.arccos(np.clip(1 - 2*(qx*qx + qy*qy), -1, 1)))
            qv_max = float(np.abs(pool['joint_vel'][idx]).max())
            av_norm = float(np.linalg.norm(pool['base_ang_vel'][idx]))
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(
                f'#{idx} z={data.qpos[2]:.2f} t={tilt:.0f}°\n'
                f'qv={qv_max:.1f} av={av_norm:.1f}',
                fontsize=7)
        axes[r, 0].set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=10)
    fig.suptitle(f'Init pool sample — {os.path.basename(args.pool)}', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    renderer.close()
    print(f'[pool] saved {args.out}')


if __name__ == '__main__':
    main()
