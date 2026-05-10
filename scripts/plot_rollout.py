"""Plot per-step state from a recorded trace .npz (output of sim2sim --record).

Plots a 2x2 grid:
  Top-left:  body_vx vs cmd_vx
  Top-right: body_vy vs cmd_vy   (the strafe reality check)
  Bot-left:  yaw_rate vs cmd_yaw  (signed yaw_rate)
  Bot-right: yaw_world (cumulative angle) vs cmd_yaw * t  (drift visualization)

The trace .npz must have base_quat, base_lin_vel, base_ang_vel, cmd_const.

Usage:
    python scripts/plot_rollout.py path/to/trace.npz
    # OR record + plot in one shot:
    python scripts/plot_rollout.py --policy <onnx> --cmd "0 0.3 0" --duration 8
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/woody/miniconda3/envs/qmini/bin/python'


def _quat_to_yaw(quat):
    """[T, 4] (w,x,y,z) → [T] yaw_world in radians."""
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _world_to_body_xy(world_xy, yaw):
    """Project world-frame [T, 2] velocity into body frame using yaw[T]."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    bx =  world_xy[:, 0] * cy + world_xy[:, 1] * sy
    by = -world_xy[:, 0] * sy + world_xy[:, 1] * cy
    return bx, by


def plot_trace(npz_path, save_to=None, show=False, title=None):
    import matplotlib
    if save_to or not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    dt = float(d['dt'])
    T = d['joint_pos'].shape[0]
    t = np.arange(T) * dt

    cmd = d['cmd_const'] if 'cmd_const' in d.files else np.array([0.0, 0.0, 0.0])
    cmd_vx, cmd_vy, cmd_yaw = float(cmd[0]), float(cmd[1]), float(cmd[2])

    yaw_world = np.unwrap(_quat_to_yaw(d['base_quat']))
    body_vx, body_vy = _world_to_body_xy(d['base_lin_vel'][:, :2], yaw_world)
    yaw_rate = d['base_ang_vel'][:, 2]  # body-frame ang vel z

    # EMA for visualization (matches training reward’s LP filter when α=0.95)
    def ema(x, alpha=0.95):
        out = np.zeros_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * out[i-1] + (1 - alpha) * x[i]
        return out

    bvx_ema = ema(body_vx, 0.95)
    bvy_ema = ema(body_vy, 0.95)
    yawr_ema = ema(yaw_rate, 0.95)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title or f'{npz_path}\ncmd=[{cmd_vx:+.2f}, {cmd_vy:+.2f}, {cmd_yaw:+.2f}]', fontsize=11)

    # vx
    ax[0,0].plot(t, body_vx, 'b-', alpha=0.4, label='body_vx (raw)')
    ax[0,0].plot(t, bvx_ema, 'b-', label='body_vx (EMA α=0.95)')
    ax[0,0].axhline(cmd_vx, color='r', ls='--', label=f'cmd_vx={cmd_vx:+.2f}')
    ax[0,0].set_ylabel('m/s'); ax[0,0].set_title('body_vx vs time'); ax[0,0].legend()
    ax[0,0].grid(alpha=0.3)

    # vy
    ax[0,1].plot(t, body_vy, 'g-', alpha=0.4, label='body_vy (raw)')
    ax[0,1].plot(t, bvy_ema, 'g-', label='body_vy (EMA α=0.95)')
    ax[0,1].axhline(cmd_vy, color='r', ls='--', label=f'cmd_vy={cmd_vy:+.2f}')
    ax[0,1].set_ylabel('m/s'); ax[0,1].set_title('body_vy vs time (strafe)'); ax[0,1].legend()
    ax[0,1].grid(alpha=0.3)

    # yaw_rate
    ax[1,0].plot(t, yaw_rate, 'm-', alpha=0.4, label='yaw_rate (raw)')
    ax[1,0].plot(t, yawr_ema, 'm-', label='yaw_rate (EMA α=0.95)')
    ax[1,0].axhline(cmd_yaw, color='r', ls='--', label=f'cmd_yaw={cmd_yaw:+.2f}')
    ax[1,0].set_ylabel('rad/s'); ax[1,0].set_xlabel('s'); ax[1,0].set_title('yaw_rate vs time')
    ax[1,0].legend(); ax[1,0].grid(alpha=0.3)

    # yaw_angle (cumulative — drift visualization)
    ax[1,1].plot(t, np.degrees(yaw_world), 'k-', label='yaw_world')
    ax[1,1].plot(t, np.degrees(cmd_yaw * t), 'r--', label=f'cmd integrated')
    ax[1,1].set_ylabel('deg'); ax[1,1].set_xlabel('s'); ax[1,1].set_title('yaw angle vs time')
    ax[1,1].legend(); ax[1,1].grid(alpha=0.3)

    # Annotations: net world displacement + means
    base_pos = d['base_pos']
    dx = float(base_pos[-1, 0] - base_pos[0, 0])
    dy = float(base_pos[-1, 1] - base_pos[0, 1])
    summary = (f'Δworld pos: (Δx={dx:+.2f}m, Δy={dy:+.2f}m)  '
               f'mean body_vx={body_vx.mean():+.3f}  '
               f'mean body_vy={body_vy.mean():+.3f}  '
               f'mean yaw_rate={yaw_rate.mean():+.3f}')
    fig.text(0.5, 0.01, summary, ha='center', fontsize=9, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_to:
        plt.savefig(save_to, dpi=120)
        print(f'  saved: {save_to}')
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()


def record_then_plot(policy, cmd_vx, cmd_vy, cmd_yaw, duration, out_dir):
    cmd_id = (f"vx{'p' if cmd_vx>=0 else 'n'}{abs(cmd_vx):.2f}"
              f"_vy{'p' if cmd_vy>=0 else 'n'}{abs(cmd_vy):.2f}"
              f"_yaw{'p' if cmd_yaw>=0 else 'n'}{abs(cmd_yaw):.2f}").replace('.', '')
    npz_path = Path(out_dir) / f'{cmd_id}.npz'
    png_path = Path(out_dir) / f'{cmd_id}.png'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sim2sim = REPO_ROOT / 'deploy' / 'sim2sim' / 'sim2sim.py'
    cmd = [PYTHON, str(sim2sim), '--policy', policy,
           '--cmd_vx', str(cmd_vx), '--cmd_vy', str(cmd_vy), '--cmd_yaw', str(cmd_yaw),
           '--duration', str(duration), '--headless',
           '--record', str(npz_path), '--record_skill', 'walk', '--record_loop',
           '--record_skip', '20']
    print(f'  recording {cmd_id} ...')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not npz_path.exists():
        print(f'  FAIL: rc={proc.returncode}, no .npz')
        print(proc.stderr[-500:])
        return None

    plot_trace(npz_path, save_to=png_path,
               title=f'{Path(policy).stem}  cmd=[{cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_yaw:+.2f}]')
    return png_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='?', help='Path to existing trace .npz (skip recording)')
    p.add_argument('--policy', help='Policy .onnx for fresh recording')
    p.add_argument('--cmd', help='Cmd as "vx vy yaw" e.g. "0 0.3 0"')
    p.add_argument('--duration', type=float, default=8.0)
    p.add_argument('--out_dir', help='Where to save .png/.npz (defaults to policy dir/plots/)')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()

    if args.npz:
        npz = Path(args.npz)
        if not npz.exists():
            sys.exit(f'no such file: {npz}')
        out_dir = args.out_dir or npz.parent
        png = Path(out_dir) / (npz.stem + '.png')
        plot_trace(npz, save_to=png, show=args.show)
    else:
        if not args.policy or not args.cmd:
            sys.exit('Either give a .npz or --policy + --cmd "vx vy yaw"')
        vx, vy, yaw = (float(x) for x in args.cmd.split())
        out_dir = args.out_dir or (Path(args.policy).parent / 'plots')
        record_then_plot(args.policy, vx, vy, yaw, args.duration, out_dir)


if __name__ == '__main__':
    main()
