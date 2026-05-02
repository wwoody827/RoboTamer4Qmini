"""
Inspect RSI trajectories visually before committing the pool to training.

For each pose class (prone/supine/side_left/side_right), pick a few init
states and run them through the v2g policy. Plot the full 5s trajectory
of tilt, base height, base angular velocity, max joint velocity. Mark
progress=0.3/0.5/0.7 sample points so you can see what state the RSI
frame snapshots are in.

Also renders MuJoCo offscreen snapshots at the three sample times for
each rollout (3 images per rollout). Trajectories that look ballistic
(tilt overshooting, ang_vel spiking, body inverted at progress=0.3)
are signs the RSI seed will be unphysical.

Usage:
    python scripts/inspect_rsi_trajectories.py \
        --policy experiments/20260429_1912_20260429_recovery_v2g/deploy/policy_5000.onnx \
        --per_class 3 \
        --out /tmp/rsi_inspect/
"""
import argparse
import os
import sys
import numpy as np
import mujoco
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'deploy', 'sim2sim'))
from sim2sim import (
    load_manifest, manifest_to_sim2sim_cfg, build_mujoco_model,
    quat_rotate_inverse, scale_transform,
)
from sim2sim_recovery import build_recovery_obs


def quat_wxyz_to_tilt_deg(q):
    qx, qy = q[1], q[2]
    cos_tilt = np.clip(1 - 2*(qx*qx + qy*qy), -1, 1)
    return float(np.degrees(np.arccos(cos_tilt)))


def run_logging(model, data, session, input_name, cfg, init_state):
    """Run one episode, return per-policy-step time series (no decimation)."""
    sim_dt        = cfg['simulation_dt']
    decimation    = cfg['control_decimation']
    ref_joint_pos = np.array(cfg['ref_joint_pos'], dtype=np.float32)
    kps           = np.array(cfg['kps'], dtype=np.float32)
    kds           = np.array(cfg['kds'], dtype=np.float32)
    abs_low       = np.array(cfg['abs_low'], dtype=np.float32)
    abs_high      = np.array(cfg['abs_high'], dtype=np.float32)
    jlim_low      = np.array(cfg['joint_limit_low'], dtype=np.float32)
    jlim_high     = np.array(cfg['joint_limit_high'], dtype=np.float32)
    lp_alpha      = float(cfg['action_lowpass_alpha'])
    duration      = cfg['episode_length_s']
    obs_history   = cfg['obs_history']
    obs_per_step  = cfg['obs_per_step']

    mujoco.mj_resetData(model, data)
    data.qpos[0:3]  = init_state['base_pos']
    data.qpos[3:7]  = init_state['base_quat']
    data.qpos[7:17] = init_state['joint_pos']
    data.qvel[0:3]  = init_state['base_lin_vel']
    data.qvel[3:6]  = init_state['base_ang_vel']
    data.qvel[6:16] = init_state['joint_vel']
    mujoco.mj_forward(model, data)

    current_joint_act = data.qpos[7:17].copy().astype(np.float32)
    lp_target = current_joint_act.copy()
    obs_hist = np.zeros((obs_history, obs_per_step), dtype=np.float32)
    n_steps = int(duration / (sim_dt * decimation))

    log = {
        't':       np.zeros(n_steps + 1),
        'z':       np.zeros(n_steps + 1),
        'tilt':    np.zeros(n_steps + 1),
        'qv_max':  np.zeros(n_steps + 1),
        'av_norm': np.zeros(n_steps + 1),
        'tau_max': np.zeros(n_steps + 1),
        'qpos17':  np.zeros((n_steps + 1, 17)),
        'qvel16':  np.zeros((n_steps + 1, 16)),
    }
    effort = np.abs(model.actuator_forcerange[:10, 1]).astype(np.float32)

    def measure(idx, step_tau_max=0.0):
        log['t'][idx]       = idx * sim_dt * decimation
        log['z'][idx]       = data.qpos[2]
        log['tilt'][idx]    = quat_wxyz_to_tilt_deg(data.qpos[3:7])
        log['qv_max'][idx]  = float(np.abs(data.qvel[6:16]).max())
        log['av_norm'][idx] = float(np.linalg.norm(data.qvel[3:6]))
        log['tau_max'][idx] = step_tau_max
        log['qpos17'][idx]  = data.qpos[:17].copy()
        log['qvel16'][idx]  = data.qvel[:16].copy()

    measure(0, 0.0)

    for t in range(n_steps):
        progress = (t + 1) / n_steps
        obs_step = build_recovery_obs(data, current_joint_act, ref_joint_pos, progress)
        obs_hist = np.roll(obs_hist, -1, axis=0)
        obs_hist[-1] = obs_step
        mu = session.run(None, {input_name: obs_hist.reshape(1, -1).astype(np.float32)})[0][0]
        target = scale_transform(mu, abs_low, abs_high, clip_val=1.0)
        lp_target = lp_alpha * target + (1.0 - lp_alpha) * lp_target
        current_joint_act = np.clip(lp_target, jlim_low, jlim_high).astype(np.float32)

        step_tau_max = 0.0
        for _ in range(decimation):
            q  = data.qpos[7:17].astype(np.float32)
            qd = data.qvel[6:16].astype(np.float32)
            tau = kps * (current_joint_act - q) - kds * qd
            data.ctrl[:10] = tau
            mujoco.mj_step(model, data)
            step_tau_max = max(step_tau_max, float((np.abs(tau) / effort).max()))
        measure(t + 1, step_tau_max)

    return log


def render_snapshot(model, data, qpos17, qvel16, renderer):
    """Set state and return RGB image."""
    data.qpos[:17] = qpos17
    data.qvel[:16] = qvel16
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera='side' if 'side' in [c.name for c in [model.cam(i) for i in range(model.ncam)]] else -1)
    return renderer.render()


def render_simple(model, data, qpos17, qvel16, width=320, height=240):
    """Offscreen render — uses default camera."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    data.qpos[:17] = qpos17
    data.qvel[:16] = qvel16
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    img = renderer.render()
    renderer.close()
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--init_pool', type=str,
                    default=os.path.join(_REPO, 'data', 'recovery_init_states.npz'))
    ap.add_argument('--per_class', type=int, default=3,
                    help='successful rollouts to plot per pose class')
    ap.add_argument('--max_attempts_per_class', type=int, default=15,
                    help='give up after N attempts per class')
    ap.add_argument('--out', type=str, default='/tmp/rsi_inspect')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_render', action='store_true', help='skip MuJoCo snapshots')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'[inspect] output dir: {args.out}')

    manifest, _ = load_manifest(args.policy)
    s2s = manifest_to_sim2sim_cfg(manifest, args.policy)
    rec = manifest['recovery']
    cfg = {
        'simulation_dt':       s2s['simulation_dt'],
        'control_decimation':  s2s['control_decimation'],
        'ref_joint_pos':       s2s['ref_joint_pos'],
        'kps':                 s2s['kps'],
        'kds':                 s2s['kds'],
        'abs_low':             manifest['action_scaling']['abs_low']  or s2s['joint_limit_low'],
        'abs_high':            manifest['action_scaling']['abs_high'] or s2s['joint_limit_high'],
        'joint_limit_low':     s2s['joint_limit_low'],
        'joint_limit_high':    s2s['joint_limit_high'],
        'action_lowpass_alpha': s2s['action_lowpass_alpha'],
        'obs_history':         s2s['obs_history'],
        'obs_per_step':        s2s['num_obs_per_step'],
        'episode_length_s':    float(rec['episode_length_s']),
        'target_height':       float(rec['target_height']),
        'target_height_ratio': float(rec['target_height_ratio']),
        'success_tilt_deg':    float(rec['success_tilt_deg']),
    }
    success_z = cfg['target_height'] * cfg['target_height_ratio']
    success_tilt = cfg['success_tilt_deg']

    model = build_mujoco_model(s2s['urdf_path'], s2s['simulation_dt'],
                               init_height=s2s['init_height'],
                               floor_friction=1.0, fix_base=False)
    data = mujoco.MjData(model)
    session = ort.InferenceSession(args.policy)
    input_name = session.get_inputs()[0].name

    pool = np.load(args.init_pool, allow_pickle=True)
    labels = pool['pose_label']
    classes = sorted(set(labels.tolist()))
    print(f'[inspect] pool classes: {classes}')

    rng = np.random.default_rng(args.seed)
    n_steps = int(cfg['episode_length_s'] / (cfg['simulation_dt'] * cfg['control_decimation']))
    progress_pts = (0.3, 0.5, 0.7)
    sample_idx_in_traj = [int(round(p * n_steps)) for p in progress_pts]

    rollouts = {}  # class -> list of (init_idx, log)
    for cls in classes:
        cand = np.where(labels == cls)[0]
        rng.shuffle(cand)
        kept = []
        for i, idx in enumerate(cand[:args.max_attempts_per_class]):
            init = {k: pool[k][idx] for k in
                    ('base_pos', 'base_quat', 'base_lin_vel', 'base_ang_vel',
                     'joint_pos', 'joint_vel')}
            log = run_logging(model, data, session, input_name, cfg, init)
            final_z, final_tilt = log['z'][-1], log['tilt'][-1]
            ok = (final_z > success_z) and (final_tilt < success_tilt)
            if ok:
                kept.append((int(idx), log))
                if len(kept) >= args.per_class:
                    break
        rollouts[cls] = kept
        print(f'[inspect] {cls}: {len(kept)} successful rollouts plotted')

    # Time-series figure: rows per class, columns: tilt, z, |qvel|max, |ang_vel|, τpeak
    metrics = [
        ('tilt',    'tilt (deg)',     [0, 180], success_tilt),
        ('z',       'base z (m)',     [0, 0.55], success_z),
        ('qv_max',  '|qvel| max (rad/s)', [0, 30], 10.0),
        ('av_norm', '|base ang_vel| (rad/s)', [0, 15], 4.0),
        ('tau_max', 'τpeak / effort', [0, 8],   1.0),
    ]
    nrows, ncols = len(classes), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.5*nrows), sharex=True)
    if nrows == 1: axes = axes[None, :]
    for r, cls in enumerate(classes):
        for c, (mkey, ylabel, ylim, thresh) in enumerate(metrics):
            ax = axes[r, c]
            for k, (idx, log) in enumerate(rollouts[cls]):
                ax.plot(log['t'], log[mkey], alpha=0.7, lw=1.0)
                # Mark RSI sample points
                for sidx in sample_idx_in_traj:
                    ax.plot(log['t'][sidx], log[mkey][sidx], 'ro', ms=4)
            ax.axhline(thresh, color='k', ls='--', lw=0.5, alpha=0.5)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            if r == 0: ax.set_title(ylabel)
            if c == 0: ax.set_ylabel(cls, rotation=0, ha='right', va='center', fontsize=10)
            if r == nrows - 1: ax.set_xlabel('t (s)')
    fig.suptitle(f'v2g iter5000 trajectories — red dots = RSI sample points (progress 0.3/0.5/0.7)')
    fig.tight_layout()
    traces_path = os.path.join(args.out, 'rsi_traces.png')
    fig.savefig(traces_path, dpi=110)
    plt.close(fig)
    print(f'[inspect] traces: {traces_path}')

    # Per-rollout snapshots at progress 0.3/0.5/0.7
    if not args.no_render:
        # Boost headlight + custom camera for visibility (URDF has no scene lighting)
        model.vis.headlight.ambient[:]  = 0.5
        model.vis.headlight.diffuse[:]  = 0.6
        model.vis.headlight.specular[:] = 0.2
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.25]
        cam.distance  = 1.4
        cam.elevation = -15.0
        cam.azimuth   = 135.0

        n_total = sum(len(v) for v in rollouts.values())
        fig, axes = plt.subplots(n_total, 4, figsize=(12, 3*n_total))
        if n_total == 1: axes = axes[None, :]
        row = 0
        renderer = mujoco.Renderer(model, height=320, width=480)
        for cls in classes:
            for k, (idx, log) in enumerate(rollouts[cls]):
                snap_steps = [0] + sample_idx_in_traj
                snap_labels = ['init'] + [f'p={p:.1f}' for p in progress_pts]
                for c, step in enumerate(snap_steps):
                    data.qpos[:17] = log['qpos17'][step]
                    data.qvel[:16] = log['qvel16'][step]
                    mujoco.mj_forward(model, data)
                    cam.lookat[0] = data.qpos[0]
                    cam.lookat[1] = data.qpos[1]
                    cam.lookat[2] = max(0.15, data.qpos[2])
                    renderer.update_scene(data, camera=cam)
                    img = renderer.render()
                    ax = axes[row, c]
                    ax.imshow(img)
                    ax.axis('off')
                    title = snap_labels[c]
                    if c == 0:
                        title = f'{cls}#{idx} {title}'
                    title += f' tilt={log["tilt"][step]:.0f}° z={log["z"][step]:.2f}'
                    if c > 0:
                        title += f' qv={log["qv_max"][step]:.0f}'
                    ax.set_title(title, fontsize=8)
                row += 1
        renderer.close()
        snap_path = os.path.join(args.out, 'rsi_snapshots.png')
        fig.tight_layout()
        fig.savefig(snap_path, dpi=110)
        plt.close(fig)
        print(f'[inspect] snapshots: {snap_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
