"""
View / record a single recovery episode for visual inspection.

Reuses the eval pipeline from sim2sim_recovery.py but runs ONE episode with
either a live MuJoCo viewer or offscreen MP4 recording.

Usage:
    # Live viewer, random fallen pose
    python deploy/sim2sim/view_recovery.py \\
        --policy experiments/recovery_v6/deploy/policy_10000.onnx

    # Filter by pose label (balance / tilted / prone / supine / side_left / side_right)
    python deploy/sim2sim/view_recovery.py \\
        --policy experiments/recovery_v6/deploy/policy_10000.onnx --label prone

    # Offscreen MP4 (SSH-friendly, requires MUJOCO_GL=egl)
    MUJOCO_GL=egl python deploy/sim2sim/view_recovery.py \\
        --policy experiments/recovery_v6/deploy/policy_10000.onnx \\
        --video /tmp/recovery_demo.mp4
"""
import argparse
import os
import sys
import time
import numpy as np
import mujoco
import onnxruntime as ort

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sim2sim import (
    load_manifest, manifest_to_sim2sim_cfg,
    build_mujoco_model, scale_transform, quat_rotate_inverse,
)
from sim2sim_recovery import build_recovery_obs


def run_one_episode(model, data, session, input_name, cfg, init_state,
                    viewer=None, video_writer=None, renderer=None,
                    video_fps=30, sim_dt=None, decimation=None):
    ref_joint_pos = np.array(cfg['ref_joint_pos'], dtype=np.float32)
    kps           = np.array(cfg['kps'], dtype=np.float32)
    kds           = np.array(cfg['kds'], dtype=np.float32)
    abs_low       = np.array(cfg['abs_low'], dtype=np.float32)
    abs_high      = np.array(cfg['abs_high'], dtype=np.float32)
    jlim_low      = np.array(cfg['joint_limit_low'], dtype=np.float32)
    jlim_high     = np.array(cfg['joint_limit_high'], dtype=np.float32)
    lp_alpha      = float(cfg['action_lowpass_alpha'])
    success_pose_err = float(cfg['success_pose_err'])
    success_tilt_rad = np.deg2rad(cfg['success_tilt_deg'])
    duration      = cfg['episode_length_s']
    obs_history   = cfg['obs_history']
    obs_per_step  = cfg['obs_per_step']
    obs_slots     = cfg['obs_slots']

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = init_state['base_pos']
    data.qpos[3:7] = init_state['base_quat']
    data.qpos[7:17] = init_state['joint_pos']
    data.qvel[0:3] = init_state['base_lin_vel']
    data.qvel[3:6] = init_state['base_ang_vel']
    data.qvel[6:16] = init_state['joint_vel']
    mujoco.mj_forward(model, data)

    current_joint_act = data.qpos[7:17].copy().astype(np.float32)
    lp_target = current_joint_act.copy()
    obs_hist = np.zeros((obs_history, obs_per_step), dtype=np.float32)
    n_policy_steps = int(duration / (sim_dt * decimation))

    cam = None
    frame_stride = 1
    if video_writer is not None:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = 90
        cam.elevation = -20
        cam.distance = 2.5
        frame_stride = max(1, int(round(1.0 / (video_fps * sim_dt))))

    sim_step = 0
    real_t0 = time.time()
    for t in range(n_policy_steps):
        episode_progress = (t + 1) / n_policy_steps
        obs_step = build_recovery_obs(data, current_joint_act, ref_joint_pos,
                                      episode_progress, obs_slots)
        obs_hist = np.roll(obs_hist, -1, axis=0)
        obs_hist[-1] = obs_step
        obs_flat = obs_hist.reshape(1, -1).astype(np.float32)

        mu = session.run(None, {input_name: obs_flat})[0][0]
        target = scale_transform(mu, abs_low, abs_high, clip_val=1.0)
        lp_target = lp_alpha * target + (1.0 - lp_alpha) * lp_target
        current_joint_act = np.clip(lp_target, jlim_low, jlim_high).astype(np.float32)

        for _ in range(decimation):
            q  = data.qpos[7:17].astype(np.float32)
            qd = data.qvel[6:16].astype(np.float32)
            tau = kps * (current_joint_act - q) - kds * qd
            data.ctrl[:10] = tau
            mujoco.mj_step(model, data)
            sim_step += 1

            if video_writer is not None and sim_step % frame_stride == 0:
                cam.lookat[:] = data.qpos[0:3]
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                bgr = rgb[:, :, ::-1]
                video_writer.write(bgr)

            if viewer is not None:
                viewer.sync()
                # real-time pacing
                target_t = real_t0 + sim_step * sim_dt
                slack = target_t - time.time()
                if slack > 0:
                    time.sleep(slack)

        # Success check
        g_body = quat_rotate_inverse(data.qpos[3:7].astype(np.float32),
                                     np.array([0., 0., -1.], dtype=np.float32))
        cos_tilt = float(np.clip(-g_body[2], -1.0, 1.0))
        tilt = float(np.arccos(cos_tilt))
        pose_err = float(np.linalg.norm(data.qpos[7:17] - ref_joint_pos))
        upright = (pose_err < success_pose_err) and (tilt < success_tilt_rad)

    return {
        'success': upright,
        'final_z': float(data.qpos[2]),
        'final_tilt_deg': float(np.rad2deg(tilt)),
        'final_pose_err': pose_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--init_pool', default=os.path.join(_HERE, '..', '..', 'data',
                                                         'recovery_init_states_mixed.npz'))
    ap.add_argument('--label', default=None,
                    help='filter init pool by pose_label (balance/tilted/prone/supine/side_left/side_right)')
    ap.add_argument('--idx', type=int, default=None,
                    help='exact index into init pool (overrides --label / random sampling)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--video', default=None, help='offscreen MP4 path; implies headless. Set MUJOCO_GL=egl over SSH.')
    ap.add_argument('--video_fps', type=int, default=30)
    ap.add_argument('--video_size', default='640x480',
                    help='WxH (default 640x480, MuJoCo default offscreen framebuffer; '
                         'larger requires patching model.vis.global_.offwidth/offheight)')
    ap.add_argument('--floor_friction', type=float, default=1.0)
    args = ap.parse_args()

    manifest, manifest_path = load_manifest(args.policy)
    if manifest is None or manifest.get('task_type') != 'Recovery':
        print('ERROR: need a Recovery-task manifest next to the policy', file=sys.stderr)
        return 1

    s2s_cfg = manifest_to_sim2sim_cfg(manifest, args.policy)
    rec = manifest['recovery']
    cfg = {
        'simulation_dt':       s2s_cfg['simulation_dt'],
        'control_decimation':  s2s_cfg['control_decimation'],
        'ref_joint_pos':       s2s_cfg['ref_joint_pos'],
        'kps':                 s2s_cfg['kps'],
        'kds':                 s2s_cfg['kds'],
        'abs_low':             manifest['action_scaling']['abs_low'] or s2s_cfg['joint_limit_low'],
        'abs_high':            manifest['action_scaling']['abs_high'] or s2s_cfg['joint_limit_high'],
        'joint_limit_low':     s2s_cfg['joint_limit_low'],
        'joint_limit_high':    s2s_cfg['joint_limit_high'],
        'action_lowpass_alpha': s2s_cfg['action_lowpass_alpha'],
        'obs_history':         s2s_cfg['obs_history'],
        'obs_per_step':        s2s_cfg['num_obs_per_step'],
        'obs_slots':           s2s_cfg['obs_slots'],
        'target_height':       float(rec['target_height']),
        'success_pose_err':    float(rec.get('success_pose_err', 0.3)),
        'success_tilt_deg':    float(rec['success_tilt_deg']),
        'episode_length_s':    float(rec['episode_length_s']),
    }

    pool = np.load(args.init_pool)
    n_pool = pool['base_pos'].shape[0]
    if args.idx is not None:
        idx = int(args.idx) % n_pool
    else:
        rng = np.random.default_rng(args.seed)
        if args.label is not None:
            labels = pool['pose_label']
            mask = np.array([str(l) == args.label for l in labels])
            candidates = np.where(mask)[0]
            if len(candidates) == 0:
                print(f"ERROR: no init states with label='{args.label}' in pool", file=sys.stderr)
                return 1
            idx = int(rng.choice(candidates))
        else:
            idx = int(rng.integers(n_pool))

    init_state = {
        'base_pos':     pool['base_pos'][idx],
        'base_quat':    pool['base_quat'][idx],
        'base_lin_vel': pool['base_lin_vel'][idx],
        'base_ang_vel': pool['base_ang_vel'][idx],
        'joint_pos':    pool['joint_pos'][idx],
        'joint_vel':    pool['joint_vel'][idx],
        'pose_label':   pool['pose_label'][idx],
    }
    print(f"[view_recovery] init pool: {args.init_pool}  (n={n_pool})")
    print(f"[view_recovery] picked idx={idx}  label={init_state['pose_label']}")

    model = build_mujoco_model(s2s_cfg['urdf_path'], s2s_cfg['simulation_dt'],
                               init_height=s2s_cfg['init_height'],
                               floor_friction=args.floor_friction, fix_base=False)
    data = mujoco.MjData(model)
    session = ort.InferenceSession(args.policy)
    input_name = session.get_inputs()[0].name

    sim_dt = s2s_cfg['simulation_dt']
    decimation = s2s_cfg['control_decimation']

    if args.video is not None:
        import cv2
        vw, vh = [int(x) for x in args.video_size.lower().split('x')]
        # MuJoCo's default offscreen framebuffer is 640x480; bump it before
        # constructing Renderer so larger video sizes work without editing XML.
        if vw > model.vis.global_.offwidth:
            model.vis.global_.offwidth = vw
        if vh > model.vis.global_.offheight:
            model.vis.global_.offheight = vh
        renderer = mujoco.Renderer(model, height=vh, width=vw)
        os.makedirs(os.path.dirname(os.path.abspath(args.video)) or '.', exist_ok=True)
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.video_fps, (vw, vh))
        print(f"[view_recovery] recording → {args.video}  ({vw}x{vh} @ {args.video_fps}fps)")
        m = run_one_episode(model, data, session, input_name, cfg, init_state,
                            video_writer=writer, renderer=renderer,
                            video_fps=args.video_fps, sim_dt=sim_dt, decimation=decimation)
        writer.release()
        renderer.close()
        print(f"[view_recovery] saved: {args.video}")
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            m = run_one_episode(model, data, session, input_name, cfg, init_state,
                                viewer=viewer, sim_dt=sim_dt, decimation=decimation)

    print(f"\n  result: success={m['success']}  final_z={m['final_z']:.3f}m  "
          f"tilt={m['final_tilt_deg']:.1f}°  pose_err={m['final_pose_err']:.3f}")
    return 0


if __name__ == '__main__':
    import mujoco.viewer  # noqa: F401  (only imported when needed)
    sys.exit(main())
