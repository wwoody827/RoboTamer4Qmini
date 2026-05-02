"""
Extract RSI (Reference State Initialization) frames from a recovery policy.

Why: pure-fallen init pool gives 0% prone success at alpha<=0.3 — the gap
between prone-pose and "halfway up" is too long for random exploration.
DeepMimic-style RSI: seed episodes from intermediate frames of a working
policy's successful trajectories, so each new policy only needs to learn
short horizons from each frame.

Source policy: v2g iter 5000 (61% overall success, 33% prone, 78% supine).
For each of N input fallen poses, run a 5s episode in MuJoCo. If the run
ends upright, sample frames at episode_progress = 0.3 / 0.5 / 0.7. These
get appended to the original fallen pool with pose_label like 'rsi_0.3'.

Usage:
    python scripts/extract_rsi_pool.py \
        --policy experiments/20260429_1912_20260429_recovery_v2g/deploy/policy_5000.onnx \
        --init_pool data/recovery_init_states.npz \
        --output    data/recovery_init_states_rsi.npz \
        --runs 2000

Output schema matches the input pool exactly so RecoveryTask._load_init_pool
needs no change. New entries get pose_label 'rsi_0.3' / 'rsi_0.5' / 'rsi_0.7'.
"""
import argparse
import os
import sys
import numpy as np
import mujoco
import onnxruntime as ort

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'deploy', 'sim2sim'))

from sim2sim import (
    load_manifest, manifest_to_sim2sim_cfg, build_mujoco_model,
    quat_rotate_inverse, scale_transform,
)
from sim2sim_recovery import build_recovery_obs


def run_and_log(model, data, session, input_name, cfg, init_state):
    """Run one 5s episode, log full state at every policy step. Returns
    (success, qpos_log[T+1, 17], qvel_log[T+1, 16], rollout_stats).
    rollout_stats: dict with peak_tau_ratio, mean_tau_ratio, max_qvel,
    max_ang_vel, monotonic_z (fraction of steps where z increases)."""
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
    target_z      = cfg['target_height']
    target_ratio  = cfg['target_height_ratio']
    tilt_deg      = cfg['success_tilt_deg']
    duration      = cfg['episode_length_s']
    obs_history   = cfg['obs_history']
    obs_per_step  = cfg['obs_per_step']

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
    qpos_log = np.zeros((n_policy_steps + 1, 17), dtype=np.float64)
    qvel_log = np.zeros((n_policy_steps + 1, 16), dtype=np.float64)
    qpos_log[0] = data.qpos[:17]
    qvel_log[0] = data.qvel[:16]

    effort = np.abs(model.actuator_forcerange[:10, 1]).astype(np.float32)
    peak_tau_ratio = 0.0
    sum_tau_ratio  = 0.0
    n_tau_samples  = 0

    for t in range(n_policy_steps):
        episode_progress = (t + 1) / n_policy_steps
        obs_step = build_recovery_obs(data, current_joint_act, ref_joint_pos, episode_progress)
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
            ratio = np.abs(tau) / effort
            peak_tau_ratio = max(peak_tau_ratio, float(ratio.max()))
            sum_tau_ratio += float(ratio.mean())
            n_tau_samples += 1

        qpos_log[t + 1] = data.qpos[:17]
        qvel_log[t + 1] = data.qvel[:16]

    # Success at final step
    final_z = qpos_log[-1, 2]
    quat = qpos_log[-1, 3:7].astype(np.float32)
    g_body = quat_rotate_inverse(quat, np.array([0., 0., -1.], dtype=np.float32))
    cos_tilt = float(np.clip(-g_body[2], -1.0, 1.0))
    tilt = float(np.arccos(cos_tilt))
    success = (final_z > target_z * target_ratio) and (tilt < np.deg2rad(tilt_deg))

    # Rollout summary stats — used to reject ballistic trajectories at the
    # rollout level (peak_tau_ratio is the strongest signal for "policy
    # weaponized PD saturation to fling the body").
    z_traj   = qpos_log[:, 2]
    qv_traj  = np.abs(qvel_log[:, 6:16]).max(axis=1)
    av_traj  = np.linalg.norm(qvel_log[:, 3:6], axis=1)
    rollout_stats = {
        'peak_tau_ratio': peak_tau_ratio,
        'mean_tau_ratio': sum_tau_ratio / max(n_tau_samples, 1),
        'max_qvel':       float(qv_traj.max()),
        'max_ang_vel':    float(av_traj.max()),
        'monotonic_z':    float(np.mean(np.diff(z_traj) >= -0.005)),  # frac steps not falling
    }
    return success, qpos_log, qvel_log, rollout_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--init_pool', type=str,
                    default=os.path.join(_REPO, 'data', 'recovery_init_states.npz'))
    ap.add_argument('--output', type=str,
                    default=os.path.join(_REPO, 'data', 'recovery_init_states_rsi.npz'))
    ap.add_argument('--runs', type=int, default=2000,
                    help='number of init poses to roll out (≤ pool size)')
    ap.add_argument('--rsi_progress', type=str, default='0.5,0.7',
                    help='comma-separated progress fractions to sample from each successful run')
    ap.add_argument('--max_rollout_tau_ratio', type=float, default=2.5,
                    help='reject all frames from rollouts whose peak τ/effort exceeds this')
    ap.add_argument('--max_qvel', type=float, default=5.0,
                    help='per-frame: max |joint_vel| (rad/s)')
    ap.add_argument('--max_ang_vel', type=float, default=2.0,
                    help='per-frame: max base |ang_vel| (rad/s)')
    ap.add_argument('--max_lin_vel', type=float, default=0.8,
                    help='per-frame: max base |lin_vel| (m/s)')
    ap.add_argument('--tilt_cap', type=str, default='0.3:60,0.5:35,0.7:20',
                    help='per-progress max tilt (deg), comma-separated p:cap')
    ap.add_argument('--per_class_cap', type=int, default=400,
                    help='cap RSI frames per source pose-class to balance the pool (0=disable)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    progress_pts = [float(x) for x in args.rsi_progress.split(',')]
    tilt_caps = {}
    for spec in args.tilt_cap.split(','):
        p, t = spec.split(':')
        tilt_caps[float(p)] = float(t)
    print(f'[rsi] policy={args.policy}')
    print(f'[rsi] sampling RSI frames at progress {progress_pts}')
    print(f'[rsi] tilt caps (deg): {tilt_caps}')
    print(f'[rsi] max_rollout_tau_ratio={args.max_rollout_tau_ratio}  '
          f'per_frame: qv<{args.max_qvel} av<{args.max_ang_vel} lv<{args.max_lin_vel}')
    print(f'[rsi] per_class_cap={args.per_class_cap}')

    manifest, manifest_path = load_manifest(args.policy)
    if manifest is None or manifest.get('task_type') != 'Recovery':
        print(f'ERROR: manifest missing or not Recovery: {manifest_path}', file=sys.stderr)
        return 1
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
        'target_height':       float(rec['target_height']),
        'target_height_ratio': float(rec['target_height_ratio']),
        'success_tilt_deg':    float(rec['success_tilt_deg']),
        'episode_length_s':    float(rec['episode_length_s']),
    }

    model = build_mujoco_model(s2s['urdf_path'], s2s['simulation_dt'],
                               init_height=s2s['init_height'],
                               floor_friction=1.0, fix_base=False)
    data = mujoco.MjData(model)
    session = ort.InferenceSession(args.policy)
    input_name = session.get_inputs()[0].name

    pool = np.load(args.init_pool, allow_pickle=True)
    n_pool = pool['base_pos'].shape[0]
    n_runs = min(args.runs, n_pool)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_pool)[:n_runs]
    print(f'[rsi] rolling out {n_runs} episodes from pool of {n_pool}')

    n_policy_steps = int(cfg['episode_length_s'] / (cfg['simulation_dt'] * cfg['control_decimation']))
    sample_steps = [int(round(p * n_policy_steps)) for p in progress_pts]
    print(f'[rsi] {n_policy_steps} policy steps/episode → sample at steps {sample_steps}')

    rsi_records = {k: [] for k in
                   ('base_pos', 'base_quat', 'base_lin_vel', 'base_ang_vel',
                    'joint_pos', 'joint_vel', 'pose_label', 'src_label')}
    success_count = 0
    rollout_kept = 0
    by_label = {}
    success_peak_tau = []   # one entry per successful rollout, regardless of filter

    for run_i, idx in enumerate(indices):
        init = {k: pool[k][idx] for k in
                ('base_pos', 'base_quat', 'base_lin_vel', 'base_ang_vel',
                 'joint_pos', 'joint_vel')}
        init['pose_label'] = pool['pose_label'][idx]
        src_label = str(init['pose_label'])

        success, qpos_log, qvel_log, stats = run_and_log(model, data, session, input_name, cfg, init)

        if src_label not in by_label:
            by_label[src_label] = [0, 0, 0]   # (success, total, kept_rollouts)
        by_label[src_label][1] += 1
        if not success:
            continue
        by_label[src_label][0] += 1
        success_count += 1
        success_peak_tau.append(stats['peak_tau_ratio'])

        # ROLLOUT-LEVEL filter: drop the entire rollout if its peak τ/effort
        # was high — those are the ballistic-flip rollouts and their frames
        # would seed the policy with that strategy.
        if stats['peak_tau_ratio'] > args.max_rollout_tau_ratio:
            continue
        rollout_kept += 1
        by_label[src_label][2] += 1

        for p, step in zip(progress_pts, sample_steps):
            rsi_records['base_pos'    ].append(qpos_log[step, 0:3].copy())
            rsi_records['base_quat'   ].append(qpos_log[step, 3:7].copy())
            rsi_records['joint_pos'   ].append(qpos_log[step, 7:17].copy())
            rsi_records['base_lin_vel'].append(qvel_log[step, 0:3].copy())
            rsi_records['base_ang_vel'].append(qvel_log[step, 3:6].copy())
            rsi_records['joint_vel'   ].append(qvel_log[step, 6:16].copy())
            rsi_records['pose_label'  ].append(f'rsi_{p:.1f}')
            rsi_records['src_label'   ].append(src_label)

        if (run_i + 1) % 50 == 0 or run_i == n_runs - 1:
            sr = success_count / (run_i + 1)
            print(f'  [{run_i+1:4d}/{n_runs}] success={sr:.2%}  '
                  f'rollouts_kept={rollout_kept}  rsi_frames={len(rsi_records["base_pos"])}')

    print()
    print('=== Per-class v2g rollout outcomes ===')
    for label, (s, n, kr) in sorted(by_label.items()):
        print(f'  {label:<10s}  success={s/n:.2%} ({s}/{n})  rollouts_after_τ_filter={kr}')

    if success_peak_tau:
        ptau = np.array(success_peak_tau)
        print(f'\n=== τpeak/effort distribution across {len(ptau)} successful rollouts ===')
        for q in (10, 25, 50, 75, 90, 95, 99):
            print(f'  p{q:02d}: {np.percentile(ptau, q):.2f}×')
        print(f'  min={ptau.min():.2f}×  max={ptau.max():.2f}×  mean={ptau.mean():.2f}×')
        for thr in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
            print(f'  rollouts with τpeak ≤ {thr:.1f}×: {(ptau <= thr).sum()}/{len(ptau)} ({(ptau <= thr).mean():.1%})')

    n_rsi_raw = len(rsi_records['base_pos'])
    print(f'\n[rsi] {n_rsi_raw} frames from {rollout_kept} rollouts '
          f'(τpeak ≤ {args.max_rollout_tau_ratio}× of {success_count} successful)')
    if n_rsi_raw == 0:
        print('[rsi] NO frames passed rollout-level τ filter — relax --max_rollout_tau_ratio')
        return 1

    rsi = {k: np.stack(rsi_records[k]) if k not in ('pose_label', 'src_label')
              else np.array(rsi_records[k], dtype='<U16')
           for k in rsi_records}

    # PER-FRAME filter: tight velocity caps + per-progress tilt caps.
    qv_max  = np.abs(rsi['joint_vel']).max(axis=1)
    av_norm = np.linalg.norm(rsi['base_ang_vel'], axis=1)
    lv_norm = np.linalg.norm(rsi['base_lin_vel'], axis=1)
    qx, qy  = rsi['base_quat'][:, 1], rsi['base_quat'][:, 2]
    tilt_deg = np.degrees(np.arccos(np.clip(1 - 2*(qx*qx + qy*qy), -1, 1)))
    vel_ok  = (qv_max < args.max_qvel) & (av_norm < args.max_ang_vel) & (lv_norm < args.max_lin_vel)
    tilt_ok = np.ones_like(vel_ok)
    for p, cap in tilt_caps.items():
        mask = rsi['pose_label'] == f'rsi_{p:.1f}'
        if mask.any():
            tilt_ok[mask] = tilt_deg[mask] < cap
    keep_rsi = vel_ok & tilt_ok
    print(f'\n[rsi] per-frame filter: {keep_rsi.sum()}/{n_rsi_raw} pass '
          f'(vel_ok={vel_ok.sum()}, tilt_ok={tilt_ok.sum()})')
    for p in progress_pts:
        mask = rsi['pose_label'] == f'rsi_{p:.1f}'
        print(f'    rsi_{p:.1f}: {(keep_rsi & mask).sum()}/{mask.sum()}')

    # PER-CLASS BALANCE: cap the number of frames sourced from each src class
    # to avoid supine domination (v2g is best at supine, ~3x prone count).
    if args.per_class_cap > 0:
        cap_keep = np.zeros_like(keep_rsi)
        rng2 = np.random.default_rng(args.seed + 1)
        # For each (src_label, pose_label) bucket, randomly pick up to cap.
        for src in sorted(set(rsi['src_label'].tolist())):
            for p in progress_pts:
                pl = f'rsi_{p:.1f}'
                bucket = np.where((rsi['src_label'] == src) &
                                  (rsi['pose_label'] == pl) & keep_rsi)[0]
                if len(bucket) > args.per_class_cap:
                    keep_idx = rng2.choice(bucket, args.per_class_cap, replace=False)
                else:
                    keep_idx = bucket
                cap_keep[keep_idx] = True
        keep_rsi = cap_keep
        print(f'[rsi] after per-class cap (≤{args.per_class_cap}): {keep_rsi.sum()}')

    rsi = {k: v[keep_rsi] for k, v in rsi.items()}
    n_rsi = keep_rsi.sum()
    rsi.pop('src_label')   # not part of pool schema

    # Build merged pool: original 2000 fallen + RSI frames
    merged = {}
    for k in ('base_pos', 'base_quat', 'base_lin_vel', 'base_ang_vel',
              'joint_pos', 'joint_vel'):
        merged[k] = np.concatenate([pool[k].astype(np.float64), rsi[k].astype(np.float64)], axis=0)
    merged['pose_label'] = np.concatenate([pool['pose_label'], rsi['pose_label']])
    # Carry forward zeros for fields we don't track for RSI entries
    n_orig = pool['base_pos'].shape[0]
    n_total = n_orig + n_rsi
    merged['impulse']     = np.concatenate([pool['impulse'],     np.zeros((n_rsi, 6))], axis=0)
    merged['settle_time'] = np.concatenate([pool['settle_time'], np.zeros((n_rsi,))])
    merged['settled']     = np.concatenate([pool['settled'],     np.ones((n_rsi,), dtype=bool)])
    # Meta scalars copied from source pool
    for k in ('meta_force_min', 'meta_force_max', 'meta_push_dt',
              'meta_settle_max_t', 'meta_seed', 'meta_attempts'):
        if k in pool.files:
            merged[k] = pool[k]

    np.savez(args.output, **merged)
    print(f'[rsi] wrote merged pool: {args.output}')
    print(f'      {n_orig} fallen + {n_rsi} RSI = {n_total} total')
    return 0


if __name__ == '__main__':
    sys.exit(main())
