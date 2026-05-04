"""
Recovery sim2sim harness.

Walking sim2sim doesn't apply: no commands, no phase, and we need to start each
episode from a *fallen* pose drawn from the init-state pool. Each episode runs
for episode_length_s; success = (base_z > target_z * ratio) AND (tilt < success_tilt_deg)
at the final step.

Usage:
    python deploy/sim2sim/sim2sim_recovery.py \\
        --policy experiments/<run>/deploy/policy_3000.onnx \\
        --runs 50

Metrics printed:
    success_rate           — fraction of runs upright at episode end
    ever_upright_rate      — fraction that hit upright at any point
    mean_final_z           — average final base height (m)
    mean_final_tilt_deg    — average final body tilt (degrees)
    mean_time_to_upright_s — mean first-upright time across runs that succeeded
"""
import argparse
import os
import sys
import numpy as np
import mujoco
import onnxruntime as ort

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sim2sim import (
    load_manifest, manifest_to_sim2sim_cfg,
    build_mujoco_model, quat_to_euler_xyz, scale_transform,
)


# Recovery obs builder — slot-driven from manifest obs_slots so the layout
# stays in sync with whatever recovery config produced the policy.
def build_recovery_obs(data, current_joint_act, ref_joint_pos, episode_progress, obs_slots):
    quat_wxyz = data.qpos[3:7].copy()
    euler = quat_to_euler_xyz(quat_wxyz)
    base_ang_vel = data.qvel[3:6].copy()
    q  = data.qpos[7:17].copy()
    qd = data.qvel[6:16].copy()
    from sim2sim import quat_rotate_inverse
    g_body = quat_rotate_inverse(quat_wxyz, np.array([0., 0., -1.], dtype=np.float32))
    # Scaling matches env/obs_builder.py: base_ang_vel × 0.5, joint_vel × 0.1
    by_name = {
        'base_euler':         euler[:2].astype(np.float32),
        'base_ang_vel':       (base_ang_vel * 0.5).astype(np.float32),
        'joint_pos_err':      (q - ref_joint_pos).astype(np.float32),
        'joint_vel':          (qd * 0.1).astype(np.float32),
        'joint_tracking_err': (current_joint_act - q).astype(np.float32),
        'projected_gravity':  g_body.astype(np.float32),
        'episode_progress':   np.array([episode_progress], dtype=np.float32),
        'joint_pos_abs':      q.astype(np.float32),
    }
    return np.concatenate([by_name[s] for s in obs_slots])


def run_recovery_episode(model, data, session, input_name, cfg, init_state, seed=0):
    """One recovery episode from a fallen pose. Returns dict of metrics."""
    rng = np.random.default_rng(seed)

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
    success_pose_err = float(cfg['success_pose_err'])
    tilt_deg      = cfg['success_tilt_deg']
    duration      = cfg['episode_length_s']
    obs_history   = cfg['obs_history']
    obs_per_step  = cfg['obs_per_step']
    obs_slots     = cfg['obs_slots']

    # Reset to fallen pose
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = init_state['base_pos']
    data.qpos[3:7] = init_state['base_quat']  # wxyz
    data.qpos[7:17] = init_state['joint_pos']
    data.qvel[0:3] = init_state['base_lin_vel']
    data.qvel[3:6] = init_state['base_ang_vel']
    data.qvel[6:16] = init_state['joint_vel']
    mujoco.mj_forward(model, data)

    current_joint_act = data.qpos[7:17].copy().astype(np.float32)
    lp_target = current_joint_act.copy()
    obs_hist = np.zeros((obs_history, obs_per_step), dtype=np.float32)

    # Track upright at every policy step
    n_policy_steps = int(duration / (sim_dt * decimation))
    z_log = np.zeros(n_policy_steps + 1, dtype=np.float32)
    tilt_log = np.zeros(n_policy_steps + 1, dtype=np.float32)
    upright_log = np.zeros(n_policy_steps + 1, dtype=bool)
    # Torque tracking for sim2real diagnostics:
    #   peak_tau     = max |τ_i| seen on any joint, in Nm (raw commanded)
    #   peak_ratio   = max |τ_i| / effort_i — most useful number: >1 means policy
    #                  is requesting more than the motor can deliver, sim is
    #                  silently clipping for it, and real robot will saturate.
    #   sum_abs_tau, n_tau_samples → mean across all (joint, sim_step) samples
    effort_limits = np.abs(model.actuator_forcerange[:10, 1]).astype(np.float32)
    peak_tau = 0.0
    peak_ratio = 0.0
    sum_abs_tau = 0.0
    n_tau_samples = 0
    success_tilt_rad = np.deg2rad(tilt_deg)
    pose_err_log = np.zeros(n_policy_steps + 1, dtype=np.float32)

    def _measure(idx):
        z_log[idx] = data.qpos[2]
        # tilt = angle between body up and world up
        # body up in world = R^T * (0,0,1). Use projected_gravity sign:
        # if gravity in body is (0,0,-1) → upright; tilt = arccos(-pg_z)
        from sim2sim import quat_rotate_inverse
        g_body = quat_rotate_inverse(data.qpos[3:7].astype(np.float32),
                                     np.array([0., 0., -1.], dtype=np.float32))
        cos_tilt = float(np.clip(-g_body[2], -1.0, 1.0))
        tilt = float(np.arccos(cos_tilt))
        tilt_log[idx] = np.rad2deg(tilt)
        # Success = pose-match AND tilt-match (z still logged, but no longer gated).
        pose_err = float(np.linalg.norm(data.qpos[7:17] - ref_joint_pos))
        pose_err_log[idx] = pose_err
        upright_log[idx] = (pose_err < success_pose_err) and (tilt < success_tilt_rad)

    _measure(0)

    for t in range(n_policy_steps):
        # Policy obs
        episode_progress = (t + 1) / n_policy_steps  # 0→1 over episode
        obs_step = build_recovery_obs(data, current_joint_act, ref_joint_pos,
                                      episode_progress, obs_slots)
        # Shift history (newest at end)
        obs_hist = np.roll(obs_hist, -1, axis=0)
        obs_hist[-1] = obs_step
        obs_flat = obs_hist.reshape(1, -1).astype(np.float32)

        # Policy forward
        mu = session.run(None, {input_name: obs_flat})[0][0]  # 10-dim
        # Absolute mode: scale mu in [-1,1] to [abs_low, abs_high]
        target = scale_transform(mu, abs_low, abs_high, clip_val=1.0)
        # Low-pass
        lp_target = lp_alpha * target + (1.0 - lp_alpha) * lp_target
        current_joint_act = np.clip(lp_target, jlim_low, jlim_high).astype(np.float32)

        # PD inner loop at sim_dt
        for _ in range(decimation):
            q  = data.qpos[7:17].astype(np.float32)
            qd = data.qvel[6:16].astype(np.float32)
            tau = kps * (current_joint_act - q) - kds * qd
            data.ctrl[:10] = tau
            mujoco.mj_step(model, data)
            abs_tau = np.abs(tau)
            peak_tau = max(peak_tau, float(abs_tau.max()))
            peak_ratio = max(peak_ratio, float((abs_tau / effort_limits).max()))
            sum_abs_tau += float(abs_tau.sum())
            n_tau_samples += abs_tau.size

        _measure(t + 1)

    # Metrics
    final_idx = n_policy_steps
    success = bool(upright_log[final_idx])
    ever_upright = bool(upright_log.any())
    if ever_upright:
        first_upright_step = int(np.argmax(upright_log))  # first True
        time_to_upright = first_upright_step * sim_dt * decimation
    else:
        time_to_upright = float('nan')

    mean_abs_tau = sum_abs_tau / max(n_tau_samples, 1)
    return {
        'success': success,
        'ever_upright': ever_upright,
        'final_z': float(z_log[final_idx]),
        'final_tilt_deg': float(tilt_log[final_idx]),
        'final_pose_err': float(pose_err_log[final_idx]),
        'time_to_upright_s': time_to_upright,
        'peak_tau_nm': peak_tau,
        'peak_tau_ratio': peak_ratio,
        'mean_abs_tau_nm': mean_abs_tau,
        'pose_label': str(init_state.get('pose_label', '?')),
    }


def quick_eval_recovery(onnx_path, sim_cfg, manifest=None, runs=100, init_pool_path=None):
    """In-training recovery eval. Returns flat dict of TB-friendly scalar metrics.
    Used by train.py instead of the walking quick_eval when task_type='Recovery'.

    `manifest` carries the recovery block + abs_low/abs_high (not exposed in sim_cfg)."""
    rec = (manifest or {}).get('recovery', {}) if manifest else {}
    abs_low = abs_high = None
    if manifest is not None:
        scaling = manifest.get('action_scaling', {}) or {}
        abs_low = scaling.get('abs_low')
        abs_high = scaling.get('abs_high')
    cfg = {
        'simulation_dt':       sim_cfg['simulation_dt'],
        'control_decimation':  sim_cfg['control_decimation'],
        'ref_joint_pos':       sim_cfg['ref_joint_pos'],
        'kps':                 sim_cfg['kps'],
        'kds':                 sim_cfg['kds'],
        'abs_low':             abs_low  or sim_cfg['joint_limit_low'],
        'abs_high':            abs_high or sim_cfg['joint_limit_high'],
        'joint_limit_low':     sim_cfg['joint_limit_low'],
        'joint_limit_high':    sim_cfg['joint_limit_high'],
        'action_lowpass_alpha': sim_cfg['action_lowpass_alpha'],
        'obs_history':         sim_cfg['obs_history'],
        'obs_per_step':        sim_cfg['num_obs_per_step'],
        'obs_slots':           sim_cfg['obs_slots'],
        'target_height':       float(rec.get('target_height', 0.45)),
        'success_pose_err':    float(rec.get('success_pose_err', 0.3)),
        'success_tilt_deg':    float(rec.get('success_tilt_deg', 25.0)),
        'episode_length_s':    float(rec.get('episode_length_s', 5.0)),
    }

    # Pool resolution priority: explicit arg → manifest's recovery.init_states_path
    # → data/recovery_init_states.npz fallback. The manifest path is what was
    # used in training, so auto-eval evaluates on the SAME pool by default.
    if init_pool_path is None:
        repo_root = os.path.normpath(os.path.join(os.path.dirname(_HERE), '..'))
        rec_path = rec.get('init_states_path') if isinstance(rec, dict) else None
        if rec_path:
            init_pool_path = rec_path if os.path.isabs(rec_path) else os.path.join(repo_root, rec_path)
        else:
            init_pool_path = os.path.join(repo_root, 'data', 'recovery_init_states.npz')

    model = build_mujoco_model(sim_cfg['urdf_path'], sim_cfg['simulation_dt'],
                               init_height=sim_cfg['init_height'],
                               floor_friction=1.0, fix_base=False)
    data = mujoco.MjData(model)
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    pool = np.load(init_pool_path)
    n_pool = pool['base_pos'].shape[0]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_pool, size=runs, replace=(runs > n_pool))

    results = []
    for run_i, idx in enumerate(sample_idx):
        init_state = {
            'base_pos':     pool['base_pos'][idx],
            'base_quat':    pool['base_quat'][idx],
            'base_lin_vel': pool['base_lin_vel'][idx],
            'base_ang_vel': pool['base_ang_vel'][idx],
            'joint_pos':    pool['joint_pos'][idx],
            'joint_vel':    pool['joint_vel'][idx],
            'pose_label':   pool['pose_label'][idx],
        }
        results.append(run_recovery_episode(model, data, session, input_name, cfg, init_state, seed=run_i))

    n = len(results)
    succ_rate = sum(r['success'] for r in results) / n
    ever_rate = sum(r['ever_upright'] for r in results) / n
    mean_z    = float(np.mean([r['final_z'] for r in results]))
    mean_tilt = float(np.mean([r['final_tilt_deg'] for r in results]))
    mean_pose_err = float(np.mean([r['final_pose_err'] for r in results]))
    ttus = [r['time_to_upright_s'] for r in results if r['ever_upright']]
    mean_ttu = float(np.mean(ttus)) if ttus else float('nan')
    mean_peak_tau   = float(np.mean([r['peak_tau_nm']    for r in results]))
    mean_avg_tau    = float(np.mean([r['mean_abs_tau_nm'] for r in results]))
    mean_peak_ratio = float(np.mean([r['peak_tau_ratio'] for r in results]))
    max_peak_ratio  = float(np.max ([r['peak_tau_ratio'] for r in results]))

    return {
        'sim2sim/recovery_success_rate': succ_rate,
        'sim2sim/recovery_ever_upright_rate': ever_rate,
        'sim2sim/recovery_mean_final_z': mean_z,
        'sim2sim/recovery_mean_final_pose_err': mean_pose_err,
        'sim2sim/recovery_mean_final_tilt_deg': mean_tilt,
        'sim2sim/recovery_mean_time_to_upright_s': mean_ttu,
        'sim2sim/recovery_mean_peak_tau_nm':    mean_peak_tau,
        'sim2sim/recovery_mean_abs_tau_nm':     mean_avg_tau,
        'sim2sim/recovery_mean_peak_tau_ratio': mean_peak_ratio,
        'sim2sim/recovery_max_peak_tau_ratio':  max_peak_ratio,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--init_pool', type=str,
                    default=os.path.join(os.path.dirname(_HERE), '..', 'data', 'recovery_init_states.npz'))
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--floor_friction', type=float, default=1.0)
    ap.add_argument('--by_label', action='store_true', help='also report success rate per pose_label')
    args = ap.parse_args()

    # Load manifest
    manifest, manifest_path = load_manifest(args.policy)
    if manifest is None:
        print(f'ERROR: no manifest found next to {args.policy}', file=sys.stderr)
        return 1
    if manifest.get('task_type') != 'Recovery':
        print(f"ERROR: expected task_type=Recovery in manifest, got {manifest.get('task_type')}",
              file=sys.stderr)
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

    print(f'[sim2sim_recovery] policy: {args.policy}')
    print(f'[sim2sim_recovery] manifest: {manifest_path}')
    print(f'[sim2sim_recovery] success: ||q-q_ref|| < {cfg["success_pose_err"]:.2f} '
          f'AND tilt < {cfg["success_tilt_deg"]:.1f}°  at t={cfg["episode_length_s"]}s')

    # Build model + ONNX
    model = build_mujoco_model(s2s_cfg['urdf_path'], s2s_cfg['simulation_dt'],
                               init_height=s2s_cfg['init_height'],
                               floor_friction=args.floor_friction, fix_base=False)
    data = mujoco.MjData(model)
    session = ort.InferenceSession(args.policy)
    input_name = session.get_inputs()[0].name

    # Load init pool
    pool = np.load(args.init_pool)
    n_pool = pool['base_pos'].shape[0]
    print(f'[sim2sim_recovery] init pool: {n_pool} states from {args.init_pool}')

    # Sample N episodes
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n_pool, size=args.runs, replace=(args.runs > n_pool))

    results = []
    for run_i, idx in enumerate(sample_idx):
        init_state = {
            'base_pos':     pool['base_pos'][idx],
            'base_quat':    pool['base_quat'][idx],
            'base_lin_vel': pool['base_lin_vel'][idx],
            'base_ang_vel': pool['base_ang_vel'][idx],
            'joint_pos':    pool['joint_pos'][idx],
            'joint_vel':    pool['joint_vel'][idx],
            'pose_label':   pool['pose_label'][idx],
        }
        m = run_recovery_episode(model, data, session, input_name, cfg, init_state, seed=run_i)
        results.append(m)
        if (run_i + 1) % 10 == 0 or run_i == args.runs - 1:
            so_far = sum(1 for r in results if r['success']) / len(results)
            print(f'  [{run_i+1:3d}/{args.runs}] success_so_far={so_far:.2%}  '
                  f'last: label={m["pose_label"]:<10s} success={m["success"]} '
                  f'final_z={m["final_z"]:.3f} tilt={m["final_tilt_deg"]:.1f}°')

    # Aggregate
    n = len(results)
    succ = sum(1 for r in results if r['success']) / n
    ever = sum(1 for r in results if r['ever_upright']) / n
    mean_z = float(np.mean([r['final_z'] for r in results]))
    mean_tilt = float(np.mean([r['final_tilt_deg'] for r in results]))
    ttus = [r['time_to_upright_s'] for r in results if r['ever_upright']]
    mean_ttu = float(np.mean(ttus)) if ttus else float('nan')
    mean_peak_tau   = float(np.mean([r['peak_tau_nm'] for r in results]))
    max_peak_tau    = float(np.max ([r['peak_tau_nm'] for r in results]))
    mean_avg_tau    = float(np.mean([r['mean_abs_tau_nm'] for r in results]))
    mean_peak_ratio = float(np.mean([r['peak_tau_ratio'] for r in results]))
    max_peak_ratio  = float(np.max ([r['peak_tau_ratio'] for r in results]))

    print()
    print('=== Recovery sim2sim results ===')
    print(f'  runs                  : {n}')
    print(f'  success_rate          : {succ:.2%}')
    print(f'  ever_upright_rate     : {ever:.2%}')
    mean_pose_err = float(np.mean([r['final_pose_err'] for r in results]))
    print(f'  mean_final_z          : {mean_z:.3f} m  (informational; not gated)')
    print(f'  mean_final_pose_err   : {mean_pose_err:.3f}    (success threshold {cfg["success_pose_err"]:.2f})')
    print(f'  mean_final_tilt       : {mean_tilt:.1f}° (success threshold {cfg["success_tilt_deg"]:.1f})')
    print(f'  mean_time_to_upright  : {mean_ttu:.2f} s  (over {len(ttus)} ever-upright runs)')
    print(f'  mean peak |τ|         : {mean_peak_tau:.1f} Nm   (worst-run peak: {max_peak_tau:.1f} Nm)')
    print(f'  mean avg  |τ|         : {mean_avg_tau:.1f} Nm')
    print(f'  mean peak |τ|/effort  : {mean_peak_ratio:.2f}×   (worst-run: {max_peak_ratio:.2f}×)')
    print(f'    >1.0 means policy is requesting more than the motor can deliver,')
    print(f'    sim is silently clipping for it, real robot will saturate.')

    if args.by_label:
        from collections import defaultdict
        per_label = defaultdict(list)
        for r in results:
            per_label[r['pose_label']].append(r)
        print('\n  per-label success_rate:')
        for label, rs in sorted(per_label.items()):
            sr = sum(1 for r in rs if r['success']) / len(rs)
            print(f'    {label:<12s} n={len(rs):3d}  success={sr:.2%}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
