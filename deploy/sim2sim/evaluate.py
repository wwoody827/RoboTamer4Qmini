"""
Policy evaluation across multiple conditions.

Runs sim2sim headlessly over a test matrix (friction × cmd_vx × cmd_yaw),
each condition repeated N times, outputs a CSV summary, and prints a
breakdown report with optional matplotlib plots.

Usage:
    python deploy/sim2sim/evaluate.py [--config deploy/sim2sim/configs/qmini_birl.yaml]
                                      [--runs 10]
                                      [--duration 10]
                                      [--out experiments/my_run/eval.csv]
                                      [--no-plots]
"""

import os
import sys
import argparse
import itertools
import csv
import math
from collections import deque

import numpy as np
import mujoco
import yaml

# reuse sim2sim helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim import (
    build_mujoco_model, PhaseModulator,
    quat_to_euler_xyz, quat_rotate_inverse, scale_transform,
)

import onnxruntime as ort

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(cfg, cmd_vx, cmd_yaw, floor_friction, duration, seed=None):
    """Run one episode. Returns a dict of scalar metrics."""
    if seed is not None:
        np.random.seed(seed)

    sim_dt     = cfg['simulation_dt']
    decimation = cfg['control_decimation']
    policy_dt  = sim_dt * decimation

    ref_joint  = np.array(cfg['ref_joint_pos'],   dtype=np.float32)
    kps        = np.array(cfg['kps'],              dtype=np.float32)
    kds        = np.array(cfg['kds'],              dtype=np.float32)
    tor_offset = np.array(cfg['joint_tor_offset'], dtype=np.float32)
    vel_sign   = np.array(cfg['joint_vel_sign'],   dtype=np.float32)
    act_low    = np.array(cfg['action_inc_low'],   dtype=np.float32)
    act_high   = np.array(cfg['action_inc_high'],  dtype=np.float32)
    jlim_low   = np.array(cfg['joint_limit_low'],  dtype=np.float32)
    jlim_high  = np.array(cfg['joint_limit_high'], dtype=np.float32)
    num_legs   = cfg['num_legs']
    obs_hist   = cfg['obs_history']
    obs_dim    = cfg['num_obs_per_step']
    static_thr = cfg['static_cmd_threshold']

    commands    = np.array([cmd_vx, cmd_yaw], dtype=np.float32)
    static_flag = float(np.linalg.norm(commands) >= static_thr)

    session    = ort.InferenceSession(cfg['policy_path'])
    input_name = session.get_inputs()[0].name

    model = build_mujoco_model(cfg['urdf_path'], sim_dt, cfg['init_height'],
                               floor_friction=floor_friction)
    data  = mujoco.MjData(model)

    NUM_JOINTS = 10
    QPOS_START = 7
    QVEL_START = 6
    imu_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'imu_in_torso')

    mujoco.mj_resetData(model, data)
    data.qpos[QPOS_START:QPOS_START + NUM_JOINTS] = ref_joint
    mujoco.mj_forward(model, data)

    pm = PhaseModulator(dt=policy_dt, num_legs=num_legs)
    pm.reset()
    current_joint_act = ref_joint.copy()
    obs_history = deque(maxlen=obs_hist)
    for _ in range(obs_hist):
        obs_history.append(np.zeros(obs_dim, dtype=np.float32))

    def get_obs():
        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        if imu_body_id >= 0:
            quat         = data.xquat[imu_body_id]
            world_angvel = data.cvel[imu_body_id][0:3]
        else:
            quat         = data.qpos[3:7]
            world_angvel = data.qvel[3:6]
        euler        = quat_to_euler_xyz(quat)
        base_euler   = euler[:2]
        base_ang_vel = quat_rotate_inverse(quat, world_angvel)
        pm_phase_val = np.concatenate([np.sin(pm.phase), np.cos(pm.phase)]) * static_flag
        pm_f_val     = (pm.frequency * 0.3 - 1.0) * static_flag
        obs = np.concatenate([
            commands, base_euler, base_ang_vel * 0.5,
            q - ref_joint, dq * 0.1, current_joint_act - q,
            pm_phase_val, pm_f_val,
        ]).astype(np.float32)
        return np.clip(obs, -3.0, 3.0)

    def compute_torques(target_q, q, dq):
        error = target_q - q
        return kps * error + kds - dq + tor_offset - 3.5 * np.sign(dq) * vel_sign

    total_steps = int(duration / sim_dt)
    fall_thresh = 0.25

    vx_errors    = []
    vy_abs       = []
    roll_rms_acc = []
    pitch_rms_acc= []
    torque_acc   = []
    survived     = True
    survive_steps= total_steps

    for step in range(total_steps):
        if step % decimation == 0:
            obs_now = get_obs()
            obs_history.append(obs_now)
            obs_stacked = np.concatenate(list(obs_history))[np.newaxis, :]
            net_out = session.run(None, {input_name: obs_stacked})[0][0]
            scaled  = scale_transform(net_out, act_low, act_high)
            pm.compute(scaled[:num_legs])
            current_joint_act[:] += scaled[num_legs:] * policy_dt
            current_joint_act[:] = np.clip(current_joint_act, jlim_low, jlim_high)
            static_flag = float(np.linalg.norm(commands) >= static_thr)

        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        torques = compute_torques(current_joint_act, q, dq)
        data.ctrl[:NUM_JOINTS] = torques
        mujoco.mj_step(model, data)

        z = data.qpos[2]
        if z < fall_thresh:
            survived = False
            survive_steps = step
            break

        if step % decimation == 0:
            vx_errors.append(abs(data.qvel[0] - cmd_vx))
            vy_abs.append(abs(data.qvel[1]))
            quat  = data.xquat[imu_body_id] if imu_body_id >= 0 else data.qpos[3:7]
            euler = quat_to_euler_xyz(quat)
            roll_rms_acc.append(euler[0] ** 2)
            pitch_rms_acc.append(euler[1] ** 2)
            torque_acc.append(np.sum(np.abs(torques * dq)))

    x_final = data.qpos[0]
    y_final = data.qpos[1]

    total_mass = 7.0
    g = 9.81
    dx = abs(x_final)
    cot = (np.sum(torque_acc) * policy_dt) / (total_mass * g * dx) if dx > 0.05 else float('nan')

    return {
        'survived':       int(survived),
        'survive_time':   survive_steps * sim_dt,
        'x_final':        x_final,
        'y_final':        y_final,
        'vx_error_mean':  float(np.mean(vx_errors)) if vx_errors else float('nan'),
        'vy_abs_mean':    float(np.mean(vy_abs))    if vy_abs    else float('nan'),
        'roll_rms':       float(np.sqrt(np.mean(roll_rms_acc)))  if roll_rms_acc  else float('nan'),
        'pitch_rms':      float(np.sqrt(np.mean(pitch_rms_acc))) if pitch_rms_acc else float('nan'),
        'cot':            cot,
    }


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(cfg, runs, duration, frictions, vx_list, yaw_list, out_path):
    conditions = list(itertools.product(frictions, vx_list, yaw_list))
    total = len(conditions) * runs
    print(f"Evaluating {len(conditions)} conditions × {runs} runs = {total} episodes")
    print(f"Policy: {cfg['policy_path']}\n")

    fieldnames = [
        'friction', 'cmd_vx', 'cmd_yaw', 'run',
        'survived', 'survive_time', 'x_final', 'y_final',
        'vx_error_mean', 'vy_abs_mean', 'roll_rms', 'pitch_rms', 'cot',
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        done = 0
        for friction, cmd_vx, cmd_yaw in conditions:
            results = []
            for run_i in range(runs):
                metrics = run_episode(cfg, cmd_vx, cmd_yaw, friction, duration, seed=run_i)
                row = {'friction': friction, 'cmd_vx': cmd_vx, 'cmd_yaw': cmd_yaw,
                       'run': run_i, **metrics}
                writer.writerow(row)
                f.flush()
                results.append(metrics)
                done += 1

            surv_rate = np.mean([r['survived'] for r in results])
            vx_err    = np.nanmean([r['vx_error_mean'] for r in results])
            vy_drift  = np.nanmean([r['vy_abs_mean'] for r in results])
            roll      = np.nanmean([r['roll_rms'] for r in results])
            print(f"friction={friction:.1f} vx={cmd_vx:+.1f} yaw={cmd_yaw:+.1f} | "
                  f"survival={surv_rate*100:.0f}%  vx_err={vx_err:.3f}  "
                  f"vy_drift={vy_drift:.3f}  roll_rms={np.degrees(roll):.1f}deg  "
                  f"[{done}/{total}]")

    print(f"\nResults saved to: {out_path}")
    return out_path


# ── report ────────────────────────────────────────────────────────────────────

def _bar(value, lo, hi, width=20):
    frac = max(0., min(1., (value - lo) / (hi - lo + 1e-9)))
    filled = round(frac * width)
    return '[' + '#' * filled + '.' * (width - filled) + ']'


def print_report(csv_path):
    try:
        import pandas as pd
    except ImportError:
        print('[report] pandas not available — skipping breakdown report')
        return

    df = pd.read_csv(csv_path)
    frictions = sorted(df['friction'].unique())
    vx_vals   = sorted(df['cmd_vx'].unique())

    sep = '+----------+---------+-----+---------+----------+-----------+-----------+------------+----------+'
    hdr = ('| friction | cmd_vx  |  N  |  Surv%  |  vx_err  |  vy_drift |  roll°rms |  pitch°rms |   CoT    |')

    print('\n' + '=' * len(sep))
    print(' Breakdown Report')
    print('=' * len(sep))
    print(sep)
    print(hdr)
    print(sep)

    for fr in frictions:
        for vx in vx_vals:
            sub = df[(df['friction'] == fr) & (df['cmd_vx'] == vx)]
            if sub.empty:
                continue
            n     = len(sub)
            surv  = sub['survived'].mean() * 100
            vxe   = sub['vx_error_mean'].mean()
            vy    = sub['vy_abs_mean'].mean()
            roll  = math.degrees(sub['roll_rms'].mean())
            pitch = math.degrees(sub['pitch_rms'].mean())
            cot   = sub['cot'].mean()
            cot_s = f'{cot:.2f}' if not math.isnan(cot) else ' nan'
            flag  = ' ' if surv == 100 else ('!' if surv >= 50 else 'X')
            print(f'| {fr:^8.1f} | {vx:^+7.1f} | {n:^3} |'
                  f' {flag}{surv:5.0f}% |'
                  f'  {vxe:6.3f}  |'
                  f'   {vy:6.3f}  |'
                  f'   {roll:6.1f}   |'
                  f'   {pitch:7.1f}   |'
                  f' {cot_s:^8} |')
        print(sep)

    # ASCII bar summaries
    print('\n── Survival rate by friction ──')
    for fr in frictions:
        rate = df[df['friction'] == fr]['survived'].mean()
        print(f'  {fr:.1f}  {_bar(rate, 0, 1)}  {rate*100:5.1f}%')

    sub_ok = df[(df['friction'] <= 1.5) & (df['survived'] == 1)]

    print('\n── vx tracking error (friction≤1.5, survived) ──')
    for vx in vx_vals:
        s = sub_ok[sub_ok['cmd_vx'] == vx]
        if s.empty:
            continue
        mu, std = s['vx_error_mean'].mean(), s['vx_error_mean'].std()
        print(f'  {vx:+.1f}  {_bar(mu, 0, 0.7)}  {mu:.3f} ± {std:.3f} m/s')

    print('\n── lateral drift vy (friction≤1.5, survived) ──')
    for vx in vx_vals:
        s = sub_ok[sub_ok['cmd_vx'] == vx]
        if s.empty:
            continue
        mu, std = s['vy_abs_mean'].mean(), s['vy_abs_mean'].std()
        print(f'  {vx:+.1f}  {_bar(mu, 0, 0.5)}  {mu:.3f} ± {std:.3f} m/s')


def make_plots(csv_path):
    if not HAS_MPL:
        print('[report] matplotlib not available — skipping plots')
        return
    try:
        import pandas as pd
    except ImportError:
        print('[report] pandas not available — skipping plots')
        return

    df = pd.read_csv(csv_path)
    frictions = sorted(df['friction'].unique())
    vx_vals   = sorted(df['cmd_vx'].unique())
    out_dir   = os.path.dirname(os.path.abspath(csv_path))

    # 1. Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Policy Evaluation Heatmaps  (rows=friction, cols=cmd_vx)', fontsize=13)

    specs = [
        ('survived',      'Survival rate (%)',  True,  plt.cm.RdYlGn),
        ('vx_error_mean', 'vx tracking err',    False, plt.cm.RdYlGn_r),
        ('vy_abs_mean',   'lateral drift (m/s)',False, plt.cm.RdYlGn_r),
        ('roll_rms',      'roll RMS (deg)',      False, plt.cm.RdYlGn_r),
    ]
    for ax, (col, title, pct, cmap) in zip(axes.flat, specs):
        mat = np.full((len(frictions), len(vx_vals)), np.nan)
        for i, fr in enumerate(frictions):
            for j, vx in enumerate(vx_vals):
                sub = df[(df['friction'] == fr) & (df['cmd_vx'] == vx)]
                if sub.empty:
                    continue
                v = sub[col].mean()
                if col in ('roll_rms', 'pitch_rms'):
                    v = math.degrees(v)
                if pct:
                    v *= 100
                mat[i, j] = v
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0,
                       vmax=(100 if pct else None))
        ax.set_xticks(range(len(vx_vals)))
        ax.set_xticklabels([f'{v:+.1f}' for v in vx_vals])
        ax.set_yticks(range(len(frictions)))
        ax.set_yticklabels([f'{f:.1f}' for f in frictions])
        ax.set_xlabel('cmd_vx (m/s)')
        ax.set_ylabel('friction')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i in range(len(frictions)):
            for j in range(len(vx_vals)):
                v = mat[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                            fontsize=8, color='black')
    plt.tight_layout()
    p = os.path.join(out_dir, 'eval_heatmaps.png')
    plt.savefig(p, dpi=120); plt.close()
    print(f'  saved: {p}')

    # 2. Grouped bar: vx_err and vy_drift
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Velocity tracking & lateral drift (survived)', fontsize=12)
    for ax, (col, ylabel) in zip(axes, [('vx_error_mean', 'vx error (m/s)'),
                                         ('vy_abs_mean',   'vy drift (m/s)')]):
        x = np.arange(len(vx_vals))
        width = 0.8 / len(frictions)
        for k, fr in enumerate(frictions):
            means, stds = [], []
            for vx in vx_vals:
                s = df[(df['friction'] == fr) & (df['cmd_vx'] == vx) & (df['survived'] == 1)]
                means.append(s[col].mean() if not s.empty else np.nan)
                stds.append(s[col].std()   if not s.empty else np.nan)
            off = (k - len(frictions) / 2 + 0.5) * width
            ax.bar(x + off, means, width, yerr=stds, label=f'fr={fr:.1f}',
                   capsize=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:+.1f}' for v in vx_vals])
        ax.set_xlabel('cmd_vx (m/s)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    p = os.path.join(out_dir, 'eval_tracking_bars.png')
    plt.savefig(p, dpi=120); plt.close()
    print(f'  saved: {p}')

    # 3. Histograms (friction≤1.5, forward, survived)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Metric distributions (friction≤1.5, vx>0, survived)', fontsize=11)
    sub_ok = df[(df['friction'] <= 1.5) & (df['cmd_vx'] > 0) & (df['survived'] == 1)]
    for ax, (col, xlabel) in zip(axes, [('vx_error_mean', 'vx error (m/s)'),
                                         ('vy_abs_mean',   'vy drift (m/s)'),
                                         ('roll_rms',      'roll RMS (rad)')]):
        data = sub_ok[col].dropna()
        ax.hist(data, bins=20, edgecolor='white', alpha=0.85, color='steelblue')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'mean={data.mean():.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'eval_histograms.png')
    plt.savefig(p, dpi=120); plt.close()
    print(f'  saved: {p}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='deploy/sim2sim/configs/qmini_birl.yaml')
    parser.add_argument('--runs',     type=int,   default=10,   help='Runs per condition')
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds per episode')
    parser.add_argument('--out',      default=None, help='Output CSV path')
    parser.add_argument('--no-plots', action='store_true', help='Skip matplotlib plots')
    parser.add_argument('--report-only', default=None, metavar='CSV',
                        help='Skip evaluation; print report for existing CSV')
    args = parser.parse_args()

    if args.report_only:
        print_report(args.report_only)
        if not args.no_plots:
            make_plots(args.report_only)
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.out is None:
        policy_dir = os.path.dirname(cfg['policy_path'])
        args.out = os.path.join(policy_dir, 'eval.csv')

    frictions = [0.5, 1.0, 1.5, 3.0]
    vx_list   = [-0.3, 0.0, 0.3, 0.5, 0.7]
    yaw_list  = [0.0]

    csv_path = evaluate(cfg, args.runs, args.duration, frictions, vx_list, yaw_list, args.out)

    print_report(csv_path)
    if not args.no_plots:
        print(f'\nGenerating plots → {os.path.dirname(os.path.abspath(csv_path))}')
        make_plots(csv_path)


if __name__ == '__main__':
    main()
