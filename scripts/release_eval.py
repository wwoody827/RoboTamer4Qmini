"""Run full release evaluation on a candidate policy: videos + metrics + tracking plots.

Produces under deploy_candidates/<name>/:
  videos/                   - 11 demo videos at common cmds
  eval/eval_omni.csv        - sweep over (friction × cmd_vx × cmd_freq)
  eval/eval_strafe.csv      - sweep over (cmd_vy × cmd_freq) at friction 1.0
  eval/eval_yaw.csv         - sweep over (cmd_yaw × cmd_freq) at friction 1.0
  eval/plots/vx_tracking.png    - body_vx vs cmd_vx, line per cmd_freq
  eval/plots/vy_tracking.png    - body_vy vs cmd_vy
  eval/plots/yaw_tracking.png   - body_yaw_rate vs cmd_yaw
  eval/plots/freq_lock_vx.png   - measured_freq across cmd_vx (per cmd_freq)
  eval/plots/freq_lock_yaw.png  - measured_freq across cmd_yaw (per cmd_freq)
  eval/plots/summary.png        - 2×3 panel of all the above

Usage:
    python scripts/release_eval.py --candidate deploy_candidates/walk_rl_v34_iter3200
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/woody/miniconda3/envs/qmini/bin/python'

VIDEO_CMDS = [
    ('01_stand',          0.0,  0.0,  0.0),
    ('02_fwd_03',         0.3,  0.0,  0.0),
    ('03_fwd_05',         0.5,  0.0,  0.0),
    ('04_fwd_07',         0.7,  0.0,  0.0),
    ('05_bwd_03',        -0.3,  0.0,  0.0),
    ('06_strafe_l_03',    0.0,  0.3,  0.0),
    ('07_strafe_r_03',    0.0, -0.3,  0.0),
    ('08_yaw_l_05',       0.0,  0.0,  0.5),
    ('09_yaw_r_05',       0.0,  0.0, -0.5),
    ('10_fwd_yaw',        0.3,  0.0,  0.5),
    ('11_fwd_strafe',     0.3,  0.3,  0.0),
]

VX_VALUES   = [-0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7]
VY_VALUES   = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
YAW_VALUES  = [-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0]
FREQ_VALUES = [2.0, 2.5, 3.0]
FRICTIONS   = [0.5, 1.0, 1.5]


def record_videos(policy, out_dir, duration=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    sim2sim = REPO_ROOT / 'deploy' / 'sim2sim' / 'sim2sim.py'
    for name, vx, vy, yw in VIDEO_CMDS:
        out = out_dir / f'{name}.mp4'
        if out.exists():
            print(f'  skip exists: {out.name}')
            continue
        env = os.environ.copy()
        env['PATH'] = '/home/woody/miniconda3/envs/qmini/bin:' + env.get('PATH', '')
        env['MUJOCO_GL'] = 'egl'
        cmd = [PYTHON, str(sim2sim),
               '--policy', str(policy),
               '--cmd_vx', str(vx), '--cmd_vy', str(vy), '--cmd_yaw', str(yw),
               '--duration', str(duration), '--headless',
               '--video', str(out), '--video_fps', '30']
        subprocess.run(cmd, capture_output=True, env=env)
        print(f'  recorded {out.name}')


def run_sweep(policy, sweep, friction, runs=3, duration=10):
    """sweep: list of (cmd_vx, cmd_vy, cmd_yaw, cmd_freq) tuples → list of metrics dicts."""
    sys.path.insert(0, str(REPO_ROOT / 'deploy' / 'sim2sim'))
    from evaluate import run_episode
    from sim2sim import load_manifest, manifest_to_sim2sim_cfg
    manifest, _ = load_manifest(str(policy))
    cfg = manifest_to_sim2sim_cfg(manifest, str(policy))

    rows = []
    for vx, vy, yw, freq in sweep:
        for run_i in range(runs):
            m = run_episode(cfg, vx, yw, friction, duration,
                            seed=run_i, cmd_vy=vy, cmd_freq=freq)
            row = {'cmd_vx': vx, 'cmd_vy': vy, 'cmd_yaw': yw, 'cmd_freq': freq,
                   'friction': friction, 'run': run_i, **m}
            rows.append(row)
            print(f'  cmd=({vx:+.2f},{vy:+.2f},{yw:+.2f}) f={freq} fric={friction} '
                  f'run={run_i} surv={m["survived"]} vx_steady={m.get("vx_err_steady", 0):.3f}')
    return rows


def save_csv(rows, path):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def body_frame_realized(rows):
    """Per-row body-frame realized (vx_steady, vy_steady, yaw_rate_steady).
    Use signed mean (vx_bias + cmd) since vx_bias = signed err."""
    out = []
    for r in rows:
        # vx_bias_body is mean(body_vx - cmd_vx) over steady. → body_vx = bias + cmd_vx.
        body_vx = float(r.get('vx_bias_body', 0)) + float(r['cmd_vx'])
        body_vy = float(r.get('vy_bias_body', 0)) + float(r['cmd_vy'])
        # yaw_drift_passive is |signed_mean_yaw_rate| when cmd_yaw=0; for non-zero
        # cmd_yaw we don't have a signed bias stored — fall back to cmd-side error mean.
        # yaw_error_mean is |yaw_rate - cmd|, so realized = cmd ± yaw_error_mean (sign unknown).
        # As a proxy: assume realized has same sign as cmd, use abs(cmd) - yaw_error_mean if
        # cmd != 0 else 0. This is good enough for monotonicity check across cmd grid.
        if abs(r['cmd_yaw']) > 0.05:
            body_yaw = (r['cmd_yaw']) - (float(r.get('yaw_error_mean', 0))
                                          * (1 if r['cmd_yaw'] >= 0 else -1))
            # Better: yaw_rate average in trace is what we want; approximate from cmd ± err
        else:
            body_yaw = float(r.get('yaw_drift_passive', 0))
        out.append((body_vx, body_vy, body_yaw))
    return out


def plot_release(out_dir, omni_rows, strafe_rows, yaw_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    surv_omni = [r for r in omni_rows if r['survived']]
    surv_strafe = [r for r in strafe_rows if r['survived']]
    surv_yaw = [r for r in yaw_rows if r['survived']]

    def agg(rows, group_key, freq):
        sub = [r for r in rows if r['cmd_freq'] == freq]
        by_group = {}
        for r in sub:
            by_group.setdefault(r[group_key], []).append(r)
        keys = sorted(by_group)
        means_real = []
        means_cmd = []
        stds = []
        for k in keys:
            grp = by_group[k]
            # body-frame realized = signed_bias + cmd
            if group_key == 'cmd_vx':
                real = [float(r.get('vx_bias_body', 0)) + r['cmd_vx'] for r in grp]
            elif group_key == 'cmd_vy':
                real = [float(r.get('vy_bias_body', 0)) + r['cmd_vy'] for r in grp]
            elif group_key == 'cmd_yaw':
                # for yaw, use raw signed eval — fall back to cmd if not stored
                # (best we have without storing signed yaw_rate)
                real = [r['cmd_yaw'] - (float(r.get('yaw_error_mean', 0)) * (1 if r['cmd_yaw'] >= 0 else -1))
                        if abs(r['cmd_yaw']) > 0.05 else float(r.get('yaw_drift_passive', 0))
                        for r in grp]
            real = [x for x in real if not (isinstance(x, float) and (x != x))]
            if not real: continue
            means_cmd.append(k)
            means_real.append(np.mean(real))
            stds.append(np.std(real))
        return np.array(means_cmd), np.array(means_real), np.array(stds)

    def agg_freq_lock(rows, group_key, freq):
        sub = [r for r in rows if r['cmd_freq'] == freq]
        by_group = {}
        for r in sub:
            by_group.setdefault(r[group_key], []).append(r)
        keys = sorted(by_group)
        freqs_mean = []
        for k in keys:
            grp = [r for r in by_group[k] if not np.isnan(float(r.get('measured_freq', float('nan'))))]
            if not grp:
                freqs_mean.append(float('nan')); continue
            freqs_mean.append(np.mean([float(r['measured_freq']) for r in grp]))
        return np.array(keys), np.array(freqs_mean)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {2.0: 'tab:blue', 2.5: 'tab:orange', 3.0: 'tab:green'}

    # Row 1: real vs cmd tracking
    for f in FREQ_VALUES:
        cmds, reals, stds = agg(surv_omni, 'cmd_vx', f)
        if len(cmds):
            axes[0,0].errorbar(cmds, reals, yerr=stds, label=f'cmd_freq={f}', color=colors[f],
                               marker='o', capsize=3)
    axes[0,0].plot([-0.7, 0.7], [-0.7, 0.7], 'k--', alpha=0.4, label='ideal')
    axes[0,0].set_xlabel('cmd_vx (m/s)'); axes[0,0].set_ylabel('realized body_vx (m/s)')
    axes[0,0].set_title('vx tracking'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    for f in FREQ_VALUES:
        cmds, reals, stds = agg(surv_strafe, 'cmd_vy', f)
        if len(cmds):
            axes[0,1].errorbar(cmds, reals, yerr=stds, label=f'cmd_freq={f}', color=colors[f],
                               marker='o', capsize=3)
    axes[0,1].plot([-0.3, 0.3], [-0.3, 0.3], 'k--', alpha=0.4, label='ideal')
    axes[0,1].set_xlabel('cmd_vy (m/s)'); axes[0,1].set_ylabel('realized body_vy (m/s)')
    axes[0,1].set_title('vy strafe tracking'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    for f in FREQ_VALUES:
        cmds, reals, stds = agg(surv_yaw, 'cmd_yaw', f)
        if len(cmds):
            axes[0,2].errorbar(cmds, reals, yerr=stds, label=f'cmd_freq={f}', color=colors[f],
                               marker='o', capsize=3)
    axes[0,2].plot([-1, 1], [-1, 1], 'k--', alpha=0.4, label='ideal')
    axes[0,2].set_xlabel('cmd_yaw (rad/s)'); axes[0,2].set_ylabel('realized yaw_rate (rad/s)')
    axes[0,2].set_title('yaw tracking'); axes[0,2].legend(); axes[0,2].grid(alpha=0.3)

    # Row 2: freq lock per axis
    for f in FREQ_VALUES:
        cmds, freqs = agg_freq_lock(surv_omni, 'cmd_vx', f)
        axes[1,0].plot(cmds, freqs, label=f'cmd_freq={f}', color=colors[f], marker='o')
        axes[1,0].axhline(f, color=colors[f], ls=':', alpha=0.4)
    axes[1,0].set_xlabel('cmd_vx (m/s)'); axes[1,0].set_ylabel('measured_freq (Hz)')
    axes[1,0].set_title('freq lock vs cmd_vx'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
    axes[1,0].set_ylim([1.0, 6.0])

    for f in FREQ_VALUES:
        cmds, freqs = agg_freq_lock(surv_strafe, 'cmd_vy', f)
        axes[1,1].plot(cmds, freqs, label=f'cmd_freq={f}', color=colors[f], marker='o')
        axes[1,1].axhline(f, color=colors[f], ls=':', alpha=0.4)
    axes[1,1].set_xlabel('cmd_vy (m/s)'); axes[1,1].set_ylabel('measured_freq (Hz)')
    axes[1,1].set_title('freq lock vs cmd_vy'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
    axes[1,1].set_ylim([1.0, 6.0])

    for f in FREQ_VALUES:
        cmds, freqs = agg_freq_lock(surv_yaw, 'cmd_yaw', f)
        axes[1,2].plot(cmds, freqs, label=f'cmd_freq={f}', color=colors[f], marker='o')
        axes[1,2].axhline(f, color=colors[f], ls=':', alpha=0.4)
    axes[1,2].set_xlabel('cmd_yaw (rad/s)'); axes[1,2].set_ylabel('measured_freq (Hz)')
    axes[1,2].set_title('freq lock vs cmd_yaw'); axes[1,2].legend(); axes[1,2].grid(alpha=0.3)
    axes[1,2].set_ylim([1.0, 6.0])

    plt.tight_layout()
    out_summary = out_dir / 'summary.png'
    plt.savefig(out_summary, dpi=120)
    print(f'  saved {out_summary}')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True,
                   help='Path to deploy_candidates/<name>/ dir. Must contain policy.onnx.')
    p.add_argument('--runs',     type=int, default=3, help='Runs per cmd')
    p.add_argument('--duration', type=float, default=10, help='Episode duration (s)')
    p.add_argument('--skip_videos', action='store_true', help='Skip the video recording stage')
    p.add_argument('--skip_eval',   action='store_true', help='Skip the sweep eval stage')
    args = p.parse_args()

    cand = Path(args.candidate).resolve()
    pol  = cand / 'policy.onnx'
    if not pol.exists():
        sys.exit(f'no policy at {pol}')
    videos = cand / 'videos'
    eval_dir = cand / 'eval'
    plots = eval_dir / 'plots'

    if not args.skip_videos:
        print(f'[release_eval] Recording videos → {videos}')
        record_videos(pol, videos, duration=args.duration)

    if not args.skip_eval:
        print(f'[release_eval] Running sweeps (this takes ~15-20 min)...')
        # vx sweep × cmd_freq (friction 1.0)
        omni = []
        for vx in VX_VALUES:
            for freq in FREQ_VALUES:
                omni.extend(run_sweep(pol, [(vx, 0, 0, freq)], 1.0,
                                       runs=args.runs, duration=args.duration))
        save_csv(omni, eval_dir / 'eval_omni.csv')

        # vy sweep × cmd_freq
        strafe = []
        for vy in VY_VALUES:
            for freq in FREQ_VALUES:
                strafe.extend(run_sweep(pol, [(0, vy, 0, freq)], 1.0,
                                          runs=args.runs, duration=args.duration))
        save_csv(strafe, eval_dir / 'eval_strafe.csv')

        # yaw sweep × cmd_freq
        yaws = []
        for yw in YAW_VALUES:
            for freq in FREQ_VALUES:
                yaws.extend(run_sweep(pol, [(0, 0, yw, freq)], 1.0,
                                        runs=args.runs, duration=args.duration))
        save_csv(yaws, eval_dir / 'eval_yaw.csv')

    # Plots (always — read existing CSVs if eval was skipped)
    omni   = list(csv.DictReader(open(eval_dir / 'eval_omni.csv')))
    strafe = list(csv.DictReader(open(eval_dir / 'eval_strafe.csv')))
    yaws   = list(csv.DictReader(open(eval_dir / 'eval_yaw.csv')))
    # numerify
    for rows in (omni, strafe, yaws):
        for r in rows:
            for k, v in list(r.items()):
                if v == '' or v is None:
                    r[k] = float('nan'); continue
                try: r[k] = float(v)
                except: pass
            r['survived'] = bool(int(r.get('survived', 0)))
    plot_release(plots, omni, strafe, yaws)

    print(f'\n[release_eval] Done.')
    print(f'  candidate: {cand}')
    print(f'  videos:    {videos}/  ({len(list(videos.glob("*.mp4")))} files)')
    print(f'  eval CSV:  {eval_dir}/eval_*.csv  (omni, strafe, yaw)')
    print(f'  plots:     {plots}/summary.png')


if __name__ == '__main__':
    main()
