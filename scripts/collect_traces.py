"""Sweep over a (vx, vy, yaw) command grid, record sim2sim trace per cmd.

Each trace is written to data/traces/<run_id>/<cmd_id>.npz and contains the
extended fields produced by deploy/sim2sim/sim2sim.py (joint_pos/vel, base_*,
cmd, obs, action_raw, action_scaled, joint_target, torque, static_flag,
phase_clock, cmd_freq_step, plus episode + config metadata).

After all traces complete, writes manifest.csv summarizing each trace's cmd +
episode metrics so downstream code (BC/MIRL/SFT) can filter by quality.

Usage:
    python scripts/collect_traces.py \\
        --policy experiments/<run>/deploy/policy_<iter>.onnx \\
        --out data/traces/walk_v27_6200 \\
        --duration 12 --cmd_freq 2.5

The grid defaults to 27 combos (vx∈{-0.3,0.3,0.5} × vy∈{-0.3,0,0.3} × yaw∈{-0.5,0,0.5})
plus 4 extras (stand, pure_yaw_±1, fast_fwd_0.7) = 31 traces.
"""

import argparse
import csv
import datetime
import hashlib
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/woody/miniconda3/envs/qmini/bin/python'


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_info(repo_root):
    """Return (commit_sha, dirty_flag) — best effort, '' / False on failure."""
    try:
        sha = subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ['git', '-C', str(repo_root), 'status', '--porcelain'],
            stderr=subprocess.DEVNULL).decode().strip())
        return sha, dirty
    except Exception:
        return '', False


def default_grid():
    """In-distribution grid — matches training regime (pure_and_pairs):
       pure_vx / pure_vy / pure_yaw / vx+vy / vx+yaw. NO 3-axis or vy+yaw
       (env never trains those, recording them produces low-quality demos)."""
    pure_vx = [(v, 0, 0) for v in (-0.3, 0.3, 0.5)]
    pure_vy = [(0, v, 0) for v in (-0.3, 0.3)]
    pure_yaw = [(0, 0, v) for v in (-0.5, 0.5)]
    vx_vy  = [(vx, vy, 0) for vx in (-0.3, 0.3, 0.5) for vy in (-0.3, 0.3)]
    vx_yaw = [(vx, 0, yw) for vx in (-0.3, 0.3, 0.5) for yw in (-0.5, 0.5)]
    extras = [
        (0.0, 0.0, 0.0),    # stand
        (0.0, 0.0, 1.0),    # pure_yaw_R extreme
        (0.0, 0.0, -1.0),   # pure_yaw_L extreme
        (0.7, 0.0, 0.0),    # fast_fwd
    ]
    return pure_vx + pure_vy + pure_yaw + vx_vy + vx_yaw + extras


def cmd_id(vx, vy, yaw):
    """Filename-safe id from a cmd triple."""
    def _fmt(x):
        return f"{'p' if x >= 0 else 'n'}{abs(x):.2f}".replace('.', '')
    return f"vx{_fmt(vx)}_vy{_fmt(vy)}_yaw{_fmt(yaw)}"


def run_one(policy, out_path, vx, vy, yaw, duration, cmd_freq, floor_friction):
    cmd = [
        PYTHON, str(REPO_ROOT / 'deploy' / 'sim2sim' / 'sim2sim.py'),
        '--policy', str(policy),
        '--cmd_vx',  f'{vx}',
        '--cmd_vy',  f'{vy}',
        '--cmd_yaw', f'{yaw}',
        '--duration', f'{duration}',
        '--headless',
        '--record', str(out_path),
        '--record_skill', 'walk',
        '--record_loop',
        '--record_skip', '20',
    ]
    if cmd_freq is not None:
        cmd += ['--cmd_freq', f'{cmd_freq}']
    if floor_friction is not None:
        cmd += ['--floor_friction', f'{floor_friction}']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc


def summarize(npz_path):
    """Return dict of cmd + key metrics from a saved trace."""
    d = np.load(npz_path, allow_pickle=True)
    out = {
        'file': npz_path.name,
        'frames': int(d['joint_pos'].shape[0]),
        'duration_s': float(d['joint_pos'].shape[0] * d['dt']),
    }
    if 'cmd_const' in d.files:
        c = d['cmd_const']
        out.update({'cmd_vx': float(c[0]), 'cmd_vy': float(c[1]), 'cmd_yaw': float(c[2])})
    for k in ['metric_mean_vx_err', 'metric_mean_vy_err', 'metric_mean_yaw_err',
              'metric_mean_height', 'metric_min_height', 'metric_max_tilt_deg',
              'metric_survival_time', 'metric_survival_full']:
        if k in d.files:
            out[k.replace('metric_', '')] = float(d[k])
    if 'metric_terminated' in d.files:
        out['terminated'] = bytes(d['metric_terminated']).decode()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', required=False, help='Path to single policy .onnx (with manifest next to it). '
                                                     'Mutually exclusive with --policies.')
    p.add_argument('--policies', nargs='+', default=None,
                   help='Multiple policy .onnx paths — sweeps each into <out>/<policy_stem>/. '
                        'Mutually exclusive with --policy.')
    p.add_argument('--out',    required=True, help='Output dir for traces + manifest.csv')
    p.add_argument('--duration', type=float, default=12.0, help='Per-trace duration (s)')
    p.add_argument('--cmd_freq', type=float, default=None, help='cmd_freq for BD_X (defaults to manifest)')
    p.add_argument('--floor_friction', type=float, default=None)
    p.add_argument('--grid', type=str, default='default', help='"default" | "small" (5 traces, smoke test)')
    p.add_argument('--skip_existing', action='store_true', help='Skip cmds whose .npz already exists')
    args = p.parse_args()

    if (args.policy is None) == (args.policies is None):
        sys.exit("Specify exactly one of --policy or --policies")
    if args.policies is not None:
        # Recurse: run one sub-sweep per policy into <out>/<stem>/
        out_root = Path(args.out).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        for pol in args.policies:
            stem = Path(pol).stem  # e.g. policy_6200
            iter_id = stem.replace('policy_', '') if stem.startswith('policy_') else stem
            sub_out = out_root / iter_id
            print(f"\n========== ckpt {iter_id}  →  {sub_out} ==========")
            sub_args = argparse.Namespace(**vars(args))
            sub_args.policy = pol
            sub_args.policies = None
            sub_args.out = str(sub_out)
            _run_single_sweep(sub_args)
        print(f"\n[collect] Multi-ckpt sweep complete: {out_root}")
        return
    _run_single_sweep(args)


def _run_single_sweep(args):

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Reproducibility recipe — written before the sweep so it lands even on partial run.
    policy_path = Path(args.policy).resolve()
    if not policy_path.exists():
        sys.exit(f"[collect] policy not found: {policy_path}")
    manifest_yaml = policy_path.with_name(policy_path.stem + '_manifest.yaml')
    git_sha, git_dirty = _git_info(REPO_ROOT)

    def _rel_or_abs(p):
        try:
            return str(Path(p).resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    recipe = {
        'created':           datetime.datetime.now().isoformat(timespec='seconds'),
        'policy_path':       _rel_or_abs(policy_path),
        'policy_sha256':     _sha256_file(policy_path),
        'policy_manifest':   (_rel_or_abs(manifest_yaml) if manifest_yaml.exists() else None),
        'git_commit':        git_sha,
        'git_dirty':         git_dirty,
        'args': {
            'duration':       float(args.duration),
            'cmd_freq':       (float(args.cmd_freq) if args.cmd_freq is not None else None),
            'floor_friction': (float(args.floor_friction) if args.floor_friction is not None else None),
            'grid':           args.grid,
        },
    }
    with open(out / 'dataset.yaml', 'w') as f:
        yaml.safe_dump(recipe, f, sort_keys=False)
    print(f"[collect] Recipe: {out / 'dataset.yaml'}")
    if git_dirty:
        print(f"[collect] WARNING: working tree dirty — git_commit={git_sha[:8]} not fully reproducible")
    print(f"[collect] Output dir: {out}")
    print(f"[collect] Policy: {args.policy}")

    if args.grid == 'small':
        grid = [(0.5, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.3, 0.0),
                (0.0, 0.0, 0.5), (0.3, 0.0, 0.5)]
    else:
        grid = default_grid()
    print(f"[collect] {len(grid)} traces in grid")

    summaries = []
    for i, (vx, vy, yaw) in enumerate(grid):
        cid = cmd_id(vx, vy, yaw)
        out_npz = out / f'{cid}.npz'
        if args.skip_existing and out_npz.exists():
            print(f"[{i+1}/{len(grid)}] SKIP {cid} (already exists)")
            try:
                summaries.append(summarize(out_npz))
            except Exception as e:
                print(f"  warn: cannot summarize existing file: {e}")
            continue
        print(f"[{i+1}/{len(grid)}] cmd={cid}  vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f}")
        proc = run_one(args.policy, out_npz, vx, vy, yaw,
                       args.duration, args.cmd_freq, args.floor_friction)
        if proc.returncode != 0:
            print(f"  FAIL rc={proc.returncode}")
            print(proc.stdout[-800:] if proc.stdout else '')
            print(proc.stderr[-800:] if proc.stderr else '')
            continue
        if not out_npz.exists():
            print(f"  FAIL: trace not saved (likely fell at start)")
            print(proc.stdout[-400:] if proc.stdout else '')
            continue
        summaries.append(summarize(out_npz))
        s = summaries[-1]
        print(f"  → frames={s.get('frames')} surv={s.get('survival_time', 0):.1f}s  "
              f"vx_err={s.get('mean_vx_err', float('nan')):.3f} "
              f"yaw_err={s.get('mean_yaw_err', float('nan')):.3f} "
              f"term={s.get('terminated', '?')}")

    # Manifest
    if summaries:
        keys = sorted({k for s in summaries for k in s.keys()})
        manifest_csv = out / 'manifest.csv'
        with open(manifest_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in summaries:
                w.writerow({k: s.get(k, '') for k in keys})
        print(f"\n[collect] Manifest: {manifest_csv}")
        print(f"[collect] {len(summaries)}/{len(grid)} traces saved")


if __name__ == '__main__':
    main()
