"""Build a BC/SFT-ready dataset from sim2sim traces.

Pipeline (one pass over all .npz under --input):
  1. Classify each trace: 'clean' | 'drift' | 'failure'
       failure  ← terminated == 'fell' (was already truncated at fall in record)
       drift    ← any uncommanded axis has |mean realized| above threshold
                  (cmd was 0 but robot moved on that axis — passive control flaw)
       clean    ← otherwise (recoverable: cmd matches realized direction)
  2. Filter by --include (default: 'clean')
  3. Relabel cmd → realized motion (per --relabel)
       cmd_axes  ← replace commanded axes with mean realized (default; small fix)
       all       ← replace all 3 axes with mean realized (incl. drift on uncmd axes)
       none      ← keep original cmd_const
       Rewriting obs[:, 0:3] keeps obs ↔ action consistent for BC training.
  4. Concatenate all kept traces along time axis → one train.npz
       + episodes.csv (provenance, class, drift, cmd_orig, cmd_relabel, frames)
       + dataset.yaml (build args + git commit)

Output is ready for BC: load train.npz, sample (obs, action_raw) pairs across
ep_starts/ep_ends, optionally re-stack with obs_history/obs_skip from
each trace's parent dataset.yaml (sim2sim already records meta_obs_history etc.).

Usage:
    python scripts/build_bc_dataset.py \\
        --input data/traces/walk_v27_multi \\
        --output data/datasets/walk_v27_bc_clean \\
        --include clean --relabel cmd_axes
"""

import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_info(repo_root):
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


def body_frame_realized(npz):
    """Per-step body-frame (vx, vy, yaw_rate) [T, 3]."""
    quat = npz['base_quat']        # [T, 4] (w, x, y, z)
    lin_vel = npz['base_lin_vel']  # world frame
    ang_vel = npz['base_ang_vel']  # body frame already

    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw_world = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cy, sy = np.cos(yaw_world), np.sin(yaw_world)
    vx_body =  lin_vel[:, 0] * cy + lin_vel[:, 1] * sy
    vy_body = -lin_vel[:, 0] * sy + lin_vel[:, 1] * cy
    yaw_rate = ang_vel[:, 2]
    return np.stack([vx_body, vy_body, yaw_rate], axis=1).astype(np.float32)


def classify(npz, drift_thresh, cmd_active_thresh=0.05, transient_frac=0.1):
    """Return (cls, drift_per_axis [3], mean_realized [3]).

    drift_per_axis[i] = |mean_realized[i]| if axis i was uncommanded (cmd[i] ≈ 0),
                       else 0 (commanded axes are not "drift" — they're tracking).
    cls is 'failure' if recorded as fell, 'drift' if any drift_per_axis exceeds
    threshold, else 'clean'.
    """
    if 'metric_terminated' in npz.files:
        term = bytes(npz['metric_terminated']).decode()
        if term == 'fell':
            return 'failure', np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    cmd = npz['cmd_const'].astype(np.float32)
    realized = body_frame_realized(npz)
    T = realized.shape[0]
    s = int(T * transient_frac)
    e = int(T * (1.0 - transient_frac))
    if e <= s:
        s, e = 0, T
    mean_realized = realized[s:e].mean(axis=0).astype(np.float32)

    drift = np.zeros(3, dtype=np.float32)
    for i in range(3):
        if abs(cmd[i]) < cmd_active_thresh:
            drift[i] = abs(mean_realized[i])

    is_drift = any(drift[i] > drift_thresh[i] for i in range(3))
    return ('drift' if is_drift else 'clean'), drift, mean_realized


def relabel_cmd(cmd_orig, mean_realized, mode, cmd_active_thresh=0.05, deadzone=0.02):
    cmd = cmd_orig.astype(np.float32).copy()
    if mode == 'none':
        return cmd
    if mode == 'all':
        cmd = mean_realized.astype(np.float32).copy()
    elif mode == 'cmd_axes':
        for i in range(3):
            if abs(cmd_orig[i]) > cmd_active_thresh:
                cmd[i] = mean_realized[i]
    cmd = np.where(np.abs(cmd) < deadzone, 0.0, cmd).astype(np.float32)
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True, help='Parent dir with traces (recursively walks for .npz)')
    p.add_argument('--output', required=True, help='Output dataset dir')
    p.add_argument('--include', default='clean',
                   help='Comma-separated classes to include: clean,drift,failure (default: clean)')
    p.add_argument('--relabel', choices=['none', 'cmd_axes', 'all'], default='cmd_axes',
                   help='How to rewrite cmd before saving (default: cmd_axes)')
    p.add_argument('--drift_thresh_vx',  type=float, default=0.05)
    p.add_argument('--drift_thresh_vy',  type=float, default=0.20)
    p.add_argument('--drift_thresh_yaw', type=float, default=0.20)
    p.add_argument('--cmd_active_thresh', type=float, default=0.05,
                   help='|cmd[i]| above this → axis is "active/commanded"')
    p.add_argument('--deadzone', type=float, default=0.02,
                   help='Relabeled cmd within deadzone is snapped to 0')
    p.add_argument('--min_tracking_eff', type=float, default=0.0,
                   help='Per-commanded-axis tracking efficiency floor: '
                        'realized[i] / cmd[i] >= floor (sign-checked). 0=off. '
                        'E.g. 0.5 keeps only traces that actually moved at '
                        'least 50%% of commanded velocity on each active axis.')
    p.add_argument('--max_uncmd_drift_yaw', type=float, default=None,
                   help='Stricter than drift_thresh_yaw: drop trace if '
                        'uncommanded yaw drift exceeds this (e.g. 0.10 for '
                        'strafe-priority). Defaults to drift_thresh_yaw.')
    p.add_argument('--max_tilt_deg', type=float, default=None,
                   help='Drop trace if metric_max_tilt_deg exceeds this '
                        '(e.g. 10°) — flags near-tipping moments even if '
                        'survived. Default: off.')
    p.add_argument('--max_peak_tau_ratio', type=float, default=None,
                   help='Drop trace if peak |torque|/effort_limit exceeds '
                        'this (e.g. 1.2) — flags motor saturation, sim2real '
                        'risk. Computed from torque trace if available. Off by default.')
    args = p.parse_args()

    # Effort limits (Nm) per joint — used for peak_tau_ratio computation.
    # Order matches NUM_JOINTS: hip_yaw, hip_roll, hip_pitch, knee, ankle (×2 legs).
    EFFORT_LIMITS = np.array([20., 60., 20., 20., 20., 20., 60., 20., 20., 20.], dtype=np.float32)

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    include = set(args.include.split(','))
    drift_thresh = (args.drift_thresh_vx, args.drift_thresh_vy, args.drift_thresh_yaw)

    # Find all trace .npz (skip top-level non-trace files like manifest.csv)
    npz_files = sorted([p for p in in_root.rglob('*.npz')])
    if not npz_files:
        sys.exit(f"[build] No .npz under {in_root}")
    print(f"[build] Scanning {len(npz_files)} candidate traces under {in_root}")

    # Per-frame chunks (concatenated at end)
    chunks = {k: [] for k in ('obs', 'action_raw', 'action_scaled', 'cmd',
                              'cmd_orig', 'joint_target', 'joint_pos', 'joint_vel',
                              'torque', 'static_flag', 'phase_clock', 'cmd_freq_step')}
    ep_starts, ep_ends, ep_meta = [], [], []
    cur = 0
    counts = {'clean': 0, 'drift': 0, 'failure': 0, 'kept': 0, 'invalid': 0}

    # Track meta consistency across kept traces (BC training expects unchanged stack/dim)
    meta_keys = ('phase_mode', 'action_mode', 'action_dim', 'obs_dim',
                 'obs_history', 'obs_skip', 'lp_alpha', 'num_legs')
    meta_seen = {}

    for npz_path in npz_files:
        try:
            d = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            counts['invalid'] += 1
            print(f"  skip {npz_path.relative_to(in_root)}: load error {e}")
            continue

        required = ('obs', 'action_raw', 'cmd_const', 'base_lin_vel', 'base_quat', 'base_ang_vel')
        if any(k not in d.files for k in required):
            counts['invalid'] += 1
            print(f"  skip {npz_path.relative_to(in_root)}: missing keys (legacy trace?)")
            continue

        cls, drift, mean_realized = classify(
            d, drift_thresh, cmd_active_thresh=args.cmd_active_thresh)
        counts[cls] += 1
        if cls not in include:
            continue

        # Optional secondary filters (after class-based include).
        cmd_orig_arr = d['cmd_const'].astype(np.float32)
        # 1) tracking efficiency on commanded axes
        if args.min_tracking_eff > 0.0 and cls != 'failure':
            ok = True
            for i in range(3):
                if abs(cmd_orig_arr[i]) > args.cmd_active_thresh:
                    eff = mean_realized[i] / cmd_orig_arr[i]
                    if eff < args.min_tracking_eff:
                        ok = False
                        break
            if not ok:
                counts.setdefault('low_eff', 0)
                counts['low_eff'] += 1
                continue
        # 2) tighter yaw drift on uncommanded yaw axis
        if args.max_uncmd_drift_yaw is not None and cls != 'failure':
            if abs(cmd_orig_arr[2]) < args.cmd_active_thresh and drift[2] > args.max_uncmd_drift_yaw:
                counts.setdefault('high_yaw_drift', 0)
                counts['high_yaw_drift'] += 1
                continue
        # 3) tilt cap (near-tipping moments even if survived)
        if args.max_tilt_deg is not None and 'metric_max_tilt_deg' in d.files:
            tilt = float(d['metric_max_tilt_deg'])
            if tilt > args.max_tilt_deg:
                counts.setdefault('high_tilt', 0)
                counts['high_tilt'] += 1
                continue
        # 4) peak torque ratio (motor saturation flag)
        if args.max_peak_tau_ratio is not None and 'torque' in d.files:
            tau = d['torque']
            peak_ratio = float(np.max(np.abs(tau) / EFFORT_LIMITS))
            if peak_ratio > args.max_peak_tau_ratio:
                counts.setdefault('high_torque', 0)
                counts['high_torque'] += 1
                continue

        T = int(d['joint_pos'].shape[0])
        cmd_orig = d['cmd_const'].astype(np.float32)
        cmd_new = relabel_cmd(cmd_orig, mean_realized, args.relabel,
                              cmd_active_thresh=args.cmd_active_thresh,
                              deadzone=args.deadzone)

        # Track meta consistency
        for k in meta_keys:
            mk = f'meta_{k}'
            if mk in d.files:
                v = d[mk].item() if d[mk].dtype.kind != 'S' else bytes(d[mk]).decode()
                if k in meta_seen and meta_seen[k] != v:
                    print(f"  WARN {npz_path.name}: meta_{k}={v} differs from prior {meta_seen[k]}")
                meta_seen[k] = v

        # Rewrite obs cmd_slot. BD_X / MIRL: first 3 dims are commands.
        # Note: phase_clock + freq_norm in obs are pre-multiplied by static_flag at
        # record time. If relabel crosses static threshold (||cmd|| ≷ 0.15), we'd
        # need to rescale; for walk_v27 traces all cmds are ≥0.3 or =0 so this is
        # safe — emit a warning if it ever happens.
        old_static = float(np.linalg.norm(cmd_orig) >= 0.15)
        new_static = float(np.linalg.norm(cmd_new) >= 0.15)
        if old_static != new_static:
            print(f"  WARN {npz_path.name}: relabel crossed static_thr "
                  f"(||cmd_orig||={np.linalg.norm(cmd_orig):.3f} → "
                  f"||cmd_new||={np.linalg.norm(cmd_new):.3f}); phase_clock obs slots not rescaled")

        obs_new = d['obs'].copy()
        obs_new[:, 0:3] = cmd_new[None, :]

        chunks['obs'].append(obs_new)
        chunks['action_raw'].append(d['action_raw'])
        chunks['action_scaled'].append(d['action_scaled'])
        chunks['cmd'].append(np.broadcast_to(cmd_new, (T, 3)).copy())
        chunks['cmd_orig'].append(np.broadcast_to(cmd_orig, (T, 3)).copy())
        chunks['joint_target'].append(d['joint_target'])
        chunks['joint_pos'].append(d['joint_pos'])
        chunks['joint_vel'].append(d['joint_vel'])
        chunks['torque'].append(d['torque'])
        chunks['static_flag'].append(d['static_flag'])
        chunks['phase_clock'].append(d['phase_clock'])
        chunks['cmd_freq_step'].append(d['cmd_freq_step'])

        ep_starts.append(cur)
        ep_ends.append(cur + T)
        ep_meta.append({
            'trace':       str(npz_path.relative_to(REPO_ROOT)),
            'ckpt':        npz_path.parent.name,
            'class':       cls,
            'frames':      T,
            'cmd_orig':    [float(x) for x in cmd_orig],
            'cmd_relabel': [float(x) for x in cmd_new],
            'mean_realized': [float(x) for x in mean_realized],
            'drift_vx':    float(drift[0]),
            'drift_vy':    float(drift[1]),
            'drift_yaw':   float(drift[2]),
        })
        cur += T
        counts['kept'] += 1

    print(f"\n[build] Classification: clean={counts['clean']}  drift={counts['drift']}  "
          f"failure={counts['failure']}  invalid={counts['invalid']}")
    print(f"[build] Kept: {counts['kept']} traces, {cur} frames "
          f"({cur * 0.015:.1f}s of demonstration)")
    if counts['kept'] == 0:
        sys.exit("[build] No traces matched filter — exiting")

    # Concatenate
    out = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    out['ep_starts'] = np.array(ep_starts, dtype=np.int64)
    out['ep_ends']   = np.array(ep_ends,   dtype=np.int64)
    train_npz = out_root / 'train.npz'
    np.savez(train_npz, **out)
    print(f"[build] Wrote {train_npz}  ({sum(v.nbytes for v in out.values()) / 1e6:.1f} MB)")

    # Per-episode CSV (cmd / class / drift)
    ep_csv = out_root / 'episodes.csv'
    keys = list(ep_meta[0].keys())
    with open(ep_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in ep_meta:
            row = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in m.items()}
            w.writerow(row)
    print(f"[build] Wrote {ep_csv}")

    # Dataset recipe
    git_sha, git_dirty = _git_info(REPO_ROOT)
    recipe = {
        'created':            datetime.datetime.now().isoformat(timespec='seconds'),
        'input':              str(in_root.relative_to(REPO_ROOT)) if str(in_root).startswith(str(REPO_ROOT)) else str(in_root),
        'git_commit':         git_sha,
        'git_dirty':          git_dirty,
        'args': {
            'include':         args.include,
            'relabel':         args.relabel,
            'drift_thresh_vx':  args.drift_thresh_vx,
            'drift_thresh_vy':  args.drift_thresh_vy,
            'drift_thresh_yaw': args.drift_thresh_yaw,
            'cmd_active_thresh': args.cmd_active_thresh,
            'deadzone':        args.deadzone,
        },
        'counts':             counts,
        'total_frames':       int(cur),
        'meta':               meta_seen,
    }
    with open(out_root / 'dataset.yaml', 'w') as f:
        yaml.safe_dump(recipe, f, sort_keys=False)
    print(f"[build] Wrote {out_root / 'dataset.yaml'}")


if __name__ == '__main__':
    main()
