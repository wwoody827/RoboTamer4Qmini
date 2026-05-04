"""
Build a mixed init pool for the two-phase recovery curriculum:
  - balance pool   (small perturbation, ~upright; from extract_balance_pool.py)
  - tilted pool    (medium perturbation, partially toppled)
  - fallen pool    (settled fallen poses; from extract_rsi_pool.py)

Default ratio 30/30/40 follows the Berkeley HumanPlus mixed-init pattern:
plenty of balance/tilted samples to keep the policy near upright, fallen samples
to retain the get-up skill once balance is solid.

Usage:
    python scripts/build_mixed_init_pool.py \
        --balance data/balance_init_states.npz \
        --tilted  data/tilted_init_states.npz \
        --fallen  data/recovery_init_states.npz \
        --out     data/recovery_init_states_mixed.npz \
        --total   5000

Output schema is the subset RecoveryTask._load_init_pool consumes:
    base_pos, base_quat (wxyz), base_lin_vel, base_ang_vel,
    joint_pos, joint_vel, pose_label
"""
import argparse
import os

import numpy as np


_KEYS = ('base_pos', 'base_quat', 'base_lin_vel', 'base_ang_vel',
        'joint_pos', 'joint_vel')


def _sample_pool(path, n, rng):
    d = np.load(path, allow_pickle=False)
    N = d['base_pos'].shape[0]
    if n > N:
        # Sample with replacement when we want more than the pool has.
        idx = rng.integers(0, N, size=n)
    else:
        idx = rng.choice(N, size=n, replace=False)
    out = {k: np.asarray(d[k])[idx] for k in _KEYS}
    if 'pose_label' in d.files:
        out['pose_label'] = np.asarray(d['pose_label'])[idx]
    else:
        # Derive from filename if missing
        guess = os.path.splitext(os.path.basename(path))[0]
        out['pose_label'] = np.array([guess] * n)
    return out, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--balance', required=True)
    ap.add_argument('--tilted',  required=True)
    ap.add_argument('--fallen',  required=True)
    ap.add_argument('--out',     default='data/recovery_init_states_mixed.npz')
    ap.add_argument('--total',   type=int, default=5000,
                    help="total mixed pool size")
    ap.add_argument('--ratio',   default='30,30,40',
                    help="balance,tilted,fallen percentages (sum=100)")
    ap.add_argument('--seed',    type=int, default=0)
    args = ap.parse_args()

    parts = [int(x) for x in args.ratio.split(',')]
    assert len(parts) == 3 and sum(parts) == 100, \
        f"--ratio must be three ints summing to 100, got {args.ratio}"
    n_bal = args.total * parts[0] // 100
    n_til = args.total * parts[1] // 100
    n_fal = args.total - n_bal - n_til  # absorb rounding

    rng = np.random.default_rng(args.seed)
    print(f"[mixed] target {args.total} states, ratio {parts} → "
          f"{n_bal} balance + {n_til} tilted + {n_fal} fallen")

    bal, N_bal = _sample_pool(args.balance, n_bal, rng)
    til, N_til = _sample_pool(args.tilted,  n_til, rng)
    fal, N_fal = _sample_pool(args.fallen,  n_fal, rng)
    print(f"[mixed] source pools: balance={N_bal}, tilted={N_til}, fallen={N_fal}")

    merged = {k: np.concatenate([bal[k], til[k], fal[k]], axis=0) for k in _KEYS}
    merged['pose_label'] = np.concatenate(
        [bal['pose_label'], til['pose_label'], fal['pose_label']], axis=0)

    perm = rng.permutation(merged['base_pos'].shape[0])
    merged = {k: v[perm] for k, v in merged.items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **merged)

    unique, counts = np.unique(merged['pose_label'], return_counts=True)
    print(f"\n[mixed] saved {merged['base_pos'].shape[0]} states → {args.out}")
    for lab, c in zip(unique, counts):
        print(f"[mixed]   {lab:>16s}: {c}")


if __name__ == '__main__':
    main()
