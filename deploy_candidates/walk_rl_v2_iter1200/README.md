# walk_rl_v2_iter1200 — current deployment candidate (supersedes walk_rl_v34_iter3200)

Walk_v34's proven recipe (`configs/walk_rl.yaml`) plus a single addition:
`reward.foot_stand: 1.0` — positive double-support reward at `‖cmd‖<0.15`.
Resumed from walk_v34's iter-8000 weights via `--init_only` (fresh optimizer)
and trained for 1200 more iters. Picked over the iter-2400 final ckpt
because of slight late-training drift on `fwd -0.3`.

## Why this checkpoint

| Metric | walk_v34 baseline | **walk_rl_v2 @1200** |
|---|---|---|
| Sim2sim survival total (6 cmds × 3 seeds × 8 s) | 43.3 / 48.0 s (89 %) | **48.0 / 48.0 s (PERFECT)** |
| Stand (cmd=0) | 8.0 / 8.0 | **8.0 / 8.0** |
| fwd 0.3 | 5.8 | **8.0** |
| fwd 0.5 | 8.0 | **8.0** |
| fwd -0.3 | 5.5 | **8.0** |
| strafe 0.3 | 8.0 | **8.0** |
| yaw 0.5 | 8.0 | **8.0** |
| mean \|vx_bias\| @ fwd 0.3 | 0.252 | **0.032** (8× tighter) |

Measured against the sim2sim infra after the **2026-05-11 delay-buffer
bug fix** (see `docs/walk_noclock_research.md` — the earlier
"walk_v34 jitters" finding was a measurement artifact, all reported numbers
above are post-fix).

## Training recipe

- `_base: walk_rl.yaml` — full walk_v34 BD_X stack
  (phase.mode=input, action_mode=absolute, history 5 × skip 2, regime
  pure_and_pairs, `cmd_track_lp_alpha 0.95`, calibrated regulators).
- Added: `reward.foot_stand: 1.0` — positive reward equal to
  `mean(foot_frc ≥ 10) × (1 − static_flag)`. Range [0, 1]. Only fires
  when commanded to stand.
- Launched via `--resume 20260509_2239_walk_v34 --init_only` — loads
  walk_v34's actor+critic but resets optimizer state (Adam moments stale
  under the new reward) and iter counter. 2500 max iters; best at 1200.

## I/O format

Identical to `walk_rl_v34_iter3200/README.md` — 215-dim obs (5 frames ×
43 dim × skip 2, includes `phase_clock` + `phase_freq_cmd` slots), 10-dim
absolute joint targets with LP α 0.75. Same PD gains, same joint order,
same ±3 obs clip, same 66.67 Hz policy rate, same external `cmd_freq`
range `[2.0, 3.0]` (deploy default 2.5).

## Files

| File | What |
|---|---|
| `policy.onnx` | exported weights (iter 1200) |
| `policy_manifest.yaml` | sim2sim config |
| `README.md` | this file |

Run `scripts/release_eval.py --candidate deploy_candidates/walk_rl_v2_iter1200`
to regenerate videos + sim2sim CSVs + plots.
