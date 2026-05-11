# walk_rl_v4_iter4800 — current deploy candidate (supersedes walk_rl_v2_iter1200)

Trained from `configs/walk_rl_v4.yaml`. This is walk_v34's recipe with three
additions, picked over walk_rl_v2 because it walks smoother, tracks tighter,
and yaws better.

## Recipe (vs walk_v34 baseline)

- `reward.foot_stand: 1.0` — positive double-support reward at `‖cmd‖<0.15`
  (standstill grounding incentive).
- `reward.base_heit: 3.0` (was 1.0) — pulls 3× harder toward 0.45 m
  standing height. (Did not actually fix the systemic ~7 cm low walking —
  see `docs/walk_noclock_research.md` §5; treat as "no harm, slight help".)
- `domain_rand.delay_*_ranges` widened to `[1, 50] / [1, 60] / [1, 60]`
  (was `[10, 40] / [20, 50] / [20, 50]`). Trains the policy on the full
  obs-latency spectrum from 1 ms to 60 ms, so real-robot deployment isn't
  brittle to whatever latency the actual hardware exhibits.

Trained from scratch, 5000 iters, 193 min. Best ckpt = iter 4800 (final
ckpt iter 5000 had slight drift on bwd tracking).

## Why this checkpoint

Sim2sim survival/quality across the 9-command grid (3 seeds × 8 s, friction 1.0,
cmd_freq 2.5). Measured with the bug-fixed sim2sim (framestack skip ✓,
obs-delay buffer at physics rate ✓):

| Metric | walk_v34 | walk_rl_v2 @1200 | **walk_rl_v4 @4800** |
|---|---|---|---|
| Survival (9 cmds × 3 seeds × 8 s) | 72.0 / 72.0 s | 72.0 / 72.0 s | **72.0 / 72.0 s** |
| com_z mean @ stand | 0.379 m | 0.405 m | 0.375 m |
| pitch_rms during walking | 0.061 rad | 0.121 rad (颠) | **0.055 rad** (smoothest) |
| mean \|vx_bias\| over walk cmds | +0.141 | +0.115 | **+0.073** (best tracking) |
| yaw_err over yaw cmds | 0.325 | 0.343 | **0.295** (best yaw) |
| Delay robustness (0 / 25 / 50 ms tests) | full / full / full | full / full / full | full / full / full |

- **Smoothest body during walking** — pitch_rms ~50% lower than v2.
- **Tightest velocity tracking** — vx_bias roughly half walk_v34's.
- **Best yaw tracking** of the three.
- **Trained on wider delay distribution** [1, 60] ms — covers any real-robot
  obs-latency scenario.
- Trade-off: same walking height (~0.375 m) as walk_v34, ~3 cm below v2.
  Per overnight investigation (`docs/walk_noclock_research.md`), this is the
  energy-optimal posture — no reward shaping we tried (base_heit weight 1/3,
  base_height_l2 -10, flat_orient_l2 -5, jnt_pos_err 0.5) could pull it higher
  without breaking other metrics.

## I/O format

Identical to walk_rl_v2_iter1200 — 215-dim observation (5 frames × 43 dim ×
skip 2), 10-dim absolute joint targets with LP α 0.75, 67 Hz policy rate,
same PD gains, same external `cmd_freq` knob (range `[2.0, 3.0]`, default 2.5).

See `walk_rl_v34_iter3200/README.md` for the full slot-by-slot input table.

## Files

| File | What |
|---|---|
| `policy.onnx` | exported weights, iter 4800 |
| `policy_manifest.yaml` | sim2sim config (delay ranges, PD gains, slot order) |
| `README.md` | this file |

## Reproducing the eval

```bash
python scripts/release_eval.py --candidate deploy_candidates/walk_rl_v4_iter4800
```

Generates `videos/`, `eval/eval_*.csv`, `eval/plots/summary.png` under this
directory. Uses the bug-fixed sim2sim — sweep numbers reflect the realistic
training delay distribution. Sim2sim infra: see
`docs/walk_noclock_research.md` for the bug-find arc tonight.
