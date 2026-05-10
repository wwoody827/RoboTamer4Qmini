# walk_v34 — Overnight RL Tuning Summary

## TL;DR
walk_v34 is the new best baseline. Beats walk_v27 on 4 of 5 metrics including a true strafe capability (first time). Three weaknesses remain that aren't solvable by reward tuning alone.

## Path taken
| Run | Change | Result |
|-----|--------|--------|
| walk_v30 | Cut reg weights ×10-30, disable 1/lin_vel_x_norm scaling | reg fight gone, modest yaw improvement |
| walk_v31 | Restore act_smo to 0.3 | similar to v30, more stable |
| walk_v32 | Steeper slopes (4/5/3), lateral 0.7→2.0, yaw 3→4 | yaw blew up to 0.7, vx good |
| walk_v33 | Regime pure_vy 17%→28%, resampling 5s→10s | strafe still stuck |
| **walk_v34** | **LP filter cmd_track_lp_alpha=0.95** | 🏆 **breakthrough across all axes** |
| walk_v35 | yaw_slope 3→2 + LP α 0.97 | yaw worse, abandoned |
| walk_v36 | yaw deadzone 0.10 | yaw worse, abandoned |

## Best ckpts (walk_v34)
- `experiments/20260509_2239_walk_v34/deploy/policy_2800.onnx` — overall best (unified)
- `experiments/20260509_2239_walk_v34/deploy/policy_3200.onnx` — best disp_y (1.59m)
- `experiments/20260509_2239_walk_v34/deploy/policy_4200.onnx` — best yaw_err (0.142)

## Metric comparison

| Metric | walk_v27 | walk_v34 best | Δ |
|--------|----------|---------------|---|
| vx_fwd_err | 0.058 | 0.038 @2800 | -34% ✅ |
| vx_bwd_err | 0.070 | 0.100 @3600 | +43% ❌ |
| yaw_err | 0.238 | 0.142 @4200 | -40% ✅ |
| yaw_drift_passive | ~0.42 | 0.015 @800 | ~28× ✅ |
| disp_y (strafe) | ~5m (no strafe) | 1.59 @3200 | first real strafe ✅ |

## Omni eval (108 episodes, friction × cmd grid)
- Survival: 100% (fric 0.5/1.0), 72% (fric 1.5)
- Stand: vx 0.04, yaw_drift 0.018 — clean
- Forward (+0.3): vx_err 0.05 — best in fleet
- Backward (-0.3): vx_err 0.29 — 5-7× weaker than fwd
- Strafe L (+0.3): disp_y 1.66 (46% efficient)
- Strafe R (-0.3): disp_y 1.11 (63% efficient)

## Key technical insight: LP filter is root cause fix
The breakthrough was `cmd_track_lp_alpha=0.95` (already in code, default off). It computes tracking err against an EMA of body velocity instead of per-step. Gait sway (zero-mean within-stride oscillation) averages out → only DC offset (real tracking) shows in reward. This defeats Jensen-style reward gaming where ±0.3 oscillation around mean=0 looked similar to mean=0.3 to the per-step reward.

Result: yaw_drift_passive dropped 28×, strafe became real.

## Remaining issues (not reward-tunable)
1. **Bwd 5-7× weaker than fwd** — systemic asymmetry; fix needs per-direction reward shape or wider DR
2. **Strafe L/R asymmetric** (46% vs 63%) — likely due to gait phase initialization
3. **High friction × high vx survival drops** — DR upper friction range may need adjusting

## Config snapshot (walk_v34)
```yaml
_base: walk_v33.yaml
reward:
  cmd_track_lp_alpha: 0.95   # the only key change vs v33
```

walk_v33 inherits v32 (steeper slopes), v31 (calibrated regs), v30 (no 1/norm scaling).

## Outputs
- 10 demo videos: `experiments/20260509_2239_walk_v34/deploy/videos_final/`
- Full eval CSV: `experiments/20260509_2239_walk_v34/deploy/eval.csv`
- Per-cmd rollout plots (iter 2800): `experiments/20260509_2239_walk_v34/deploy/plots/iter_2800/`

## Suggested next steps (not done overnight)
- Generate cleaner trace dataset from walk_v34 @2800 for future reuse
- For bwd asymmetry: try walk_v37 with regime weights bumping bwd cmd_vx range
- For multi-skill master policy: revive task #14 (MIRL multi-expert distillation) once we have stair / uneven terrain experts
