# walk_clean: clean-slate, budget-driven reward design

## Why a new track

v5-v16 walk_noclock experiments built up by piling fix on fix. Cumulative
result was tangled: v16 inherits weights from v15←v14←...←v5, each layer
covering for the previous layer's gaps. End state was "great backward
walker, broken forward" — only revealed by full release_eval (training
quick_eval grid by accident only tested one cmd_vx that survived).

walk_clean restarts from MINIMAL rewards, builds up one term at a time,
verifies each addition with `scripts/reward_budget.py` before training.

## Methodology (per phase)

For each phase A→D:

1. Define **goal** (one capability — e.g. "stand stably").
2. Pick **MINIMAL reward set** required for that goal (no extras).
3. Run `reward_budget.py` with the relevant trajectory state pair
   (e.g. WALK vs FALL_OVER for Phase A).
4. Verify Walk/Other ratio ≥ 2× — if not, tune weight magnitudes via
   physics estimates (NOT empirical tweaking).
5. Train. Acceptance criterion: capability achieved + ratio holds in
   measurement.
6. Run release_eval to verify full picture (not just training cmd subset).
7. Promote ckpt as baseline for next phase.

## Phase progression

| Phase | Goal | cmd range | New rewards |
|---|---|---|---|
| **A** | Stable stand | all zero | constant, base_heit, upright, foot_stand, light regs |
| **B** | Forward walk at fixed cmd_vx=0.3 | lin_vel_x [0, 0.3] | + fwd_vel, foot_phase, foot_clr, foot_supt, air_time |
| **C** | Walking with good gait | lin_vel_x [0, 0.3] | + stride_length, feet_swing_height_peak, vertical_vel bell |
| **D** | Multi-direction | full grid | + lateral, yaw, regime weighting |

## Rules

- ONE config-level change per version. If you want to bump 3 weights, run 3
  configs.
- Track BEST ckpt by `release_eval` walk_quality (omni grid), NOT training
  quick_eval. Training quick_eval is too narrow.
- Compare each new config to the previous BASELINE in budget AND release_eval.
- If budget predicts no improvement but training shows one (or vice versa),
  STOP and debug the budget model before moving on.

## Anti-patterns (lessons from v5-v16)

1. **Don't inherit too many layers**: walk_clean_vN should inherit only from
   walk_clean_v{N-1}, not 5 layers deep.
2. **Don't trust training quick_eval for direction-asymmetric failures**: it
   tests only the cmd subset you pick. Always release_eval before declaring success.
3. **Don't `getattr(..., default) or default`**: 0.0 is falsy, silently
   becomes default. Use explicit `is None` check.
4. **Don't pile on reward terms**: each one adds an axis the policy can
   exploit. Start minimal.
