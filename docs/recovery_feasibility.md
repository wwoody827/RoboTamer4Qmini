# Qmini Recovery — Phase 1 Feasibility Study

**Status**: v0 — passive baseline only. Awaiting hand-designed recovery scripts and/or viewer inspection.
**Tool**: `tests/recovery_feasibility.py` (MuJoCo, no GPU required).
**Robot**: 10 DoF (5 per leg: hip_yaw, hip_roll, hip_pitch, knee, ankle). No arms, no upper body, no ankle roll.

## Method

For each candidate fallen pose:
1. Spawn the robot in MuJoCo at the specified base orientation + height + joint angles.
2. Run a fixed PD policy that simply holds `ref_joint_pos` (nominal standing posture).
3. Step physics for 4 s (`sim_dt=2 ms`).
4. Record max base height, final base height, and base tilt angle (vs +Z).

Success criterion: `z_end > 0.85 × 0.45 = 0.383 m` AND `tilt_end < 25°`.

Note: this is the *passive* baseline. RL recovery is expected to expand the recoverable set well beyond what passive PD can achieve. Failure here is informative, not disqualifying.

## Results (passive PD-to-nominal, 4 s)

| Pose | Outcome | z_end (m) | max z (m) | tilt_end (deg) | max tilt drop (deg) | Note |
|---|---|---|---|---|---|---|
| `prone` (face down) | FAIL | 0.062 | 0.100 | 93.9 | 1.4 | No motion — body inertia + ground contact lock the pose |
| `supine` (face up) | FAIL | 0.005 | 0.100 | 123.8 | 72.8 | Falls flat; body rotates further, doesn't lift |
| `side_left` (rolled +90° X) | FAIL | 0.196 | 0.262 | 83.6 | 13.6 | Slight motion; legs scissor but no lift |
| `side_right` (rolled −90° X) | FAIL | 0.196 | 0.262 | 83.6 | 13.7 | Mirror of side_left |
| `forward_kneel` | FAIL | 0.062 | 0.455 | 93.9 | 0.0 | Initial pose itself is taller (z_init=0.18), then collapses |
| `back_sit` | FAIL | 0.062 | 0.729 | 93.9 | 6.1 | Briefly bounces upward, then falls; no controlled stand-up |

## Interpretation

- **Passive PD recovery is infeasible from every tested pose.** Recovery requires dynamic torque sequences (push-up phases, momentum transfer), not stiffness alone.
- **`max_tilt_drop` > 10° on side poses** suggests the body *can* move toward upright with the right motion — promising for RL.
- **Supine + zero arms = expected hardest case.** Tilt actually grows (123.8°) — body just rolls onto its head.
- **Forward kneel `max_z=0.455m`** is misleading — initial spawn z is high then collapses. Need to rerun starting from a stable settled pose, not idealised joint angles.

## Open questions before declaring feasibility

1. **Hand-designed recovery scripts.** Need a human (or expert) to design 1–2 trajectories per pose (e.g. prone: "tuck both legs → straighten hips → push to seated → stand"). Wire these into `--scripted <name>` mode.
2. **Settled initial poses.** Replace hand-set joint angles for `forward_kneel` / `back_sit` with poses obtained by *simulating* a fall (Phase 2's fall-state generator, then re-using its outputs here).
3. **Viewer inspection.** Run `python tests/recovery_feasibility.py --pose <name>` (no `--headless`) to watch each pose for ~5 s and judge visually whether any leg/torso motion suggests a viable recovery strategy.

## Provisional decision

Treat **all 4 "lying" poses (prone, supine, side_L, side_R)** as candidates for RL training, with the understanding that:

- `supine` is the highest-risk infeasibility candidate — no arms to roll over, hardware can't generate the needed torque-vs-mass moment around the spine. Expect to drop it.
- `prone`, `side_L`, `side_R` are likely the recoverable set. Side poses may need the policy to first roll to prone, then push up.
- `forward_kneel` and `back_sit` should be re-derived from real fall dynamics (Phase 2), not synthesized by hand.

Rerun this study with hand-designed scripted recovery before committing reward design choices in Phase 3.
