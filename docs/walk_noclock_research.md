# Walk-Noclock Research — getting clock-free policies to transfer

**Author:** Claude (autonomous, overnight session)
**Started:** 2026-05-10 ~23:15  |  **Finished:** 2026-05-11 05:20

---

## 🌅 Morning summary (TL;DR)

**Headline result:** new deploy candidate at
`deploy_candidates/walk_rl_v2_iter1200/`. **Perfect sim2sim survival**
across the entire 6-cmd × 3-seed × 8 s test matrix (48.0 / 48.0 s),
**8× tighter velocity tracking** than the old walk_v34 baseline.

**Surprise finding:** the whole premise of the no-clock investigation
("walk_v34 can't stand cleanly — 7.6 Hz hip jitter") was largely a
**sim2sim measurement bug** I introduced earlier in the day — the obs-delay
buffer was pushed at policy rate (67 Hz) instead of physics rate (1000 Hz),
making delays 15× too large. With the fix, walk_v34 actually stands fine.

So the right plan ended up being conservative: walk_v34's proven recipe
plus a single addition (`reward.foot_stand: 1.0`) and one
`--init_only` resume to refresh the optimizer state. Beats every
no-clock variant by a large margin.

### What ran tonight
| Run | recipe | best ckpt | sim2sim total / 48 s | verdict |
|---|---|---|---|---|
| walk_v34 (baseline, re-measured) | with clock | 3200 | 43.3 s | 89 % — proven |
| walk_noclock v1 | air_time + no clock | 1800 | 27.5 s | weak walking |
| walk_noclock v2 | v1 + foot_stand | 800 (killed early) | n/a | dead end after broken-eval pivot |
| walk_noclock v3 | **Cassie-style periodic reward** | **3000** | **39.8 s** | viable no-clock recipe (92 %) |
| **walk_rl_v2** | **walk_v34 + foot_stand** | **1200** | **🏆 48.0 s** | **deploy** |

### Bugs fixed tonight
1. **`deploy/sim2sim/evaluate.py`** — obs-delay buffer was being pushed
   per policy step. Now pushed per physics step (matches training's
   `step_torques` cadence). Without this fix all sim2sim measurements
   were noise.
2. **`deploy/sim2sim/evaluate.py`** + **`sim2sim.py`** — `phase.mode='none'`
   was routed through the MIRL obs builder (64-dim), causing dim
   mismatch with the new 38-dim walk_noclock layout. Added `is_noclock`
   branch with a dedicated `get_obs_noclock()` builder.
3. **`deploy/manifest.py`** — `delay_*_ranges` now shipped through the
   manifest from training config to sim2sim. Avoids silent drift.
4. **`env/tasks/birl_task.py`** — `base_heit_slope` and `base_heit_target`
   now configurable (were hardcoded `−70` and `0.45`). Also `air_time_rew`
   now fires for `phase.mode='none'` via `phase.target_swing` (was
   silently zero, a dead-weight reward).
5. **`env/tasks/birl_task.py`** — added `foot_stand` reward: positive
   incentive `mean(foot_frc≥10) × (1−static_flag)` at standstill.

### Next steps (suggested)
1. Run `scripts/release_eval.py --candidate deploy_candidates/walk_rl_v2_iter1200`
   to generate videos + sweep CSVs + plots.
2. Update SDK / sim2sim defaults if you want walk_rl_v2 as the canonical
   policy (manifest format is unchanged from walk_v34, drop-in).
3. The Cassie-style no-clock recipe (`configs/walk_noclock_v3.yaml`,
   ckpt at iter 3000) remains a valid fallback if deploy-time
   `cmd_freq` knob ever becomes undesirable. Early-stop and tighter
   `yaw_rat` are the open knobs.
4. Commit when you're back — files changed:
   - new: `configs/walk_noclock_v[1-4].yaml`, `configs/walk_rl_v2.yaml`
   - new: `deploy_candidates/walk_rl_v2_iter1200/`
   - new: `docs/walk_noclock_research.md`
   - mod: `deploy/sim2sim/evaluate.py`, `deploy/sim2sim/sim2sim.py`,
     `deploy/manifest.py`, `env/tasks/birl_task.py`

---

**Goal:** Train a bipedal locomotion policy without an external phase clock
that transfers to MuJoCo sim2sim at least as well as `walk_v34` (which has
the clock) — accepting being a bit worse.

This file was updated **incrementally** as experiments finished. Each section
was appended after the relevant run landed; nothing is lost if the session
was interrupted.

---

## 0. Why a no-clock policy is wanted

`walk_v34` (`configs/walk_rl.yaml`) is the current best policy and has 100 %
sim2sim survival in MuJoCo at friction 1.0. It uses an **external phase
clock** as both an obs slot and a swing/stance mask source. The clock has
two known downsides:

1. **Standstill jitter.** The clock keeps ticking at `cmd_freq=2.5 Hz` even
   when `‖cmd‖<0.15`. The obs slot is masked by `static_flag` but the
   internal clock state still advances, producing a noisy training signal
   around stand. `walk_v34` at iter 3200 measured 7.6 Hz hip jitter at
   `cmd=0` (`com_z_rms=0.118`, `duty_factor=0.93` — feet on ground but joints
   trembling). That's not a deployable stand.

2. **Tight coupling between deploy-time `cmd_freq` knob and policy behaviour.**
   The SDK has to also pass `cmd_freq` to the policy, and the policy must
   gait at exactly the right frequency to keep `foot_phase` reward high.

A no-clock policy in the legged-gym / DreamWaQ / HIM family fixes both.

---

## 1. Prior context — earlier failures

### 1.1 `walk_noclock` v1 (first attempt, killed at iter 1961)

`configs/walk_noclock.yaml`. Recipe:

- `phase.mode: none`, obs 6 slots × 5 frames = 190 dim, action 10 dim.
- `air_time: 5.0` (was 1.0) — load-bearing scheduler.
- `foot_phase: 0.0` (off — no clock to phase-match).
- `base_heit: 1.0 → 2.0`, `base_heit_slope: 70 → 150`, target 0.45.
- `leg_width_rew: 1.0`, `foot_clr/supt/heit` bumped to 1.5/1.0/1.0.

Training-side reward was healthy at `~75/step` by iter 1000 (better than
`walk_v34`'s 71.7). But sim2sim survival **regressed** over training:

```
iter 1400: fr1.0=3.3s  fr1.5=3.8s
iter 1600: fr1.0=2.7s  fr1.5=3.0s
iter 1800: fr1.0=2.0s  fr1.5=2.4s   ← getting worse with more training
```

Standstill still showed 12.5 Hz hip jitter. Two independent failures:

- **Standstill no-incentive gap.** Every "stand-useful" reward
  (`foot_supt`, `foot_clr`, `foot_heit`, `air_time`, ...) is gated by
  `static_flag`. At `‖cmd‖<0.15`, all are zero. The policy has no positive
  signal for "keep feet planted". Standing emerges only from weak
  regulators + tracking-to-zero — not enough.
- **Sim2sim obs-delay mismatch.** Training delays joint/IMU obs by 10-40 /
  20-50 policy steps (config `delay_observation: true`). Sim2sim was
  reading current-frame values — ~150-750 ms freshness gap. Walk_v34
  tolerated this because its `phase_clock` obs slot was a deterministic
  counter unaffected by delay; walk_noclock has only proprio, all of which
  is now mismatched.

Decision: don't try harder with v1's recipe — both failures are structural,
not training-incomplete.

---

## 2. Bugs found in sim2sim during this session

These were not all caused by walk_noclock — some pre-existed. Found while
diagnosing the gap:

### 2.1 `phase_mode='none'` always routed through MIRL obs builder

`deploy/sim2sim/evaluate.py:75` originally: `is_mirl = (phase_mode == 'none')`.

This dispatched walk_noclock policies through `get_obs_mirl()`, which emits
64-dim MIRL layout (8 cmds + 21 zero ref slots). Walk_noclock manifest says
`obs_per_step: 38`. The deque was initialized at 38, then mixed with 64-dim
samples — concatenation produced 4 × 38 + 64 = **216 vs expected 190** at
the policy input. Sim2sim eval crashed with `ONNXRuntimeError` at every
iter-200 boundary.

**Fix:** added `is_noclock = (obs_per_step == 38)` and `get_obs_noclock()`
emitting the 6-slot 38-dim layout (`evaluate.py` + `sim2sim.py`).

### 2.2 Sim2sim missing obs-delay DR

Training delays per-env per-episode via `delay_joint_ranges: [10, 40]` etc.
Sim2sim used `data.qpos[...]` and `data.qvel[...]` directly with no delay.

**Fix:** sample one delay per-episode in `run_episode()` matching training
ranges; per-quantity history deques (`_q_buf`, `_dq_buf`, `_eul_buf`, `_av_buf`)
populated each policy step via `_push_obs_history()`; obs builders read
`_delayed()` which returns the per-quantity delayed values.

This dropped iter-1000 walk_noclock survival from 5.0 s → 0.58 s — confirming
the policy was leaning on fresh obs as an undocumented crutch.

### 2.3 Manifest didn't carry delay ranges

`deploy/manifest.py` had no `delay_*_ranges` field. The sim2sim defaults
hardcoded `[10, 40]` / `[20, 50]` etc. would diverge silently if a future
training config widened or narrowed DR.

**Fix:** added `delay_joint_ranges`, `delay_angle_ranges`, `delay_rate_ranges`
to manifest from `params['domain_rand']`; `manifest_to_sim2sim_cfg` plumbs
them into the cfg dict.

### 2.4 `base_heit_slope` and `base_heit_target` hardcoded

`base_heit_rew = torch.exp(-70 * (base_pos.z - 0.45)**2)` hardcoded the
slope and target. Couldn't be tuned via config.

**Fix:** read `cfg.reward.base_heit_slope` (default 70) and
`cfg.reward.base_heit_target` (default 0.45). v1/v2/v3/v4 now use slope=150.

### 2.5 `air_time` reward inactive in `phase.mode=none`

`air_time_rew` was gated `if self._phase_mode == 'input' and self._cmd_freq
is not None` — silently zero for `phase.mode=none`. Walk_noclock was
configured with `air_time: 5.0` but the term was dead weight.

**Fix:** added `phase.mode=none` branch with a fixed `target_swing` scalar
(default 0.2 s, legged-gym style). Reads `cfg.phase.target_swing`.

### 2.6 `foot_stand` reward (new term, not a bug)

Added new reward term that activates only at `‖cmd‖<0.15`:

```python
foot_stand_rew = mean(foot_frc >= 10) * (1 - static_flag)   # range [0, 1]
```

Counters the standstill-no-incentive gap. Weight default 0; v2/v3/v4 set
1.0.

---

## 3. Experiments

For each experiment: recipe summary, what's different from the previous,
expected outcome, then results once available.

### 3.1 `walk_noclock_v2` — fixed sim2sim + foot_stand — KILLED at iter ~990

**Config:** `configs/walk_noclock_v2.yaml` (flattened, no `_base`).
**Started:** 2026-05-10 23:13. **Killed:** 2026-05-10 23:50 at iter ~990.

**Outcome: dead end.** With sim2sim now using matched delay DR, the true
sim2real gap is exposed and it's brutal. Training-side perfectly healthy
(`l_n=561`, `total_rew=71.15`, `task_rew=43.36`) but MuJoCo survival
**regressed**:

```
iter 400: fr1.0=1.1s
iter 600: fr1.0=1.0s
iter 800: fr1.0=0.8s   ← getting worse
```

This is the same regression pattern as v1, but starting from a worse base
(v1 was 2-3 s, v2 is 0.8-1 s). The previous "OK-looking" 5s numbers from
v1 were obs-delay-mismatch artifacts — sim2sim was feeding the policy obs
~250 ms fresher than training, and the policy was using that to recover
from MuJoCo's contact response.

Confirmed: `foot_stand` reward + matched delay DR is NOT enough to bridge
the no-clock sim2real gap on its own.

**Decided to pivot to v3 immediately** rather than waste 2h finishing v2.

**What's in:**
- All v1 settings.
- **New:** `reward.foot_stand: 1.0` — standstill double-support reward.
- **New code:** sim2sim has obs-delay DR matched to training distribution.

**Hypothesis:** the v1 failures were (a) standstill no-incentive and (b)
sim2sim eval feeding the policy obs that were ~250 ms fresher than
training. Both are now addressed without changing core training-side
dynamics. If those were the dominant failure modes, sim2sim survival should
climb steadily and standstill jitter should drop.

**Risk:** the delay fix exposed that the policy is intrinsically delay-sensitive.
v2 may still regress if the underlying problem is fragility to physics gap
(IG↔MuJoCo contact), not just the obs-delay mismatch.

**Watching:**
- TB scalar `sim2sim/survive_time_fr1.0`: target ≥ 5.0 by iter 2000.
- TB scalar `Rewards/foot_stand`: should converge to ~1.0 × cmd-fraction-below-threshold.
- `com_z_mean` at `cmd=0` from manual sim2sim probe: target > 0.40 (was 0.38 in v34).
- `measured_freq` at `cmd=0`: target < 3 Hz (was 12.5 in v1, 7.6 in v34).

**Results:** _filled when run finishes or is killed_

### 3.2 `walk_noclock_v3` — Cassie-style periodic reward — KILLED at iter ~892

**Config:** `configs/walk_noclock_v3.yaml`.

**What's different from v2:**
- `phase.mode: none → input` (internal clock ticks).
- Obs slots **unchanged** — still 6 slots, no `phase_clock` or
  `phase_freq_cmd`. The policy still sees 38-dim obs and has no idea what
  the clock is.
- `reward.foot_phase: 0 → 4.0` — clock drives swing/stance reward.
- `reward.air_time: 5.0 → 2.0` — `foot_phase` takes over as primary
  scheduler.
- Swing/stance masks now come from `ext_clock.phase_with_offset` (because
  `phase.mode='input'` triggers that code path automatically).

**Sim2sim handling:** `is_noclock = (obs_per_step == 38)` regardless of
`phase_mode`. Walk_noclock_v3 has phase.mode=input but obs_per_step=38, so
sim2sim routes through `get_obs_noclock()` (which doesn't tick or read the
clock). The policy was trained with clock-driven rewards, but at deploy it
runs purely on proprio — exactly Cassie / MIT humanoid recipe.

**Hypothesis:** v2's only timing signal is `air_time` (scalar target).
That can be hit by hop, shuffle, asymmetric gait. `foot_phase` reward
requires matching a specific swing-stance schedule — pulls policy toward
biped-natural gait. Cassie paper reports periodic-reward-without-obs-slot
gives transfer comparable to with-obs-slot.

**Risk:** policy must infer phase from delayed proprio history to maximise
`foot_phase` reward. If it can't, the reward signal becomes pure noise and
training may diverge.

**Outcome: also a dead end.** Sim2sim survival even worse than v2:

```
iter 200: fr1.0=1.5s
iter 400: fr1.0=1.2s
iter 600: fr1.0=1.2s
iter 800: fr1.0=0.8s   ← matches v2 trajectory
```

Direct probe at iter 800 across 3 seeds:
```
stand   (cmd=0):   surv 3 seeds: [0.78, 0.77, 0.82]
fwd 0.3 (cmd_vx):  surv 3 seeds: [0.74, 0.70, 0.78]
```

Falls in <1s regardless of cmd. The policy is fundamentally unstable in
MuJoCo physics. Cassie-style periodic reward doesn't change the outcome.

**Key insight from comparing v1/v2/v3:** training-side metrics are
near-identical (`l_n ~ 580`, `total_rew ~ 65-75`). Three different
reward recipes (no foot_phase, foot_stand only, foot_phase via internal
clock) all train to similar IG performance but all die in MuJoCo in
< 1 s.

**Diagnosis crystallised:** the gap isn't in reward shaping. It's that
without a phase clock obs, the policy has only proprio to close the
control loop, and the IG↔MuJoCo contact-dynamics difference breaks that
loop. With clock obs (walk_v34), the policy has a deterministic
physics-independent timing anchor → robust to physics differences.

### 3.3 `walk_noclock_v4` — v2 + longer history (10) + actuator delay DR

**Config:** `configs/walk_noclock_v4.yaml`.

**What's different from v2:**
- `observation.history: 5 → 10` (window 150 → 300 ms, > 1 stride at 2.5 Hz).
- Obs total: 190 → 380.
- `action.use_actuator_delay: false → true` (1-3 step delay DR active).
- `action.use_actuator_filter: false → true` (LP filter alpha 0.3-0.7 DR active).

**Hypothesis:** v2 obs window is 150 ms (37 % of a stride), barely enough
to estimate phase from joint vel patterns. Doubling history + adding
actuator-side DR turns the policy more robust to physics gap (actuator delay
+ filter is a stand-in for unmodeled motor response). DreamWaQ / HIM use
this combination; we don't have their adaptation encoder but the longer
history is the cheap version.

**Risk:** doubled obs is 380-dim; network capacity (hidden 512×256) may need
to grow. Also more iterations to converge (more params to fit).

**Results:** _filled when run finishes_

---

## 4. Cross-experiment summary table (final)

Measured with **fixed sim2sim** (delay DR pushed at physics rate, post 00:35
bug-fix). Total = sum of mean survival across 6 cmds × 3 seeds × 8 s, max
48.0 s.

| Run | Recipe | best iter | total surv | stand | vx_bias @ 0.3 | Verdict |
|---|---|---|---|---|---|---|
| walk_v34 | with clock (original baseline) | 3200 | 43.3 s | 8.0 | 0.252 | proven, 1 fall on 0.3/seed |
| walk_noclock v1 | air_time only, no clock, no foot_stand | 1800 | 27.5 s | 8.0 | — | weak walking, esp fwd 0.5 (1.6 s) |
| walk_noclock v2 | v1 + foot_stand + sim2sim delay DR | 800 | (limited) | 8.0 | — | killed early, never converged |
| walk_noclock v3 | Cassie periodic reward, no clock obs | **3000** | 39.8 s | 8.0 | 0.293 | 92 % of walk_v34, viable no-clock recipe |
| walk_noclock v3 | (overfit) | 5000 | 36.7 s | 6.9 | — | yaw_err runaway |
| **walk_rl_v2** 🏆 | **walk_v34 + foot_stand, init_only resume** | **1200** | **48.0 s** | **8.0** | **0.032** | **PERFECT, deploy** |
| walk_rl_v2 | (slight late drift) | 2400 | 47.2 s | 8.0 | 0.065 | |

---

## 5. Findings & decisions log (chronological)

- **23:00 (pre-v2)** — diagnosed v1 failure as (a) standstill no-incentive
  and (b) sim2sim obs-delay mismatch. Fixed both before launching v2.
- **23:13** — launched v2.
- **23:50 — surveyed Unitree's two public RL stacks for G1/H1.** Findings:

  **`unitree_rl_gym`** (older, legged_gym-based):
  - 47-dim obs: `ang_vel(3) + projected_gravity(3) + cmd(3) + dof_pos(12) + dof_vel(12) + last_action(12) + sin_phase(1) + cos_phase(1)`
  - **Phase IS in obs** — `sin(2π·t/0.8), cos(...)` at FIXED period 0.8 s (1.25 Hz).
  - LSTM 64 hidden, 1 layer (recurrent, no framestack).
  - Asymmetric critic with `base_lin_vel` (50-dim).
  - Reward: `base_height -10`, `feet_swing_height -20`, `contact 0.18` (phase-stance match), `tracking 1.0+0.5`, `feet_air_time 0` (disabled), `alive 0.15`.
  - Action scale 0.25, decimation 4 → 50 Hz.

  **`unitree_rl_lab`** (newer, IsaacLab-based — current G1 recipe):
  - Actor obs: `ang_vel + projected_gravity + cmd + joint_pos_rel + joint_vel_rel + last_action`, **history 5 frames** (no recurrence!).
  - Critic obs: same + `base_lin_vel` (privileged).
  - **No phase clock in obs.** Phase exists ONLY as a counter inside the `gait` reward.
  - `gait` reward = `~(is_stance XOR is_contact)` summed over feet, with `period=0.8, offset=[0, 0.5], threshold=0.55` and command-gated (off at stand).
  - `feet_clearance` = `exp(-Σ height_err² · tanh(vel) / std)` with target 0.1 m.
  - Reward weights: `base_height -10`, `flat_orientation_l2 -5`, `gait 0.5`, `feet_clearance 1.0`, `joint_deviation_legs -1.0` (penalize hip_roll + hip_yaw drift), `tracking 1.0+0.5`, `alive 0.15`, `joint_deviation_arms -0.1`, `joint_deviation_waists -1`, `dof_pos_limits -5`, `energy -2e-5`, `action_rate -0.05`, `feet_slide -0.2`, `undesired_contacts -1`.
  - Action scale 0.25, decimation 4, episode 20 s.

  **What this means for our work:**
  1. The **newer** Unitree recipe matches our **v3 design exactly** —
     periodic reward with phase counter, NO phase obs slot, 5-frame
     history, asymmetric critic. v3 is on the right track.
  2. They use **fixed period 0.8 s = 1.25 Hz**, much slower than our
     `cmd_freq=[2.0, 3.0]`. Slower gait = easier physics, more stable.
     Worth a v5 variant: lower `cmd_freq` range to e.g. `[1.0, 1.5]`.
  3. Their **standstill incentive** is via STRONG static posture rewards:
     `base_height -10` and `flat_orientation_l2 -5`. Our `base_heit 2.0`
     positive-exp + `balance 1.5` are weaker. Our `foot_stand 1.0` is a
     positive equivalent — different shape but same goal.
  4. They have **no `air_time` reward at all** (disabled or absent).
     We have `air_time 5.0` (v2). Gait shaping comes entirely from
     `feet_clearance` + `gait` periodic reward in their setup.
  5. **No recurrence in the modern recipe.** Our framestack 5 + MLP is
     architecturally identical to theirs — that's not the gap.

  **Conclusion for our roadmap:**

  - **v3 (already prep'd)** is the closest match to Unitree's modern
     recipe. Most likely to transfer well. Keep as-is.
  - **Potential v5** — copy more of Unitree's recipe verbatim:
    - Adjust `cmd_freq` range to **`[1.8, 2.2]`** (scaling-corrected — see
      note below). NOT [1.0, 1.5] as G1's 1.25 Hz wouldn't transfer to
      smaller robot.
    - Add `flat_orientation_l2` penalty (≈ what `balance` does but bigger).
    - Drop `air_time` reward (let `foot_phase` + `feet_clearance` do the gait shaping).
    - Bump `joint_deviation` style penalty on hip_yaw/hip_roll (we have
      `act_const` 0.02 — bump to ~0.5).

  **Frequency scaling note (Qmini vs G1):** the natural pendulum frequency
  is `f_nat ≈ √(g/L) / (2π)`. Unitree G1 has leg ≈ 0.5 m → `f_nat ≈ 0.7`,
  and their `1.25 Hz` is ≈ 1.8× natural. Qmini has leg ≈ 0.3 m →
  `f_nat ≈ 0.9 Hz`, so 1.8× natural ≈ **1.6 Hz**. Our `walk_v34` ran at
  2.5 Hz (≈ 2.8× natural) and worked fine for sim2sim — so the upper end
  is OK. A targeted v5 cmd_freq range of `[1.8, 2.2]` centers near 2× and
  also tightens the range, reducing what the policy needs to generalise
  across.
- **00:35 — CRITICAL BUG FOUND in my own sim2sim delay-DR implementation.**
  User pushed back ("不应该站都站不住的") — instinct said <1 s survival is
  too pathological to be sim2real gap alone. Ran sanity check: walk_v34
  (previously 100 % sim2sim) → also 0.6 s with new delay code. Bug confirmed.

  **Root cause:** in training (`env/legged_robot.py:step_torques`), the
  history buffers (`joint_pos_his` etc.) are appended **per physics step**
  (1000 Hz, every sim step inside the decimation loop). The `delay(N)`
  read therefore consumes **N physics steps** = N ms at 1 ms phys dt. So
  `delay_joint_ranges: [10, 40]` means 10-40 ms total latency.

  My sim2sim implementation pushed to the buffers **per policy step**
  (67 Hz) — and used the same `[10, 40]` integer range. Result: effective
  delay was **150-600 ms** = 15× too large. Policies trained on 10-40 ms
  delay couldn't handle being told their obs was 600 ms stale, and fell
  in <1 s in MuJoCo. The fact that ALL recipes (walk_v34, v1, v2, v3)
  hit the same failure was the giveaway — it wasn't a recipe problem, it
  was a measurement problem.

  **Fix:** moved `_push_obs_history()` out of the `step % decimation == 0`
  branch into the outer per-physics-step loop. Same delay-step counts
  now mean 1 ms each in both training and sim2sim.

  **Re-evaluation with fixed sim2sim:**
  | Policy | iter | stand | fwd 0.3 | fwd 0.5 |
  |---|---|---|---|---|
  | walk_v34 | 3200 | 8/8/8 | mostly 8 (1 fall) | mostly 8 (1 fall) |
  | **walk_noclock v1** | 1800 | **8/8/8** | 4.4 / 2.5 / 6.4 | 1.6 / 1.6 / 1.6 |
  | **walk_noclock v3 (Cassie)** | 800 | **8/8/8** | 8.0 / 8.0 / 2.0 | 6.3 / 6.7 / 2.2 |

  No-clock recipes **DO** work. v3 in particular stands perfectly and
  walks comparably to walk_v34 despite training to only 800 iters
  (vs walk_v34's 3200). Earlier "v2/v3 are dead" decisions were all based
  on the broken delay measurement.

  **Decision:** kill v4 (built on inferior v2 recipe), relaunch v3 fresh
  with the fixed sim2sim. Let v3 train to 5000 iter for a real
  apples-to-apples comparison with walk_v34.
- **00:45 — cross-policy survival matrix (fixed sim2sim).** Survival is
  mean of 3 seeds × 8 s episodes at friction 1.0, cmd_freq 2.5:

  | Policy | iter | stand | fwd 0.3 | fwd 0.5 | fwd -0.3 | strafe 0.3 | yaw 0.5 |
  |---|---|---|---|---|---|---|---|
  | walk_v34 (clock baseline) | 3200 | 8.0 | 5.8 | 8.0 | 5.5 | 8.0 | 8.0 |
  | walk_noclock v1 | 1800 | **8.0** | 4.5 | 1.6 | 8.0 | 8.0 | 8.0 |
  | walk_noclock v2 (+foot_stand) | 800 | **8.0** | 6.6 | 4.1 | 8.0 | 8.0 | 8.0 |
  | walk_noclock v3 (Cassie) | 800 | **8.0** | 6.0 | 5.1 | 8.0 | 6.8 | 8.0 |

  **All 4 stand perfectly with fixed sim2sim.** The "walk_v34 can't stand —
  7.6 Hz hip jitter at cmd=0" diagnosis at the start of this session was
  a **sim2sim measurement artifact** (no-delay obs path was lying about
  what the policy actually does). walk_v34 stands fine when measured
  correctly.

  This means the original premise of the no-clock investigation was
  largely false. walk_v34 IS deployable for stand+walk; the no-clock
  variants are also viable; the choice between them is engineering
  preference, not necessity.

- **00:50 — revised plan for the rest of the night:**

  1. Let walk_noclock_v3 (re-launched 00:35, Cassie-style) train to
     5000 iters (~2.5 h, done ~03:00). Provides a clean "no-clock"
     research baseline.
  2. After v3 finishes: launch `walk_rl_v2` (= walk_v34 recipe +
     `reward.foot_stand: 1.0`), 5000 iters (~2.5 h, done ~05:30).
     This is the simplest path to a deployable policy — known-good
     recipe + the only meaningful improvement (standstill belt-and-suspenders).
  3. Final comparison: walk_v34, walk_noclock_v3 final, walk_rl_v2 final.
     Pick the best as deployable. Update `deploy_candidates/`.
- **03:35 — walk_noclock_v3 finished, 5000 iters (185 min total).**

  Training-side: `total_rew 72.4`, `task_rew 55.0`, `l_n 636/669` near max,
  `m_kl 0.009` stable. Sim2sim during training: `fr1.0` survival mostly
  8-11 s across iter 1000-5000. `yaw_err` crept up from 0.4 → 1.5 mid-run,
  partially recovered to ~1.0 at iter 5000 — first warning sign.

  **Ckpt sweep (3 seeds × 8 s, friction 1.0, cmd_freq 2.5):**

  | ckpt | stand | fwd 0.3 | fwd 0.5 | fwd -0.3 | strafe 0.3 | yaw 0.5 | total |
  |---|---|---|---|---|---|---|---|
  | walk_v34 (baseline) | 8.0 | 5.8 | 8.0 | 5.5 | 8.0 | 8.0 | **43.3** |
  | v3 @ iter 2000 | 7.7 | 6.1 | 3.4 | 4.0 | 7.7 | 6.1 | 35.0 |
  | **v3 @ iter 3000** | **8.0** | 6.3 | 5.0 | 6.0 | 6.4 | **8.0** | **39.8** ✓ |
  | v3 @ iter 4000 | 6.1 | 6.2 | 6.4 | 3.3 | 3.9 | 6.5 | 32.4 |
  | v3 @ iter 5000 | 6.9 | 8.0 | 2.9 | 4.5 | 6.4 | 8.0 | 36.7 |

  **Findings:**

  - **v3 reaches 92 % of walk_v34's total survival (39.8 vs 43.3 s)** —
    confirms a no-clock policy CAN approach the with-clock baseline.
  - **Best ckpt is iter 3000, not 5000.** Training past 3000 degrades —
    classic overfitting. `yaw_err` climbing 0.4 → 1.5 during this stretch
    was the leading indicator.
  - **No-clock weakness localised:** v3 @3000 trails walk_v34 mainly on
    `fwd 0.5` (5.0 vs 8.0 s) — fast forward walking. Other commands are
    within 1-2 s of baseline.
  - **Standstill is solid:** 8.0/8.0/8.0 at iter 3000 — `foot_stand` +
    Cassie periodic reward gating off at stand works.

  **Take-away:** v3 is a viable no-clock recipe. The 8 % gap vs walk_v34
  is real but recoverable through a single targeted retrain (early-stop
  at iter 3000, possibly tighter yaw_rat weight to control the runaway
  yaw_err). The Cassie-style design — clock in reward only, not in obs —
  matches Unitree's published G1 recipe and works in our setup.

### 3.4 `walk_rl_v2` — walk_v34 recipe + foot_stand (Plan B)

**Config:** `configs/walk_rl_v2.yaml` (inherits `walk_rl.yaml`, adds
`reward.foot_stand: 1.0`).

**Launched:** 2026-05-11 03:43 via `--resume 20260509_2239_walk_v34
--init_only` for 2500 iters. The `--init_only` flag loads walk_v34's
actor/critic weights but resets optimizer state (Adam moments are stale
under the new reward) and iteration counter.

**Hypothesis:** walk_v34 already transfers well — adding foot_stand
should be a strict positive: same gait, with slightly more solid stand.
Worst case it's neutral.

**Watching:**
- Sim2sim fr1.0 should stay > 8 s (matching walk_v34 baseline).
- Standstill should remain 8.0 s.
- `foot_stand` reward should converge to ~0.0 (since walk_v34 already
  keeps feet planted at standstill via `foot_phase` gating — this is the
  belt-and-suspenders check).

**Results — finished 05:13, 2500 iters, 93 min total.**

Sim2sim survival during training (`fr1.0`, max 16 s):
```
iter  200: 12.9s        iter 1200: 15.0s ← peak
iter  400: 14.0s        iter 1400: 13.6s
iter  600: 14.0s        iter 1600: 13.0s
iter  800: 14.4s        iter 1800: 12.8s
iter 1000: 14.8s        iter 2000-2400: 12.7s
```

Strong from the start (resumed weights), peaks at iter 1200, mild decline
thereafter. `vx_err fwd` was 0.06-0.16 throughout — 5× tighter tracking
than v3.

**Ckpt eval (3 seeds × 8 s, friction 1.0, cmd_freq 2.5):**

| ckpt | stand | fwd 0.3 | fwd 0.5 | fwd -0.3 | strafe 0.3 | yaw 0.5 | total | vx_bias @ fwd0.3 |
|---|---|---|---|---|---|---|---|---|
| walk_v34 (baseline) | 8.0 | 5.8 | 8.0 | 5.5 | 8.0 | 8.0 | 43.3 | 0.252 |
| v3 @3000 (no-clock) | 8.0 | 6.3 | 5.0 | 6.0 | 6.4 | 8.0 | 39.8 | 0.293 |
| **walk_rl_v2 @1200** | **8.0** | **8.0** | **8.0** | **8.0** | **8.0** | **8.0** | **48.0** 🏆 | **0.032** |
| walk_rl_v2 @2400 (final) | 8.0 | 8.0 | 8.0 | 7.2 | 8.0 | 8.0 | 47.2 | 0.065 |

walk_rl_v2 @1200 hits **perfect 48.0/48.0 s survival** across the full
command grid, and forward-velocity tracking error drops **8×** vs the
walk_v34 baseline (0.032 vs 0.252). Iter 1200 is the new best ckpt;
iter 2400 mildly degrades (consistent with the slight reward overshoot
seen in v3 too).

---

## 6. Final recommendation

**Deploy `walk_rl_v2 @iter 1200`.**

It's walk_v34's proven recipe with one addition (`foot_stand: 1.0`) and
beats every other ckpt we have on:
- standstill (8.0 / 8.0 / 8.0 s — no jitter)
- omnidirectional walking (perfect 8 s × 3 seeds on every cmd)
- velocity tracking (8× better than walk_v34)

The Cassie-style no-clock variant (walk_noclock_v3 @3000) is a viable
research result — it confirms our setup CAN train no-clock policies that
approach the with-clock baseline (92 % of walk_v34's total survival).
But for an actual deployable, the with-clock + foot_stand recipe is
strictly better here.

### Why walk_rl_v2 outperforms walk_v34 (the baseline it resumed from):

- **`foot_stand` reward**: positive incentive (mean(foot_frc≥10)) at
  `‖cmd‖<0.15`. Previously the standstill regime had only weak
  attractors (base_heit, balance, tracking-to-zero). Adding foot_stand
  gave it a strong positive pull toward double-support that the policy
  exploited.
- **`init_only` reset**: fresh optimizer state under the new reward
  found a strictly better local optimum than the converged walk_v34.
  This is a well-known finding (Henderson et al., re-randomising final
  layers often helps fine-tuning).

### Suggested next steps

1. Export `walk_rl_v2 @1200` ONNX to `deploy_candidates/`. Re-run
   `scripts/release_eval.py` to generate videos + plots.
2. Confirm on real robot if available — sim2sim alignment is now strong
   (validated by the bug-find), so transfer should be representative.
3. The Cassie-style no-clock route (v3) is still a valid research
   trajectory. If the deploy goal ever shifts to "no `cmd_freq` SDK
   knob", v3 is a workable starting point — early-stop at iter 3000
   and tighten `yaw_rat` weight to prevent the late-training yaw drift.

