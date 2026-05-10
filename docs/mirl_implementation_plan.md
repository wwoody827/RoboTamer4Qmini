# MIRL upgrade: full Motion-Imitation-RL pipeline for walk_v27 traces

## Why

We have **144 clean BC-style demonstrations** (`data/datasets/walk_v27_bc_clean/`,
51.7 MB, 28 min) recorded from walk_v27's best ckpts. Pure BC on this would
suffer covariate shift + catastrophic forgetting. Modern locomotion RL uses
**MIRL** instead: keep RL exploration, add imitation as a *soft* constraint
that decays over training.

The codebase already has 2/6 MIRL pieces wired up. This plan implements the
remaining 4. Implementing this turns our trace dataset into a **real
training signal** rather than a passive archive.

---

## Audit: current state vs full MIRL

| # | Component | Current | Gap |
|---|-----------|---------|-----|
| 1 | Hybrid reward `w_imit·imit + w_task·task` | ✅ `birl_task.py:1022-1037` | weights are constants — no decay |
| 2a | RSI: random start frame | ✅ `_assign_ref_clips:339` | frame index assigned |
| 2b | RSI: init robot **state** from clip frame | ❌ `_reset_dofs` uses default ref pose, ignores clip | **secret weapon** per Gemini |
| 3 | BC regularization in PPO actor loss | ❌ — | optional Phase 2 |
| — | w_imit / w_task annealing schedule | ❌ — | constant in current code |
| — | Cmd-matched clip selection | ❌ random regardless of cmd | clips from cmd=−0.3 trace assigned to env with cmd=+0.5 → noise |

---

## Phase 1 — must-have (A + B + C, ~75 LOC)

Run order: A first (verify RSI alone), then B+C together.

### A. Full RSI (state initialization from clip frame)

**Why**: kicks the robot into "interesting" intermediate states (mid-stride,
mid-strafe) that PPO with default reset takes thousands of iters to discover.
This is the well-documented core of DeepMimic / AMP.

**Files**

- `env/tasks/birl_task.py` — extend `_load_ref_clips` to also load
  `base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel` from the .npz (our
  trace format already has all these keys, see `data/README.md`):
  ```python
  clips.append({
      'joint_pos': ..., 'joint_vel': ...,
      'base_pos':     torch.tensor(raw['base_pos']),     # NEW
      'base_quat':    torch.tensor(raw['base_quat']),    # NEW (w,x,y,z)
      'base_lin_vel': torch.tensor(raw['base_lin_vel']), # NEW (world)
      'base_ang_vel': torch.tensor(raw['base_ang_vel']), # NEW (body frame)
      'T': ..., 'dt': ..., 'loop': ..., 'skill': ...,
  })
  ```
  Pad and stack as `_ref_bp_all`, `_ref_bq_all`, `_ref_blv_all`, `_ref_bav_all`
  alongside `_ref_jp_all` / `_ref_jv_all`.

- `env/legged_robot.py` — `_reset_dofs` and `_reset_root_states` need a
  task-aware code path. Cleanest: expose a hook the task can override.
  Option 1 (preferred): add `task.get_reset_state(env_ids) -> dict|None`.
  When non-None, `_reset_dofs`/`_reset_root_states` use those tensors
  instead of defaults.
  ```python
  # legged_robot.py
  override = self.task.get_reset_state(env_ids) if hasattr(self, 'task') else None
  if override is not None:
      self.joint_pos[env_ids] = override['joint_pos']
      self.joint_vel[env_ids] = override['joint_vel']
      base_state = override['base_state']  # [N, 13]
  else:
      # existing default reset
  ```
  ```python
  # birl_task.py
  def get_reset_state(self, env_ids):
      if not self._has_ref:
          return None
      cid, fid = self._ref_clip_id[env_ids], self._ref_frame_idx[env_ids]
      base_state = torch.cat([
          self._ref_bp_all[cid, fid],     # [N, 3] world pos
          self._ref_bq_all[cid, fid],     # [N, 4] quat
          self._ref_blv_all[cid, fid],    # [N, 3] world lin vel
          self._ref_bav_all[cid, fid],    # [N, 3] body ang vel
      ], dim=-1)
      return {
          'joint_pos': self._ref_jp_all[cid, fid],
          'joint_vel': self._ref_jv_all[cid, fid],
          'base_state': base_state,
      }
  ```
- Keep DR perturbations on top: add small Gaussian noise to the loaded state
  so env diversity isn't lost (e.g. ±0.05 rad on joints, ±0.05 m on base z,
  ±0.1 rad/s on velocities).

**Risk**: clip stride ≠ env stride → robot may load into a configuration the
policy cannot continue. Mitigation: only RSI **clean** traces (already
filtered in `data/datasets/walk_v27_bc_clean/`).

### B. Cmd-matched clip selection

**Why**: random clip assignment poisons imitation when env's `cmd_vx=+0.5`
gets paired with a clip recorded for `cmd_vx=-0.3`. This is the dominant
source of imitation noise with a multi-cmd dataset.

**Files**

- `env/tasks/birl_task.py` — `_load_ref_clips`:
  - Read `cmd_const` from each .npz, store as `self._ref_cmd_all` shape
    `[n_clips, 3]`.
  - If a .npz lacks `cmd_const` (legacy MIRL clips), fall back to all-zero
    cmd → matches stand cmd preferentially.

- `env/tasks/birl_task.py` — `_assign_ref_clips`:
  ```python
  cmds = self.commands[env_ids, :3]                                 # [N, 3]
  weights = torch.tensor([1.0, 1.0, 0.5], device=self.device)        # yaw less weighted
  d = ((cmds.unsqueeze(1) - self._ref_cmd_all.unsqueeze(0)) * weights).pow(2).sum(-1)
  self._ref_clip_id[env_ids] = d.argmin(dim=-1)
  ```

- For variety, add **soft assignment**: pick uniformly among top-K nearest
  clips (K=3) so envs with the same cmd don't all imitate the *same* clip
  → preserves diversity in trajectories.

**Risk**: only 144 clips covering 27 unique cmds. Fine for the cmd grid
we sweep over; if training samples cmd outside the grid, nearest-neighbor
may be far. Mitigation: include `stand` clip + interpolation note for
later.

### C. w_imit annealing

**Why**: imitation should be a **scaffold**, not a destination. Start
strong (policy follows demos closely), end weak (PPO exploration owns the
late-game optimization).

**Files**

- `configs/base.yaml` — add defaults:
  ```yaml
  task:
    w_imit_start: 0.8       # initial weight on imitation
    w_imit_end:   0.1       # final weight
    w_imit_decay_until: 5000  # iters; weights interpolate linearly to this
    w_task: 1.0             # task reward stays constant (or set
                             # w_task_start/end if want it to ramp up)
  ```

- `env/tasks/birl_task.py` — replace constant `getattr(self.cfg.task, 'w_imit', 0.5)`
  with a method:
  ```python
  def _w_imit_now(self):
      it = self.train_iter            # set externally by train.py each iter
      d  = self.cfg.task.w_imit_decay_until
      a  = min(it / max(d, 1), 1.0)
      return (1 - a) * self.cfg.task.w_imit_start + a * self.cfg.task.w_imit_end
  ```

- `train.py` — push current iter into task each iteration:
  ```python
  task.train_iter = it
  ```

- TB logging: log `task/w_imit` per iter — verify schedule is what we want.

**Risk**: too aggressive decay (decay_until = 1000) → imitation gone before
policy stabilizes. Too slow (decay_until = 7500) → late-stage stuck
imitating walk_v27's bugs. Default 5000 (62.5 % of 8000-iter run) is a
sane starting point.

### Phase 1 deliverables

- New config `configs/walk_v29_imit.yaml`:
  ```yaml
  _base: walk_v27.yaml
  task:
    ref_clip_paths:
      # Curated subset — top per-cmd clean traces (built from episodes.csv).
      - data/traces/walk_v27_multi/6200/vxp030_vyp000_yawp000.npz
      - data/traces/walk_v27_multi/6200/vxn030_vyp000_yawp000.npz
      - data/traces/walk_v27_multi/6200/vxp000_vyp030_yawp000.npz
      - data/traces/walk_v27_multi/6200/vxp000_vyn030_yawp000.npz
      - data/traces/walk_v27_multi/6200/vxp000_vyp000_yawp050.npz
      - data/traces/walk_v27_multi/6200/vxp000_vyp000_yawn050.npz
      - data/traces/walk_v27_multi/6200/vxp000_vyp000_yawp000.npz
      # ... + best-per-cmd from other ckpts via episodes.csv selection
    w_imit_start: 0.8
    w_imit_end:   0.1
    w_imit_decay_until: 5000
    w_task: 1.0
  ```

- A small script `scripts/select_best_clips.py` that reads
  `data/traces/walk_v27_multi/*/manifest.csv`, picks the lowest-error
  clean trace per cmd, and emits the YAML list. Use it to (re-)generate
  the `ref_clip_paths` block.

---

## Phase 2 — optional (D, ~60 LOC)

### D. BC regularization in PPO actor loss

**Why**: imitation reward is a per-step pull, but BC loss is a direct
gradient on the policy distribution. When PPO exploration drifts off the
demo manifold, BC loss yanks it back. Useful when imitation reward gets
overwhelmed by task reward.

**Files**

- `rl/alg/ppo.py`:
  - Accept an `expert_buffer` (obs, action_raw) pairs from
    `data/datasets/walk_v27_bc_clean/train.npz`.
  - In `update()`, sample `bc_batch_size` expert pairs per minibatch:
    ```python
    expert_obs, expert_act = expert_buffer.sample(bc_batch_size)
    pred = self.actor.mean(expert_obs)
    bc_loss = F.mse_loss(pred, expert_act)
    actor_loss = ppo_loss + self.bc_lambda * bc_loss
    ```
  - Anneal `bc_lambda` symmetrically with `w_imit`.

- `train.py`: load `data/datasets/walk_v27_bc_clean/train.npz` if config
  asks; build a circular buffer / dataset; pass to PPO.

- New config fields:
  ```yaml
  ppo:
    bc_dataset_path: data/datasets/walk_v27_bc_clean/train.npz
    bc_lambda_start: 1.0
    bc_lambda_end:   0.0
    bc_decay_until:  5000
    bc_batch_size:   2048
  ```

**Risk**: BC and PPO gradients can fight (different objectives at the same
parameters). Watch `actor_loss` and `bc_loss` separately on TB; if BC
dominates, lower `bc_lambda_start`.

---

## Implementation order + branching

Strict sequential — each phase trains for ≥1k iters before moving on.

```
[branch: mirl_phase1a]
  A: full RSI
  → spot-check: train walk_v29_imit 500 iters, plot reset state distribution
[merge to main once verified]

[branch: mirl_phase1bc]
  B: cmd-matched clip
  C: w_imit annealing
  → train walk_v29_imit 8000 iters end-to-end
[merge once it beats walk_v27 on yaw_err]

[optional: mirl_phase2]
  D: BC reg in PPO
  → train walk_v29_imit_bc 8000 iters
```

---

## Test plan

| Test | Pass criterion |
|---|---|
| `tests/test_load_ref_clips.py` | Loads walk_v27 trace, retrieves correct base_quat/base_lin_vel at frame i |
| `tests/test_rsi_state.py` | After reset_dofs, joint_pos matches selected clip frame within 1e-5 |
| `tests/test_cmd_matched_clip.py` | env with cmd=(0,0,0.5) gets pure_yaw clip, not pure_fwd |
| `tests/test_w_imit_anneal.py` | At iter 0: w_imit≈0.8; at iter 5000: w_imit≈0.1; at iter 7000: w_imit=0.1 |
| Smoke run (200 iters) | No NaN / crash. TB shows `task/w_imit` decaying. |
| Full run vs walk_v27 baseline | yaw_err improves (target: < 0.20 vs walk_v27's 0.238) **and** vx_fwd ≤ 0.10 |

---

## Success criteria — when do we ship?

`walk_v29_imit` should beat walk_v27 best on at least 2 of 3:
- vx_fwd_err < 0.07 (walk_v27 best 0.058 — match)
- vx_bwd_err < 0.10 (walk_v27 best 0.070)
- yaw_err < 0.20 (walk_v27 best 0.238 — clear win)

If yaw doesn't drop below 0.20 with Phase 1 alone, add Phase 2 (BC reg).
If still no gain, the trace dataset itself is the bottleneck — collect
fresh traces from a yaw-stronger ckpt (e.g. once we have one).

---

## What this plan does NOT do

- Doesn't change BC dataset format — `data/datasets/walk_v27_bc_clean/`
  is reused as-is for both Phase 1 (ref_clip_paths) and Phase 2 (BC reg).
- Doesn't fix walk_v27's intrinsic yaw drift via reward redesign — that
  belongs to Tasks #11 (yaw_motion_proxy_scale) / #12 (per-term regulation).
- Doesn't touch sim2sim deploy path — MIRL config produces a normal
  walk_v27-shaped policy (43-dim obs, 10-dim action), backwards-compatible
  with the existing manifest pipeline.

---

## Estimated effort

| Phase | LOC | Code time | Train time | Total wall-clock |
|---|---|---|---|---|
| 1A | ~30 | 1 hr | 0.5 hr smoke | 1.5 hr |
| 1B+1C | ~45 | 1.5 hr | 5 hr full run | 6.5 hr |
| 2 (optional) | ~60 | 2.5 hr | 5 hr | 7.5 hr |
| **Total Phase 1** | **~75** | **~2.5 hr** | **5.5 hr** | **~8 hr** |
| Total Phase 1+2 | ~135 | ~5 hr | ~10 hr | ~15 hr |
