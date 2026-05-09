# `data/` — traces and datasets

Demonstration data harvested from trained policies in MuJoCo sim2sim, plus
BC/SFT-ready datasets built from those traces. **All `.npz` blobs are
gitignored** — only the recipe files (`dataset.yaml`, `manifest.csv`,
`episodes.csv`) are tracked, and any blob is regenerable from those plus
the policy checkpoint.

---

## Layout

```
data/
├── README.md              ← this file
├── reference_clips/       ← legacy MIRL-format clips (state-only)
│   └── walk_fwd.npz
├── *_init_states.npz      ← legacy reset-pose pools (RSI / balance)
│
├── traces/                ← per-policy sim2sim rollouts (extended format)
│   └── walk_v27_multi/
│       ├── 3600/
│       │   ├── <cmd>.npz           gitignored
│       │   ├── manifest.csv        per-trace metrics + cmd
│       │   └── dataset.yaml        recipe (policy sha256, git commit, args)
│       ├── 6200/  ...
│       ├── 6800/  ...
│       ├── 7800/  ...
│       └── 8000/  ...
│
└── datasets/              ← filtered + relabeled BC training sets
    └── walk_v27_bc_clean/
        ├── train.npz       gitignored — concatenated obs/action pairs
        ├── episodes.csv    per-episode provenance + class + drift
        └── dataset.yaml    recipe (build args, filter counts, meta)
```

---

## Trace format (`traces/<run>/<ckpt>/<cmd>.npz`)

Recorded by `deploy/sim2sim/sim2sim.py --record`. Per-step at policy rate
(default 67 Hz, `dt = 0.015 s`):

| Key | Shape | Notes |
|-----|-------|-------|
| `joint_pos`, `joint_vel`     | `[T, 10]` | actual robot state |
| `base_pos`, `base_quat`      | `[T, 3]`, `[T, 4]` (`[w,x,y,z]`) | world-frame pose |
| `base_lin_vel`, `base_ang_vel` | `[T, 3]` each | world / body frame |
| `cmd`                          | `[T, 3]`  | command per step (constant within trace) |
| `obs`                          | `[T, obs_dim]` | **single-frame** obs — re-stack downstream |
| `action_raw`                   | `[T, action_dim]` | network output ∈ `[-1, 1]` — BC target |
| `action_scaled`                | `[T, action_dim]` | physical units after `scale_transform` |
| `joint_target`                 | `[T, 10]` | actual PD target sent to MuJoCo |
| `torque`                       | `[T, 10]` | snapshot at policy step |
| `static_flag`                  | `[T]`     | 1 if `‖cmd‖ ≥ 0.15` else 0 |
| `phase_clock`                  | `[T, 4]`  | `[sinL, sinR, cosL, cosR]` × static_flag |
| `cmd_freq_step`                | `[T]`     | BD_X external phase frequency |

Plus 0-dim metadata: `cmd_const`, `cmd_freq`, `policy_path`, `meta_phase_mode`,
`meta_action_mode`, `meta_action_dim`, `meta_obs_dim`, **`meta_obs_history`**,
**`meta_obs_skip`**, `meta_lp_alpha`, `meta_num_legs`, `meta_static_thr`.

Episode-level metrics: `metric_mean_vx_err`, `metric_mean_vy_err`,
`metric_mean_yaw_err`, `metric_mean_height`, `metric_min_height`,
`metric_max_tilt_deg`, `metric_survival_time`, `metric_terminated`
(`'completed'` or `'fell'`). Recording is auto-truncated at fall.

---

## BC dataset format (`datasets/<name>/train.npz`)

Concatenated frames from filtered traces. Episode boundaries via
`ep_starts`, `ep_ends`. The `obs` field has its **cmd slot rewritten to the
relabeled cmd** (`cmd_axes` mode by default) so obs ↔ action remains
self-consistent.

```python
import numpy as np
d = np.load('data/datasets/walk_v27_bc_clean/train.npz')
obs        = d['obs']         # [N, 43] — cmd slot already relabeled
action_raw = d['action_raw']  # [N, 10] — BC target ∈ [-1, 1]
starts, ends = d['ep_starts'], d['ep_ends']
# For frame-stacking (per dataset.yaml meta_obs_history / meta_obs_skip):
# stack obs[i - k*skip] for k in range(history) within (start, end) bounds.
```

Per-episode metadata in `episodes.csv` (cmd_orig, cmd_relabel,
mean_realized, drift_vx/vy/yaw, class, ckpt) lets you filter further at
training time.

---

## Reproduce a trace set

```bash
# Single ckpt (31 traces × 12 s ≈ 5 min, ~12 MB):
python scripts/collect_traces.py \
    --policy experiments/<run>/deploy/policy_<iter>.onnx \
    --out data/traces/<run>_<iter> \
    --duration 12

# Multi-ckpt sweep (5 ckpts × 31 cmds ≈ 25 min, ~60 MB):
python scripts/collect_traces.py \
    --policies experiments/<run>/deploy/policy_{3600,6200,6800,7800,8000}.onnx \
    --out data/traces/<run>_multi \
    --duration 12
```

`collect_traces.py` writes `dataset.yaml` containing `policy_sha256`,
`git_commit`, and `args` — sweeps under the same SHA + same args + same
policy checksum reproduce byte-for-byte (sim2sim is deterministic given
fixed cmd, no DR).

---

## Reproduce a BC dataset

```bash
python scripts/build_bc_dataset.py \
    --input  data/traces/walk_v27_multi \
    --output data/datasets/walk_v27_bc_clean \
    --include clean \
    --relabel cmd_axes
```

### Trace classification

For each trace, `build_bc_dataset.py` computes mean **body-frame realized**
velocity over the middle 80 % of the trace and classifies:

| Class | Definition | Default action |
|---|---|---|
| `failure` | `metric_terminated == 'fell'` | filter out |
| `drift`   | uncommanded axis (`\|cmd[i]\| < 0.05`) has `\|mean_realized[i]\|` exceeding threshold | filter out |
| `clean`   | otherwise — recoverable: cmd matches realized direction | **keep** |

Drift thresholds: `vx > 0.05 m/s`, `vy > 0.20 m/s` (gait swaying baseline
~0.15 makes vy noisier), `yaw > 0.20 rad/s`.

### Cmd relabel modes (`--relabel`)

| Mode | What changes |
|---|---|
| `none`     | keep `cmd_const` exactly (use for evaluation, not BC) |
| `cmd_axes` | replace commanded axes with mean realized; uncommanded axes stay 0 (default) |
| `all`      | replace all 3 axes with mean realized (incl. uncmd drift — be careful) |

Whichever mode is chosen, **`obs[:, 0:3]` is rewritten to match** so BC
input/output stays consistent.

---

## Versioning approach

Traces are deterministic from `(policy.onnx, sweep args, sim2sim code)`,
so we version the **recipe**, not the artifact:

- `dataset.yaml` carries `policy_sha256` + `git_commit` (+ `git_dirty` flag)
- `manifest.csv` / `episodes.csv` give per-trace summaries for filtering
- `.npz` blobs are gitignored — regenerate on demand

If the dataset later needs to be shared across machines or grows past a
few hundred MB, graduate to **DVC** (local SSH remote is a 5-line config)
or **Git LFS**. For now, recipe-only is the cheapest path.
