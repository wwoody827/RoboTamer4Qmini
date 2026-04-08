# Training & Visualization Tutorial

## 1. Activate the environment

```bash
conda activate qmini
cd ~/code/RoboTamer4Qmini
```

---

## 2. Start training

### New run

```bash
python train.py --config BIRL --name my_run
```

### Resume from a checkpoint

```bash
python train.py --config BIRL --name my_run --resume my_run
```

### Useful training flags

| Flag | Default | Description |
|---|---|---|
| `--name` | `test` | Experiment name. Saves to `experiments/<name>/` |
| `--config` | `config.Base` | Config file to use (`BIRL` or `Base`) |
| `--resume` | None | Resume from `experiments/<name>/model/policy.pt` |
| `--num_envs` | from config | Override number of parallel environments |
| `--max_iterations` | from config | Override max training iterations |
| `--render` | False | Show Isaac Gym viewer during training |
| `--fix_cam` | False | Lock camera on robot 0 |
| `--rl_device` | `cuda:0` | Device to run on (`cuda:0`, `cpu`, etc.) |

Models are saved to `experiments/<name>/model/` every `save_interval` iterations (default: 200).

---

## 3. Monitor training with TensorBoard

Open a second terminal:

```bash
conda activate qmini
cd ~/code/RoboTamer4Qmini
python -m tensorboard.main --logdir experiments/
```

Open [http://localhost:6006](http://localhost:6006) in your browser.

### What to look at

| Group | Metric | What it tells you |
|---|---|---|
| `1:Train` | `mean_reward` | Overall reward per episode |
| `1:Train` | `mean_task_reward` | Task-specific reward (velocity tracking) |
| `1:Train` | `mean_episode_time` | How long episodes last (longer = better balance) |
| `2:Loss` | `value` / `surrogate` | Should decrease and stabilize |
| `2:Loss` | `mean_kl` | Should stay near `desired_kl` (0.01) |
| `2:Loss` | `mean_noise_std` | Exploration noise — decreases as policy converges |
| `3:Perf` | `total_fps` | Simulation throughput |
| `4:Rewards` | `fwd_vel`, `balance`, etc. | Per-term reward breakdown |

---

## 4. Visualize with play

### Play the latest checkpoint

```bash
python play.py --name my_run --render
```

### Play with fixed velocity commands

```bash
# Walk forward at 0.5 m/s
python play.py --name my_run --render --cmd_vx 0.5 --cmd_yaw 0.0

# Turn in place
python play.py --name my_run --render --cmd_vx 0.0 --cmd_yaw 0.5

# Walk and turn
python play.py --name my_run --render --cmd_vx 0.4 --cmd_yaw 0.3
```

### Play a specific checkpoint iteration

```bash
python play.py --name my_run --render --iter 1000
```

### Record a video

```bash
python play.py --name my_run --render --time 10 --video --out my_clip
# Output: experiments/my_run/debug/my_clip.mp4
```

### Useful play flags

| Flag | Default | Description |
|---|---|---|
| `--name` | `test` | Experiment to load |
| `--render` | False | Show Isaac Gym viewer |
| `--cmd_vx` | None | Fix forward velocity (m/s). None = random commands |
| `--cmd_yaw` | None | Fix yaw rate (rad/s). None = random commands |
| `--iter` | None | Load checkpoint at this iteration. None = latest |
| `--time` | 10s | How long to run the evaluation |
| `--video` | False | Save viewer as `.mp4` |
| `--out` | `<name>` | Output video filename (no extension) |
| `--fix_cam` | False | Lock camera on robot 0 |
| `--num_envs` | from config | Number of robots to simulate |

---

## 5. Export to ONNX (for deployment)

```bash
python export_pt2onnx.py --name my_run
# Output: experiments/my_run/deploy/policy.onnx
```

---

## Typical workflow

```
train  →  tensorboard (watch rewards converge)  →  play (sanity check)  →  export
```

1. Run `train.py` and open TensorBoard in parallel
2. Watch `mean_task_reward` and `mean_episode_time` — both should rise
3. Once converged (or at a good checkpoint), run `play.py --render` to visually confirm
4. Export with `export_pt2onnx.py` for deployment on the real robot
