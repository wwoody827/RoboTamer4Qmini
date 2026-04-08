# Project Setup

## Conda Environment

This project uses the `qmini` conda environment (Python 3.8).

> **Note:** Python 3.8 is required. Isaac Gym's compiled bindings are built against `libpython3.8.so.1.0` and will not work with newer Python versions.

### Create and activate

```bash
conda create -n qmini python=3.8 -y
conda activate qmini
```

---

## Isaac Gym

Isaac Gym Preview 4 must be downloaded manually from [developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) (requires NVIDIA developer account). Place it anywhere — recommended: `~/code/isaacgym/`.

```bash
tar -zxvf IsaacGym_Preview_4_Package.tar.gz -C ~/code/
cd ~/code/isaacgym/isaacgym/python
pip install -e .
```

### Fix: `np.float` removed in NumPy 1.24+

Edit `~/code/isaacgym/isaacgym/python/isaacgym/torch_utils.py` line 135:

```python
# Change:
def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
# To:
def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
```

### Fix: `libpython3.8.so.1.0` not found

The conda env's lib directory must be on the library path. This is set automatically on `conda activate` via:

```bash
mkdir -p ~/miniconda3/envs/qmini/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=~/miniconda3/envs/qmini/lib:$LD_LIBRARY_PATH' \
  > ~/miniconda3/envs/qmini/etc/conda/activate.d/env_vars.sh
```

---

## Python Dependencies

```bash
pip install torch==2.4.1 torchvision torchaudio
pip install -r requirements.txt
pip install opencv-python pandas tensorboard matplotlib onnxruntime onnx openpyxl
```

> `requirements.txt` pins torch 2.0.0 + CUDA 11, but torch 2.4.1 + CUDA 12 works fine.

---

## Running

### Activate environment

```bash
conda activate qmini
# LD_LIBRARY_PATH is set automatically by the activate hook above
```

### Train

```bash
cd ~/code/RoboTamer4Qmini
python train.py --config BIRL --name <run_name>
```

### Play (evaluate pre-trained weights)

```bash
cd ~/code/RoboTamer4Qmini
python play.py --name q2 --render --cmd_vx 0.5 --cmd_yaw 0.0
```

| Argument | Default | Description |
|---|---|---|
| `--name` | `test` | Experiment name (loads from `experiments/<name>/`) |
| `--render` | False | Show Isaac Gym viewer |
| `--cmd_vx` | None (free) | Fix forward velocity command (m/s) |
| `--cmd_yaw` | None (free) | Fix yaw rate command (rad/s) |
| `--time` | 10s | Evaluation duration |
| `--video` | False | Record video |

### TensorBoard

```bash
cd ~/code/RoboTamer4Qmini
python -m tensorboard.main --logdir experiments/
```

Then open [http://localhost:6006](http://localhost:6006).

> If you see `TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`, fix with:
> ```bash
> pip install protobuf==3.20.3
> ```

---

## Visualizer

[visualizer.py](visualizer.py) receives live UDP broadcasts from the robot and displays real-time plots.

### Usage

```bash
python3 visualizer.py [--port 9870] [--history 200]
```

| Argument | Default | Description |
|---|---|---|
| `--port` | `9870` | UDP port to listen on |
| `--history` | `200` | Frames to keep (200 = ~10s at 20Hz) |

### Requirements

- Robot machine must be on the same LAN subnet
- Robot must be running `run_interface` in mode 1, 2, 3, or 5
- Firewall must allow inbound UDP on the chosen port

### Troubleshooting

| Problem | Fix |
|---|---|
| No data received | Check same subnet; check firewall allows UDP 9870 |
| `TkAgg` backend error | `sudo apt install python3-tk` or change `matplotlib.use('TkAgg')` to `'Qt5Agg'` |
| Plots lag | Reduce `--history` (e.g. `--history 50`) |
