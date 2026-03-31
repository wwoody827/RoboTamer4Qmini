# Project Setup

## Conda Environment

This project uses the `qmini` conda environment (Python 3.10).

### Create and activate

```bash
conda create -n qmini python=3.10 -y
conda activate qmini
pip install matplotlib numpy
```

### Or just activate (if already created)

```bash
conda activate qmini
```

---

## Visualizer

[visualizer.py](visualizer.py) receives live UDP broadcasts from the robot and displays real-time plots.

### Usage

```bash
conda activate qmini
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
