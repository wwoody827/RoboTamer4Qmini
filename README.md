# RoboTamer4Qmini_v1.0

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.8%2B-green.svg)
[![Powered by Isaac Gym](https://img.shields.io/badge/Powered%20by-Isaac%20Gym-blue.svg)](https://developer.nvidia.com/isaac-gym)
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-green.svg)](https://arxiv.org/abs/1707.06347)
![Version](https://img.shields.io/badge/Version-1.0-blue.svg)  

**This is currently Version 1.0.**  
✅ Initial release complete.  
🚀 Future updates are planned to add new features and optimize performance.  
Stay tuned for changelogs!

This repository offers an open-source framework for bipedal robot motion control, leveraging deep reinforcement learning within NVIDIA's Isaac Gym environment. It enables training robots like Unitree Qmini to walk on rough terrain and includes critical domain randomization and random pushes during training for sim-to-real transfer. The repository contains the complete codebase for both training and deploying the biped robots in simulation environments.

**Maintainer**: Yanyun Chen, Tiyu Fang, Wenhao Tan, Xing Fang, Kaiwen Li, Kunqi Zhang, and Wei zhang<br>
**Affiliation**: Visual Sensing and Intelligent System Lab (VSISLab),
School of Control Science and Engineering,
Shandong University, China<br>
**Website**: www.vsislab.com  
**Contact**: info@vsislab.com

## Features
- **Advanced Control Algorithms**: Implements PPO algorithm using Isaac Gym for stable and efficient locomotion control.<br>
- **Modular Design**: Easy-to-use API for customizing robot models, environments, and reward functions.<br>
- **Real-world Deployment**: Includes tools for transferring learned policies from simulation to physical robots.<br>
- **Comprehensive Documentation**: Detailed tutorials and documentation for quick start and customization.<br>

## Code Structure
   ```
RoboTamer4Qmini/
   ├── assets/                 # URDF model of the robot
   ├── configs/                # YAML configuration files (with `_base:` inheritance)
   ├── config/                 # Config loader (loads configs/*.yaml)
   ├── env/                    # Simulation environments + unified BIRLTask
   ├── experiments/            # Trained models and evaluation results
   ├── model/                  # Neural network architectures
   ├── rl/                     # PPO algorithm + runner
   ├── deploy/                 # ONNX export + MuJoCo sim2sim
   ├── utils/                  # Utility functions (mirror aug, etc.)
   ├── docs/rewards.md         # Reward term reference
   ├── export_pt2onnx.py       # Export `.pt` checkpoint to `.onnx`
   ├── play.py                 # Evaluate a trained policy in Isaac Gym
   ├── train.py                # Train a policy
   ├── requirements.txt        # Python dependencies
   └── README.md
   ```

### Notes
* This repository is not maintained anymore. If you have any question, please send emails to info@vsislab.com.
* The project can only be run after successful installation.


## Installation

### Prerequisites
- Ubuntu 18.04 or 20.04.
- NVIDIA driver version 470+
- Hardware: NVIDIA Pascal or later GPU with at least 8 gb of VRAM
- Cuda 11.4+ 
- Python 3.8+
- PyTorch 2.0.0+
- Isaac Gym 1.0rc3+ (for simulation environments)
- Additional dependencies (see `requirements.txt` and `Install dependencies`)

### Steps
1. Create a new conda environment:
```bash
$  conda create -n isaac python==3.8 && conda activate isaac
```

2. Install dependencies:
```bash
    pip3 install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0
    tar -zxvf IsaacGym_Preview_3_Package.tar.gz && cd ./isaacgym/python && pip install -e . 
    pip3 install requirements.txt
    pip3 install matplotlib pandas tensorboard opencv-python numpy==1.23.5 openpyxl onnxruntime onnx
 ```

## Usage:

### **To train (default:test):**
```bash
$ python train.py --config configs/birl_fwd.yaml --name <name>
```
Other configs: `configs/bdx.yaml` (BD_X style, external phase clock) or `configs/mirl_fwd.yaml` (MIRL, no phase).

  - --name <str> # Experiment name (Default: 'test'), overrides settings in the config file
  - --config <str> # Path to YAML config (Default: 'configs/birl_fwd.yaml'), supports `_base:` inheritance
  - --resume <str> # Resume training from checkpoint (Default: None), requires specifying experiment 'name'
  - --render # Boolean flag (Default: False), force display off at all times
  - --fix_cam # Boolean flag (Default: False), fix camera view on the robot in environment 0
  - --rl_device <str> # RL device (Default: 'cuda:0'), supports formats like 'cpu'/'cuda:0'
  - --num_envs <int> # Number of environments (Default: None), overrides config file settings
  - --seed <int> # Random seed (Default: None), overrides config file settings
  - --max_iterations <int> # Maximum number of iterations (Default: None), overrides config file settings
  - --sim2sim_interval <int> # Run MuJoCo sim2sim eval every N training iterations (Default: 0, disabled)

### **To visualize the training logs in a browser:**
```bash
$ tensorboard --logdir experiments/ 
```    

### **To play (default:test):**
```bash
$ python play.py --render --name <name>
```
#### **To open viewer when training or playing (default:False):**
```bash
$ python play.py --name <name> --render
```
#### **To change display time when playing (default:4s)**
```bash
$ python play.py --name <name> --render --time 10
```
#### **To record video when playing (default:False)**
```bash
$ python play.py --name <name> --render --time 10 --video
```
#### **To save data in Excel when playing (default:False)**
```bash
$ python play.py --name <name> --render --time 10 --video --debug
```
  - --name <str> # Experiment name (Default: 'test'), overrides settings in the config file
  - --render # Boolean flag (Default: False), force display off at all times
  - --fix_cam # Boolean flag (Default: False), fix camera view on the robot in environment 0
  - --cmp_real # Boolean flag (Default: False), plot curves compared to the real robot
  - --plt_sim # Boolean flag (Default: False), plot curves in the simulation environment
  - --num_envs <int> # Number of environments (Default: None), overrides config file settings
  - --video # Boolean flag (Default: False), record display as video
  - --time <float> # Evaluation duration (seconds, Default: 10s)
  - --iter <int> # Specify pre-trained policy by training iteration (Default: None, loads the latest policy in the current directory)
  - --epochs <int> # Number of evaluation epochs (Default: 1)
  - --debug # Boolean flag (Default: False), save data to Excel

### **To export the pre-trained models to onnx models (default:test):**
```bash
$ python export_pt2onnx.py --name <name>
```
  - --name <str> # Experiment name (Default: 'test')，the policy.onnx is saved to directory 'name/deploy'

### **To load a urdf model:**
```bash
$ python tune_urdf.py
```
### **To optimize PID parameters to minimize the discrepancy between simulation and real-world robot behavior:**
```bash
$ python tune_pid.py
```
  - --mode <str> # Test mode: {'sin,', 'real,', 'reset,'}. Select test mode (simulation, real-world, or reset).

## Contributing

We welcome contributions from the community! Please contact us at info@vsislab.com before submitting pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License —— see the [LICENSE] file —— for details.

## Citation

If you use this code in your research, please cite our work:
```
@article{Chen2025GALA,
  author={Yanyun Chen, Ran Song, Jiapeng Sheng, Xing Fang, Wenhao Tan, Wei Zhang and Yibin Li},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={A Generalist Agent Learning Architecture for Versatile Quadruped Locomotion}, 
  year={2025},
  keywords={Quadruped Robots, Versatile Locomotion, Deep Reinforcement Learning, A Single Policy Network, Multiple Critic Networks}
}

@article{Sheng2022BioInspiredRL,
  title={Bio-Inspired Rhythmic Locomotion for Quadruped Robots},
  author={Jiapeng Sheng and Yanyun Chen and Xing Fang and Wei Zhang and Ran Song and Yuan-hua Zheng and Yibin Li},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  volume={7},
  pages={6782-6789}
}

@article{Liu2024MCLER,
  author={Liu, Maoqi and Chen, Yanyun and Song, Ran and Qian, Longyue and Fang, Xing and Tan, Wenhao and Li, Yibin and Zhang, Wei},
  journal={IEEE Robotics and Automation Letters}, 
  title={MCLER: Multi-Critic Continual Learning With Experience Replay for Quadruped Gait Generation}, 
  year={2024},
  volume={9},
  number={9},
  pages={8138-8145},
  keywords={Quadrupedal robots;Task analysis;Continuing education;Optimization;Legged locomotion;Training;Motors;Continual learning;legged robots},
  doi={10.1109/LRA.2024.3418310}
}

```








