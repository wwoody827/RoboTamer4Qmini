import os
import copy
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch



def set_seed(seed):
    if seed is None:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, sim_cfg=None):
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if sim_cfg is not None:
        gymutil.parse_sim_config(sim_cfg, sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params



def get_args():
    custom_parameters = [
        {"name": "--name", "type": str, "action": "store_true", "default": 'test',
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--tcn", "type": str, "action": "store_true", "default": None,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--encoder", "type": str, "action": "store_true", "default": None,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--dynamics", "type": str, "action": "store_true", "default": None,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--residuals", "type": str, "action": "store_true", "default": None,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--config", "type": str, "action": "store_true", "default": 'configs/birl_fwd.yaml',
         "help": "Path to YAML config file."},
        {"name": "--resume", "type": str, "action": "store_true", "default": None,
         "help": "Resume training from a checkpoint"},
        {"name": "--render", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--fix_cam", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--cmp_real", "action": "store_true", "default": False,
         "help": "Plot curves compared to the real robot"},
        {"name": "--plt_sim", "action": "store_true", "default": False,
         "help": "Plot curves compared to the real robot"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
         "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": None,
         "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int,
         "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--video", "action": "store_true", "default": False, "help": "record your display"},
        {"name": "--time", "type": float, "default": 10, "help": "display time(seconds)."},
        {"name": "--iter", "type": int, "default": None, "help": "display epoch times."},
        {"name": "--epochs", "type": int, "default": 1, "help": "display epoch times."},
        {"name": "--debug", "action": "store_true", "default": False, "help": "save data to excel"},
        {"name": "--mode", "type": str, "action": "store_true", "default": 'reset',
         "help": "Test mode. Overrides config file if provided."},
        {"name": "--cmd_vx", "type": float, "default": None, "help": "Fix forward velocity command (m/s)."},
        {"name": "--cmd_yaw", "type": float, "default": None, "help": "Fix yaw rate command (rad/s)."},
        {"name": "--out", "type": str, "default": None, "help": "Output video filename (without extension)."},
        {"name": "--sim2sim_config", "type": str, "default": None,
         "help": "Path to sim2sim YAML config for periodic evaluation (default: auto-detect qmini_birl.yaml)"},
        {"name": "--sim2sim_interval", "type": int, "default": 0,
         "help": "Run sim2sim eval every N training iterations (0 = disabled)"},

    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
