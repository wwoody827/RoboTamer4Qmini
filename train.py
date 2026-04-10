import importlib
import os
import tempfile
from os.path import join

from env.utils import get_args
from env.utils.helpers import update_cfg_from_args, class_to_dict, set_seed, parse_sim_params
from env import LeggedRobotEnv, GymEnvWrapper
from env.tasks import load_task_cls
from model import load_actor, load_critic
from rl.alg import PPO
import time
from collections import deque
import collections
import statistics
from utils.common import clear_dir
from utils.yaml import ParamsProcess
from isaacgym.torch_utils import *
from torch.utils.tensorboard import SummaryWriter
import torch
import yaml

# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train():
    torch.cuda.empty_cache()
    args = get_args()
    device = args.rl_device
    cfg = getattr(importlib.import_module('.'.join(['config', args.config])), args.config)
    cfg = update_cfg_from_args(cfg, args)
    cfg.runner.num_envs = args.num_envs if args.num_envs is not None else cfg.runner.num_envs
    exp_dir = join('experiments', args.name)
    model_dir = join(exp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    all_model_dir = join(exp_dir, 'model', 'all')
    os.makedirs(all_model_dir, exist_ok=True)
    log_dir = join(exp_dir, 'log')
    clear_dir(log_dir)
    writer = SummaryWriter(log_dir, flush_secs=10)
    num_steps_per_env = cfg.runner.num_steps_per_env
    num_learning_iterations = cfg.runner.max_iterations
    set_seed(seed=None)

    sim_params = parse_sim_params(args, class_to_dict(cfg.sim))
    env = LeggedRobotEnv(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         render=args.render,
                         fix_cam=args.fix_cam)
    task = load_task_cls(cfg.task.cfg)(env)
    gym_env = GymEnvWrapper(env, task)
    task.num_observations = len(gym_env.task.pure_observation()[0]) * gym_env.task.obs_history.maxlen
    task.num_actions = len(gym_env.task.action_low)

    cfg_dict = collections.OrderedDict()
    paramProcess = ParamsProcess()
    cfg_dict.update(paramProcess.class2dict(cfg))
    cfg_dict['policy'].update({'num_observations': task.num_observations, 'num_actions': task.num_actions,
                               'num_critic_obs': len(gym_env.task.critic_observation()[0])})
    cfg_dict['action'].update({'action_limit_low': env.dof_pos_limits[:, 0].cpu().numpy(), 'action_limit_up': env.dof_pos_limits[:, 1].cpu().numpy()})
    cfg_dict['action'].update({'action_scale_low': cfg.action.low_ranges[2:], 'action_scale_up': cfg.action.high_ranges[2:]})

    paramProcess.write_param(join(model_dir, "cfg.yaml"), cfg_dict)

    # ── sim2sim eval setup ────────────────────────────────────────────────────
    sim2sim_cfg = None
    sim2sim_interval = getattr(args, 'sim2sim_interval', 0)
    if sim2sim_interval > 0:
        sim2sim_config_path = getattr(args, 'sim2sim_config', None)
        if sim2sim_config_path is None:
            sim2sim_config_path = 'deploy/sim2sim/configs/qmini_birl.yaml'
        if os.path.exists(sim2sim_config_path):
            with open(sim2sim_config_path, encoding='utf-8') as _f:
                sim2sim_cfg = yaml.safe_load(_f)
            # add sim2sim helpers to path
            import sys as _sys
            _sim2sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deploy', 'sim2sim')
            if _sim2sim_dir not in _sys.path:
                _sys.path.insert(0, _sim2sim_dir)
            from evaluate import quick_eval as _quick_eval
            print(f'[sim2sim] eval enabled every {sim2sim_interval} iters using {sim2sim_config_path}')
        else:
            print(f'[sim2sim] config not found: {sim2sim_config_path} — sim2sim eval disabled')
            sim2sim_interval = 0
    # ─────────────────────────────────────────────────────────────────────────

    actor = load_actor(cfg_dict['policy'], device).train()
    critic = load_critic(cfg_dict['policy'], device).train()


    alg = PPO(actor, critic, device=device, **class_to_dict(cfg.algorithm))
    alg.init_storage(cfg.runner.num_envs, num_steps_per_env, [len(gym_env.task.critic_observation()[0])],
                     [task.num_observations], [task.num_actions])
    if args.resume is not None:
        resume_model_dir = join(join('experiments', args.resume), 'model')
        saved_model_state_dict = torch.load(join(resume_model_dir, 'policy.pt'))
        alg.actor.load_state_dict(saved_model_state_dict['actor'])
        alg.critic.load_state_dict(saved_model_state_dict['critic'])
        alg.optimizer.load_state_dict(saved_model_state_dict['optimizer'])
        current_learning_iteration = saved_model_state_dict['iteration']
    else:
        current_learning_iteration = 1

    total_time, total_timesteps = 0., 0
    total_iteration = current_learning_iteration + num_learning_iterations
    rew_buffer, len_buffer,task_rew_buffer = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    cur_reward_sum = torch.zeros(cfg.runner.num_envs, dtype=torch.float, device=device)
    cur_task_rew_sum = torch.zeros(cfg.runner.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(cfg.runner.num_envs, dtype=torch.float, device=device)

    obs, cri_obs = gym_env.reset(torch.arange(cfg.runner.num_envs, device=device))
    rew_component_acc = None
    rew_component_steps = 0
    for it in range(current_learning_iteration, total_iteration):

        env.update_friction_curriculum(it)
        start = time.time()
        rew_component_acc = None
        rew_component_steps = 0
        for i in range(num_steps_per_env):
            act = alg.act(obs, cri_obs)
            obs, cri_obs, rew, done, info, eval_rew = gym_env.step(act,it)
            alg.process_env_step(rew, done, info)
            cur_reward_sum += rew
            cur_task_rew_sum+=eval_rew
            cur_episode_length += 1
            if hasattr(gym_env.task, '_last_rew_components') and gym_env.task._last_rew_components is not None:
                step_mean = gym_env.task._last_rew_components.mean(dim=0).detach()
                if rew_component_acc is None:
                    rew_component_acc = step_mean
                else:
                    rew_component_acc += step_mean
                rew_component_steps += 1
            reset_env_ids = (done > 0).nonzero(as_tuple=False)[:, [0]].flatten()
            if len(reset_env_ids) > 0:
                rew_buffer.extend(cur_reward_sum[reset_env_ids].cpu().numpy().tolist())
                task_rew_buffer.extend(cur_task_rew_sum[reset_env_ids].cpu().numpy().tolist())
                len_buffer.extend(cur_episode_length[reset_env_ids].cpu().numpy().tolist())
                cur_reward_sum[reset_env_ids] = 0
                cur_task_rew_sum[reset_env_ids] = 0
                cur_episode_length[reset_env_ids] = 0
        alg.compute_returns(cri_obs)
        stop = time.time()
        collection_time = stop - start
        start = stop
        mean_value_loss, mean_surrogate_loss, mean_kl = alg.update()
        saved_model_state_dict = {
            'actor': alg.actor.state_dict(),
            'critic': alg.critic.state_dict(),
            'optimizer': alg.optimizer.state_dict(),
            'iteration': current_learning_iteration,
        }
        try:
            torch.save(saved_model_state_dict, join(model_dir, 'policy.pt'))
        except OSError as e:
            print('Failed to save policy.')
            print(e)
        if it % cfg.runner.save_interval == 0:
            try:
                torch.save(saved_model_state_dict, join(all_model_dir, f'policy_{it}.pt'))
            except OSError as e:
                print('Failed to save policy.')
                print(e)

        # ── sim2sim eval ──────────────────────────────────────────────────────
        if sim2sim_interval > 0 and it % sim2sim_interval == 0 and sim2sim_cfg is not None:
            try:
                _t0 = time.time()
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as _tf:
                    _onnx_path = _tf.name
                alg.actor.eval()
                _dummy = torch.zeros(1, task.num_observations, device='cpu')
                torch.onnx.export(alg.actor.cpu(), _dummy, _onnx_path,
                                  opset_version=12, input_names=['input'], output_names=['output'],
                                  verbose=False)
                alg.actor.to(device).train()
                _metrics = _quick_eval(_onnx_path, sim2sim_cfg)
                os.unlink(_onnx_path)
                for k, v in _metrics.items():
                    if not (isinstance(v, float) and (v != v)):  # skip nan
                        writer.add_scalar(f'5:{k}', v, it)
                _elapsed = time.time() - _t0
                _surv_str = '  '.join(
                    f'fr{k.split("fr")[1]}={v:.1f}s' for k, v in _metrics.items()
                    if 'survive_time' in k
                )
                print(f'[sim2sim@{it}] {_surv_str}  '
                      f'vx_err fwd={_metrics.get("sim2sim/vx_err_fwd", float("nan")):.3f} '
                      f'bwd={_metrics.get("sim2sim/vx_err_bwd", float("nan")):.3f}  '
                      f'({_elapsed:.0f}s)')
            except Exception as _e:
                print(f'[sim2sim] eval failed at iter {it}: {_e}')
        # ─────────────────────────────────────────────────────────────────────

        stop = time.time()
        learn_time = stop - start
        iteration_time = collection_time + learn_time
        total_time += iteration_time
        total_timesteps += num_steps_per_env * cfg.runner.num_envs
        fps = int(num_steps_per_env * cfg.runner.num_envs / iteration_time)
        mean_std = alg.actor.std.mean()
        mean_reward = statistics.mean(rew_buffer) if len(rew_buffer) > 0 else 0.
        mean_task_reward = statistics.mean(task_rew_buffer) if len(task_rew_buffer) > 0 else 0.

        mean_episode_length = statistics.mean(len_buffer) if len(len_buffer) > 0 else 0.
        writer.add_scalar('1:Train/mean_reward', mean_reward, it)
        writer.add_scalar('1:Train/mean_task_reward', mean_task_reward, it)
        writer.add_scalar('1:Train/mean_episode_length', mean_episode_length, it)
        writer.add_scalar('1:Train/mean_episode_time', mean_episode_length * gym_env.env.dt, it)

        writer.add_scalar('2:Loss/value', mean_value_loss, it)
        writer.add_scalar('2:Loss/surrogate', mean_surrogate_loss, it)
        writer.add_scalar('2:Loss/learning_rate', alg.learning_rate, it)
        writer.add_scalar('2:Loss/mean_kl', mean_kl, it)
        writer.add_scalar('2:Loss/mean_noise_std', mean_std.item(), it)

        writer.add_scalar('3:Perf/total_fps', fps, it)
        writer.add_scalar('3:Perf/collection_time', collection_time, it)
        writer.add_scalar('3:Perf/learning_time', learn_time, it)

        if rew_component_acc is not None and rew_component_steps > 0 and gym_env.task.rew_names is not None:
            rew_component_mean = rew_component_acc / rew_component_steps
            for name, val in zip(gym_env.task.rew_names, rew_component_mean.cpu().tolist()):
                writer.add_scalar(f'4:Rewards/{name}', val, it)

        print(f"{args.name}#{it}:",
              f"{'t'} {total_time / 60:.1f}m({iteration_time:.1f}s)",
              f"col {collection_time:.2f}s",
              f"lnt {learn_time:.2f}s",
              f"nm {fps:.0f}",
              f"m_kl {mean_kl:.3f}",
              f"{'v_lss:'} {mean_value_loss:.3f}",
              f"{'a_lss:'} {mean_surrogate_loss:.3f}",
              # f"l_t {mean_episode_length * gym_env.env.dt:.2f}s",
              f"l_n {int(mean_episode_length)}",
              f"total_rew {mean_reward:.2f}",
              f"task_rew {mean_task_reward:.2f}",
              sep='  ')


if __name__ == '__main__':
    train()
