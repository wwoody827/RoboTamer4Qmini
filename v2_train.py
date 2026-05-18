"""v2_train.py — minimal trainer for V2Task walk_clean experiments.

Key differences from train.py:
  - NO separate stand-eval rollout — stand metrics computed inline from the
    training rollout itself (training cmd is always 0 for walk_clean_v2, so
    train rollout IS a stand eval). Avoids post-eval reset artifact and the
    GPU stall of a dedicated 9s rollout every 200 iter.
  - Optional sim2sim eval (--sim2sim_interval > 0): exports ONNX + manifest
    every N iter and runs deploy/sim2sim/evaluate.quick_eval. Logs
    `sim2sim/*` to TB.
  - TB metrics: train rewards + per-reward components + `stand/*` per-iter
  - Simpler logging, no MIRL/Recovery hooks

Usage:
    python v2_train.py --config configs/walk_clean_v2.yaml --name v2_run1 \
        --sim2sim_interval 200
"""

import collections
import os
import time
from os.path import join

# IMPORTANT: isaacgym must be imported before torch
import isaacgym  # noqa: F401

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env.utils import get_args
from env.utils.helpers import set_seed, parse_sim_params
from env import LeggedRobotEnv, GymEnvWrapper
from env.tasks import load_task_cls
from rl.policy import load_actor, load_critic
from rl.alg import PPO
from config.loader import load_config, CfgNode, config_to_dict, save_config
from utils.common import clear_dir
from deploy.manifest import build_manifest, save_manifest
from isaacgym.torch_utils import *  # required: imports get_axis_params, to_torch, etc.


CKPT_INTERVAL = 200


def main():
    torch.cuda.empty_cache()
    args = get_args()
    device = args.rl_device

    cfg = load_config(args.config)
    if args.num_envs is not None:
        cfg.runner.num_envs = args.num_envs
    if args.max_iterations is not None:
        cfg.runner.max_iterations = args.max_iterations
    if args.seed is not None:
        cfg.runner.seed = args.seed
    set_seed(seed=cfg.runner.seed)

    # Experiment dir
    exp_name = time.strftime('%Y%m%d_%H%M') + '_' + args.name
    exp_dir = join('experiments', exp_name)
    log_dir = join(exp_dir, 'log')
    model_dir = join(exp_dir, 'model')
    all_dir = join(model_dir, 'all')
    clear_dir(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)
    writer = SummaryWriter(log_dir, flush_secs=10)
    print(f'[v2_train] exp dir: {exp_dir}')

    # Build env + task
    sim_params = parse_sim_params(args, cfg.sim.to_dict())
    env = LeggedRobotEnv(cfg=cfg, sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device, render=False, fix_cam=False)
    task = load_task_cls(cfg.task.cfg)(env)
    gym_env = GymEnvWrapper(env, task)

    # Resolve sizes (after pure_observation runs once)
    cfg_dict = collections.OrderedDict(config_to_dict(cfg))
    task.num_observations = len(task.pure_observation()[0]) * task._obs_history_n
    task.num_actions = len(task.action_low)
    cfg_dict['policy'].update({
        'num_observations': task.num_observations,
        'num_actions':      task.num_actions,
        'num_critic_obs':   len(task.critic_observation()[0]),
    })
    save_config(CfgNode(cfg_dict), join(model_dir, 'cfg.yaml'))

    # sim2sim eval setup (manifest-based, runs every --sim2sim_interval iters)
    sim2sim_cfg = None
    sim2sim_interval = getattr(args, 'sim2sim_interval', 0) or 0
    _quick_eval = None
    _manifest = None
    if sim2sim_interval > 0:
        import sys as _sys
        _sim2sim_dir = join(os.path.dirname(os.path.abspath(__file__)), 'deploy', 'sim2sim')
        if _sim2sim_dir not in _sys.path:
            _sys.path.insert(0, _sim2sim_dir)
        from sim2sim import manifest_to_sim2sim_cfg
        from evaluate import quick_eval_v2 as _quick_eval
        _manifest = build_manifest(cfg_dict)
        sim2sim_cfg = manifest_to_sim2sim_cfg(_manifest, policy_path='<will be set per eval>')
        print(f'[sim2sim] eval enabled every {sim2sim_interval} iters (manifest-based)')

    # Build PPO
    actor = load_actor(cfg_dict['policy'], device).train()
    critic = load_critic(cfg_dict['policy'], device).train()
    alg = PPO(actor, critic, device=device, **cfg.algorithm.to_dict())
    alg.init_storage(cfg.runner.num_envs, cfg.runner.num_steps_per_env,
                     [cfg_dict['policy']['num_critic_obs']],
                     [task.num_observations], [task.num_actions])

    # Initial reset
    obs, cri_obs = gym_env.reset(torch.arange(cfg.runner.num_envs, device=device))

    rew_buf = collections.deque(maxlen=100)
    len_buf = collections.deque(maxlen=100)
    cur_rew = torch.zeros(cfg.runner.num_envs, device=device)
    cur_len = torch.zeros(cfg.runner.num_envs, device=device)

    # Per-env episode-relative XY origin (for drift). Updated on each reset.
    episode_init_xy = env.base_pos[:, :2].clone()

    print(f'[v2_train] starting training: {cfg.runner.max_iterations} iters, '
          f'{cfg.runner.num_envs} envs, ckpt every {CKPT_INTERVAL} iter')

    for it in range(1, cfg.runner.max_iterations + 1):
        t0 = time.time()

        # Inline stand-metric accumulators (cleared per iter).
        st_base_z, st_pitch_sq, st_roll_sq = [], [], []
        st_vx, st_vy, st_yaw = [], [], []
        st_drift = []
        st_fall = torch.zeros(cfg.runner.num_envs, device=device)
        st_timeout = torch.zeros(cfg.runner.num_envs, device=device)
        st_n_steps = 0

        # Rollout
        for _ in range(cfg.runner.num_steps_per_env):
            act = alg.act(obs, cri_obs)
            obs, cri_obs, rew_clipped, done, info, eval_rew = gym_env.step(act)
            # Bypass wrapper's clip(min=0) — use unclipped weighted-component sum.
            rew = eval_rew.sum(dim=1)
            alg.process_env_step(rew, done, info)
            cur_rew += rew
            cur_len += 1

            # Stand metrics: collect per-step env state.
            st_base_z.append(env.base_pos[:, 2].clone())
            st_pitch_sq.append(env.base_euler[:, 1] ** 2)
            st_roll_sq.append(env.base_euler[:, 0] ** 2)
            st_vx.append(env.base_lin_vel[:, 0].clone())
            st_vy.append(env.base_lin_vel[:, 1].clone())
            st_yaw.append(env.base_ang_vel[:, 2].clone())
            st_drift.append((env.base_pos[:, :2] - episode_init_xy).norm(dim=1))
            st_n_steps += 1

            done_mask = done.squeeze(-1).bool()
            timeout_mask = task.extra_info.get('timeouts',
                            torch.zeros_like(done_mask)).bool()
            fall_mask = done_mask & ~timeout_mask
            st_fall += fall_mask.float()
            st_timeout += (done_mask & timeout_mask).float()

            done_ids = done.squeeze(-1).nonzero(as_tuple=False).flatten()
            for ei in done_ids:
                rew_buf.append(cur_rew[ei].item())
                len_buf.append(cur_len[ei].item())
                cur_rew[ei] = 0
                cur_len[ei] = 0
            if len(done_ids) > 0:
                # env has already reset for these — capture new XY as episode origin
                episode_init_xy[done_ids] = env.base_pos[done_ids, :2]

        alg.compute_returns(cri_obs)
        v_loss, s_loss, kl = alg.update()

        # Save policy.pt
        torch.save({
            'actor': alg.actor.state_dict(),
            'critic': alg.critic.state_dict(),
            'optimizer': alg.optimizer.state_dict(),
            'iteration': it,
        }, join(model_dir, 'policy.pt'))

        # TB train metrics
        if rew_buf:
            writer.add_scalar('train/mean_reward', float(np.mean(rew_buf)), it)
            writer.add_scalar('train/mean_ep_length', float(np.mean(len_buf)), it)
        writer.add_scalar('train/value_loss', float(v_loss), it)
        writer.add_scalar('train/surrogate_loss', float(s_loss), it)
        writer.add_scalar('train/kl', float(kl), it)
        writer.add_scalar('train/iter_time', time.time() - t0, it)

        # Per-reward component (mean over envs and rollout steps)
        if task._last_rew_components is not None:
            for i, name in enumerate(task.reward_names):
                writer.add_scalar(f'reward/{name}',
                                  task._last_rew_components[:, i].mean().item(), it)

        # Stand metrics from this iter's rollout — no separate eval needed.
        bz_t = torch.stack(st_base_z)
        vx_t = torch.stack(st_vx)
        vy_t = torch.stack(st_vy)
        yaw_t = torch.stack(st_yaw)
        drift_t = torch.stack(st_drift)
        n_env_steps = cfg.runner.num_envs * st_n_steps
        writer.add_scalar('stand/base_z_mean',   bz_t.mean().item(), it)
        writer.add_scalar('stand/base_z_rms',    bz_t.std(dim=0).mean().item(), it)
        writer.add_scalar('stand/pitch_rms',
                          torch.stack(st_pitch_sq).mean().sqrt().item(), it)
        writer.add_scalar('stand/roll_rms',
                          torch.stack(st_roll_sq).mean().sqrt().item(), it)
        writer.add_scalar('stand/vx_dc_bias',    vx_t.mean().item(), it)
        writer.add_scalar('stand/vx_abs_mean',   vx_t.abs().mean().item(), it)
        writer.add_scalar('stand/vy_dc_bias',    vy_t.mean().item(), it)
        writer.add_scalar('stand/vy_abs_mean',   vy_t.abs().mean().item(), it)
        writer.add_scalar('stand/yaw_rate_bias', yaw_t.mean().item(), it)
        writer.add_scalar('stand/yaw_rate_abs',  yaw_t.abs().mean().item(), it)
        writer.add_scalar('stand/xy_drift_mean', drift_t.mean().item(), it)
        writer.add_scalar('stand/xy_drift_max',  drift_t.max().item(), it)
        # Falls/timeouts per env-step (rate, not per-episode).
        writer.add_scalar('stand/fall_per_step', st_fall.sum().item() / n_env_steps, it)
        writer.add_scalar('stand/timeout_per_step',
                          st_timeout.sum().item() / n_env_steps, it)

        # Save snapshot ckpt every CKPT_INTERVAL iter (no eval, no env reset).
        if it % CKPT_INTERVAL == 0:
            torch.save({
                'actor': alg.actor.state_dict(),
                'critic': alg.critic.state_dict(),
                'iteration': it,
            }, join(all_dir, f'policy_{it}.pt'))

        # sim2sim eval — export ONNX + run MuJoCo quick_eval, log to TB.
        if sim2sim_interval > 0 and it % sim2sim_interval == 0 and sim2sim_cfg is not None:
            try:
                _t0 = time.time()
                _deploy_dir = join(exp_dir, 'deploy')
                os.makedirs(_deploy_dir, exist_ok=True)
                _onnx_path = join(_deploy_dir, f'policy_{it}.onnx')
                alg.actor.eval()
                _dummy = torch.zeros(1, task.num_observations, device='cpu')
                torch.onnx.export(alg.actor.cpu(), _dummy, _onnx_path,
                                  opset_version=12, input_names=['input'], output_names=['output'],
                                  verbose=False)
                alg.actor.to(device).train()
                save_manifest(_manifest, join(_deploy_dir, f'policy_{it}_manifest.yaml'))
                sim2sim_cfg['policy_path'] = _onnx_path
                _metrics = _quick_eval(_onnx_path, sim2sim_cfg)
                for k, v in _metrics.items():
                    if not (isinstance(v, float) and (v != v)):  # skip nan
                        writer.add_scalar(k, v, it)
                _surv = '  '.join(f'fr{k.split("fr")[1]}={v:.1f}s'
                                  for k, v in _metrics.items() if 'survive_time' in k)
                _drift = _metrics.get('sim2sim/xy_drift_final', float('nan'))
                _pitch = _metrics.get('sim2sim/pitch_rms_deg', float('nan'))
                _q = _metrics.get('sim2sim/quality_score', float('nan'))
                print(f'[sim2sim@{it}] {_surv}  drift={_drift:.2f}m  pitch={_pitch:.1f}°  q={_q:.2f}  ({time.time() - _t0:.0f}s)', flush=True)
            except Exception as _e:
                print(f'[sim2sim@{it}] eval failed: {_e}', flush=True)

        if it % 50 == 0 and rew_buf:
            print(f'[iter {it}] rew={np.mean(rew_buf):6.2f} '
                  f'len={np.mean(len_buf):5.0f} '
                  f'fall/step={st_fall.sum().item()/n_env_steps:.4f} '
                  f'drift={drift_t.mean().item():.2f}m '
                  f'kl={kl:.4f} ({time.time() - t0:.1f}s)', flush=True)

    print(f'[v2_train] done. Final ckpt: {model_dir}/policy.pt')
    writer.close()


if __name__ == '__main__':
    main()
