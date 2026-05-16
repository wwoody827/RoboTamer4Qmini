"""Isaac Gym eval: load trained policy, force cmd=0, record base_z trajectory.

Diagnostic for sim-to-sim gap. If Isaac Gym z stays at 0.44 but sim2sim z drops
to 0.20, the gap is between sims (physics). If Isaac Gym also gives z=0.20, the
training reward signal was misleading and policy is genuinely bad.

Usage:
    python scripts/isaac_eval.py --resume <run_name> --duration 10
"""

import os
import sys
from os.path import join
import collections
import argparse
import numpy as np

from env.utils import get_args
from env.utils.helpers import set_seed, parse_sim_params
from env import LeggedRobotEnv, GymEnvWrapper
from env.tasks import load_task_cls
from rl.policy import load_actor
from config.loader import load_config, CfgNode, config_to_dict, save_config
from isaacgym.torch_utils import *
import torch


def _cfg_section_to_dict(section):
    return section.to_dict()


def main():
    torch.cuda.empty_cache()
    args = get_args()
    device = args.rl_device

    # Use saved config from the resume dir (matches what was trained)
    if args.resume is None:
        print("ERROR: --resume <run_name> required")
        sys.exit(1)
    resume_dir = join('experiments', args.resume)
    cfg_path = join(resume_dir, 'model', 'cfg.yaml')
    cfg = load_config(cfg_path)

    # Override num_envs to small for fast eval
    cfg.runner.num_envs = 4
    cfg.runner.seed = 42
    set_seed(seed=42)

    # Force NO randomization for diagnostic (match sim2sim clean init)
    cfg.init_state.random_rot = False  # disable random init
    if hasattr(cfg, 'domain_rand'):
        cfg.domain_rand.randomize_friction = False
        cfg.domain_rand.randomize_mass = False
        cfg.domain_rand.randomize_torque = False
        if hasattr(cfg.domain_rand, 'randomize_gains'):
            cfg.domain_rand.randomize_gains = False

    sim_params = parse_sim_params(args, _cfg_section_to_dict(cfg.sim))
    env = LeggedRobotEnv(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         render=False, fix_cam=False)
    task = load_task_cls(cfg.task.cfg)(env)
    gym_env = GymEnvWrapper(env, task)

    cfg_dict = collections.OrderedDict(config_to_dict(cfg))
    task.num_observations = len(gym_env.task.pure_observation()[0]) * gym_env.task._obs_history_n
    task.num_actions = len(gym_env.task.action_low)
    cfg_dict['policy'].update({'num_observations': task.num_observations,
                               'num_actions': task.num_actions,
                               'num_critic_obs': len(gym_env.task.critic_observation()[0])})

    actor = load_actor(cfg_dict['policy'], device).eval()
    saved = torch.load(join(resume_dir, 'model', 'policy.pt'), map_location=device)
    actor.load_state_dict(saved['actor'])
    print(f"Loaded policy from {resume_dir}")

    obs, cri_obs = gym_env.reset(torch.arange(cfg.runner.num_envs, device=device))

    # Force cmd to ZERO every step
    duration = 10.0
    policy_dt = cfg.sim.dt * cfg.pd_gains.decimation
    n_steps = int(duration / policy_dt)

    # Storage
    base_pos_log = []
    joint_pos_log = []
    joint_act_log = []

    with torch.no_grad():
        for step in range(n_steps):
            # FORCE cmd=0
            task.commands[:, :] = 0.0
            task.static_flag[:, :] = 0.0

            # Run policy deterministically. In eval mode, actor returns
            # {'logits': ..., 'act': mu} where mu is deterministic.
            res = actor(obs.to(torch.float32))
            action = res['act'].detach()
            obs, cri_obs, rew, done, info, eval_rew = gym_env.step(action)

            # Record env 0 only (others will diverge due to internal stochasticity)
            base_pos_log.append(env.base_pos[0].clone().cpu().numpy())
            joint_pos_log.append(env.joint_pos[0].clone().cpu().numpy())
            joint_act_log.append(task.current_joint_act[0].clone().cpu().numpy())

    base_pos = np.array(base_pos_log)
    joint_pos = np.array(joint_pos_log)
    joint_act = np.array(joint_act_log)

    # Skip first 1s transient
    skip = int(1.0 / policy_dt)
    bp = base_pos[skip:]
    jp = joint_pos[skip:]
    ja = joint_act[skip:]

    print(f"\n=== Isaac Gym eval @ cmd=0 (env 0, deterministic policy) ===")
    print(f"Duration: {duration}s  ({n_steps} policy steps)")
    print(f"\nBase position (last 9s):")
    print(f"  x:  start={bp[0,0]:+.3f}  end={bp[-1,0]:+.3f}  mean={bp[:,0].mean():+.3f}")
    print(f"  y:  start={bp[0,1]:+.3f}  end={bp[-1,1]:+.3f}  mean={bp[:,1].mean():+.3f}")
    print(f"  z:  mean={bp[:,2].mean():.3f}  RMS={bp[:,2].std():.4f}  min={bp[:,2].min():.3f}  max={bp[:,2].max():.3f}")

    drift_xy = np.linalg.norm(bp[-1, :2] - bp[0, :2])
    print(f"\n  |XY| drift over 9s: {drift_xy:.3f}m")

    ref = np.array(cfg.action.ref_joint_pos)
    print(f"\nJoint divergence from ref (steady state):")
    names = ['hip_y_l', 'hip_r_l', 'hip_p_l', 'knee_l', 'ank_l',
             'hip_y_r', 'hip_r_r', 'hip_p_r', 'knee_r', 'ank_r']
    print(f"  {'name':>8} {'ref':>8} {'pos_mean':>9} {'act_mean':>9} {'pos-ref':>8}")
    for i, n in enumerate(names):
        print(f"  {n:>8} {ref[i]:>+8.3f} {jp[:, i].mean():>+9.3f} {ja[:, i].mean():>+9.3f} {jp[:, i].mean() - ref[i]:>+8.3f}")


if __name__ == '__main__':
    main()
