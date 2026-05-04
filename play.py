import time
import cv2
import isaacgym
import numpy as np
import os
from os.path import join, exists
import shutil
import torch

from env import LeggedRobotEnv, GymEnvWrapper
from env.tasks import load_task_cls
from env.utils import get_args
from env.utils.helpers import set_seed, parse_sim_params
from rl.policy import load_actor
from config.loader import load_config, CfgNode
from env.utils.math import scale_transform
import pandas as pd
from collections import deque
from isaacgym.torch_utils import *
import matplotlib.pyplot as plt
from utils.common import safe_cv2_crop

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def play(args):
    device = args.rl_device
    if args.fix_cam:
        args.render = True
    exp_dir = join('experiments', args.name)

    model_dir = join(exp_dir, 'model')
    debug_dir = join(exp_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    if args.cmp_real:
        debug_real_dir = join(debug_dir, 'realdata')
        os.makedirs(debug_real_dir, exist_ok=True)
        xl = pd.read_csv(join(debug_real_dir, 'general.txt'),
                         sep='\t+', header=None, engine='python').values[1:, :].astype(float)
        num_joint = 10
        start_idx = 2
        joint_act_real = xl[start_idx:, :num_joint][:, :num_joint]
        joint_pos_real = xl[start_idx:, num_joint:2 * num_joint][:, :num_joint]
        joint_vel_real = xl[start_idx:, 2 * num_joint:3 * num_joint][:, :num_joint]
        joint_tau_real = -xl[start_idx:, 3 * num_joint:4 * num_joint][:, :num_joint]
        base_eul_real = xl[start_idx:, 4 * num_joint:4 * num_joint + 3][:, :3]
        ang_vel_real = xl[start_idx:, 4 * num_joint + 3:4 * num_joint + 6][:, :3]
        lin_acc_real = xl[start_idx:, 4 * num_joint + 6:4 * num_joint + 9][:, :3]

        ts_real = np.linspace(0, len(joint_act_real[:, [0]]), len(joint_act_real[:, [0]]))  # .reshape(1, -1)
    if args.video:
        args.render = True
        pic_folder = os.path.join(debug_dir, 'picture')
        crop_folder = os.path.join(debug_dir, 'cropped')
        if not os.path.exists(pic_folder):
            os.makedirs(pic_folder)
        if not os.path.exists(crop_folder):
            os.makedirs(crop_folder)
    cfg = load_config(join(model_dir, 'cfg.yaml'))

    # cfg.runner.num_envs = min(cfg.runner.num_envs, 1)
    cfg.runner.num_envs = args.num_envs if args.num_envs is not None else 1
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    cfg.runner.episode_length_s = args.time

    # cfg.terrain.mesh_type = 'plane'
    cfg.noise_values.randomize_noise = False
    cfg.domain_rand.delay_observation = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_damping = False
    cfg.domain_rand.randomize_mass = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_gains = False
    cfg.domain_rand.randomize_torque = False
    cfg.init_state.random_rot = False

    set_seed(seed=None)
    # set_seed(seed=3985)

    sim_params = parse_sim_params(args, cfg.sim.to_dict())
    env = LeggedRobotEnv(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         render=args.render,
                         debug=args.debug,
                         fix_cam=args.fix_cam,
                         tcn_name=args.tcn,
                         epochs=args.epochs)
    task = load_task_cls(cfg.task.cfg)(env)
    gym_env = GymEnvWrapper(env, task, debug=args.debug)
    pure_obs_len = len(gym_env.task.pure_observation()[0])
    stack_obs_len = gym_env.task.obs_history.maxlen
    task.num_observations = pure_obs_len * stack_obs_len
    task.num_actions = len(gym_env.task.action_low)

    actor = load_actor(cfg.policy.to_dict(), device).eval()
    if args.iter is not None:
        saved_model_state_dict = torch.load(join(model_dir, f'all/policy_{args.iter}.pt'))
    else:
        saved_model_state_dict = torch.load(join(model_dir, 'policy.pt'))
    actor.load_state_dict(saved_model_state_dict['actor'])

    print(f'--------------------------------------------------------------------------------------')
    print(f'Start to evaluate policy `{exp_dir}`.')

    # loaded_rl_data = pd.read_csv(join(join(exp_dir, 'real'), 'rl.txt'), sep='\t+', header=None, engine='python').values[1:, :].astype(float)
    # rl_data_list = to_torch(np.array(loaded_rl_data), dtype=torch.float, device=device, requires_grad=False)
    live_time_count = []
    for epoch in range(args.epochs):
        print(f'#The `{epoch + 1}st/(total {args.epochs} times)` rollout......................................')

        if args.cmd_vx is not None:
            gym_env.task.fixed_commands[0] = args.cmd_vx
        if args.cmd_yaw is not None:
            gym_env.task.fixed_commands[2] = args.cmd_yaw
        obs, cri_obs = gym_env.reset(torch.arange(env.num_envs, device=device).detach())
        obs,cri_obs = obs.type(torch.float32), cri_obs.type(torch.float32)
        for i in range(int(args.time / (cfg.sim.dt * cfg.pd_gains.decimation))):
            with torch.inference_mode():
                act = actor(obs)['act'].detach().clone()

            obs, cri_obs, rew, done, info, eval_rew = gym_env.step(act)
            if args.video:
                if i >= 3:
                    filename = os.path.join(pic_folder, f"{i}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
            if any(done):
                break
        print(f'Evaluation finished after {i} timesteps ({i * gym_env.env.dt:.1f} seconds).')
        if args.epochs > 1:
            live_time_count.append(i)
            if epoch == (args.epochs - 1):
                data_path = os.path.join(debug_dir, f'live_time.xlsx')
                with pd.ExcelWriter(data_path) as f:
                    pd.DataFrame({'live_time': live_time_count}).to_excel(f, sheet_name='live time', index=False)
                    print(f'#The live count data has been written into `{data_path}`.')
        if args.debug and gym_env.env.common_step_counter >= 1:
            debug_data = gym_env.save_debug_data(debug_dir)
            if args.cmp_real or args.plt_sim:
                joint_act, joint_pos, joint_vel, joint_tau = debug_data['joint_act'], debug_data['joint_pos'], debug_data['joint_vel'], debug_data['joint_tau']
                base_eul, ang_vel, lin_acc = debug_data['base_eul'], debug_data['ang_vel'], debug_data['lin_acc']
                ts_sim = np.linspace(0, len(joint_act[0, :]), len(joint_act[0, :]))
                num = min(len(joint_act[0, :]), len(ts_real)) if args.cmp_real else len(joint_act[0, :])
                plt_id = list(range(0, num_joint))  # [4, 5, 10, 11]  # list(range(0, 6))
                for j in plt_id:
                    for i, motor_id in enumerate([j]):
                        plt.plot(ts_sim[:num], joint_act[0, :num, motor_id], linestyle='-.', c='k')
                        plt.plot(ts_sim[:num], joint_pos[0, :num, motor_id], linestyle='-', c='b')
                        if args.cmp_real:
                            plt.plot(ts_real[:num], joint_act_real[:num, motor_id], linestyle=':', c='k')
                            plt.plot(ts_real[:num], joint_pos_real[:num, motor_id], linestyle='-', c='r')
                            plt.title(gym_env.env.dof_names[j] + ':' + 'joint_pos')
                            plt.legend(['sim_act', 'sim_pos', 'real_act', 'real_pos'])
                        else:
                            plt.legend(['sim_act', 'sim_pos'])
                        plt.grid()
                        plt.savefig(join(debug_dir, f'joint_pos_{motor_id}.png'))
                        plt.show()

                        plt.plot(ts_sim[:num], joint_act[0, :num, motor_id] - joint_pos[0, :num, motor_id], linestyle='-', c='b')
                        if args.cmp_real:
                            plt.plot(ts_real[:num], joint_act_real[:num, motor_id] - joint_pos_real[:num, motor_id], linestyle='-', c='r')
                            plt.legend(['sim', 'real'])
                        plt.title(gym_env.env.dof_names[j] + ':' + 'joint_err')
                        plt.grid()
                        plt.savefig(join(debug_dir, f'joint_err_{motor_id}.png'))
                        plt.show()

                        plt.plot(ts_sim[:num], joint_vel[0, :num, motor_id], linestyle='-', c='b')
                        if args.cmp_real:
                            plt.plot(ts_real[:num], joint_vel_real[:num, motor_id], linestyle='-', c='r')
                            plt.legend(['sim_vel', 'real_vel'])
                        plt.title(gym_env.env.dof_names[j] + ':' + 'joint_vel')
                        plt.grid()
                        plt.savefig(join(debug_dir, f'joint_vel_{motor_id}.png'))
                        plt.show()

                        plt.plot(ts_sim[:num], joint_tau[0, :num, motor_id], linestyle='-', c='b')
                        if args.cmp_real:
                            plt.plot(ts_real[:num], joint_tau_real[:num, motor_id], linestyle='-', c='r')
                            plt.legend(['sim_tau', 'real_tau'])
                        plt.title(gym_env.env.dof_names[j] + ':' + 'joint_tau')
                        plt.grid()
                        plt.savefig(join(debug_dir, f'joint_tau_{motor_id}.png'))
                        plt.show()
                plt_base_id = list(range(0, 2))
                axis_names = gym_env.debug_name['base_eul']
                for j in plt_base_id:
                    plt.plot(ts_sim[:num], base_eul[0, :num, j], linestyle='--', c='b')
                    if args.cmp_real:
                        plt.plot(ts_real[:num], base_eul_real[:num, j], linestyle='-', c='r')
                        plt.legend([f'sim_{axis_names[j]}', f'real_{axis_names[j]}'])
                    plt.title(axis_names[j] + ':' + f'base_euler {axis_names[j]}')
                    plt.grid()
                    plt.savefig(join(debug_dir, f'base_eul_{j}.png'))
                    plt.show()
                for j in plt_base_id:
                    plt.plot(ts_sim[:num], ang_vel[0, :num, j], linestyle='--', c='b')
                    if args.cmp_real:
                        plt.plot(ts_real[:num], ang_vel_real[:num, j], linestyle='-', c='r')
                        plt.legend([f'sim_{axis_names[j]}', f'real_{axis_names[j]}'])
                    plt.title(axis_names[j] + ':' + f'ang_vel {axis_names[j]}')
                    plt.grid()
                    plt.savefig(join(debug_dir, f'ang_vel_{j}.png'))
                    plt.show()
                for j in plt_base_id:
                    plt.plot(ts_sim[:num], lin_acc[0, :num, j], linestyle='--', c='b')
                    if args.cmp_real:
                        plt.plot(ts_real[:num], lin_acc_real[:num, j], linestyle='-', c='r')
                        plt.legend([f'sim_{axis_names[j]}', f'real_{axis_names[j]}'])
                    plt.title(axis_names[j] + ':' + f'lin_acc {axis_names[j]}')
                    plt.grid()
                    plt.savefig(join(debug_dir, f'lin_acc_{j}.png'))
                    plt.show()

        if args.video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # low quality mp4
            cap_fps = int(1 / (cfg.sim.dt * cfg.pd_gains.decimation))
            ckpt_tag = f'iter{args.iter}' if args.iter is not None else 'latest'
            vx_tag = f'vx{args.cmd_vx}' if args.cmd_vx is not None else 'vxfree'
            yaw_tag = f'yaw{args.cmd_yaw}' if args.cmd_yaw is not None else 'yawfree'
            auto_suffix = f'_{ckpt_tag}_{vx_tag}_{yaw_tag}'
            video_name = (args.out if args.out is not None else args.name) + auto_suffix
            video_path = join(debug_dir, f'{video_name}.mp4')
            videoWriter = cv2.VideoWriter(video_path, fourcc, cap_fps, (1600, 900))
            file_lst = os.listdir(pic_folder)
            file_lst.sort(key=lambda x: int(x[:-4]))
            for i, filename in enumerate(file_lst):
                img = cv2.imread(join(pic_folder, filename))
                videoWriter.write(img)
                crop = safe_cv2_crop(img, crop_size=600)
                cv2.imwrite(join(crop_folder, f"{i}.png"), crop)
            videoWriter.release()
            shutil.rmtree(pic_folder)
            print(f'#The video has been saved into `{video_path}`.')
    print(f'--------------------------------------------------------------------------------------')


if __name__ == '__main__':
    args = get_args()
    play(args)
