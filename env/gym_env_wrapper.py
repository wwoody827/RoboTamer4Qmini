from collections import OrderedDict
import numpy as np
import pandas as pd
import os

import env
from .legged_robot import LeggedRobotEnv
from .tasks import NullTask
from isaacgym.torch_utils import *
import collections
import torch


class GymEnvWrapper:
    def __init__(self, env: LeggedRobotEnv, task: NullTask, residual_task=None,
                 dynamic_params=None, debug: bool = False):
        self.env = env
        self.task = task
        self.task.debug = debug
        self.device = env.device
        self.debug = debug
        self.debug_data = {name: [] for name in self.debug_name} if debug else None
        if dynamic_params is not None:
            self.dynamic_params = dynamic_params
            self.tcn_obs_buf = collections.deque(maxlen=dynamic_params['seq_length'])
        self.residual_task = residual_task if residual_task is not None else None

    def reset(self, env_ids, reset_joint_pos=None, reset_joint_vel=None, reset_base_quat=None):
        if reset_joint_pos is not None:
            self.env.reset_joint_pos = reset_joint_pos
        if reset_joint_vel is not None:
            self.env.reset_joint_vel = reset_joint_vel
        if reset_base_quat is not None:
            self.env.reset_base_quat = reset_base_quat
        self.env.step_torques(torch.clone(self.task.current_joint_act))
        self.env.step_states()
        obs = self.task.observation()
        cri_obs = self.task.critic_observation()
        if self.debug:
            self.clear_debug_data()
        self.env.reset(env_ids)
        self.task.reset(env_ids)
        if self.residual_task is not None:
            self.residual_task.reset(env_ids)
        return obs, cri_obs

    def step(self, net_out, it=None):
        joint_act = self.task.action(net_out)
        for m in range(self.env.cfg.pd_gains.decimation):
            if self.env.cfg.noise_values.randomize_noise and m % 5 == 0:
                self.env.lin_vel_noise = self.env.cfg.noise_values.lin_vel * (2. * torch.rand_like(self.env.base_lin_vel) - 1.)
                self.env.gravity_noise = self.env.cfg.noise_values.gravity * (2. * torch.rand_like(self.env.base_euler) - 1.)
                self.env.ang_vel_noise = self.env.cfg.noise_values.ang_vel * (2. * torch.rand_like(self.env.base_ang_vel) - 1.)
                self.env.foot_frc_noise = self.env.cfg.noise_values.foot_frc * (2. * torch.rand_like(self.env.foot_frc) - 1.)
                self.env.joint_pos_noise = self.env.cfg.noise_values.dof_pos * (2. * torch.rand_like(self.env.joint_pos) - 1.)
                self.env.joint_vel_noise = self.env.cfg.noise_values.dof_vel * (2. * torch.rand_like(self.env.joint_vel) - 1.)
                self.env.base_acc_noise = self.env.cfg.noise_values.base_acc * (2. * torch.rand_like(self.env.base_acc) - 1.)
            self.env.step_torques(joint_act)
        self.env.step_states(it)  # 10ms update!
        self.task.step()
        obs = self.task.observation()
        cri_obs = self.task.critic_observation()
        rew, eval_rew = self.task.reward()
        done = self.task.terminate()
        info = self.task.info()
        rew_buf = torch.clip(rew.sum(dim=1), min=0.)
        # rew_buf = rew.sum(dim=1)
        if self.debug and self.env.common_step_counter >= 8:
            self.record_debug_data(rew, obs)
        reset_env_ids = (done > 0).nonzero(as_tuple=False)[:, [0]].flatten()
        if len(reset_env_ids) > 0:
            self.env.reset(reset_env_ids)
            self.task.reset(reset_env_ids)
        return obs, cri_obs, rew_buf, done, info, eval_rew


    def close(self):
        pass

    def record_debug_data(self,rew, obs):
        self.debug_data['reward'].append(rew / self.env.dt)
        self.debug_data['command'].append(self.task.commands.clone())
        self.debug_data['lin_vel'].append(self.task.base_lin_vel.clone())
        self.debug_data['lin_acc'].append(self.task.base_acc.clone())


        self.debug_data['base_eul'].append(self.task.base_euler.clone())
        self.debug_data['ang_vel'].append(self.task.base_ang_vel.clone())
        self.debug_data['base_pos'].append(self.env.base_pos_hd.clone())

        self.debug_data['foot_pos'].append(self.env.foot_pos_hd.clone())
        self.debug_data['foot_vel'].append(self.env.foot_vel.clone())
        self.debug_data['foot_frc'].append(self.task.foot_frc.clone())
        self.debug_data['foot_rpy'].append(self.env.foot_euler.clone())
        self.debug_data['foot_phs'].append(self.task.phase_modulator.phase.clone())
        self.debug_data['joint_act'].append(self.task.current_joint_act.clone())
        self.debug_data['joint_pos'].append(self.task.joint_pos.clone())


        self.debug_data['joint_vel'].append(self.task.joint_vel.clone())
        self.debug_data['joint_tau'].append(self.env.react_tau.clone())
        self.debug_data['joint_acc'].append(self.env.joint_acc.clone())
        self.debug_data['netout_a'].append(self.task.net_out_history[-1].clone())
        self.debug_data['netout_f'].append(self.task.pm_f.clone())
        self.debug_data['obs_state'].append(obs)

    @property
    def debug_name(self):
        d = OrderedDict()
        axises = ['x', 'y', 'z']
        foot_names = ['L', 'R']
        d['reward'] = self.task.rew_names
        d['command'] = ['fwd_vel', 'lat_vel', 'yaw_rate', 'heading']
        d['lin_vel'] = [n for n in axises]
        d['lin_acc'] = [n for n in axises]

        d['base_eul'] = [n for n in axises]
        d['ang_vel'] = [n for n in axises]
        d['joint_act'] = self.env.dof_names[:10]
        d['joint_pos'] = [n for n in self.env.dof_names[:10]]


        d['netout_f'] = [f for f in foot_names]  # + [n for n in self.env.dof_names]
        d['netout_a'] = [f for f in foot_names]+[n for n in self.env.dof_names]

        d['foot_phs'] = [n for n in foot_names]
        d['foot_frc'] = [n for n in foot_names]
        d['foot_pos'] = [f'{o}_{n}' for o in foot_names for n in axises]
        d['obs_state'] = ['obs' + '_' + str(i) for i in range(self.env.num_observations)]
        d['foot_rpy'] = [f'{o}_{n}' for o in foot_names for n in axises]
        d['base_pos'] = [n for n in axises]

        d['foot_vel'] = [f'{o}_{n}' for o in foot_names for n in axises]

        d['joint_vel'] = [n for n in self.env.dof_names]
        d['joint_acc'] = [n for n in self.env.dof_names]
        d['joint_tau'] = [n for n in self.env.dof_names]
        return d


    def clear_debug_data(self):
        for k, v in self.debug_data.items():
            self.debug_data[k].clear()

    def save_debug_data(self, debug_dir: str):
        debug_data = {key: torch.stack(self.debug_data[key], dim=1).cpu().numpy() for key in self.debug_name.keys()}
        for i in range(min(self.env.num_envs, 2)):
            data_path = os.path.join(debug_dir, f'debug_{i}.xlsx')
            with pd.ExcelWriter(data_path) as f:
                for key in self.debug_name.keys():
                    pd.DataFrame(np.asarray(debug_data[key][i]), columns=self.debug_name[key]).to_excel(f, key, index=False)
            print(f'#The debug data has been written into `{data_path}`.')
        return debug_data

