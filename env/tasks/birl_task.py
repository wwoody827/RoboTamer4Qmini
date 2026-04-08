from math import pi, sin, cos, exp, tau
import numpy as np
from scipy.linalg import toeplitz
from collections import OrderedDict
from env.legged_robot import LeggedRobotEnv
from env.utils.helpers import class_to_dict
from env.utils.math import wrap_to_pi, smallest_signed_angle_between
from env.utils.phase_modulator import PhaseModulator
from env.tasks.null_task import NullTask, register
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R
from env.tasks.base_task import BaseTask
import random
from env.utils.math import scale_transform, smallest_signed_angle_between_torch
from collections import deque
import statistics
import torch


@register
class BIRLTask(BaseTask):
    def __init__(self, env: LeggedRobotEnv):
        super(BIRLTask, self).__init__(env)
        self.env = env
        self.cmd_id = 0
        self.rew_names = None
        self.num_envs = env.num_envs
        self.num_legs = 2

        self.commands = torch.zeros(self.num_envs, self.cfg.command.num_commands, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # x vel, y vel, yaw vel, heading

        self.command_cfgs = class_to_dict(self.cfg.command)
        self.resampling_interval = int(self.cfg.command.resampling_time / self.env.dt)
        self.static_flag = torch.where(torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.15, False,
                                       True).float()
        self.zero_command_env_ids = (torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.15).nonzero(as_tuple=False)[:, [0]].flatten()
        self._resample_commands(torch.arange(env.num_envs, device=self.device))

        if self.cfg.domain_rand.delay_observation:
            self.delay_joint_steps = random.randint(self.cfg.domain_rand.delay_joint_ranges[0],
                                                    self.cfg.domain_rand.delay_joint_ranges[1])
            self.delay_rate_steps = random.randint(self.cfg.domain_rand.delay_rate_ranges[0],
                                                   self.cfg.domain_rand.delay_rate_ranges[1])
            self.delay_angle_steps = random.randint(self.cfg.domain_rand.delay_angle_ranges[0],
                                                    self.cfg.domain_rand.delay_angle_ranges[1])
        else:
            self.delay_joint_steps = 1
            self.delay_rate_steps = 1
            self.delay_angle_steps = 1
        self.convert_phi = 1.2 * pi

        self.phase_modulator = PhaseModulator(time_step=env.dt, num_envs=self.num_envs, num_legs=self.num_legs,device=self.device)
        self.phase_modulator.reset(convert_phi=self.convert_phi, env_ids=torch.arange(self.num_envs),
                                   render=self.env.render or self.env.debug or self.env.epochs > 1 or self.env.tcn_name is not None)
        self.foot_phase = self.phase_modulator.phase
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)

        if self.cfg.action.use_increment:
            self.action_low = to_torch(self.cfg.action.inc_low_ranges, device=self.device)
            self.action_high = to_torch(self.cfg.action.inc_high_ranges, device=self.device)
        else:
            self.action_low = to_torch(self.cfg.action.low_ranges, device=self.device)
            self.action_high = to_torch(self.cfg.action.high_ranges, device=self.device)
            self.action_low[self.num_legs:self.num_legs + self.env.num_dofs] = torch.as_tensor(self.env.dof_pos_limits[:, 0], device=self.device)
            self.action_high[self.num_legs:self.num_legs + self.env.num_dofs] = torch.as_tensor(self.env.dof_pos_limits[:, 1], device=self.device)
        self.current_joint_act = to_torch(self.env.default_dof_pos, device=self.device).repeat(self.num_envs, 1)
        self.previous_joint_act = self.current_joint_act.clone()

        self.ref_joint_action = to_torch(self.cfg.action.ref_joint_pos, device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_low_over = torch.as_tensor(self.env.dof_pos_limits[:, 0]).repeat(self.num_envs, 1)
        self.joint_action_limit_high_over = torch.as_tensor(self.env.dof_pos_limits[:, 1]).repeat(self.num_envs, 1)

        # self.joint_action_limit_low = torch.as_tensor(self.cfg.action.low_ranges[self.num_legs:], device=self.device).repeat(self.num_envs, 1)
        # self.joint_action_limit_high = torch.as_tensor(self.cfg.action.high_ranges[self.num_legs:], device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_low = torch.as_tensor(self.env.dof_pos_limits[:, 0], device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_high = torch.as_tensor(self.env.dof_pos_limits[:, 1], device=self.device).repeat(self.num_envs, 1)

        self.obs_history = deque(maxlen=3)
        self.cri_obs_history = deque(maxlen=3)

        self.action_history = deque(maxlen=3)
        self.net_out_history = deque(maxlen=3)

        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_joint_act)

        for _ in range(self.net_out_history.maxlen):
            self.net_out_history.append(torch.zeros_like(self.action_low).repeat(self.num_envs, 1))

        foot_support_mask_1 = torch.where(self.foot_phase >= 0, True, False)
        foot_support_mask_2 = torch.where(self.foot_phase < self.convert_phi, True, False)
        self.foot_support_mask = torch.logical_and(foot_support_mask_1, foot_support_mask_2)
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone()

        self.last_foot_frc = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.foot_frc_acc = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device,
                                        requires_grad=False)

        self.last_foot_vel = torch.zeros(self.num_envs, self.num_legs * 3, dtype=torch.float, device=self.device,
                                         requires_grad=False)

        self.joint_vel = self.env.joint_vel_his.delay(self.delay_joint_steps)
        self.joint_pos = self.env.joint_pos_his.delay(self.delay_joint_steps)
        self.base_acc = self.env.base_acc_his.delay(self.delay_rate_steps)

        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.joint_tau = self.env.p_gains * self.joint_pos_error - self.env.d_gains * self.joint_vel
        self.foot_pos_hd = self.env.foot_pos_hd
        if self.cfg.terrain.mesh_type in ['trimesh','heightfield']:
            self.foot_height = self.env.get_foot_height_to_ground()
        else:
            self.foot_height =  self.env.foot_pos_hd[:, [2, 5]]

        self.foot_vel = self.env.foot_vel

        self.foot_frc = self.env.foot_frc
        self.base_ang_vel = self.env.base_ang_vel_his.delay(self.delay_rate_steps)

        self.base_euler = self.env.base_eul_his.delay(self.delay_angle_steps)
        self.base_lin_vel = self.env.base_lin_vel

        for _ in range(self.obs_history.maxlen):
            self.obs_history.append(self.pure_observation())
        for _ in range(self.cri_obs_history.maxlen):
            self.cri_obs_history.append(self.pure_critic_observation())

        self.extra_info["task"] = {}
        if self.cfg.terrain.curriculum:
            self.extra_info["task"]["terrain_level"] = torch.mean(self.env.terrain_levels.float())
        if self.cfg.runner.send_timeouts:
            self.extra_info["timeouts"] = self.env.time_out_buf

    def reset(self, env_ids):
        self.base_acc[env_ids] = self.env.base_acc_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_vel[env_ids] = self.env.joint_vel_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_pos[env_ids] = self.env.joint_pos_his.delay(self.delay_joint_steps)[env_ids]
        self.current_joint_act[env_ids] = self.env.default_dof_pos
        self.previous_joint_act[env_ids] = self.current_joint_act[env_ids].clone()

        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.phase_modulator.reset(convert_phi=self.convert_phi, env_ids=env_ids,
                                   render=self.env.render or self.env.epochs > 1 or self.env.tcn_name is not None)
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)
        self.static_flag = torch.where(torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.15, False,True).float()
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        self._resample_commands(env_ids)
        foot_support_mask_1 = torch.where(self.foot_phase >= 0, True, False)
        foot_support_mask_2 = torch.where(self.foot_phase < self.convert_phi, True, False)
        self.foot_support_mask = torch.logical_and(foot_support_mask_1, foot_support_mask_2)
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone()



    def step(self):

        self.joint_pos = self.env.joint_pos_his.delay(self.delay_joint_steps)
        self.joint_vel = self.env.joint_vel_his.delay(self.delay_joint_steps)
        self.base_acc = self.env.base_acc_his.delay(self.delay_rate_steps).clip(min=-30., max=30.)
        self.joint_tau = self.env.joint_tau_his.delay(1)

        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.foot_pos_hd = self.env.foot_pos_hd
        if self.cfg.terrain.mesh_type in ['trimesh', 'heightfield']:
            self.foot_height = self.env.get_foot_height_to_ground()
        else:
            self.foot_height = self.env.foot_pos_hd[:, [2, 5]]

        self.foot_vel = self.env.foot_vel

        self.foot_frc = self.env.foot_frc

        self.base_euler = self.env.base_eul_his.delay(self.delay_angle_steps)
        self.base_ang_vel = self.env.base_ang_vel_his.delay(self.delay_rate_steps)
        self.base_lin_vel = self.env.base_lin_vel
        self.foot_phase = self.phase_modulator.phase
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)

        foot_support_mask_1 = torch.where(self.foot_phase >= 0., True, False)
        foot_support_mask_2 = torch.where(self.foot_phase < self.convert_phi, True, False)
        self.foot_support_mask = torch.logical_and(foot_support_mask_1, foot_support_mask_2)
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone().detach()
        env_ids = ((self.env.episode_length_buf) % self.resampling_interval == 0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

        if self.cfg.domain_rand.delay_observation and self.env.common_step_counter % 200 == 0:
            self.delay_joint_steps = random.randint(self.cfg.domain_rand.delay_joint_ranges[0],
                                                    self.cfg.domain_rand.delay_joint_ranges[1])
            self.delay_rate_steps = random.randint(self.cfg.domain_rand.delay_rate_ranges[0],
                                                   self.cfg.domain_rand.delay_rate_ranges[1])
            self.delay_angle_steps = random.randint(self.cfg.domain_rand.delay_angle_ranges[0],
                                                    self.cfg.domain_rand.delay_angle_ranges[1])


    def observation(self):
        self.obs_buf_pure = self.pure_observation()
        self.obs_history.append(self.obs_buf_pure)
        return  torch.cat([obs for obs in self.obs_history], dim=-1)

    def critic_observation(self):
        pure_obs_buf = self.pure_critic_observation()
        self.cri_obs_history.append(pure_obs_buf)
        return  torch.cat([obs for obs in self.cri_obs_history], dim=-1)

    def pure_critic_observation(self):
        obs_buf = torch.cat([
            self.commands[:, [0,2]],
            self.commands[:, [0]] - self.env.base_lin_vel[:, [0]],
            self.commands[:, [2]] - self.env.base_ang_vel[:, [2]],
            self.env.base_lin_vel,
            self.env.base_euler[:, :2],
            self.env.base_ang_vel * 0.5,
            (self.env.joint_pos - self.ref_joint_action),
            self.env.joint_vel * 0.1,
            (self.current_joint_act - self.ref_joint_action),
            self.joint_pos_error,
            self.pm_phase * self.static_flag,
            (self.pm_f * 0.3 - 1.) * self.static_flag,
            self.net_out_history[-1][:, self.num_legs:] / 15.,
            self.foot_height.clip(min=-0.5, max=0.5) * 10.,
            (self.env.base_pos_hd[:, [2]] - 0.4) * 10.,
            self.env.foot_vel.clip(min=-8., max=8.) * 0.5,
            self.env.base_acc.clip(min=-20., max=20.) * 0.2,
            self.env.foot_frc.clip(min=0., max=200.) * 0.01,
            self.net_out_history[-1][:, self.num_legs:] / 15.,
            self.base_euler[:, :2] * 1.,
            self.base_ang_vel * 0.5,
            self.joint_pos - self.ref_joint_action,
            self.joint_vel * 0.1,
            self.joint_pos_error,
        ], dim=1)
        return obs_buf

    def pure_observation(self):
        self.obs_buf = torch.cat([
            self.commands[:, [0,2]],
            self.base_euler[:, :2] * 1.,
            self.base_ang_vel * 0.5,
            self.joint_pos - self.ref_joint_action,
            self.joint_vel * 0.1,
            self.joint_pos_error,
            self.pm_phase * self.static_flag,
            (self.pm_f * 0.3 - 1.) * self.static_flag,
            # (self.base_acc.clip(min=-20., max=20.)) * 0.05,
            # self.net_out_history[-1][:, self.num_legs:] / 15.,
        ], dim=1).clip(min=-3., max=3.)
        return self.obs_buf

    def action(self, net_out):
        net_out = scale_transform(net_out, self.action_low, self.action_high)
        self.net_out_history.append(net_out)
        self.phase_modulator.compute(net_out[:, :self.num_legs])
        if self.env.render and self.env.common_step_counter <= 1:
            pass
        else:
            if self.cfg.action.use_increment:
                self.current_joint_act += net_out[:, self.num_legs:] * self.env.dt
            else:
                self.current_joint_act =net_out[:, self.num_legs:]
        self.current_joint_act = torch.clip(self.current_joint_act, self.joint_action_limit_low,self.joint_action_limit_high)
        self.action_history.append(self.current_joint_act.clone())
        self.previous_joint_act = self.current_joint_act.clone()
        return self.current_joint_act

    def reward(self):
        constant_rew = to_torch([1.]).repeat(self.num_envs, 1)
        lin_vel_x_norm = torch.clip(torch.abs(self.commands[:, [0]]), min=0.3, max=2.) + 0.2
        yaw_rate_norm = torch.clip(torch.abs(self.commands[:, [2]]), min=0.3, max=1.5) + 0.2
        base_heit_rew = torch.exp(-70 * (self.env.base_pos[:, [2]] - 0.45) ** 2)

        balance_rew = 0.5 * (base_heit_rew * torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2, max=8.) * torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)) + 1.)

        forward_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2., max=10.) * (
                self.commands[:, [0]] - self.env.base_lin_vel[:, [0]]) ** 2) #* balance_rew
        lateral_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=3., max=15.) * torch.norm(self.env.base_lin_vel[:, [1]], dim=1, keepdim=True) ** 2)

        yaw_rate_rew = torch.exp(-torch.clip(2. / yaw_rate_norm, min=2., max=6.) * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2)

        lateral_vel_rew += -0.6 / lin_vel_x_norm * torch.norm(self.env.base_lin_vel[:, [1]], dim=1, keepdim=True) * self.static_flag

        ang_vel_rew = torch.exp(
            -torch.clip(2. / lin_vel_x_norm, min=0.7, max=6.) * torch.norm(self.env.base_ang_vel[:, :2], dim=1,
                                                                            keepdim=True) ** 2)
        base_acc_rew = -0.4 / lin_vel_x_norm * torch.norm((self.env.base_acc - to_torch([0, 0, 9.81], device=self.device)) * 0.1, dim=1, keepdim=True)
        base_acc_rew *= self.static_flag

        vertical_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2., max=10.) * torch.norm(self.env.base_lin_vel[:, [2]], dim=1,
                                                                           keepdim=True) ** 2)
        vertical_vel_rew -= 0.2 / lin_vel_x_norm * torch.norm(self.env.base_lin_vel[:, 1:], dim=1, keepdim=True) * self.static_flag

        support_foot_index = torch.where(self.env.foot_frc >= 10., True, False)
        swing_foot_index = torch.where(self.env.foot_frc < 1., True, False)

        foot_clear_rew = torch.sum(torch.logical_and(swing_foot_index, self.foot_swing_mask), dtype=torch.float, dim=1,keepdim=True) / self.num_legs

        foot_support_rew = torch.sum(torch.logical_and(support_foot_index, self.foot_support_mask), dtype=torch.float,dim=1,keepdim=True) / self.num_legs
        foot_support_rew *= self.static_flag
        foot_clear_rew *= self.static_flag

        foot_heit_score = 40. * torch.clip(self.foot_height, min=0.0, max=0.05)
        foot_height_rew = torch.sum(self.foot_swing_mask * foot_heit_score, dim=1,keepdim=True).clip(max=2.) * self.static_flag

        foot_height_rew += -20. * torch.sum((self.foot_height - 0.06).clip(min=0.), dim=1, keepdim=True)
        foot_height_rew += -0.2 * torch.sum(self.foot_support_mask * foot_heit_score, dim=1,keepdim=True) * self.static_flag
        foot_height_rew += -0.2 * torch.sum(support_foot_index * foot_heit_score, dim=1, keepdim=True) * self.static_flag

        twist_rew = -torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)

        self.foot_frc_acc = (self.env.foot_frc - self.last_foot_frc).clone()
        foot_soft_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.5) * torch.norm(self.foot_frc_acc, dim=1, keepdim=True) / 100.

        self.last_foot_frc = self.env.foot_frc.clone().detach()

        feet_contact_frc_rew = -torch.norm(self.env.foot_frc * self.foot_swing_mask, dim=1, keepdim=True) * self.static_flag
        feet_contact_frc_rew += -torch.norm((torch.abs(self.env.foot_frc - 55.) * support_foot_index).clip(min=0.), dim=1, keepdim=True)

        clip_foot_h = torch.abs(self.foot_height) + 0.03

        foot_slip_rew = 2. * (lin_vel_x_norm * torch.sum(
            (self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, 0]) * self.commands[:, [0]].sign() * self.foot_swing_mask,
            dim=1, keepdim=True)).clip(min=-0., max=1.) * self.static_flag

        foot_slip_rew += -0.5 * torch.norm(torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [1]], dim=-1), dim=1,
                                           keepdim=True) * self.static_flag

        foot_slip_rew += 0.3 * torch.norm(torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, :2], dim=-1), dim=1, keepdim=True) * (
                self.static_flag - 1.)

        foot_slip_rew += -0.3 / lin_vel_x_norm * torch.norm(
            0.1 * torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, :2], dim=-1) / clip_foot_h * self.foot_support_mask, dim=1,
            keepdim=True) * self.static_flag

        foot_vz_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(
            torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [2]].clip(max=0.), dim=-1) / clip_foot_h,
            dim=1, keepdim=True) * self.static_flag

        foot_vz_rew += 0.8 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(
            torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [2]].clip(max=0.), dim=-1),
            dim=1, keepdim=True) * (self.static_flag - 1.)

        foot_acc_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.norm(self.env.foot_vel[:, [2, 5]], dim=1, keepdim=True)

        action_smooth_rew = -0.3 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.norm(
            self.action_history[-3] - 2. * self.action_history[-2] + self.action_history[-1], dim=1, keepdim=True)
        net_out_smooth_rew = -0.2 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.norm(
            (self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1])[:, self.num_legs:],dim=1,keepdim=True) ** 2

        action_constraint_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, 0, 1.) * torch.norm((self.current_joint_act - self.ref_joint_action), dim=1, keepdim=True)
        action_constraint_rew += -3. * torch.norm(((self.current_joint_act - self.ref_joint_action)[:, [0, 1, 5, 6]]), dim=1, keepdim=True) * self.static_flag

        sa_constraint_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.current_joint_act - self.ref_joint_action, dim=1,keepdim=True) ** 2 * self.static_flag

        sa_constraint_rew += -self.static_flag * torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(
            ((self.env.joint_pos - self.ref_joint_action)[:, :5] * support_foot_index[:, [0]]), dim=1,
            keepdim=True) ** 2
        sa_constraint_rew += -self.static_flag * torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(
            ((self.env.joint_pos - self.ref_joint_action)[:, 5:] * support_foot_index[:, [1]]), dim=1,
            keepdim=True) ** 2

        joint_pos_error_rew = - 0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm((self.current_joint_act - self.env.joint_pos), dim=1,keepdim=True) ** 2

        joint_velocity_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.env.joint_vel[:, :], dim=1,keepdim=True) ** 2
        joint_velocity_rew += -torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(self.env.joint_vel[:, [0, 1, 5, 6]], dim=1,keepdim=True) ** 2

        joint_tor_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.sum(
            (torch.abs(self.env.react_tau[:, :]) - self.env.torque_limits[:]).clip(min=0.), dim=1, keepdim=True)

        joint_tor_rew *= self.static_flag

        self.last_foot_vel = self.env.foot_vel.clone().detach()
        pmf_rew = -0.02 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(
            (self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1])[:, :self.num_legs],
            dim=1, keepdim=True)
        pmf_rew += -0.5 * torch.clip(1 / lin_vel_x_norm, 0, 1.) * torch.norm(self.net_out_history[-1][:, :self.num_legs] * self.foot_support_mask, dim=1,keepdim=True) ** 2
        pmf_rew *= self.static_flag

        net_out_val_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.net_out_history[-1][:, self.num_legs:], dim=1,keepdim=True) ** 2
        foot_py_rew = -0.5 * (torch.norm(self.env.foot_euler[:, [1, 4]], dim=1, keepdim=True))

        leg_width_rew = -torch.norm(torch.abs(self.env.foot_pos_hd[:, [1, 4]] - self.env.base_pos_hd[:, [1]]) - 0.14, dim=1, keepdim=True)

        lsin = torch.sin(self.foot_phase.clone())
        lcos = torch.cos(self.foot_phase.clone())
        foot_phase_rew = -torch.norm(lsin[:, [0]] + lsin[:, [1]], dim=1, keepdim=True) ** 2
        foot_phase_rew += -torch.norm(lcos[:, [0]] + lcos[:, [1]], dim=1, keepdim=True) ** 2
        foot_phase_rew *= self.static_flag

        rew_dict = dict(
            constant=constant_rew * 0.3,
            base_heit=base_heit_rew,
            balance=balance_rew * 1.5,
            fwd_vel=forward_vel_rew * 2.3,
            yaw_rat=yaw_rate_rew * 2.5,
            lateral_vel=lateral_vel_rew * 0.7,
            vertical_vel=vertical_vel_rew * 0.6,
            ang_vel=ang_vel_rew * 0.6,
            twist=twist_rew * 2.5,
            base_acc=base_acc_rew * balance_rew * 0.1,
            foot_clr=foot_clear_rew * 1.,
            foot_supt=foot_support_rew * 0.7,
            foot_heit=foot_height_rew * 0.7,
            leg_width_rew=leg_width_rew * balance_rew * 0.5,
            act_const=action_constraint_rew * balance_rew * 0.2,
            sa_const=sa_constraint_rew * balance_rew * 0.1,
            foot_phase=foot_phase_rew * balance_rew * 0.3,
            jnt_pos_err=joint_pos_error_rew * balance_rew * 0.2,
            act_smo=action_smooth_rew * balance_rew * 1.5,
            net_smo=net_out_smooth_rew * balance_rew * 0.001,
            net_out_val=net_out_val_rew * balance_rew * 0.0001,
            foot_slip=foot_slip_rew * balance_rew * 0.5,
            foot_vz=foot_vz_rew * 0.2 * balance_rew,
            foot_acc=foot_acc_rew * balance_rew * 0.05,
            foot_sft=foot_soft_rew * 2.7 * balance_rew,
            jnt_vel=joint_velocity_rew * balance_rew * 0.003,
            feet_py=foot_py_rew * balance_rew * 0.5,
            feet_frc=feet_contact_frc_rew * 0.001,
            joint_tor=joint_tor_rew  * 0.001,
            pmf=pmf_rew * balance_rew * 0.03
        )
        if self.debug:
            self.rew_names = [name for name in rew_dict.keys()]
            self.debug = None
        if self.rew_names is None:
            self.rew_names = list(rew_dict.keys())
        rewards = torch.cat(
            [torch.clip(value.to(self.device), min=-4., max=5.) * self.env.dt for value in rew_dict.values()], dim=1)
        self._last_rew_components = rewards.detach()
        eval_rew = torch.cat([rew_dict[key] * self.env.dt for key in
                              ['fwd_vel', 'yaw_rat', 'ang_vel', 'lateral_vel', 'vertical_vel', 'twist']],
                             dim=1).sum(dim=1)
        return rewards, eval_rew
