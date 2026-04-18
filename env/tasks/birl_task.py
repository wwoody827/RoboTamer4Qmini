from math import pi, sin, cos, exp, tau
import numpy as np
from scipy.linalg import toeplitz
from collections import OrderedDict
from env.legged_robot import LeggedRobotEnv
from env.utils.math import wrap_to_pi, smallest_signed_angle_between
from env.utils.phase_modulator import PhaseModulator, ExternalPhaseClock
from env.tasks.null_task import NullTask, register
from isaacgym.torch_utils import *  # includes to_torch, torch_rand_float
from scipy.spatial.transform import Rotation as R
from env.tasks.base_task import BaseTask
from env.obs_builder import ObsBuilder
import random
from env.utils.math import scale_transform, smallest_signed_angle_between_torch
from collections import deque
import statistics
import torch


@register
class BIRLTask(BaseTask):

    # Class-level defaults so pure_observation() works when called from
    # BaseTask.__init__ before our __init__ has finished setting instance attrs.
    _has_ref = False
    _phase_mode = 'output'
    _foot_mask_mode = 'phase'
    _action_mode = 'increment'
    _ext_clock = None

    def __init__(self, env: LeggedRobotEnv):
        self.obs_builder = None  # set before super().__init__ which calls pure_observation()
        super(BIRLTask, self).__init__(env)
        self.env = env
        self.cmd_id = 0
        self.rew_names = None
        self.num_envs = env.num_envs
        self.num_legs = 2
        self.rew_weights = self.cfg.reward.to_dict() if self.cfg.reward is not None else {}

        # --- Phase mode (explicit config, not inferred from action dim) ---
        _phase_cfg = self.cfg.phase
        self._phase_mode = _phase_cfg.mode if _phase_cfg is not None and _phase_cfg.mode is not None else 'output'
        assert self._phase_mode in ('output', 'input', 'none'), \
            f"Unknown phase.mode: '{self._phase_mode}'. Must be 'output', 'input', or 'none'."

        # --- Foot mask mode ---
        self._foot_mask_mode = getattr(self.cfg.task, 'foot_mask_mode', 'phase')
        assert self._foot_mask_mode in ('phase', 'contact'), \
            f"Unknown foot_mask_mode: '{self._foot_mask_mode}'. Must be 'phase' or 'contact'."

        self.commands = torch.zeros(self.num_envs, self.cfg.command.num_commands, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # x vel, y vel, yaw vel, heading

        self.command_cfgs = self.cfg.command.to_dict()
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

        # Phase modulator — always created (for pm_phase/pm_f tensors), but only
        # used in action/obs when _phase_mode == 'output'.
        self.phase_modulator = PhaseModulator(time_step=env.dt, num_envs=self.num_envs, num_legs=self.num_legs,device=self.device)
        self.phase_modulator.reset(convert_phi=self.convert_phi, env_ids=torch.arange(self.num_envs),
                                   render=self.env.render or self.env.debug or self.env.epochs > 1 or self.env.tcn_name is not None)
        self.foot_phase = self.phase_modulator.phase
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)

        # External phase clock (BD_X style, phase.mode == 'input')
        if self._phase_mode == 'input':
            _base_freq = getattr(_phase_cfg, 'base_freq', 1.0) or 1.0
            _vel_scale = getattr(_phase_cfg, 'vel_scale', 1.0) or 1.0
            self._ext_clock = ExternalPhaseClock(
                dt=env.dt, num_envs=self.num_envs, num_legs=self.num_legs,
                device=self.device, base_freq=_base_freq, vel_scale=_vel_scale,
            )
            self._ext_clock.reset(torch.arange(self.num_envs, device=self.device),
                                  render=self.env.render or self.env.debug)
        else:
            self._ext_clock = None

        # --- Action mode: increment or absolute (BD_X style) ---
        self._action_mode = self.cfg.action.action_mode
        assert self._action_mode in ('increment', 'absolute'), \
            f"Unknown action.action_mode: '{self._action_mode}'. Must be 'increment' or 'absolute'."

        self._lp_alpha = getattr(self.cfg.action, 'action_lowpass_alpha', 1.0)

        if self._action_mode == 'increment':
            self.action_low = to_torch(self.cfg.action.inc_low_ranges, device=self.device)
            self.action_high = to_torch(self.cfg.action.inc_high_ranges, device=self.device)
        else:
            # Absolute mode: scale network output to joint position range.
            # Use abs_low/high_ranges if set, otherwise fall back to URDF limits.
            abs_low = getattr(self.cfg.action, 'abs_low_ranges', None)
            abs_high = getattr(self.cfg.action, 'abs_high_ranges', None)
            dof_low = self.env.dof_pos_limits[:, 0]
            dof_high = self.env.dof_pos_limits[:, 1]
            joint_low = to_torch(abs_low, device=self.device) if abs_low else torch.as_tensor(dof_low, device=self.device)
            joint_high = to_torch(abs_high, device=self.device) if abs_high else torch.as_tensor(dof_high, device=self.device)
            if self._phase_mode == 'output':
                # 12-dim: [freq_low(2), joint_low(10)]
                freq_low = to_torch(self.cfg.action.low_ranges[:self.num_legs], device=self.device)
                freq_high = to_torch(self.cfg.action.high_ranges[:self.num_legs], device=self.device)
                self.action_low = torch.cat([freq_low, joint_low])
                self.action_high = torch.cat([freq_high, joint_high])
            else:
                # 10-dim: joints only
                self.action_low = joint_low
                self.action_high = joint_high

        # Validate action dim matches phase mode
        _action_dim = len(self.action_low)
        if self._phase_mode == 'output':
            assert _action_dim == self.env.num_dofs + self.num_legs, \
                f"phase.mode=output requires {self.env.num_dofs + self.num_legs}-dim action, got {_action_dim}"
        else:
            assert _action_dim == self.env.num_dofs, \
                f"phase.mode={self._phase_mode} requires {self.env.num_dofs}-dim action, got {_action_dim}"

        self.current_joint_act = to_torch(self.env.default_dof_pos, device=self.device).repeat(self.num_envs, 1)
        self.previous_joint_act = self.current_joint_act.clone()
        # Lowpass filter target (absolute mode only)
        self._lp_target = self.current_joint_act.clone()

        self.ref_joint_action = to_torch(self.cfg.action.ref_joint_pos, device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_low_over = torch.as_tensor(self.env.dof_pos_limits[:, 0]).repeat(self.num_envs, 1)
        self.joint_action_limit_high_over = torch.as_tensor(self.env.dof_pos_limits[:, 1]).repeat(self.num_envs, 1)

        self.joint_action_limit_low = torch.as_tensor(self.env.dof_pos_limits[:, 0], device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_high = torch.as_tensor(self.env.dof_pos_limits[:, 1], device=self.device).repeat(self.num_envs, 1)

        obs_cfg = self.cfg.observation
        obs_history_len = obs_cfg.history if obs_cfg is not None and obs_cfg.history is not None else 3
        self.obs_history = deque(maxlen=obs_history_len)
        self.cri_obs_history = deque(maxlen=obs_history_len)

        # Build obs from config slots (falls back to hardcoded if no config)
        if obs_cfg is not None and obs_cfg.slots is not None:
            self.obs_builder = ObsBuilder(self, slot_names=obs_cfg.slots)
        else:
            self.obs_builder = None

        self._use_act_delay  = getattr(self.cfg.action, 'use_actuator_delay',  False)
        self._use_act_filter = getattr(self.cfg.action, 'use_actuator_filter', False)

        _delay_range = getattr(self.cfg.action, 'actuator_delay_range', [1, 3])
        _delay_max = max(_delay_range) if self._use_act_delay else 0
        self._act_delay_steps = random.randint(*_delay_range) if self._use_act_delay else 0

        self.action_history = deque(maxlen=3 + _delay_max)
        self.net_out_history = deque(maxlen=3)

        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_joint_act)

        # joint_act_for_pd: what actually gets sent to the PD controller (after delay + filter)
        self.joint_act_for_pd = self.current_joint_act.clone()
        if self._use_act_filter:
            _alpha_range = getattr(self.cfg.action, 'actuator_filter_alpha_range', [0.3, 0.7])
            self.act_filter_alpha = torch.FloatTensor(self.num_envs, 1).uniform_(*_alpha_range).to(self.device)
        else:
            self.act_filter_alpha = None

        for _ in range(self.net_out_history.maxlen):
            self.net_out_history.append(torch.zeros_like(self.action_low).repeat(self.num_envs, 1))

        # --- Foot masks (phase-based, contact-force-based, or ext-clock-based) ---
        if self._phase_mode == 'input':
            # BD_X: derive desired swing/stance from external clock phase
            ext_phase = self._ext_clock.phase_with_offset  # [num_envs, num_legs]
            self.foot_support_mask = (ext_phase < self.convert_phi)
            self.foot_swing_mask = ~self.foot_support_mask
        elif self._foot_mask_mode == 'contact':
            self.foot_swing_mask = (self.env.foot_frc < 1.0)
            self.foot_support_mask = (self.env.foot_frc >= 10.0)
        else:
            foot_support_mask_1 = torch.where(self.foot_phase >= 0, True, False)
            foot_support_mask_2 = torch.where(self.foot_phase < self.convert_phi, True, False)
            self.foot_support_mask = torch.logical_and(foot_support_mask_1, foot_support_mask_2)
            self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone()

        # --- Air time tracking (always initialized; reward weight gates usage) ---
        self.foot_air_time = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self._pre_reset_air_time = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self._prev_foot_in_air = torch.zeros(self.num_envs, self.num_legs, dtype=torch.bool, device=self.device)

        # --- Reference clip state (populated by _load_ref_clips if paths provided) ---
        self._has_ref = False
        self._ref_joint_pos_now = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device)
        self._ref_joint_vel_now = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device)
        self._ref_phase_progress = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        ref_paths = getattr(self.cfg.task, 'ref_clip_paths', []) or []
        if ref_paths:
            self._load_ref_clips(ref_paths)

        self.heading_ref    = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.last_ang_vel_z = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

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

    # ------------------------------------------------------------------
    # Reference clip loading (active when cfg.task.ref_clip_paths != [])
    # ------------------------------------------------------------------

    def _load_ref_clips(self, paths):
        """Load reference clips, pad to max length, build per-env tracking tensors."""
        clips = []
        for p in paths:
            raw = np.load(p, allow_pickle=True)
            clips.append({
                'joint_pos': torch.tensor(raw['joint_pos'], dtype=torch.float, device=self.device),
                'joint_vel': torch.tensor(raw['joint_vel'], dtype=torch.float, device=self.device),
                'T':         raw['joint_pos'].shape[0],
                'dt':        float(raw['dt']),
                'loop':      bool(raw['loop']),
                'skill':     str(raw['skill']),
            })
        if not clips:
            return

        max_T = max(c['T'] for c in clips)
        num_clips = len(clips)

        # Pad all clips to max_T → [num_clips, max_T, 10]
        jp = torch.zeros(num_clips, max_T, 10, dtype=torch.float, device=self.device)
        jv = torch.zeros(num_clips, max_T, 10, dtype=torch.float, device=self.device)
        for i, c in enumerate(clips):
            jp[i, :c['T']] = c['joint_pos']
            jv[i, :c['T']] = c['joint_vel']

        self._ref_jp_all = jp                                                              # [n_clips, max_T, 10]
        self._ref_jv_all = jv                                                              # [n_clips, max_T, 10]
        self._ref_clip_lengths = torch.tensor([c['T'] for c in clips], dtype=torch.long,  # [n_clips]
                                              device=self.device)
        self._ref_num_clips = num_clips

        # Per-env clip assignment and frame index
        self._ref_clip_id  = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ref_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ref_frame_frac = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Randomly assign clips to envs
        self._assign_ref_clips(torch.arange(self.num_envs, device=self.device))
        self._update_ref_state()

        self._has_ref = True
        print(f"[BIRLTask] Loaded {num_clips} reference clip(s): "
              + ", ".join(f"{c['skill']}({c['T']} frames)" for c in clips))

    def _assign_ref_clips(self, env_ids):
        """Randomly assign one of the loaded clips to each env (RSI: random start frame)."""
        self._ref_clip_id[env_ids] = torch.randint(
            0, self._ref_num_clips, (len(env_ids),), device=self.device
        )
        lengths = self._ref_clip_lengths[self._ref_clip_id[env_ids]]
        rand_frames = (torch.rand(len(env_ids), device=self.device) * lengths.float()).long()
        self._ref_frame_idx[env_ids] = rand_frames
        self._ref_frame_frac[env_ids] = 0.0

    def _update_ref_state(self):
        """Vectorised lookup: read current ref joint_pos/vel for every env."""
        if not self._has_ref:
            return
        cid = self._ref_clip_id       # [num_envs]
        fid = self._ref_frame_idx     # [num_envs]
        self._ref_joint_pos_now = self._ref_jp_all[cid, fid]   # [num_envs, 10]
        self._ref_joint_vel_now = self._ref_jv_all[cid, fid]   # [num_envs, 10]
        lengths = self._ref_clip_lengths[cid].float()
        self._ref_phase_progress[:, 0] = fid.float() / lengths.clamp(min=1.)

    def _advance_ref_frames(self):
        """Advance each env's frame by 1 per policy step (fractional accumulation)."""
        if not self._has_ref:
            return
        self._ref_frame_frac += 1.0
        advance = self._ref_frame_frac.long()
        lengths = self._ref_clip_lengths[self._ref_clip_id]
        self._ref_frame_idx = (self._ref_frame_idx + advance) % lengths
        self._ref_frame_frac -= advance.float()
        self._update_ref_state()

    # ------------------------------------------------------------------
    # Command resampling — per-slot, supports any single direction or combined
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids):
        """Sample each command slot independently based on its configured range."""
        n = len(env_ids)
        self.commands[env_ids, :] = torch.zeros(
            n, self.cfg.command.num_commands, dtype=torch.float,
            device=self.device, requires_grad=False
        )

        vx_lo,  vx_hi  = self.command_cfgs["lin_vel_x_range"]
        vy_lo,  vy_hi  = self.command_cfgs["lin_vel_y_range"]
        yaw_lo, yaw_hi = self.command_cfgs["ang_vel_yaw_range"]

        if vx_lo != 0 or vx_hi != 0:
            self.commands[env_ids, 0] = torch_rand_float(
                vx_lo, vx_hi, (n, 1), device=self.device
            ).squeeze(1)

        if vy_lo != 0 or vy_hi != 0:
            self.commands[env_ids, 1] = torch_rand_float(
                vy_lo, vy_hi, (n, 1), device=self.device
            ).squeeze(1)

        if yaw_lo != 0 or yaw_hi != 0:
            self.commands[env_ids, 2] = torch_rand_float(
                yaw_lo, yaw_hi, (n, 1), device=self.device
            ).squeeze(1)

        # First 96 envs stand still (curriculum stability)
        if not self.fixed_commands:
            self.commands[:96, [0, 2]] = 0

        for dim, val in self.fixed_commands.items():
            self.commands[:, dim] = val

        # Update static_flag and zero commands below threshold
        self.static_flag[env_ids] = torch.where(
            torch.norm(self.commands[env_ids, :3], dim=1, keepdim=True) < 0.15,
            False, True
        ).float()
        self.commands[env_ids, :3] *= self.static_flag[env_ids]
        self.commands[env_ids, 0:1] *= torch.where(
            torch.norm(self.commands[env_ids, 0:1], dim=1, keepdim=True) < 0.15,
            False, True
        ).float()
        self.commands[env_ids, 2:3] *= torch.where(
            torch.norm(self.commands[env_ids, 2:3], dim=1, keepdim=True) < 0.15,
            False, True
        ).float()

    def reset(self, env_ids):
        self.base_acc[env_ids] = self.env.base_acc_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_vel[env_ids] = self.env.joint_vel_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_pos[env_ids] = self.env.joint_pos_his.delay(self.delay_joint_steps)[env_ids]
        self.current_joint_act[env_ids] = self.env.default_dof_pos
        self.previous_joint_act[env_ids] = self.current_joint_act[env_ids].clone()
        self._lp_target[env_ids] = self.current_joint_act[env_ids].clone()

        self.joint_pos_error = self.joint_act_for_pd - self.joint_pos
        self.phase_modulator.reset(convert_phi=self.convert_phi, env_ids=env_ids,
                                   render=self.env.render or self.env.epochs > 1 or self.env.tcn_name is not None)
        if self._ext_clock is not None:
            self._ext_clock.reset(env_ids, render=self.env.render)
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)
        self.static_flag = torch.where(torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.15, False, True).float()
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        self._resample_commands(env_ids)

        # Foot masks: ext-clock, contact-force, or phase-based
        if self._phase_mode == 'input':
            ext_phase = self._ext_clock.phase_with_offset
            self.foot_support_mask = (ext_phase < self.convert_phi)
            self.foot_swing_mask = ~self.foot_support_mask
        elif self._foot_mask_mode == 'contact':
            self.foot_swing_mask = (self.env.foot_frc < 1.0)
            self.foot_support_mask = (self.env.foot_frc >= 10.0)
        else:
            foot_support_mask_1 = torch.where(self.foot_phase >= 0, True, False)
            foot_support_mask_2 = torch.where(self.foot_phase < self.convert_phi, True, False)
            self.foot_support_mask = torch.logical_and(foot_support_mask_1, foot_support_mask_2)
            self.foot_swing_mask = torch.logical_not(self.foot_support_mask)

        self.pm_f = self.phase_modulator.frequency.clone()
        self.heading_ref[env_ids]    = self.env.base_euler[env_ids, 2].unsqueeze(-1)
        self.last_ang_vel_z[env_ids] = self.env.base_ang_vel[env_ids, 2].unsqueeze(-1)
        self.joint_act_for_pd[env_ids] = self.current_joint_act[env_ids]
        if self._use_act_filter:
            _alpha_range = getattr(self.cfg.action, 'actuator_filter_alpha_range', [0.3, 0.7])
            self.act_filter_alpha[env_ids] = torch.FloatTensor(len(env_ids), 1).uniform_(*_alpha_range).to(self.device)

        # Air time reset
        self.foot_air_time[env_ids] = 0.0

        # RSI: assign new clip and randomise start frame for reset envs
        if self._has_ref:
            self._assign_ref_clips(env_ids)
            self._update_ref_state()

    def step(self):
        self.joint_pos = self.env.joint_pos_his.delay(self.delay_joint_steps)
        self.joint_vel = self.env.joint_vel_his.delay(self.delay_joint_steps)
        self.base_acc = self.env.base_acc_his.delay(self.delay_rate_steps).clip(min=-30., max=30.)
        self.joint_tau = self.env.joint_tau_his.delay(1)

        self.joint_pos_error = self.joint_act_for_pd - self.joint_pos
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

        # Foot masks: ext-clock, contact-force, or phase-based
        if self._phase_mode == 'input':
            ext_phase = self._ext_clock.phase_with_offset
            self.foot_support_mask = (ext_phase < self.convert_phi)
            self.foot_swing_mask = ~self.foot_support_mask
        elif self._foot_mask_mode == 'contact':
            self.foot_swing_mask = (self.env.foot_frc < 1.0)
            self.foot_support_mask = (self.env.foot_frc >= 10.0)
        else:
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
        self.last_ang_vel_z = self.base_ang_vel[:, [2]].clone()
        if self._use_act_delay and self.env.common_step_counter % 200 == 0:
            self._act_delay_steps = random.randint(*getattr(self.cfg.action, 'actuator_delay_range', [1, 3]))

        # External phase clock: advance from velocity command
        if self._ext_clock is not None:
            cmd_vel_norm = torch.norm(self.commands[:, :2], dim=1, keepdim=True)
            self._ext_clock.update(cmd_vel_norm)

        # Air time accumulation: always uses actual contact forces (not clock masks),
        # so it measures real time in the air.
        actual_in_air = (self.env.foot_frc < 1.0)
        self.foot_air_time += self.env.dt
        self._pre_reset_air_time = self.foot_air_time.clone()  # snapshot before reset
        self.foot_air_time *= actual_in_air.float()            # reset to 0 on ground contact

        # Advance reference frame
        self._advance_ref_frames()


    def observation(self):
        self.obs_buf_pure = self.pure_observation()
        self.obs_history.append(self.obs_buf_pure)
        return  torch.cat([obs for obs in self.obs_history], dim=-1)

    def critic_observation(self):
        pure_obs_buf = self.pure_critic_observation()
        self.cri_obs_history.append(pure_obs_buf)
        return  torch.cat([obs for obs in self.cri_obs_history], dim=-1)

    def pure_critic_observation(self):
        _jo = self.num_legs if self._phase_mode == 'output' else 0  # joint offset in net_out
        _cmd = self.commands[:, [0,1,2]] if self._phase_mode == 'output' else self.commands[:, :8]

        parts = [
            _cmd,
            self.commands[:, [0]] - self.env.base_lin_vel[:, [0]],
            self.commands[:, [1]] - self.env.base_lin_vel[:, [1]],
            self.commands[:, [2]] - self.env.base_ang_vel[:, [2]],
            self.env.base_lin_vel,
            self.env.base_euler[:, :2],
            self.env.base_ang_vel * 0.5,
            (self.env.joint_pos - self.ref_joint_action),
            self.env.joint_vel * 0.1,
            (self.current_joint_act - self.ref_joint_action),
            self.joint_pos_error,
        ]

        # Phase slots only for output mode
        if self._phase_mode == 'output':
            parts.extend([
                self.pm_phase * self.static_flag,
                (self.pm_f * 0.3 - 1.) * self.static_flag,
            ])

        parts.extend([
            self.net_out_history[-1][:, _jo:] / 15.,
            self.foot_height.clip(min=-0.5, max=0.5) * 10.,
            (self.env.base_pos_hd[:, [2]] - 0.4) * 10.,
            self.env.foot_vel.clip(min=-8., max=8.) * 0.5,
            self.env.base_acc.clip(min=-20., max=20.) * 0.2,
            self.env.foot_frc.clip(min=0., max=200.) * 0.01,
            self.base_euler[:, :2] * 1.,
            self.base_ang_vel * 0.5,
            self.joint_pos - self.ref_joint_action,
            self.joint_vel * 0.1,
            self.joint_pos_error,
        ])

        obs_buf = torch.cat(parts, dim=1)
        return obs_buf

    def pure_observation(self):
        if self.obs_builder is not None:
            self.obs_buf = self.obs_builder.build()
            return self.obs_buf
        # Legacy fallback (no observation.slots in config)
        if self._phase_mode == 'output':
            # BIRL legacy: 44-dim
            parts = [
                self.commands[:, [0,1,2]],
                self.base_euler[:, :2] * 1.,
                self.base_ang_vel * 0.5,
                self.joint_pos - self.ref_joint_action,
                self.joint_vel * 0.1,
                self.joint_pos_error,
                self.pm_phase * self.static_flag,
                (self.pm_f * 0.3 - 1.) * self.static_flag,
            ]
            if getattr(self.cfg.task, 'use_teacher_obs', False):
                parts.append(self.base_lin_vel)
        else:
            # MIRL legacy: 64-dim (no phase slots, ref clip slots instead)
            if self._has_ref:
                ref_pos_err = (self._ref_joint_pos_now - self.joint_pos).clip(-3., 3.)
                ref_vel     = self._ref_joint_vel_now.clip(-3., 3.)
                ref_slots   = torch.cat([ref_pos_err, ref_vel, self._ref_phase_progress], dim=1)  # 21
            else:
                ref_slots = torch.zeros(self.num_envs, 21, dtype=torch.float, device=self.device)
            parts = [
                self.commands[:, :8],
                self.base_euler[:, :2] * 1.0,
                self.base_ang_vel * 0.5,
                self.joint_pos - self.ref_joint_action,
                self.joint_vel * 0.1,
                self.joint_pos_error,
                ref_slots,
            ]
        self.obs_buf = torch.cat(parts, dim=1).clip(min=-3., max=3.)
        return self.obs_buf

    def action(self, net_out):
        net_out = scale_transform(net_out, self.action_low, self.action_high)
        self.net_out_history.append(net_out)

        # Phase modulator: only consume freq prefix when phase.mode == 'output'
        if self._phase_mode == 'output':
            self.phase_modulator.compute(net_out[:, :self.num_legs])
            joint_out = net_out[:, self.num_legs:]
        else:
            joint_out = net_out  # all dims are joint outputs

        if self.env.render and self.env.common_step_counter <= 1:
            pass
        else:
            if self._action_mode == 'increment':
                self.current_joint_act += joint_out * self.env.dt
            else:
                # Absolute mode: lowpass filter the raw position target
                if self._lp_alpha < 1.0:
                    self._lp_target = self._lp_alpha * joint_out + (1.0 - self._lp_alpha) * self._lp_target
                else:
                    self._lp_target = joint_out
                self.current_joint_act = self._lp_target
        self.current_joint_act = torch.clip(self.current_joint_act, self.joint_action_limit_low,self.joint_action_limit_high)
        self.action_history.append(self.current_joint_act.clone())
        self.previous_joint_act = self.current_joint_act.clone()

        # --- actuator lag ---
        if self._use_act_delay:
            delayed = self.action_history[-(self._act_delay_steps + 1)]
        else:
            delayed = self.current_joint_act

        if self._use_act_filter:
            self.joint_act_for_pd = self.act_filter_alpha * self.joint_act_for_pd + (1. - self.act_filter_alpha) * delayed
        else:
            self.joint_act_for_pd = delayed
        # --------------------

        return self.joint_act_for_pd

    def reward(self):
        constant_rew = to_torch([1.]).repeat(self.num_envs, 1)
        lin_vel_x_norm = torch.clip(torch.norm(self.commands[:, [0, 1]], dim=1, keepdim=True), min=0.3, max=2.) + 0.2
        yaw_rate_norm = torch.clip(torch.abs(self.commands[:, [2]]), min=0.3, max=1.5) + 0.2
        base_heit_rew = torch.exp(-70 * (self.env.base_pos[:, [2]] - 0.45) ** 2)

        balance_rew = 0.5 * (base_heit_rew * torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2, max=8.) * torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)) + 1.)

        forward_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2., max=10.) * (
                self.commands[:, [0]] - self.env.base_lin_vel[:, [0]]) ** 2) #* balance_rew
        lateral_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=3., max=15.) * (self.commands[:, [1]] - self.env.base_lin_vel[:, [1]]) ** 2)

        yaw_rate_rew = torch.exp(-torch.clip(2. / yaw_rate_norm, min=2., max=6.) * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2)

        lateral_vel_rew += -0.6 / lin_vel_x_norm * torch.abs(self.commands[:, [1]] - self.env.base_lin_vel[:, [1]]) * self.static_flag

        ang_vel_rew = torch.exp(
            -torch.clip(2. / lin_vel_x_norm, min=0.7, max=6.) * torch.norm(self.env.base_ang_vel[:, :2], dim=1,
                                                                            keepdim=True) ** 2)
        base_acc_rew = -0.4 / lin_vel_x_norm * torch.norm((self.env.base_acc - to_torch([0, 0, 9.81], device=self.device)) * 0.1, dim=1, keepdim=True)
        base_acc_rew *= self.static_flag

        vertical_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2., max=10.) * torch.norm(self.env.base_lin_vel[:, [2]], dim=1,
                                                                           keepdim=True) ** 2)
        vertical_vel_rew -= 0.2 / lin_vel_x_norm * torch.norm(self.env.base_lin_vel[:, [2]], dim=1, keepdim=True) * self.static_flag

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

        vy_walking = (torch.abs(self.commands[:, [1]]) > 0.1).float()
        foot_slip_rew += -0.5 * torch.norm(torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [1]], dim=-1), dim=1,
                                           keepdim=True) * self.static_flag * (1. - vy_walking)

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
        # net_out smoothness: skip freq prefix when phase.mode == 'output'
        _jo = self.num_legs if self._phase_mode == 'output' else 0  # joint offset
        net_out_smooth_rew = -0.2 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.norm(
            (self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1])[:, _jo:], dim=1, keepdim=True) ** 2

        action_constraint_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, 0, 1.) * torch.norm((self.current_joint_act - self.ref_joint_action), dim=1, keepdim=True)
        action_constraint_rew += -3. * torch.norm(((self.current_joint_act - self.ref_joint_action)[:, [0, 1, 5, 6]]), dim=1, keepdim=True) * self.static_flag * (1. - vy_walking)

        sa_constraint_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.current_joint_act - self.ref_joint_action, dim=1,keepdim=True) ** 2 * self.static_flag

        sa_constraint_rew += -self.static_flag * torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(
            ((self.env.joint_pos - self.ref_joint_action)[:, :5] * support_foot_index[:, [0]]), dim=1,
            keepdim=True) ** 2
        sa_constraint_rew += -self.static_flag * torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(
            ((self.env.joint_pos - self.ref_joint_action)[:, 5:] * support_foot_index[:, [1]]), dim=1,
            keepdim=True) ** 2

        joint_pos_error_rew = - 0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm((self.current_joint_act - self.env.joint_pos), dim=1,keepdim=True) ** 2

        joint_velocity_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.env.joint_vel[:, :], dim=1,keepdim=True) ** 2
        joint_velocity_rew += -torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(self.env.joint_vel[:, [0, 1, 5, 6]], dim=1,keepdim=True) ** 2 * (1. - vy_walking)

        joint_tor_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.sum(
            (torch.abs(self.env.react_tau[:, :]) - self.env.torque_limits[:]).clip(min=0.), dim=1, keepdim=True)

        joint_tor_rew *= self.static_flag

        self.last_foot_vel = self.env.foot_vel.clone().detach()

        # PMF reward: only meaningful when phase.mode == 'output' (freq prefix exists)
        if self._phase_mode == 'output':
            pmf_rew = -0.02 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(
                (self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1])[:, :self.num_legs],
                dim=1, keepdim=True)
            pmf_rew += -0.5 * torch.clip(1 / lin_vel_x_norm, 0, 1.) * torch.norm(self.net_out_history[-1][:, :self.num_legs] * self.foot_support_mask, dim=1, keepdim=True) ** 2
            pmf_rew *= self.static_flag
        else:
            pmf_rew = torch.zeros(self.num_envs, 1, device=self.device)

        net_out_val_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.net_out_history[-1][:, _jo:], dim=1, keepdim=True) ** 2
        foot_py_rew = -0.5 * (torch.norm(self.env.foot_euler[:, [1, 4]], dim=1, keepdim=True))

        leg_width_rew = -torch.norm(torch.abs(self.env.foot_pos_hd[:, [1, 4]] - self.env.base_pos_hd[:, [1]]) - 0.14, dim=1, keepdim=True)

        # Foot phase reward: penalizes legs being in-phase (should be anti-phase)
        if self._phase_mode == 'output':
            lsin = torch.sin(self.foot_phase.clone())
            lcos = torch.cos(self.foot_phase.clone())
            foot_phase_rew = -torch.norm(lsin[:, [0]] + lsin[:, [1]], dim=1, keepdim=True) ** 2
            foot_phase_rew += -torch.norm(lcos[:, [0]] + lcos[:, [1]], dim=1, keepdim=True) ** 2
            foot_phase_rew *= self.static_flag
        elif self._phase_mode == 'input':
            # BD_X: external clock guarantees anti-phase. Reward matching
            # actual foot contact to clock-defined swing/stance schedule.
            # foot_swing_mask is already set from ext_clock above.
            actual_swing = (self.env.foot_frc < 1.0)     # feet actually in air
            actual_support = (self.env.foot_frc >= 10.0)  # feet actually on ground
            phase_match = (
                torch.sum((actual_swing & self.foot_swing_mask).float(), dim=1, keepdim=True)
                + torch.sum((actual_support & self.foot_support_mask).float(), dim=1, keepdim=True)
            ) / self.num_legs - 1.0  # normalized: 0 when no match, 1 when perfect
            foot_phase_rew = phase_match * self.static_flag
        else:
            foot_phase_rew = torch.zeros(self.num_envs, 1, device=self.device)

        # Air time reward: fires at actual touchdown, proportional to real time airborne
        actual_in_air = (self.env.foot_frc < 1.0)
        actual_on_ground = (self.env.foot_frc >= 10.0)
        just_landed = self._prev_foot_in_air & actual_on_ground
        air_time_rew = torch.sum(
            (self._pre_reset_air_time - 0.1).clip(min=0.) * just_landed.float(),
            dim=1, keepdim=True
        ) * self.static_flag
        self._prev_foot_in_air = actual_in_air.clone()

        # Mechanical power penalty: |torque × joint_vel|
        power_rew = -torch.sum(
            torch.abs(self.env.torques * self.env.joint_vel), dim=1, keepdim=True
        ) / 100.

        if self.rew_weights.get('yaw_smooth', 0.0) > 0:
            yaw_smooth_rew = -torch.abs(
                self.base_ang_vel[:, [2]] - self.last_ang_vel_z
            ) * self.static_flag
        else:
            yaw_smooth_rew = torch.zeros(self.num_envs, 1, device=self.device)

        if self.rew_weights.get('heading', 0.0) > 0:
            heading_err = wrap_to_pi(self.env.base_euler[:, [2]] - self.heading_ref)
            heading_rew = torch.exp(-3.0 * heading_err ** 2)
            heading_rew *= (self.commands[:, [2]].abs() < 0.05).float() * self.static_flag
        else:
            heading_rew = torch.zeros(self.num_envs, 1, device=self.device)

        w = self.rew_weights
        rew_dict = dict(
            constant=constant_rew * w.get('constant', 0.3),
            base_heit=base_heit_rew * w.get('base_heit', 1.0),
            balance=balance_rew * w.get('balance', 1.5),
            fwd_vel=forward_vel_rew * w.get('fwd_vel', 2.3),
            yaw_rat=yaw_rate_rew * w.get('yaw_rat', 2.5),
            lateral_vel=lateral_vel_rew * w.get('lateral_vel', 0.7),
            vertical_vel=vertical_vel_rew * w.get('vertical_vel', 0.6),
            ang_vel=ang_vel_rew * w.get('ang_vel', 0.6),
            twist=twist_rew * w.get('twist', 2.5),
            base_acc=base_acc_rew * balance_rew * w.get('base_acc', 0.1),
            foot_clr=foot_clear_rew * w.get('foot_clr', 1.0),
            foot_supt=foot_support_rew * w.get('foot_supt', 0.7),
            foot_heit=foot_height_rew * w.get('foot_heit', 0.7),
            leg_width_rew=leg_width_rew * balance_rew * w.get('leg_width_rew', 0.5),
            act_const=action_constraint_rew * balance_rew * w.get('act_const', 0.2),
            sa_const=sa_constraint_rew * balance_rew * w.get('sa_const', 0.1),
            foot_phase=foot_phase_rew * w.get('foot_phase', 0.3),
            jnt_pos_err=joint_pos_error_rew * balance_rew * w.get('jnt_pos_err', 0.2),
            act_smo=action_smooth_rew * balance_rew * w.get('act_smo', 1.5),
            net_smo=net_out_smooth_rew * balance_rew * w.get('net_smo', 0.001),
            net_out_val=net_out_val_rew * balance_rew * w.get('net_out_val', 0.0001),
            foot_slip=foot_slip_rew * balance_rew * w.get('foot_slip', 0.5),
            foot_vz=foot_vz_rew * balance_rew * w.get('foot_vz', 0.2),
            foot_acc=foot_acc_rew * balance_rew * w.get('foot_acc', 0.05),
            foot_sft=foot_soft_rew * balance_rew * w.get('foot_sft', 2.7),
            jnt_vel=joint_velocity_rew * balance_rew * w.get('jnt_vel', 0.003),
            feet_py=foot_py_rew * balance_rew * w.get('feet_py', 0.5),
            feet_frc=feet_contact_frc_rew * w.get('feet_frc', 0.001),
            joint_tor=joint_tor_rew * w.get('joint_tor', 0.001),
            pmf=pmf_rew * balance_rew * w.get('pmf', 0.03),
            heading=heading_rew * w.get('heading', 0.0),
            yaw_smooth=yaw_smooth_rew * w.get('yaw_smooth', 0.0),
            power=power_rew * w.get('power', 0.0),
            air_time=air_time_rew * w.get('air_time', 0.0),
        )

        # Imitation rewards (only active when reference clips are loaded)
        if self._has_ref:
            w_imit = getattr(self.cfg.task, 'w_imit', 0.5)
            w_task = getattr(self.cfg.task, 'w_task', 0.5)

            jp_imit = torch.exp(
                -5.0 * torch.norm(self.joint_pos - self._ref_joint_pos_now, dim=1, keepdim=True) ** 2
            )
            jv_imit = torch.exp(
                -0.1 * torch.norm(self.joint_vel - self._ref_joint_vel_now, dim=1, keepdim=True) ** 2
            )
            imit_rew = (jp_imit + jv_imit) * 0.5

            # Scale task rewards by w_task, add imitation by w_imit
            rew_dict = {k: v * w_task for k, v in rew_dict.items()}
            rew_dict['jp_imit'] = imit_rew * w_imit * 2.0
            rew_dict['jv_imit'] = jv_imit * w_imit * 0.5

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
