"""
MIRLTask — Motion Imitation RL task for Qmini.

Key differences from BIRLTask:
  - 10-dim action (no leg-frequency outputs)
  - 64-dim observation per step × 3 history = 192 total
  - foot_swing/support masks from contact forces, not phase modulator
  - Per-slot command resampling (supports forward-only, strafe-only, turn-only, combined)
  - Optional reference clip support:
      ref_clip_paths = []     → pure RL (obs slots 43-63 = zero)
      ref_clip_paths = [...]  → MIRL (imitation reward + RSI + obs slots populated)

Observation layout (64-dim per step):
  [0-7]   8 command slots: [vx, vy, yaw, height, 0, 0, 0, 0]
  [8-9]   roll, pitch
  [10-12] angular velocity × 0.5
  [13-22] joint_pos − ref_joint_pos
  [23-32] joint_vel × 0.1
  [33-42] joint_act − joint_pos  (tracking error)
  [43-52] ref_joint_pos[t] − joint_pos  (zeros if no clip)
  [53-62] ref_joint_vel[t]               (zeros if no clip)
  [63]    phase_progress 0→1             (zeros if no clip)

Reference clip format (.npz):
  joint_pos:    [T, 10]  joint_vel:    [T, 10]
  base_pos:     [T, 3]   base_quat:    [T, 4]
  base_lin_vel: [T, 3]   base_ang_vel: [T, 3]
  dt: float, source: str, skill: str, loop: bool

Training flow (walking experts):
  Step 1: MIRLTask, ref_clip_paths=[]        → mirl_fwd_v1 / mirl_strafe_v1 / mirl_turn_v1
  Step 2: collect clips via sim2sim --record from step 1 policies
  Step 3: MIRLTask, ref_clip_paths=[...]     → mirl_combined_v1 (resume or fresh)
"""

import numpy as np
import torch
from env.tasks.birl_task import BIRLTask
from env.tasks.null_task import register
from env.legged_robot import LeggedRobotEnv
from env.utils.math import scale_transform
from isaacgym.torch_utils import to_torch, torch_rand_float


@register
class MIRLTask(BIRLTask):

    # Class-level defaults so pure_observation() works when called from BaseTask.__init__
    # before MIRLTask.__init__ has finished setting up instance attributes.
    _has_ref = False

    def __init__(self, env: LeggedRobotEnv):
        # BIRLTask.__init__ sets up action_low/high (10-dim from MIRL config),
        # commands tensor (8-slot), obs_history (populated by our pure_observation()),
        # and phase_modulator (kept alive but not used for obs/action).
        super(MIRLTask, self).__init__(env)

        # Override foot masks: use contact forces, not phase modulator
        self.foot_swing_mask = (self.env.foot_frc < 1.0)
        self.foot_support_mask = (self.env.foot_frc >= 10.0)

        # Reference clip state (populated by _load_ref_clips if paths provided)
        self._has_ref = False
        self._ref_joint_pos_now = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device)
        self._ref_joint_vel_now = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device)
        self._ref_phase_progress = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        ref_paths = getattr(self.cfg.task, 'ref_clip_paths', [])
        if ref_paths:
            self._load_ref_clips(ref_paths)

    # ------------------------------------------------------------------
    # Reference clip loading
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
        print(f"[MIRLTask] Loaded {num_clips} reference clip(s): "
              + ", ".join(f"{c['skill']}({c['T']} frames)" for c in clips))

    def _assign_ref_clips(self, env_ids):
        """Randomly assign one of the loaded clips to each env."""
        self._ref_clip_id[env_ids] = torch.randint(
            0, self._ref_num_clips, (len(env_ids),), device=self.device
        )
        # RSI: start from random frame within the assigned clip's length
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
        """Advance each env's frame by policy_dt / clip_dt steps (fractional accumulation)."""
        if not self._has_ref:
            return
        # Assume all clips recorded at policy rate → advance 1 frame per policy step.
        # (clip dt ≈ policy dt because sim2sim --record records at policy rate)
        self._ref_frame_frac += 1.0   # advance by 1 frame
        advance = self._ref_frame_frac.long()
        lengths = self._ref_clip_lengths[self._ref_clip_id]
        self._ref_frame_idx = (self._ref_frame_idx + advance) % lengths
        self._ref_frame_frac -= advance.float()
        self._update_ref_state()

    # ------------------------------------------------------------------
    # Command resampling — per-slot, supports any single direction or combined
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids):
        """Sample each command slot independently based on its configured range.

        Unlike base_task._resample_commands which uses a case/switch approach
        (and crashes on turn-only configs because random.choice([]) raises),
        this version samples each slot independently if its range is non-zero.
        """
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

        # First 96 envs stand still (keep for curriculum stability)
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

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self, env_ids):
        self.base_acc[env_ids] = self.env.base_acc_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_vel[env_ids] = self.env.joint_vel_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_pos[env_ids] = self.env.joint_pos_his.delay(self.delay_joint_steps)[env_ids]
        self.current_joint_act[env_ids] = self.env.default_dof_pos
        self.previous_joint_act[env_ids] = self.current_joint_act[env_ids].clone()
        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.static_flag = torch.where(
            torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.15, False, True
        ).float()
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        self._resample_commands(env_ids)
        self.foot_swing_mask = (self.env.foot_frc < 1.0)
        self.foot_support_mask = (self.env.foot_frc >= 10.0)
        # RSI: assign new clip and randomise start frame for reset envs
        if self._has_ref:
            self._assign_ref_clips(env_ids)
            self._update_ref_state()

    def step(self):
        super(MIRLTask, self).step()
        # Override foot masks from contact forces
        self.foot_swing_mask = (self.env.foot_frc < 1.0)
        self.foot_support_mask = (self.env.foot_frc >= 10.0)
        # Advance reference frame
        self._advance_ref_frames()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def pure_observation(self):
        """64-dim obs per step."""
        if self._has_ref:
            ref_pos_err = (self._ref_joint_pos_now - self.joint_pos).clip(-3., 3.)
            ref_vel     = self._ref_joint_vel_now.clip(-3., 3.)
            ref_slots   = torch.cat([ref_pos_err, ref_vel, self._ref_phase_progress], dim=1)  # 21
        else:
            ref_slots = torch.zeros(self.num_envs, 21, dtype=torch.float, device=self.device)

        self.obs_buf = torch.cat([
            self.commands[:, :8],                    # [0-7]
            self.base_euler[:, :2] * 1.0,            # [8-9]
            self.base_ang_vel * 0.5,                  # [10-12]
            self.joint_pos - self.ref_joint_action,  # [13-22]
            self.joint_vel * 0.1,                     # [23-32]
            self.joint_pos_error,                     # [33-42]
            ref_slots,                                # [43-63]
        ], dim=1).clip(min=-3.0, max=3.0)
        return self.obs_buf

    def pure_critic_observation(self):
        obs_buf = torch.cat([
            self.commands[:, :8],
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
            self.net_out_history[-1] / 15.0,
            self.foot_height.clip(min=-0.5, max=0.5) * 10.0,
            (self.env.base_pos_hd[:, [2]] - 0.4) * 10.0,
            self.env.foot_vel.clip(min=-8.0, max=8.0) * 0.5,
            self.env.base_acc.clip(min=-20.0, max=20.0) * 0.2,
            self.env.foot_frc.clip(min=0.0, max=200.0) * 0.01,
            self.base_euler[:, :2] * 1.0,
            self.base_ang_vel * 0.5,
            self.joint_pos - self.ref_joint_action,
            self.joint_vel * 0.1,
            self.joint_pos_error,
        ], dim=1)
        return obs_buf

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def action(self, net_out):
        """All 10 outputs are joint position increments. No phase modulator."""
        net_out = scale_transform(net_out, self.action_low, self.action_high)
        self.net_out_history.append(net_out)
        if self.env.render and self.env.common_step_counter <= 1:
            pass
        else:
            if self.cfg.action.use_increment:
                self.current_joint_act += net_out * self.env.dt
            else:
                self.current_joint_act = net_out
        self.current_joint_act = torch.clip(
            self.current_joint_act, self.joint_action_limit_low, self.joint_action_limit_high
        )
        self.action_history.append(self.current_joint_act.clone())
        self.previous_joint_act = self.current_joint_act.clone()
        return self.current_joint_act

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self):
        """Task rewards identical to BIRLTask minus pmf/foot_phase (phase-dependent).
        Adds imitation rewards when reference clips are loaded.

        Weight schedule for imitation (set in config via w_imit / w_task):
          iters   0-500:  w_imit=0.8, w_task=0.2  — learn reference first
          iters 500-2000: decay to w_imit=0.3, w_task=0.7
          iters  2000+:   w_imit=0.2, w_task=0.8  — optimise task
        Weights are controlled by config; not annealed automatically here.
        """
        constant_rew = to_torch([1.]).repeat(self.num_envs, 1)
        lin_vel_x_norm = torch.clip(
            torch.norm(self.commands[:, [0, 1]], dim=1, keepdim=True), min=0.3, max=2.
        ) + 0.2
        yaw_rate_norm = torch.clip(
            torch.abs(self.commands[:, [2]]), min=0.3, max=1.5
        ) + 0.2

        base_heit_rew = torch.exp(-70 * (self.env.base_pos[:, [2]] - 0.45) ** 2)
        balance_rew = 0.5 * (
            base_heit_rew * torch.exp(
                -torch.clip(5.0 / lin_vel_x_norm, min=2, max=8.)
                * torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)
            ) + 1.
        )

        forward_vel_rew = torch.exp(
            -torch.clip(5. / lin_vel_x_norm, min=2., max=10.)
            * (self.commands[:, [0]] - self.env.base_lin_vel[:, [0]]) ** 2
        )
        lateral_vel_rew = torch.exp(
            -torch.clip(5. / lin_vel_x_norm, min=3., max=15.)
            * (self.commands[:, [1]] - self.env.base_lin_vel[:, [1]]) ** 2
        )
        yaw_rate_rew = torch.exp(
            -torch.clip(2. / yaw_rate_norm, min=2., max=6.)
            * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2
        )
        lateral_vel_rew += (
            -0.6 / lin_vel_x_norm
            * torch.abs(self.commands[:, [1]] - self.env.base_lin_vel[:, [1]])
            * self.static_flag
        )

        ang_vel_rew = torch.exp(
            -torch.clip(2. / lin_vel_x_norm, min=0.7, max=6.)
            * torch.norm(self.env.base_ang_vel[:, :2], dim=1, keepdim=True) ** 2
        )
        base_acc_rew = (
            -0.4 / lin_vel_x_norm
            * torch.norm(
                (self.env.base_acc - to_torch([0, 0, 9.81], device=self.device)) * 0.1,
                dim=1, keepdim=True
            ) * self.static_flag
        )

        vertical_vel_rew = torch.exp(
            -torch.clip(5. / lin_vel_x_norm, min=2., max=10.)
            * torch.norm(self.env.base_lin_vel[:, [2]], dim=1, keepdim=True) ** 2
        )
        vertical_vel_rew -= (
            0.2 / lin_vel_x_norm
            * torch.norm(self.env.base_lin_vel[:, [2]], dim=1, keepdim=True)
            * self.static_flag
        )

        support_foot_index = torch.where(self.env.foot_frc >= 10., True, False)
        swing_foot_index = torch.where(self.env.foot_frc < 1., True, False)

        foot_clear_rew = torch.sum(
            torch.logical_and(swing_foot_index, self.foot_swing_mask),
            dtype=torch.float, dim=1, keepdim=True
        ) / self.num_legs * self.static_flag

        foot_support_rew = torch.sum(
            torch.logical_and(support_foot_index, self.foot_support_mask),
            dtype=torch.float, dim=1, keepdim=True
        ) / self.num_legs * self.static_flag

        foot_heit_score = 40. * torch.clip(self.foot_height, min=0.0, max=0.05)
        foot_height_rew = torch.sum(
            self.foot_swing_mask * foot_heit_score, dim=1, keepdim=True
        ).clip(max=2.) * self.static_flag
        foot_height_rew += -20. * torch.sum((self.foot_height - 0.06).clip(min=0.), dim=1, keepdim=True)
        foot_height_rew += -0.2 * torch.sum(self.foot_support_mask * foot_heit_score, dim=1, keepdim=True) * self.static_flag
        foot_height_rew += -0.2 * torch.sum(support_foot_index * foot_heit_score, dim=1, keepdim=True) * self.static_flag

        twist_rew = -torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)

        self.foot_frc_acc = (self.env.foot_frc - self.last_foot_frc).clone()
        foot_soft_rew = (
            -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.5)
            * torch.norm(self.foot_frc_acc, dim=1, keepdim=True) / 100.
        )
        self.last_foot_frc = self.env.foot_frc.clone().detach()

        feet_contact_frc_rew = (
            -torch.norm(self.env.foot_frc * self.foot_swing_mask, dim=1, keepdim=True) * self.static_flag
            - torch.norm(
                (torch.abs(self.env.foot_frc - 55.) * support_foot_index).clip(min=0.),
                dim=1, keepdim=True
            )
        )

        clip_foot_h = torch.abs(self.foot_height) + 0.03
        vy_walking = (torch.abs(self.commands[:, [1]]) > 0.1).float()

        foot_slip_rew = 2. * (
            lin_vel_x_norm * torch.sum(
                (self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, 0])
                * self.commands[:, [0]].sign() * self.foot_swing_mask,
                dim=1, keepdim=True
            )
        ).clip(min=-0., max=1.) * self.static_flag
        foot_slip_rew += (
            -0.5 * torch.norm(
                torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [1]], dim=-1),
                dim=1, keepdim=True
            ) * self.static_flag * (1. - vy_walking)
        )
        foot_slip_rew += 0.3 * torch.norm(
            torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, :2], dim=-1),
            dim=1, keepdim=True
        ) * (self.static_flag - 1.)
        foot_slip_rew += (
            -0.3 / lin_vel_x_norm * torch.norm(
                0.1 * torch.norm(
                    self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, :2], dim=-1
                ) / clip_foot_h * self.foot_support_mask,
                dim=1, keepdim=True
            ) * self.static_flag
        )

        foot_vz_rew = (
            -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.)
            * torch.norm(
                torch.norm(
                    self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [2]].clip(max=0.),
                    dim=-1
                ) / clip_foot_h, dim=1, keepdim=True
            ) * self.static_flag
        )
        foot_vz_rew += 0.8 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(
            torch.norm(
                self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [2]].clip(max=0.),
                dim=-1
            ), dim=1, keepdim=True
        ) * (self.static_flag - 1.)

        foot_acc_rew = (
            -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.)
            * torch.norm(self.env.foot_vel[:, [2, 5]], dim=1, keepdim=True)
        )

        action_smooth_rew = (
            -0.3 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.)
            * torch.norm(
                self.action_history[-3] - 2. * self.action_history[-2] + self.action_history[-1],
                dim=1, keepdim=True
            )
        )
        # MIRL: net_out is 10-dim (no freq prefix), use all dims
        net_out_smooth_rew = (
            -0.2 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.)
            * torch.norm(
                self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1],
                dim=1, keepdim=True
            ) ** 2
        )

        action_constraint_rew = (
            -0.1 * torch.clip(1. / lin_vel_x_norm, 0, 1.)
            * torch.norm((self.current_joint_act - self.ref_joint_action), dim=1, keepdim=True)
        )
        action_constraint_rew += (
            -3. * torch.norm(
                (self.current_joint_act - self.ref_joint_action)[:, [0, 1, 5, 6]],
                dim=1, keepdim=True
            ) * self.static_flag * (1. - vy_walking)
        )

        sa_constraint_rew = (
            -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.)
            * torch.norm(self.current_joint_act - self.ref_joint_action, dim=1, keepdim=True) ** 2
            * self.static_flag
        )
        sa_constraint_rew += (
            -self.static_flag * torch.clip(1. / lin_vel_x_norm, 0, 1)
            * torch.norm(
                ((self.env.joint_pos - self.ref_joint_action)[:, :5] * support_foot_index[:, [0]]),
                dim=1, keepdim=True
            ) ** 2
        )
        sa_constraint_rew += (
            -self.static_flag * torch.clip(1. / lin_vel_x_norm, 0, 1)
            * torch.norm(
                ((self.env.joint_pos - self.ref_joint_action)[:, 5:] * support_foot_index[:, [1]]),
                dim=1, keepdim=True
            ) ** 2
        )

        joint_pos_error_rew = (
            -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.)
            * torch.norm((self.current_joint_act - self.env.joint_pos), dim=1, keepdim=True) ** 2
        )

        joint_velocity_rew = (
            -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.)
            * torch.norm(self.env.joint_vel, dim=1, keepdim=True) ** 2
        )
        joint_velocity_rew += (
            -torch.clip(1. / lin_vel_x_norm, 0, 1)
            * torch.norm(self.env.joint_vel[:, [0, 1, 5, 6]], dim=1, keepdim=True) ** 2
            * (1. - vy_walking)
        )

        joint_tor_rew = (
            -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.)
            * torch.sum(
                (torch.abs(self.env.react_tau) - self.env.torque_limits).clip(min=0.),
                dim=1, keepdim=True
            ) * self.static_flag
        )

        # Mechanical power penalty: |torque × joint_vel| — minimizes energy consumption.
        # Also naturally reduces step frequency (rapid shuffling wastes power).
        power_rew = (
            -torch.sum(torch.abs(self.env.react_tau * self.env.joint_vel), dim=1, keepdim=True)
            / 100.
        )

        self.last_foot_vel = self.env.foot_vel.clone().detach()

        net_out_val_rew = (
            -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.)
            * torch.norm(self.net_out_history[-1], dim=1, keepdim=True) ** 2
        )

        foot_py_rew = -0.5 * torch.norm(self.env.foot_euler[:, [1, 4]], dim=1, keepdim=True)
        leg_width_rew = -torch.norm(
            torch.abs(self.env.foot_pos_hd[:, [1, 4]] - self.env.base_pos_hd[:, [1]]) - 0.14,
            dim=1, keepdim=True
        )

        rew_dict = dict(
            constant       = constant_rew * 0.3,
            base_heit      = base_heit_rew,
            balance        = balance_rew * 1.5,
            fwd_vel        = forward_vel_rew * 2.3,
            yaw_rat        = yaw_rate_rew * 2.5,
            lateral_vel    = lateral_vel_rew * 0.7,
            vertical_vel   = vertical_vel_rew * 0.6,
            ang_vel        = ang_vel_rew * 0.6,
            twist          = twist_rew * 2.5,
            base_acc       = base_acc_rew * balance_rew * 0.1,
            foot_clr       = foot_clear_rew * 1.,
            foot_supt      = foot_support_rew * 0.7,
            foot_heit      = foot_height_rew * 0.7,
            leg_width_rew  = leg_width_rew * balance_rew * 0.5,
            act_const      = action_constraint_rew * balance_rew * 0.2,
            sa_const       = sa_constraint_rew * balance_rew * 0.1,
            jnt_pos_err    = joint_pos_error_rew * balance_rew * 0.2,
            act_smo        = action_smooth_rew * balance_rew * 1.5,
            net_smo        = net_out_smooth_rew * balance_rew * 0.001,
            net_out_val    = net_out_val_rew * balance_rew * 0.0001,
            foot_slip      = foot_slip_rew * balance_rew * 0.5,
            foot_vz        = foot_vz_rew * 0.2 * balance_rew,
            foot_acc       = foot_acc_rew * balance_rew * 0.05,
            foot_sft       = foot_soft_rew * 2.7 * balance_rew,
            jnt_vel        = joint_velocity_rew * balance_rew * 0.003,
            feet_py        = foot_py_rew * balance_rew * 0.5,
            feet_frc       = feet_contact_frc_rew * 0.001,
            joint_tor      = joint_tor_rew * 0.001,
            power          = power_rew * 0.01,
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
            self.rew_names = list(rew_dict.keys())
            self.debug = None
        if self.rew_names is None:
            self.rew_names = list(rew_dict.keys())

        rewards = torch.cat(
            [torch.clip(value.to(self.device), min=-4., max=5.) * self.env.dt
             for value in rew_dict.values()],
            dim=1
        )
        self._last_rew_components = rewards.detach()

        eval_rew = torch.cat(
            [rew_dict[key] * self.env.dt
             for key in ['fwd_vel', 'yaw_rat', 'ang_vel', 'lateral_vel', 'vertical_vel', 'twist']],
            dim=1
        ).sum(dim=1)
        return rewards, eval_rew
