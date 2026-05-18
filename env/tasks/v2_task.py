"""V2Task — minimal stand task for walk_clean v2 series.

14 reward components — see REWARD_NAMES below.

Bell rewards use robust_bell(): exp(-err²/σ²) − l2·clip(err²/(4σ)², 0, 1).
Always has long-range gradient unlike pure exp bells (which saturate at 0).

Foot terms are contact-gated: multiplied by contact_avg ∈ {0, 0.5, 1.0} so
airborne foot → 0 reward (prevents the "hopping" exploit where lifting feet
maximizes exp(-0·in_contact) = 1).

foot_geom constrains stance: foot xy positions relative to base in heading
frame. Lets policy freely tune sagittal-plane joints (hip_pitch/knee/ankle
for body lean/height) while keeping foot placement (hip_yaw + hip_roll) at
ref. Subtracting base_pos_hd makes it invariant to where each env spawns.

Action: residual mode only. target = ref_joint + lp_filtered(scaled_offset).
Phase: none. No clock signal in obs.
"""

import random
from typing import Tuple
from collections import deque

import torch

from env.legged_robot import LeggedRobotEnv
from env.tasks.null_task import NullTask, register
from env.utils.math import scale_transform
from isaacgym.torch_utils import to_torch, torch_rand_float


def robust_bell(err_sq: torch.Tensor, sigma_bell: float,
                sigma_l2_ratio: float = 4.0, l2_weight: float = 0.1) -> torch.Tensor:
    """Robust-bell reward (RL analog of Huber loss).

    Pure exp() bell loses gradient when |err| >> σ_bell: exp(-9)≈1e-4 with
    near-zero slope. PPO can't escape regions far from the target. The "lock"
    L2 trick (yaw_lock, body_lock) fixes this by adding a parallel L2 penalty
    on the same quantity. This helper bakes both into one expression so any
    bell-shaped reward gets long-range gradient for free.

    r(err) = exp(-err²/σ_bell²) − l2_weight · clip(err²/(σ_l2_ratio·σ_bell)², 0, 1)

      err = 0:                  bell = 1, l2 = 0 → r = +1.0   (max)
      err = σ_bell:             bell ≈ 0.37     → r ≈ +0.35
      err = σ_l2_ratio·σ_bell:  bell ≈ 0, l2 = 1 → r = −l2_weight (floor)
      err → ∞:                  reward saturates at −l2_weight; gradient is
                                the clipped L2 (constant inside the band).

    Args:
        err_sq: squared error per env, shape [N, 1] or [N].
        sigma_bell: tight bell scale (where the bell decays).
        sigma_l2_ratio: σ_l2 = ratio · σ_bell (wider than bell for coarse pull).
        l2_weight: depth of the L2 floor (default 0.1 — small enough that 8
            bells stacked don't dominate the per-step return; bump higher if
            a specific bell has plateau issues).

    Returns:
        Reward in [−l2_weight, 1.0]. Always differentiable wrt err.
    """
    bell = torch.exp(-err_sq / (sigma_bell ** 2))
    sigma_l2_sq = (sigma_bell * sigma_l2_ratio) ** 2
    l2 = torch.clip(err_sq / sigma_l2_sq, 0.0, 1.0)
    return bell - l2_weight * l2


# Reward names in fixed order (for TB logging)
REWARD_NAMES = (
    'base_height',         # exp(-((z - target_z)/σ)^2)                bounded (0,1]
    'upright',             # exp(-||g_xy||^2/σ^2)                      bounded (0,1]
    'linear_velocity',     # exp(-||v - v_cmd||^2/σ^2)                 bounded (0,1]
    'angular_velocity',    # robust_bell w/ strong L2 floor (l2_w=1.0) — replaces yaw_rate_lock
    'body_rate_lock',      # -clip(rp_avel²/σ², 0, 1)                  L2: stops pitch/roll hover
    'pose_match',          # robust_bell on Σ (q - ref)^2 (10-dim joint pose tracking to ref)
    'foot_slip',           # exp(-Σ ||v_xy(foot)||²)·contact_avg        contact-gated (×0 if airborne)
    'foot_rotation',       # exp(-Σ ||ω(foot)||²)·contact_avg           contact-gated (×0 if airborne)
    'action_smoothness',   # -||a[t] - a[t-1]||^2 / 10                 small negative reg
    'termination',         # -1 on fall (not timeout)                  large terminal penalty
)


@register
class V2Task(NullTask):

    def __init__(self, env: LeggedRobotEnv):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.num_legs = 2
        self.num_dofs = env.num_dofs

        # ── action: residual mode only ─────────────────────────────────────
        assert getattr(self.cfg.action, 'action_mode', 'residual') == 'residual', \
            "V2Task requires action.action_mode='residual'"
        self._lp_alpha = float(getattr(self.cfg.action, 'action_lowpass_alpha', 0.75))
        self.action_low  = to_torch(self.cfg.action.residual_low_ranges,  device=self.device)
        self.action_high = to_torch(self.cfg.action.residual_high_ranges, device=self.device)
        assert len(self.action_low) == self.num_dofs

        self.ref_joint = to_torch(self.cfg.action.ref_joint_pos, device=self.device).repeat(self.num_envs, 1)
        self.current_joint_act = self.ref_joint.clone()
        self._lp_target = torch.zeros_like(self.current_joint_act)
        self._action_mode = 'residual'

        # ── commands (per-env target velocity) ────────────────────────────
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.static_flag = torch.zeros(self.num_envs, 1, device=self.device)
        self._vx_range  = self.cfg.command.lin_vel_x_range
        self._vy_range  = self.cfg.command.lin_vel_y_range
        self._yaw_range = self.cfg.command.ang_vel_yaw_range
        self._cmd_resample_dt = float(getattr(self.cfg.command, 'resampling_time', 5.0))
        policy_dt = env.cfg.sim.dt * env.cfg.pd_gains.decimation
        self._cmd_resample_steps = max(1, int(self._cmd_resample_dt / policy_dt))

        # ── obs delay (DR for sim2real) ────────────────────────────────────
        self.delay_joint_steps = 0
        self.delay_angle_steps = 0
        self.delay_rate_steps  = 0
        # Range in physics steps (1ms each). Defaults match BIRL.
        self._dj_range = [10, 40]
        self._da_range = [20, 50]
        self._dr_range = [20, 50]
        self._resample_delays()

        # ── proprio buffers (refreshed each step) ─────────────────────────
        self.joint_pos    = env.joint_pos.clone()
        self.joint_vel    = env.joint_vel.clone()
        self.base_euler   = env.base_euler.clone()
        self.base_ang_vel = env.base_ang_vel.clone()
        self.base_lin_vel = env.base_lin_vel.clone()
        self.joint_pos_error = torch.zeros_like(self.current_joint_act)

        # ── obs history (frame-stack) ─────────────────────────────────────
        self._obs_history_n = int(getattr(self.cfg.observation, 'history', 5))
        self._obs_skip      = int(getattr(self.cfg.observation, 'skip', 2))
        buf_len = (self._obs_history_n - 1) * self._obs_skip + 1
        obs_dim = self._compute_obs_dim()
        cri_dim = self._compute_cri_obs_dim()
        self.obs_history = deque(
            [torch.zeros(self.num_envs, obs_dim, device=self.device) for _ in range(buf_len)],
            maxlen=buf_len)
        self.cri_obs_history = deque(
            [torch.zeros(self.num_envs, cri_dim, device=self.device) for _ in range(buf_len)],
            maxlen=buf_len)

        # ── action history (for smoothness reward) ─────────────────────────
        self.action_history = deque(
            [self.current_joint_act.clone() for _ in range(2)], maxlen=2)

        # ── termination thresholds ────────────────────────────────────────
        self._tilt_term_rad = float(getattr(self.cfg.task, 'tilt_term_rad', 0.7))
        self._base_z_min    = float(getattr(self.cfg.task, 'base_z_min', 0.2))
        self._episode_length_s = float(getattr(self.cfg.runner, 'episode_length_s', 15.0))
        self._max_steps = int(self._episode_length_s / policy_dt)
        self.episode_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # ── reward weights ────────────────────────────────────────────────
        r = self.cfg.reward
        self._w = torch.tensor([
            float(getattr(r, 'w_base_height',       2.0)),
            float(getattr(r, 'w_upright',           2.0)),
            float(getattr(r, 'w_linear_velocity',   2.0)),
            float(getattr(r, 'w_angular_velocity',  2.0)),
            float(getattr(r, 'w_body_rate_lock',    3.0)),
            float(getattr(r, 'w_pose_match',        3.0)),
            float(getattr(r, 'w_foot_slip',         1.5)),
            float(getattr(r, 'w_foot_rotation',     1.5)),
            float(getattr(r, 'w_action_smoothness', 1.0)),
            float(getattr(r, 'w_termination',      50.0)),
        ], device=self.device)
        self._target_z                = float(getattr(r, 'target_z',              0.45))
        self._base_height_scale       = float(getattr(r, 'base_height_scale',     0.10))
        self._upright_scale           = float(getattr(r, 'upright_scale',         0.30))
        self._linear_velocity_scale   = float(getattr(r, 'linear_velocity_scale', 0.25))
        self._angular_velocity_scale  = float(getattr(r, 'angular_velocity_scale', 0.50))
        self._angular_velocity_l2_weight = float(getattr(r, 'angular_velocity_l2_weight', 1.0))  # deep L2 floor — absorbs yaw_rate_lock role
        self._body_rate_lock_scale    = float(getattr(r, 'body_rate_lock_scale',  1.0))   # rad/s combined rp_avel
        self._pose_match_scale        = float(getattr(r, 'pose_match_scale',      1.5))   # rad on Σ(q-ref)² bell
        self._foot_slip_scale         = float(getattr(r, 'foot_slip_scale',       0.05))
        self._foot_rotation_scale     = float(getattr(r, 'foot_rotation_scale',   0.30))
        self._contact_threshold       = float(getattr(r, 'contact_threshold',     5.0))

        # Episode-init XY for drift reward. Set in reset(); init for first rollout.
        self._episode_init_xy = env.base_pos[:, :2].clone()

        # Cache foot body indices for angular-velocity lookup.
        self._foot_idx_L = int(env.feet_indices[0].item())
        self._foot_idx_R = int(env.feet_indices[1].item())

        self._phase_mode = 'none'
        self._last_rew_components = None
        self.reward_names = REWARD_NAMES

        # First command sample
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    # ── dims ──────────────────────────────────────────────────────────────
    def _compute_obs_dim(self):
        # commands(3) + base_ang_vel(3) + proj_grav(3) + joint_pos_err(10)
        # + joint_vel(10) + joint_tracking_err(10) = 39
        return 3 + 3 + 3 + self.num_dofs * 3

    def _compute_cri_obs_dim(self):
        # actor obs + base_lin_vel(3) + foot_height(2) + foot_frc(2) + base_z(1)
        # + DR priv (friction, p_gain_mean, d_gain_mean, tau_gain_mean = 4) = +12
        return self._compute_obs_dim() + 12

    # ── helpers ───────────────────────────────────────────────────────────
    def _resample_delays(self):
        self.delay_joint_steps = random.randint(*self._dj_range)
        self.delay_angle_steps = random.randint(*self._da_range)
        self.delay_rate_steps  = random.randint(*self._dr_range)

    def _resample_commands(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(self._vx_range[0],  self._vx_range[1],  (n, 1), device=self.device).squeeze(-1)
        self.commands[env_ids, 1] = torch_rand_float(self._vy_range[0],  self._vy_range[1],  (n, 1), device=self.device).squeeze(-1)
        self.commands[env_ids, 2] = torch_rand_float(self._yaw_range[0], self._yaw_range[1], (n, 1), device=self.device).squeeze(-1)
        sf = (torch.norm(self.commands[env_ids, :3], dim=1, keepdim=True) >= 0.15).float()
        self.commands[env_ids, :3] *= sf
        self.static_flag[env_ids] = sf

    # ── core methods ──────────────────────────────────────────────────────
    def step(self):
        # delayed proprio
        self.joint_pos    = self.env.joint_pos_his.delay(self.delay_joint_steps)
        self.joint_vel    = self.env.joint_vel_his.delay(self.delay_joint_steps)
        self.base_euler   = self.env.base_eul_his.delay(self.delay_angle_steps)
        self.base_ang_vel = self.env.base_ang_vel_his.delay(self.delay_rate_steps)
        self.base_lin_vel = self.env.base_lin_vel
        self.joint_pos_error = self.current_joint_act - self.joint_pos

        # periodic command resample
        self.episode_step += 1
        resample_ids = (self.episode_step % self._cmd_resample_steps == 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        # Surface timeouts to PPO so it bootstraps value at episode end
        # (timeouts != falls; without this, terminal timeout looks like a crash to the critic).
        self.extra_info["timeouts"] = (self.episode_step >= self._max_steps)

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.joint_pos[env_ids]    = self.env.joint_pos_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_vel[env_ids]    = self.env.joint_vel_his.delay(self.delay_joint_steps)[env_ids]
        self.base_euler[env_ids]   = self.env.base_eul_his.delay(self.delay_angle_steps)[env_ids]
        self.base_ang_vel[env_ids] = self.env.base_ang_vel_his.delay(self.delay_rate_steps)[env_ids]

        self.current_joint_act[env_ids] = self.ref_joint[env_ids]
        self._lp_target[env_ids] = 0.0
        for h in self.action_history:
            h[env_ids] = self.ref_joint[env_ids]

        self.episode_step[env_ids] = 0
        # env.reset() already ran in the wrapper, so env.base_pos reflects the new pose.
        self._episode_init_xy[env_ids] = self.env.base_pos[env_ids, :2]
        self._resample_delays()
        self._resample_commands(env_ids)

    def action(self, net_out):
        offset = scale_transform(net_out, self.action_low, self.action_high)
        self._lp_target = self._lp_alpha * offset + (1.0 - self._lp_alpha) * self._lp_target
        self.current_joint_act = self.ref_joint + self._lp_target
        self.current_joint_act = torch.clip(
            self.current_joint_act, self.env.dof_pos_limits[:, 0], self.env.dof_pos_limits[:, 1])
        self.action_history.append(self.current_joint_act.clone())
        return self.current_joint_act

    # ── observations ─────────────────────────────────────────────────────
    def pure_observation(self):
        parts = [
            self.commands,                          # 3
            self.base_ang_vel * 0.5,                 # 3
            self.env.projected_gravity,              # 3
            self.joint_pos - self.ref_joint,         # 10
            self.joint_vel * 0.1,                    # 10
            self.joint_pos_error,                    # 10
        ]
        return torch.cat(parts, dim=1).clip(min=-3., max=3.)

    def observation(self):
        self.obs_history.append(self.pure_observation())
        buf = list(self.obs_history)
        idx = [i * self._obs_skip for i in range(self._obs_history_n)]
        return torch.cat([buf[i] for i in idx], dim=-1)

    def pure_critic_observation(self):
        actor_obs = self.pure_observation()
        # foot_pos is flat [n, 6] = (xyz_L, xyz_R); indices [2, 5] are z.
        foot_z = self.env.foot_pos[:, [2, 5]]
        # DR scalars — centered at 0 (subtract nominal 1.0). Lets critic see
        # the actual physics it's evaluating value under, so PPO advantage
        # signal is clean instead of being corrupted by DR variance.
        z0 = torch.zeros(self.num_envs, 1, device=self.device)
        fric = (self.env.friction_coeffs.to(self.device) - 1.0
                if hasattr(self.env, 'friction_coeffs') else z0)
        pg = (self.env.p_gains_rand.mean(dim=-1, keepdim=True) - 1.0
              if hasattr(self.env, 'p_gains_rand') else z0)
        dg = (self.env.d_gains_rand.mean(dim=-1, keepdim=True) - 1.0
              if hasattr(self.env, 'd_gains_rand') else z0)
        tg = (self.env.tau_gains.mean(dim=-1, keepdim=True) - 1.0
              if hasattr(self.env, 'tau_gains') else z0)
        priv = torch.cat([
            self.base_lin_vel,                                  # 3
            foot_z.clip(-0.5, 0.5) * 10.0,                       # 2
            self.env.foot_frc.clip(0., 200.) * 0.01,            # 2
            (self.env.base_pos[:, [2]] - 0.4) * 10.0,           # 1
            fric, pg, dg, tg,                                    # 4 — DR priv
        ], dim=1)
        return torch.cat([actor_obs, priv], dim=1)

    def critic_observation(self):
        self.cri_obs_history.append(self.pure_critic_observation())
        buf = list(self.cri_obs_history)
        idx = [i * self._obs_skip for i in range(self._obs_history_n)]
        return torch.cat([buf[i] for i in idx], dim=-1)

    # ── reward (10 terms; see REWARD_NAMES at top for layout) ────────────
    # Bells use robust_bell() helper: exp(-err²/σ²) − 0.5·clip(err²/(4σ)², 0, 1).
    # Always has long-range gradient (Huber-like), no plateau beyond ±3σ.
    def reward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bounded positive tracking terms.
        bz = self.env.base_pos[:, [2]]
        height_err_sq = (bz - self._target_z) ** 2
        r_base_height = robust_bell(height_err_sq, self._base_height_scale)

        g_xy_sq = torch.sum(self.env.projected_gravity[:, :2] ** 2, dim=-1, keepdim=True)
        r_upright = robust_bell(g_xy_sq, self._upright_scale)

        vel_err_sq = torch.sum((self.base_lin_vel[:, :2] - self.commands[:, :2]) ** 2,
                               dim=-1, keepdim=True)
        r_linear_velocity = robust_bell(vel_err_sq, self._linear_velocity_scale)

        yaw_err_sq = (self.base_ang_vel[:, [2]] - self.commands[:, [2]]) ** 2
        rp_avel_sq = torch.sum(self.base_ang_vel[:, :2] ** 2, dim=-1, keepdim=True)
        ang_err_sq = yaw_err_sq + rp_avel_sq * 0.3
        # angular_velocity uses deep L2 floor (l2_weight=1.0) so it absorbs the
        # yaw_rate_lock role: at large yaw_rate (>= ~0.8) the L2 cap saturates
        # to -1.0, × w=2 = -2.0/step — matches yaw_rate_lock's L2 exactly.
        # Bonus: bell still gives +reward near zero rate. One reward, two roles.
        r_angular_velocity = robust_bell(ang_err_sq, self._angular_velocity_scale,
                                         l2_weight=self._angular_velocity_l2_weight)

        # L2 body-rate penalty (roll/pitch rate) — gradient exists at ANY rate
        # (unlike bell which saturates beyond ~3σ). Without this,
        # angular_velocity bell saturates at 0 for high rp_avel → no gradient → policy
        # hovers via pitch/roll oscillation.
        r_body_rate_lock = -torch.clip(rp_avel_sq / (self._body_rate_lock_scale ** 2), 0.0, 1.0)

        # Pose match: full 10-dim joint tracking toward ref. σ on Σ(q-ref)² controls
        # how tightly policy is anchored to the captured stand pose. Larger σ allows
        # more joint excursion for active balance; smaller σ enforces stricter ref
        # tracking but reduces policy's recovery freedom.
        joint_err_sq = torch.sum((self.env.joint_pos - self.ref_joint) ** 2,
                                  dim=-1, keepdim=True)
        r_pose_match = robust_bell(joint_err_sq, self._pose_match_scale)

        # Foot terms: contact-gated. swing-leg motion is free (walk-compatible).
        # foot_vel is heading-frame [N, 6] = (vx,vy,vz)_L | (vx,vy,vz)_R.
        # Multiply final bell by contact_avg so airborne foot gets 0 reward
        # (not 1.0). Without this multiplier policy maximizes by lifting feet
        # → "hopping" stand exploit.
        in_contact = (self.env.foot_frc > self._contact_threshold).float()  # [N, 2]
        contact_avg = ((in_contact[:, 0] + in_contact[:, 1]) * 0.5).unsqueeze(-1)  # 0/0.5/1.0
        vxy_L = self.env.foot_vel[:, 0:2]
        vxy_R = self.env.foot_vel[:, 3:5]
        slip_sq = (torch.sum(vxy_L ** 2, dim=-1) * in_contact[:, 0] +
                   torch.sum(vxy_R ** 2, dim=-1) * in_contact[:, 1]).unsqueeze(-1)
        r_foot_slip = robust_bell(slip_sq, self._foot_slip_scale) * contact_avg

        # World-frame angular velocity of each foot from rigid_body_param.
        avel_L = self.env.rigid_body_param[:, self._foot_idx_L, 10:13]
        avel_R = self.env.rigid_body_param[:, self._foot_idx_R, 10:13]
        rot_sq = (torch.sum(avel_L ** 2, dim=-1) * in_contact[:, 0] +
                  torch.sum(avel_R ** 2, dim=-1) * in_contact[:, 1]).unsqueeze(-1)
        r_foot_rotation = robust_bell(rot_sq, self._foot_rotation_scale) * contact_avg

        # Small negative regularization — must not dominate positives.
        da = self.action_history[-1] - self.action_history[-2]
        r_action_smoothness = -torch.sum(da ** 2, dim=-1, keepdim=True) / 10.0

        # Termination penalty: only when this step causes a fall (not timeout).
        tilt = torch.norm(self.env.base_euler[:, :2], dim=1, keepdim=True)
        fall = ((tilt > self._tilt_term_rad) |
                (self.env.base_pos[:, [2]] < self._base_z_min)).float()
        r_termination = -fall

        components = torch.cat([
            r_base_height, r_upright, r_linear_velocity, r_angular_velocity,
            r_body_rate_lock,
            r_pose_match,
            r_foot_slip, r_foot_rotation,
            r_action_smoothness, r_termination,
        ], dim=1)  # [n, 10]
        weighted = components * self._w.unsqueeze(0)
        self._last_rew_components = weighted
        return weighted, weighted

    def terminate(self) -> torch.Tensor:
        tilt = torch.norm(self.env.base_euler[:, :2], dim=1)
        too_tilted = tilt > self._tilt_term_rad
        too_low = self.env.base_pos[:, 2] < self._base_z_min
        timeout = self.episode_step >= self._max_steps
        return (too_tilted | too_low | timeout).float().unsqueeze(-1)
