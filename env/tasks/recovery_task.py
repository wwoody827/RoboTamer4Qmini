"""
Fall-recovery task (Phase 3 scaffold).

Architecturally distinct from BIRLTask:
    - Episodic (no rolling time): 5s episodes, terminal reward at end / on hard fail
    - No phase signal, no phase modulator
    - No velocity command
    - 10-dim absolute action with low-pass filter (smoothness via filtering)
    - obs_history = 1 (Markovian)
    - Init pool from data/recovery_init_states.npz (settled fallen poses)
    - Reward composed of orthogonal *groups*, gated by `cfg.reward.groups_enabled`

Reward groups (orthogonal concerns, not progressive layers):
    shaping_static: dense exp-shaped pose-match + orientation tracking
    shaping_phased: phase-conditioned shaping (orient early, pose late)
    stability:      bonus when upright with low base velocity
    smoothness:     action-rate / jerk / torque penalties
    safety:         joint-limit and contact-force penalties
    terminal:       once-per-episode bonus / penalty at episode end

Any subset of groups can be enabled per config; the historical "layer" naming
(layer1/layer2/...) is gone — these are orthogonal groups, not stages.
"""

import numpy as np
import torch
from collections import deque
from isaacgym.torch_utils import to_torch, torch_rand_float

from env.legged_robot import LeggedRobotEnv
from env.tasks.null_task import NullTask, register
from env.obs_builder import ObsBuilder, obs_slot
from env.recovery_curriculum import RecoveryCurriculum


# Recovery-specific obs slots (registered globally; only consumed if listed
# in observation.slots).

@obs_slot('projected_gravity', dim=3)
def _projected_gravity(task):
    """Gravity vector projected into body frame. Encodes orientation directly:
    upright → ~[0, 0, -1]; lying on back → [0, 0, +1]; on a side → [0, ±1, 0]."""
    return task.env.projected_gravity


@obs_slot('episode_progress', dim=1)
def _episode_progress(task):
    """0 → 1 over the episode horizon. Lets the policy plan the terminal phase."""
    return (task.env.episode_length_buf.float() / max(task.env.max_episode_length, 1)).unsqueeze(-1)


@obs_slot('joint_pos_abs', dim=10)
def _joint_pos_abs(task):
    """Absolute joint positions (no nominal subtraction). Useful for recovery
    where the *current* angle matters more than the offset from a stand pose."""
    return task.joint_pos


# ─── RecoveryTask ───────────────────────────────────────────────────────────

# Reward group keys (for cfg.reward.groups_enabled):
GROUP_SHAPING_STATIC      = 'shaping_static'
GROUP_SHAPING_PHASED      = 'shaping_phased'
GROUP_STABILITY           = 'stability'
GROUP_SMOOTHNESS          = 'smoothness'
GROUP_SAFETY              = 'safety'
GROUP_TERMINAL            = 'terminal'


@register
class RecoveryTask(NullTask):

    def __init__(self, env: LeggedRobotEnv):
        super().__init__(env)
        self.env = env
        self.num_envs = env.num_envs
        self.num_legs = 2
        self.device = env.device
        self.cfg = env.cfg

        # Recovery is structurally different from BIRL: no phase, no commands.
        # _phase_mode='none' signals to train.py / mirror code to skip phase plumbing.
        self._phase_mode = 'none'
        self._action_mode = 'absolute'

        # ─── Action setup (10-dim absolute + low-pass) ─────────────────────
        self._lp_alpha = float(getattr(self.cfg.action, 'action_lowpass_alpha', 0.5))
        abs_low = getattr(self.cfg.action, 'abs_low_ranges', None)
        abs_high = getattr(self.cfg.action, 'abs_high_ranges', None)
        if abs_low is not None and abs_high is not None:
            self.action_low = to_torch(abs_low, device=self.device)
            self.action_high = to_torch(abs_high, device=self.device)
        else:
            # Fall back to URDF joint limits.
            self.action_low = torch.as_tensor(self.env.dof_pos_limits[:, 0], device=self.device)
            self.action_high = torch.as_tensor(self.env.dof_pos_limits[:, 1], device=self.device)
        assert len(self.action_low) == self.env.num_dofs, \
            f"RecoveryTask requires {self.env.num_dofs}-dim action, got {len(self.action_low)}"

        self.ref_joint_action = to_torch(self.cfg.action.ref_joint_pos, device=self.device).repeat(self.num_envs, 1)
        self.current_joint_act = self.ref_joint_action.clone()
        self.previous_joint_act = self.current_joint_act.clone()
        self._lp_target = self.current_joint_act.clone()
        self.joint_act_for_pd = self.current_joint_act.clone()
        self.joint_action_limit_low = torch.as_tensor(self.env.dof_pos_limits[:, 0], device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_high = torch.as_tensor(self.env.dof_pos_limits[:, 1], device=self.device).repeat(self.num_envs, 1)

        # Action / net_out histories for smoothness rewards
        self.action_history = deque(maxlen=4)
        self.net_out_history = deque(maxlen=3)
        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_joint_act.clone())
        for _ in range(self.net_out_history.maxlen):
            self.net_out_history.append(torch.zeros_like(self.action_low).repeat(self.num_envs, 1))

        # ─── Stubs to keep wrapper / debug paths happy ─────────────────────
        # Not all are used by recovery, but several are referenced by env / wrapper.
        self.commands = torch.zeros(self.num_envs, max(self.cfg.command.num_commands, 8),
                                    dtype=torch.float, device=self.device)
        self.static_flag = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.heading_ref = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.last_ang_vel_z = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.foot_air_time = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self.last_foot_frc = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self.foot_frc_acc = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self.last_foot_vel = torch.zeros(self.num_envs, self.num_legs * 3, dtype=torch.float, device=self.device)
        # Phase-shaped tensors (zeros) — kept so any shared code paths survive
        self.foot_phase = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self.pm_phase = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.pm_f = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self.foot_swing_mask = torch.zeros(self.num_envs, self.num_legs, dtype=torch.bool, device=self.device)
        self.foot_support_mask = torch.ones(self.num_envs, self.num_legs, dtype=torch.bool, device=self.device)
        self.phase_modulator = None  # never used; obs has no phase slot
        self._has_ref = False

        # ─── Sensor mirrors (reset in step) ────────────────────────────────
        self.joint_pos = self.env.joint_pos.clone()
        self.joint_vel = self.env.joint_vel.clone()
        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.joint_tau = torch.zeros_like(self.joint_pos)
        self.base_acc = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.base_ang_vel = self.env.base_ang_vel.clone()
        self.base_euler = self.env.base_euler.clone()
        self.base_lin_vel = self.env.base_lin_vel.clone()
        self.foot_frc = self.env.foot_frc.clone()
        self.foot_height = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self.foot_pos_hd = self.env.foot_pos_hd.clone()
        self.foot_vel = self.env.foot_vel.clone()

        # ─── Init-state pool (settled fallen poses) ────────────────────────
        init_path = getattr(self.cfg.task, 'init_states_path', None)
        if init_path is None:
            raise ValueError("RecoveryTask requires cfg.task.init_states_path "
                             "(e.g. data/recovery_init_states.npz)")
        self._load_init_pool(init_path)
        # PD target on episode reset: 'sampled' = match injected qpos (limp),
        # 'ref' = standing reference (PD actively supports). See _inject_init_states.
        self._init_pd_target = getattr(self.cfg.task, 'init_pd_target', 'sampled')
        if self._init_pd_target not in ('sampled', 'ref'):
            raise ValueError(f"task.init_pd_target must be 'sampled' or 'ref', "
                             f"got '{self._init_pd_target}'")
        print(f"[RecoveryTask] init_pd_target={self._init_pd_target}")

        # ─── Curriculum (optional) ─────────────────────────────────────────
        cur_cfg = getattr(self.cfg.task, 'curriculum', None)
        cur_enabled = True if cur_cfg is None else bool(getattr(cur_cfg, 'enabled', True))
        replay_old_frac = 0.20 if cur_cfg is None else float(getattr(cur_cfg, 'replay_old_frac', 0.20))
        nominal_h = float(getattr(self.cfg.task, 'target_height', 0.45))
        if cur_enabled:
            self._curriculum = RecoveryCurriculum(
                pool_quat_wxyz=self._pool_quat_wxyz_np,
                pool_base_pos=self._pool_base_pos_np,
                pool_pose_label=self._pool_pose_label,
                nominal_height=nominal_h,
                replay_old_frac=replay_old_frac,
                device=self.device,
            )
        else:
            self._curriculum = None
        # Tracks which envs were sampled this episode, to record outcomes on done.
        self._episode_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ─── Reward groups ─────────────────────────────────────────────────
        groups = list(getattr(self.cfg.reward, 'groups_enabled', []) or [])
        if not groups:
            # default: dense static shaping only
            groups = [GROUP_SHAPING_STATIC]
        self._groups_enabled = set(groups)
        self.rew_weights = self.cfg.reward.to_dict() if self.cfg.reward is not None else {}

        # Targets. Success criterion is now ||q - q_ref|| AND tilt — z height is
        # observable but not gated, since PD-to-ref already lands at z≈0.44m
        # (the issue was *policy* squatting, not PD pulling low). _target_height
        # is kept only as an informational reference for logging.
        self._target_height = float(getattr(self.cfg.task, 'target_height', 0.45))
        self._success_pose_err = float(getattr(self.cfg.task, 'success_pose_err', 0.3))
        self._success_tilt = float(getattr(self.cfg.task, 'success_tilt_deg', 25.0))

        # ─── Obs builder ───────────────────────────────────────────────────
        obs_cfg = self.cfg.observation
        obs_history_len = obs_cfg.history if obs_cfg is not None and obs_cfg.history is not None else 1
        self.obs_history = deque(maxlen=obs_history_len)
        self.cri_obs_history = deque(maxlen=obs_history_len)

        if obs_cfg is None or obs_cfg.slots is None:
            raise ValueError("RecoveryTask requires explicit observation.slots in config")
        self.obs_builder = ObsBuilder(self, slot_names=obs_cfg.slots)

        # Bookkeeping
        self._last_rew_components = None
        self.rew_names = None
        self.extra_info = {}

        # Sample initial states for all envs (the wrapper will call reset later
        # to actually inject them; but we pre-populate so first observation has
        # the right state structure).
        self._sampled_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._inject_init_states(torch.arange(self.num_envs, device=self.device))

        # Prime obs history
        for _ in range(self.obs_history.maxlen):
            self.obs_history.append(self.pure_observation())
        for _ in range(self.cri_obs_history.maxlen):
            self.cri_obs_history.append(self.pure_critic_observation())

    # ──────────────────────────────────────────────────────────────────────
    # Init pool
    # ──────────────────────────────────────────────────────────────────────

    def _load_init_pool(self, path):
        d = np.load(path, allow_pickle=False)
        # NPZ keys: base_pos, base_quat (wxyz), base_lin_vel, base_ang_vel,
        #           joint_pos, joint_vel, pose_label, ...
        N = d['base_pos'].shape[0]
        # Convert quaternion wxyz → xyzw (Isaac convention).
        q_wxyz = torch.as_tensor(d['base_quat'], dtype=torch.float, device=self.device)
        q_xyzw = torch.stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]], dim=-1)

        self._pool_base_pos     = torch.as_tensor(d['base_pos'],     dtype=torch.float, device=self.device)
        self._pool_base_quat    = q_xyzw
        self._pool_base_lin_vel = torch.as_tensor(d['base_lin_vel'], dtype=torch.float, device=self.device)
        self._pool_base_ang_vel = torch.as_tensor(d['base_ang_vel'], dtype=torch.float, device=self.device)
        self._pool_joint_pos    = torch.as_tensor(d['joint_pos'],    dtype=torch.float, device=self.device)
        self._pool_joint_vel    = torch.as_tensor(d['joint_vel'],    dtype=torch.float, device=self.device)
        self._pool_size = N
        # Keep raw numpy copies of the metadata used by RecoveryCurriculum
        # (curriculum filters indices once at init; cheap to keep these around).
        self._pool_quat_wxyz_np = np.asarray(d['base_quat'])
        self._pool_base_pos_np  = np.asarray(d['base_pos'])
        # Optional: per-pose-label index for curriculum sampling later.
        if 'pose_label' in d.files:
            self._pool_pose_label = np.array(d['pose_label'])
        else:
            self._pool_pose_label = None
        print(f"[RecoveryTask] loaded {N} init states from {path}")

    def _sample_indices(self, env_ids):
        n = len(env_ids)
        if self._curriculum is not None:
            return self._curriculum.sample_indices(n)
        return torch.randint(0, self._pool_size, (n,), device=self.device)

    def _inject_init_states(self, env_ids):
        """Override env state for env_ids with samples from the init pool.
        Called on reset *after* env.reset has run; we then push the modified
        tensors back to the simulator."""
        idx = self._sample_indices(env_ids)
        self._sampled_idx[env_ids] = idx

        # Joint state (writing to joint_pos/joint_vel which are views into env.dof_states).
        self.env.joint_pos[env_ids] = self._pool_joint_pos[idx]
        self.env.joint_vel[env_ids] = self._pool_joint_vel[idx]
        # Pre-populate "reset_joint_pos" so future internal env resets match — defensive.
        self.env.reset_joint_pos[env_ids] = self._pool_joint_pos[idx]
        self.env.reset_joint_vel[env_ids] = self._pool_joint_vel[idx]

        # Root state. Layout: [pos(3) | quat_xyzw(4) | lin_vel(3) | ang_vel(3)]
        # Add env_origins so the robots stay in their assigned cells.
        base_pos = self._pool_base_pos[idx].clone()
        base_pos[:, :3] = base_pos[:, :3] + self.env.env_origins[env_ids]
        self.env.root_states[env_ids, 0:3]  = base_pos
        self.env.root_states[env_ids, 3:7]  = self._pool_base_quat[idx]
        self.env.root_states[env_ids, 7:10] = self._pool_base_lin_vel[idx]
        self.env.root_states[env_ids, 10:13] = self._pool_base_ang_vel[idx]

        # Push to simulator.
        from isaacgym import gymtorch
        env_id_int32 = env_ids.to(dtype=torch.int32)
        self.env.gym.set_dof_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.dof_states),
            gymtorch.unwrap_tensor(env_id_int32),
            len(env_id_int32),
        )
        self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.root_states),
            gymtorch.unwrap_tensor(env_id_int32),
            len(env_id_int32),
        )

        # Reset action buffers so PD doesn't snap from old target.
        # init_pd_target='sampled' (default, backward-compat): PD target = current
        # joint pose → zero PD error → zero torque at reset → robot is "limp"
        # and gravity acts before policy can react. OK for fallen poses (already
        # on ground) but disastrous for upright/perturbed init.
        # init_pd_target='ref': PD target = standing reference → PD actively
        # pulls toward standing → robot is supported from step 0 → policy learns
        # small modulations around a stable baseline (Cassie/H1 community std).
        if self._init_pd_target == 'ref':
            target = self.ref_joint_action[env_ids]
        else:
            target = self._pool_joint_pos[idx]
        self.current_joint_act[env_ids] = target.clone()
        self._lp_target[env_ids] = target.clone()
        self.joint_act_for_pd[env_ids] = target.clone()

    # ──────────────────────────────────────────────────────────────────────
    # NullTask interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, env_ids):
        # Inject sampled fallen poses.
        self._inject_init_states(env_ids)

        # Reset action history and bookkeeping.
        for k in range(self.action_history.maxlen):
            self.action_history[k] = self.current_joint_act.clone()
        self.previous_joint_act[env_ids] = self.current_joint_act[env_ids].clone()
        self.foot_air_time[env_ids] = 0.0

        # Refresh sensor mirrors for these envs.
        self.joint_pos = self.env.joint_pos
        self.joint_vel = self.env.joint_vel
        self.base_ang_vel = self.env.base_ang_vel
        self.base_euler = self.env.base_euler
        self.base_lin_vel = self.env.base_lin_vel
        self.foot_frc = self.env.foot_frc

    def step(self):
        self.joint_pos = self.env.joint_pos
        self.joint_vel = self.env.joint_vel
        self.joint_pos_error = self.joint_act_for_pd - self.joint_pos
        self.base_acc = self.env.base_acc.clip(min=-30., max=30.)
        self.base_euler = self.env.base_euler
        self.base_ang_vel = self.env.base_ang_vel
        self.base_lin_vel = self.env.base_lin_vel
        self.foot_frc = self.env.foot_frc
        if self.cfg.terrain.mesh_type in ['trimesh', 'heightfield']:
            self.foot_height = self.env.get_foot_height_to_ground()
        else:
            self.foot_height = self.env.foot_pos_hd[:, [2, 5]]
        self.foot_pos_hd = self.env.foot_pos_hd
        self.foot_vel = self.env.foot_vel

    def action(self, net_out):
        from env.utils.math import scale_transform
        scaled = scale_transform(net_out, self.action_low, self.action_high)
        self.net_out_history.append(scaled)

        # Absolute mode + low-pass filter
        if self._lp_alpha < 1.0:
            self._lp_target = self._lp_alpha * scaled + (1.0 - self._lp_alpha) * self._lp_target
        else:
            self._lp_target = scaled
        self.current_joint_act = torch.clip(
            self._lp_target, self.joint_action_limit_low, self.joint_action_limit_high,
        )

        self.action_history.append(self.current_joint_act.clone())
        self.previous_joint_act = self.current_joint_act.clone()
        self.joint_act_for_pd = self.current_joint_act
        return self.joint_act_for_pd

    # ──────────────────────────────────────────────────────────────────────
    # Observations
    # ──────────────────────────────────────────────────────────────────────

    def pure_observation(self):
        return self.obs_builder.build()

    def pure_critic_observation(self):
        # Privileged critic obs: concatenate the actor obs with raw base
        # linear velocity + base position (privileged in real deploy).
        actor_obs = self.pure_observation()
        privileged = torch.cat([
            self.env.base_lin_vel,
            self.env.base_pos[:, [2]],          # base z
            self.env.foot_frc.clip(0., 200.) * 0.01,
            self.env.foot_pos_hd[:, [2, 5]],     # foot heights
        ], dim=1)
        return torch.cat([actor_obs, privileged], dim=1)

    def observation(self):
        self.obs_buf_pure = self.pure_observation()
        self.obs_history.append(self.obs_buf_pure)
        return torch.cat(list(self.obs_history), dim=-1)

    def critic_observation(self):
        cri = self.pure_critic_observation()
        self.cri_obs_history.append(cri)
        return torch.cat(list(self.cri_obs_history), dim=-1)

    # ──────────────────────────────────────────────────────────────────────
    # Termination / reward
    # ──────────────────────────────────────────────────────────────────────

    def _is_upright(self):
        # Success = body +Z aligned with world +Z (tilt < threshold) AND
        #          joint angles close to the nominal standing pose.
        # Replaces the old z-height gate, since PD-to-ref already produces
        # z≈0.44m; the failure mode we want to reject is *policy-induced*
        # squatting at q ≠ q_ref.
        pg = self.env.projected_gravity
        cos_tilt = -pg[:, 2:3]
        cos_thresh = float(np.cos(np.deg2rad(self._success_tilt)))
        upright = cos_tilt > cos_thresh
        ref_jp = self.ref_joint_action
        pose_err = torch.norm(self.env.joint_pos - ref_jp, dim=1, keepdim=True)
        pose_ok = pose_err < self._success_pose_err
        return (upright & pose_ok).float()

    def terminate(self):
        # Termination conditions for recovery:
        #   - timeout (handled by env.time_out_buf)
        #   - hard joint limit violation (any joint at limit AND tracking)
        #   - termination_contact (excessive contact force on non-foot bodies)
        time_out = torch.unsqueeze(self.env.time_out_buf, 1)
        con_over = torch.any(
            torch.norm(
                self.env.contact_forces[:, self.env.termination_contact_indices, :],
                dim=-1, keepdim=True,
            ) > 100.0,  # tolerate normal ground contact with body
            dim=1,
        )
        done = time_out | con_over

        # Curriculum bookkeeping: when an env finishes, record success/failure.
        # Episode is "successful" iff the robot ended upright.
        if self._curriculum is not None:
            done_mask = done.squeeze(-1)
            if done_mask.any():
                upright = self._is_upright().squeeze(-1).bool()
                outcomes = upright[done_mask]
                self._curriculum.record_outcomes(outcomes)
                self._curriculum.maybe_advance()

        return done

    def reward(self):
        components = []
        names = []

        is_upright = self._is_upright()  # [num_envs, 1] in {0., 1.}

        # ─── shaping_static: pose-match + orientation (always-on baseline) ─
        if GROUP_SHAPING_STATIC in self._groups_enabled:
            # Pose match: exp(-k * ||q - q_ref||^2). Replaces height tracking —
            # PD-to-ref equilibrium produces z≈0.44m anyway, so chasing height
            # was redundant; the real failure mode is the policy *moving* the
            # joints into a squat to dodge pushes. Penalizing ||q-q_ref||
            # directly attacks that. k=8 gives ~0.5 reward at ||err||=0.3
            # (success threshold) and ~0 at ||err||=0.8 (deep squat).
            ref_jp = self.ref_joint_action
            jp_err_sq = torch.sum((self.env.joint_pos - ref_jp) ** 2, dim=1, keepdim=True)
            k_p = float(self.rew_weights.get('pose_match_k', 8.0))
            pose_rew = torch.exp(-k_p * jp_err_sq)

            # Orientation: exp(-k * tilt^2). k=0.5 → ~0.85 reward at 25° tilt.
            pg_z = self.env.projected_gravity[:, [2]]
            cos_tilt = (-pg_z).clip(-1., 1.)
            tilt = torch.acos(cos_tilt)
            k_o = float(self.rew_weights.get('orientation_k', 0.5))
            orient_rew = torch.exp(-k_o * tilt ** 2)

            w_p = float(self.rew_weights.get('pose_match', 1.0))
            w_o = float(self.rew_weights.get('orientation', 1.0))
            components += [w_p * pose_rew, w_o * orient_rew]
            names += ['pose_match', 'orientation']

            # Height tracking: exp(-k * (z - target)^2). Re-added for v6 — v5
            # by_label eval showed pose_match is satisfied by lying flat with
            # legs straight (joints match ref at z≈0.17m), so without a height
            # signal nothing rewards the *act of getting up*. Same shape as
            # the pre-pose_match version. Default w=0 keeps it off for older
            # configs; v6 turns it on.
            w_h = float(self.rew_weights.get('height', 0.0))
            if w_h != 0.0:
                z = self.env.base_pos[:, [2]]
                k_h = float(self.rew_weights.get('height_k', 2.0))
                height_rew = torch.exp(-k_h * (z - self._target_height) ** 2)
                components.append(w_h * height_rew)
                names.append('height')

        # ─── shaping_phased: phase-conditioned shaping (BD_X-inspired) ────
        # Reward profile changes over the episode using episode_progress p∈[0,1]:
        #   early (p→0): orient_w high, pose_w low → "get torso upright first"
        #   late  (p→1): orient_w low,  pose_w high → "match nominal pose"
        #   foot_contact bonus only kicks in for p > foot_phase_gate
        # Same physical signals as shaping_static, but weights schedule over time.
        if GROUP_SHAPING_PHASED in self._groups_enabled:
            p = (self.env.episode_length_buf.float()
                 / max(self.env.max_episode_length, 1)).unsqueeze(-1)

            ref_jp = self.ref_joint_action
            jp_err_sq = torch.sum((self.env.joint_pos - ref_jp) ** 2, dim=1, keepdim=True)
            k_p = float(self.rew_weights.get('pose_match_k', 8.0))
            pose_rew_b = torch.exp(-k_p * jp_err_sq)

            pg_z = self.env.projected_gravity[:, [2]]
            cos_tilt = (-pg_z).clip(-1., 1.)
            tilt = torch.acos(cos_tilt)
            k_o = float(self.rew_weights.get('orientation_k', 0.5))
            orient_rew_b = torch.exp(-k_o * tilt ** 2)

            # Linear schedule: weight slides between early and late values.
            o_early = float(self.rew_weights.get('orient_w_early', 2.0))
            o_late  = float(self.rew_weights.get('orient_w_late',  0.5))
            p_early = float(self.rew_weights.get('pose_w_early',   0.5))
            p_late  = float(self.rew_weights.get('pose_w_late',    2.0))
            orient_w = o_early + (o_late - o_early) * p
            pose_w = p_early + (p_late - p_early) * p

            # Foot contact bonus: both feet in contact, only in late phase.
            foot_thresh = float(self.rew_weights.get('foot_contact_force_thresh', 5.0))
            both_feet = (self.env.foot_frc > foot_thresh).all(dim=1, keepdim=True).float()
            gate = float(self.rew_weights.get('foot_phase_gate', 0.7))
            late_mask = (p > gate).float()
            foot_w = float(self.rew_weights.get('foot_contact', 1.0))
            foot_rew = both_feet * late_mask

            components += [
                orient_w * orient_rew_b,
                pose_w * pose_rew_b,
                foot_w * foot_rew,
            ]
            names += ['orient_phased', 'pose_phased', 'foot_contact_late']

        # ─── stability: bonus when upright with low base velocity ─────────
        if GROUP_STABILITY in self._groups_enabled:
            base_lin_v = torch.norm(self.env.base_lin_vel, dim=1, keepdim=True)
            base_ang_v = torch.norm(self.env.base_ang_vel, dim=1, keepdim=True)
            stability = torch.exp(-1.0 * base_lin_v) * torch.exp(-0.5 * base_ang_v)
            stab_rew = stability * is_upright

            w_s = float(self.rew_weights.get('stability', 1.0))
            components.append(w_s * stab_rew)
            names.append('stability')

        # ─── smoothness: action-rate / jerk / torque penalties ───────────
        if GROUP_SMOOTHNESS in self._groups_enabled:
            a_now = self.action_history[-1]
            a_prev = self.action_history[-2]
            a_pp   = self.action_history[-3]
            act_rate = torch.sum((a_now - a_prev) ** 2, dim=1, keepdim=True)
            act_jerk = torch.sum((a_now - 2 * a_prev + a_pp) ** 2, dim=1, keepdim=True)
            joint_v = torch.sum(self.env.joint_vel ** 2, dim=1, keepdim=True)
            torque_pen = torch.sum(self.env.react_tau ** 2, dim=1, keepdim=True)
            # Hard threshold saturation penalty: only counts force above
            # sat_thresh × torque_limit. Sharp barrier — gradient only fires
            # above the threshold; policy gets nothing for staying below it.
            sat_thresh = float(self.rew_weights.get('torque_sat_thresh', 0.8))
            over = (torch.abs(self.env.react_tau)
                    - sat_thresh * self.env.torque_limits).clip(min=0.)
            torque_sat = torch.sum(over, dim=1, keepdim=True)
            # Soft normalized-squared penalty: Σ (|τ_i| / τ_max_i)². Always
            # provides gradient (no threshold), dimensionless across joints
            # with different effort limits. Use this when training from
            # scratch — the hard barrier is for breaking already-trained
            # policies out of "torque-addicted" local optima, but lacks the
            # smooth signal needed to shape a fresh policy.
            tau_norm = self.env.react_tau / self.env.torque_limits.unsqueeze(0)
            torque_norm_sq = torch.sum(tau_norm ** 2, dim=1, keepdim=True)
            # Hinge penalty: only fires when commanded ratio > 1.0× rated
            # effort. Below rated: zero gradient (free budget). Above rated:
            # quadratic growth — matches real motor saturation, since hardware
            # caps at ~rated and there's no user-configurable limit on the
            # Unitree GO-M8010-6. Training under this penalty produces a
            # policy that doesn't *rely* on saturation tolerance.
            tau_excess = (torch.abs(tau_norm) - 1.0).clip(min=0.)
            torque_excess = torch.max(tau_excess ** 2, dim=1, keepdim=True).values
            # Smooth saturation surrogate: τ - τ_max·tanh(τ/τ_max). For |τ| ≪
            # τ_max it's ≈ 0 (no penalty in normal range); as |τ| grows past
            # rated, it asymptotes to τ - τ_max·sign(τ). Crucially, the
            # gradient is non-zero everywhere (including past the MuJoCo
            # forcerange clip), so PPO can pull a "torque-addicted" policy
            # out of a ballistic basin — unlike the hinge above, whose
            # gradient over the clipped portion is killed by the simulator.
            tau_smooth_excess = self.env.react_tau - self.env.torque_limits.unsqueeze(0) * torch.tanh(tau_norm)
            torque_smooth_sat = torch.sum(tau_smooth_excess ** 2, dim=1, keepdim=True)

            w_rate    = float(self.rew_weights.get('action_rate',     -0.01))
            w_jerk    = float(self.rew_weights.get('action_jerk',     -0.005))
            w_jvel    = float(self.rew_weights.get('joint_vel_pen',   -0.0005))
            w_torque  = float(self.rew_weights.get('torque_pen',      -0.0001))
            w_sat     = float(self.rew_weights.get('torque_sat_pen',  -0.01))
            w_normsq  = float(self.rew_weights.get('torque_norm_sq_pen', 0.0))
            w_excess  = float(self.rew_weights.get('torque_excess_pen', 0.0))
            w_smooth  = float(self.rew_weights.get('torque_smooth_sat_pen', 0.0))
            components += [
                w_rate    * act_rate,
                w_jerk    * act_jerk,
                w_jvel    * joint_v,
                w_torque  * torque_pen,
                w_sat     * torque_sat,
                w_normsq  * torque_norm_sq,
                w_excess  * torque_excess,
                w_smooth  * torque_smooth_sat,
            ]
            names += ['action_rate', 'action_jerk', 'joint_vel_pen',
                      'torque_pen', 'torque_sat', 'torque_norm_sq',
                      'torque_excess', 'torque_smooth_sat']

        # ─── safety: joint-limit and contact-force penalties ─────────────
        if GROUP_SAFETY in self._groups_enabled:
            # Joint position margin to limits (penalize approaching either side).
            jp = self.env.joint_pos
            margin_lo = (jp - self.joint_action_limit_low).clip(min=0., max=0.05)
            margin_hi = (self.joint_action_limit_high - jp).clip(min=0., max=0.05)
            limit_pen = -torch.sum((0.05 - margin_lo) ** 2 + (0.05 - margin_hi) ** 2, dim=1, keepdim=True)

            # Excessive contact force (any rigid body)
            cf = torch.norm(self.env.contact_forces[:, self.env.termination_contact_indices, :], dim=-1)
            cf_max = cf.max(dim=1, keepdim=True).values
            contact_pen = -torch.clip(cf_max - 50.0, min=0., max=200.) * 0.001

            w_lim = float(self.rew_weights.get('joint_limit',  1.0))
            w_con = float(self.rew_weights.get('contact_force',1.0))
            components += [w_lim * limit_pen, w_con * contact_pen]
            names += ['joint_limit', 'contact_force']

        # ─── Terminal reward (applied only on the last step of episode) ───
        # Continuous pose match against ref_joint_pos: penalizes "upright but
        # squatting" sinks where height/tilt are partially satisfied without
        # leg extension.
        if GROUP_TERMINAL in self._groups_enabled:
            terminal = (self.env.episode_length_buf >= self.env.max_episode_length - 1).unsqueeze(-1).float()
            ref_jp = self.ref_joint_action
            jp_err = torch.sum((self.env.joint_pos - ref_jp) ** 2, dim=1, keepdim=True)
            k_jp = float(self.rew_weights.get('terminal_pose_k', 1.0))
            pose_match = torch.exp(-k_jp * jp_err)  # ∈ (0, 1], 1 when matching ref
            success_bonus = is_upright * pose_match * terminal
            failure_pen = (1.0 - is_upright) * terminal * (-1.0)
            w_t = float(self.rew_weights.get('terminal', 5.0))
            components.append(w_t * (success_bonus + failure_pen))
            names.append('terminal')

        if not components:
            zero = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            components = [zero]
            names = ['zero']

        # Per-component rewards × dt, clipped (same convention as BIRLTask).
        rew = torch.cat(
            [torch.clip(c, min=-4., max=5.) * self.env.dt for c in components],
            dim=1,
        )
        self.rew_names = names
        self._last_rew_components = rew.detach()
        self._last_is_upright = is_upright.detach()
        # eval_rew: 1D sum over components (used for task-reward logging).
        eval_rew = rew.sum(dim=1)
        return rew, eval_rew

    # ──────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────

    def info(self):
        out = dict(self.extra_info)
        # Always surface timeouts so PPO can credit value at episode end.
        out['timeouts'] = self.env.time_out_buf
        # Live success rate proxy: fraction of envs currently upright.
        # (Per-layer reward means are already logged via task.rew_names in train.py.)
        if hasattr(self, '_last_is_upright') and self._last_is_upright is not None:
            out['recovery/upright_frac'] = self._last_is_upright.float().mean()
        # Curriculum stats.
        if self._curriculum is not None:
            cinfo = self._curriculum.info()
            out['curriculum/stage'] = float(cinfo['curriculum/stage'])
            out['curriculum/rolling_success'] = float(cinfo['curriculum/rolling_success'])
            out['curriculum/episodes_in_stage'] = float(cinfo['curriculum/episodes_in_stage'])
        return out
