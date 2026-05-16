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
from env.obs_builder import ObsBuilder
import random
from env.utils.math import scale_transform, smallest_signed_angle_between_torch
from collections import deque
from pathlib import Path
import statistics
import torch


@register
class BIRLTask(NullTask):

    def __init__(self, env: LeggedRobotEnv):
        super().__init__(env)
        self.env = env
        self.cmd_id = 0
        self.rew_names = None
        self.num_envs = env.num_envs
        self.num_legs = 2
        self.fixed_commands = {}  # set by play.py to lock specific command dims
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

        # --- Phase frequency command (phase.mode == 'input' only) ---
        # Allocated before _resample_commands so the sampler can populate it.
        if self._phase_mode == 'input':
            _freq_lo = getattr(_phase_cfg, 'freq_low', 2.0) or 2.0
            _freq_hi = getattr(_phase_cfg, 'freq_high', 3.0) or 3.0
            _freq_default = getattr(_phase_cfg, 'freq_default', None)
            if _freq_default is None:
                _freq_default = 0.5 * (_freq_lo + _freq_hi)
            self._freq_low = float(_freq_lo)
            self._freq_high = float(_freq_hi)
            self._freq_default = float(_freq_default)
            # Normalize [freq_low, freq_high] → [-1, 1] for phase_freq_cmd obs.
            self._freq_mid = 0.5 * (self._freq_low + self._freq_high)
            self._freq_scale = max(0.5 * (self._freq_high - self._freq_low), 1e-6)
            self._cmd_freq = torch.full(
                (self.num_envs, 1), self._freq_default,
                dtype=torch.float, device=self.device,
            )
        else:
            self._cmd_freq = None

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
        _support_ratio = getattr(_phase_cfg, 'support_ratio', 0.6) if _phase_cfg is not None else 0.6
        self.convert_phi = tau * _support_ratio

        # Phase modulator — always created (for pm_phase/pm_f tensors), but only
        # used in action/obs when _phase_mode == 'output'.
        self.phase_modulator = PhaseModulator(time_step=env.dt, num_envs=self.num_envs, num_legs=self.num_legs,device=self.device)
        self.phase_modulator.reset(convert_phi=self.convert_phi, env_ids=torch.arange(self.num_envs),
                                   render=self.env.render or self.env.debug or self.env.epochs > 1 or self.env.tcn_name is not None)
        self.foot_phase = self.phase_modulator.phase
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)

        # External phase clock (BD_X style, phase.mode == 'input').
        # _cmd_freq and freq range are set earlier so _resample_commands can use them.
        if self._phase_mode == 'input':
            self._ext_clock = ExternalPhaseClock(
                dt=env.dt, num_envs=self.num_envs, num_legs=self.num_legs,
                device=self.device, default_freq=self._freq_default,
            )
            self._ext_clock.reset(torch.arange(self.num_envs, device=self.device),
                                  render=self.env.render or self.env.debug)
        else:
            self._ext_clock = None

        # --- Action mode: increment, absolute, or residual ---
        self._action_mode = self.cfg.action.action_mode
        assert self._action_mode in ('increment', 'absolute', 'residual'), \
            f"Unknown action.action_mode: '{self._action_mode}'. Must be 'increment', 'absolute', or 'residual'."

        self._lp_alpha = getattr(self.cfg.action, 'action_lowpass_alpha', 1.0)

        if self._action_mode == 'increment':
            self.action_low = to_torch(self.cfg.action.inc_low_ranges, device=self.device)
            self.action_high = to_torch(self.cfg.action.inc_high_ranges, device=self.device)
        elif self._action_mode == 'residual':
            # Residual mode: joint_target = ref + scaled_offset. action_low/high define
            # the symmetric offset band around ref (e.g., ±0.5 rad).
            res_low = self.cfg.action.residual_low_ranges
            res_high = self.cfg.action.residual_high_ranges
            self.action_low = to_torch(res_low, device=self.device)
            self.action_high = to_torch(res_high, device=self.device)
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
        # Lowpass filter state. For absolute: stores the position target.
        # For residual: stores the offset (starts at zero so target = ref + 0 = ref).
        if self._action_mode == 'residual':
            self._lp_target = torch.zeros_like(self.current_joint_act)
        else:
            self._lp_target = self.current_joint_act.clone()

        self.ref_joint_action = to_torch(self.cfg.action.ref_joint_pos, device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_low_over = torch.as_tensor(self.env.dof_pos_limits[:, 0]).repeat(self.num_envs, 1)
        self.joint_action_limit_high_over = torch.as_tensor(self.env.dof_pos_limits[:, 1]).repeat(self.num_envs, 1)

        self.joint_action_limit_low = torch.as_tensor(self.env.dof_pos_limits[:, 0], device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_high = torch.as_tensor(self.env.dof_pos_limits[:, 1], device=self.device).repeat(self.num_envs, 1)

        obs_cfg = self.cfg.observation
        obs_history_len = obs_cfg.history if obs_cfg is not None and obs_cfg.history is not None else 3
        obs_skip = getattr(obs_cfg, 'skip', None) if obs_cfg is not None else None
        if obs_skip is None or obs_skip < 1:
            obs_skip = 1
        self._obs_history_n = obs_history_len
        self._obs_skip = obs_skip
        buf_len = (obs_history_len - 1) * obs_skip + 1
        self.obs_history = deque(maxlen=buf_len)
        self.cri_obs_history = deque(maxlen=buf_len)

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
        # Snapshot of swing duration at touchdown event for cmd_freq-aware
        # air_time reward (legged_gym style). _held_air_delta holds the most
        # recent (swing_dur - target) value per leg, refreshed at each
        # touchdown event — gives continuous per-step reward (vs sparse
        # spike-only which averaged to ~0 in TB and had no learning effect).
        self._td_swing_duration = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)
        self._td_event = torch.zeros(self.num_envs, self.num_legs, dtype=torch.bool, device=self.device)
        self._held_air_delta = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)

        # BDX-R-MjLab style peak-tracking foot swing height: holds the max
        # foot height reached during the current swing per leg. Sparse
        # reward fires at touchdown (=`_td_event`) for (peak/target - 1)².
        self._foot_peak_z = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device)

        # Last-touchdown world-frame xy position per foot, used by the
        # stride_length reward (direct gradient toward LIPM-target step
        # size — added in v11 after no v6-v10 config/form change unlocked
        # stride past ~6 cm). Seeded to current foot position at reset so
        # the first TD doesn't fire spuriously against stale state.
        self._last_td_foot_xy = torch.zeros(self.num_envs, self.num_legs, 2, dtype=torch.float, device=self.device)

        # --- Reference clip state (populated by _load_ref_clips if paths provided) ---
        self._has_ref = False
        self._ref_joint_pos_now = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device)
        self._ref_joint_vel_now = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device)
        self._ref_phase_progress = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        # Resolve ref_clip_paths. Either an explicit list, or auto-loaded from
        # a BC dataset directory (reads episodes.csv → trace paths + relabeled cmds).
        ref_paths = getattr(self.cfg.task, 'ref_clip_paths', []) or []
        ref_dataset = getattr(self.cfg.task, 'ref_clip_dataset', None) or None
        ref_cmd_override = None
        if ref_dataset and not ref_paths:
            ref_paths, ref_cmd_override = self._resolve_ref_clip_dataset(ref_dataset)
        if ref_paths:
            self._load_ref_clips(ref_paths, cmd_override=ref_cmd_override)
            # MIRL-specific config (cmd matching, RSI, annealing)
            self._ref_cmd_match     = bool(getattr(self.cfg.task, 'ref_cmd_match', True) or False)
            self._ref_cmd_topk      = int(getattr(self.cfg.task, 'ref_cmd_topk', 3) or 3)
            self._ref_rsi_state     = bool(getattr(self.cfg.task, 'ref_rsi_state', True) or False)
            self._ref_rsi_jnt_noise = float(getattr(self.cfg.task, 'ref_rsi_jnt_noise', 0.05) or 0.0)
            self._ref_rsi_vel_noise = float(getattr(self.cfg.task, 'ref_rsi_vel_noise', 0.10) or 0.0)
            self._w_imit_start      = float(getattr(self.cfg.task, 'w_imit_start', 0.5) or 0.0)
            self._w_imit_end        = float(getattr(self.cfg.task, 'w_imit_end', 0.5) or 0.0)
            self._w_imit_decay_iter = int(getattr(self.cfg.task, 'w_imit_decay_iter', 5000) or 5000)
            # Iteration counter set externally by train.py — controls w_imit annealing.
            self.train_iter = 0

        self.heading_ref    = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.last_ang_vel_z = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        # Measurement low-pass for cmd-tracking rewards. When > 0, fwd/lateral/yaw
        # tracking errors are computed against the LP-filtered measurement instead
        # of the raw per-step value, so natural gait oscillation isn't penalized.
        # 0.0 = no filter (use raw, backward compat). 0.9 ≈ 150ms time constant.
        # CfgNode returns None for missing keys (not raise AttributeError), so
        # getattr default never fires — explicit `or 0.0` covers both missing and
        # explicit-None cases.
        self._meas_lp_alpha = float(getattr(self.cfg.reward, 'cmd_track_lp_alpha', 0.0) or 0.0)
        self._lp_lin_vel = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)  # [vx, vy]
        self._lp_yaw_rate = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

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

    def _resolve_ref_clip_dataset(self, dataset_dir):
        """Read episodes.csv from a BC dataset dir and return (paths, cmd_override).

        cmd_override: per-clip relabeled cmd from the BC dataset
            (cmd_relabel column) — this is the *realized* mean-velocity
            tuple, not the originally-issued command. Using it as the
            cmd-match key + reward target avoids the "imitation pulls
            student toward 70%-tracking demo" pathology where the student
            inherits the demo's tracking error as a ceiling.

        Used by ref_clip_dataset: directly point to a curated dataset
        (e.g. data/datasets/walk_v34_high_quality) without listing every
        trace in the YAML.
        """
        import csv as _csv
        import json as _json
        repo = Path(__file__).resolve().parents[2]
        ep_csv = repo / dataset_dir / 'episodes.csv'
        if not ep_csv.exists():
            print(f"[BIRLTask] WARN: ref_clip_dataset={dataset_dir!r} has no episodes.csv at {ep_csv}; ignoring")
            return [], None
        paths = []
        relabeled = []
        for row in _csv.DictReader(open(ep_csv)):
            p = row.get('trace')
            if not p:
                continue
            full = repo / p
            if not full.exists():
                continue
            paths.append(str(full))
            cmd_str = row.get('cmd_relabel') or row.get('cmd_orig') or '[0,0,0]'
            try:
                relabeled.append(_json.loads(cmd_str))
            except Exception:
                relabeled.append([0.0, 0.0, 0.0])
        print(f"[BIRLTask] ref_clip_dataset={dataset_dir!r} → {len(paths)} trace paths, "
              f"using cmd_relabel for cmd-match")
        return paths, relabeled

    def _load_ref_clips(self, paths, cmd_override=None):
        """Load reference clips, pad to max length, build per-env tracking tensors.

        Loads joint_pos/vel (always, used by imitation reward) and base
        pos/quat/lin_vel/ang_vel + cmd_const (when present, used by full
        RSI and cmd-matched clip selection).

        cmd_override: optional list of [vx, vy, yaw] triples (one per path)
            that overrides cmd_const from the .npz. Used when traces come
            from a BC dataset that has relabeled cmds (mean realized rather
            than issued cmd). Length must match `paths`.
        """
        clips = []
        if cmd_override is not None:
            assert len(cmd_override) == len(paths), \
                f"cmd_override length {len(cmd_override)} != paths {len(paths)}"
        for i, p in enumerate(paths):
            raw = np.load(p, allow_pickle=True)
            clip = {
                'joint_pos': torch.tensor(raw['joint_pos'], dtype=torch.float, device=self.device),
                'joint_vel': torch.tensor(raw['joint_vel'], dtype=torch.float, device=self.device),
                'T':         raw['joint_pos'].shape[0],
                'dt':        float(raw['dt']),
                'loop':      bool(raw['loop']),
                'skill':     str(raw['skill']),
            }
            # Base state for full RSI (optional — fall back to default reset if absent).
            if 'base_pos' in raw.files:
                clip['base_pos']     = torch.tensor(raw['base_pos'],     dtype=torch.float, device=self.device)
                clip['base_quat']    = torch.tensor(raw['base_quat'],    dtype=torch.float, device=self.device)  # (w,x,y,z)
                clip['base_lin_vel'] = torch.tensor(raw['base_lin_vel'], dtype=torch.float, device=self.device)
                clip['base_ang_vel'] = torch.tensor(raw['base_ang_vel'], dtype=torch.float, device=self.device)
            # cmd: used by cmd-matched assignment. Prefer cmd_override (e.g.
            # relabeled mean-realized cmd from BC dataset) over raw cmd_const
            # (issued cmd) — student trained to match relabeled cmd will
            # inherit the *demonstrated* tracking quality, not the issued/ideal.
            if cmd_override is not None:
                clip['cmd'] = torch.tensor(cmd_override[i], dtype=torch.float, device=self.device)
            elif 'cmd_const' in raw.files:
                clip['cmd'] = torch.tensor(raw['cmd_const'], dtype=torch.float, device=self.device)
            else:
                clip['cmd'] = torch.zeros(3, dtype=torch.float, device=self.device)
            # cmd_freq used by cmd_freq-aware matching: env clock runs at
            # cmd_freq, so we should imitate demos recorded at that freq.
            clip['cmd_freq'] = float(raw['cmd_freq']) if 'cmd_freq' in raw.files else 2.5
            clips.append(clip)
        if not clips:
            return

        max_T = max(c['T'] for c in clips)
        num_clips = len(clips)

        # Pad joint state
        jp = torch.zeros(num_clips, max_T, 10, dtype=torch.float, device=self.device)
        jv = torch.zeros(num_clips, max_T, 10, dtype=torch.float, device=self.device)
        for i, c in enumerate(clips):
            jp[i, :c['T']] = c['joint_pos']
            jv[i, :c['T']] = c['joint_vel']
        self._ref_jp_all = jp                                                              # [n_clips, max_T, 10]
        self._ref_jv_all = jv                                                              # [n_clips, max_T, 10]

        # Pad base state if any clip has it (for full RSI)
        self._ref_has_base = all('base_pos' in c for c in clips)
        if self._ref_has_base:
            self._ref_bp_all  = torch.zeros(num_clips, max_T, 3, dtype=torch.float, device=self.device)
            self._ref_bq_all  = torch.zeros(num_clips, max_T, 4, dtype=torch.float, device=self.device)
            self._ref_blv_all = torch.zeros(num_clips, max_T, 3, dtype=torch.float, device=self.device)
            self._ref_bav_all = torch.zeros(num_clips, max_T, 3, dtype=torch.float, device=self.device)
            for i, c in enumerate(clips):
                self._ref_bp_all[i,  :c['T']] = c['base_pos']
                self._ref_bq_all[i,  :c['T']] = c['base_quat']
                self._ref_blv_all[i, :c['T']] = c['base_lin_vel']
                self._ref_bav_all[i, :c['T']] = c['base_ang_vel']

        # Per-clip cmd_const for cmd-matched selection
        self._ref_cmd_all = torch.stack([c['cmd'] for c in clips], dim=0)                  # [n_clips, 3]
        self._ref_freq_all = torch.tensor([c['cmd_freq'] for c in clips],
                                          dtype=torch.float, device=self.device)            # [n_clips]

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
        print(f"[BIRLTask] Loaded {num_clips} reference clip(s) "
              f"(base_state: {'yes' if self._ref_has_base else 'no'}, "
              f"cmd_const: yes), max_T={max_T}")

    def _assign_ref_clips(self, env_ids):
        """Assign each env a clip (cmd-matched if enabled) + start frame.

        Match key includes cmd_freq when available — env clock runs at this
        rate so we should imitate demos recorded at the same rate (otherwise
        the imitation signal fights the external clock).

        Start frame is phase-aligned to env clock when possible (so the
        demo's stride phase agrees with env clock at t=0), else random (RSI).
        """
        n = len(env_ids)
        if getattr(self, '_ref_cmd_match', False) and self._ref_num_clips > 1:
            # 4D similarity: (vx, vy, yaw, cmd_freq). Yaw weight 0.5, freq weight 1.0.
            cmds = self.commands[env_ids, :3]                                              # [n, 3]
            w = torch.tensor([1.0, 1.0, 0.5], device=self.device)
            d_cmd = ((cmds.unsqueeze(1) - self._ref_cmd_all.unsqueeze(0)) * w).pow(2).sum(-1)  # [n, n_clips]
            if self._cmd_freq is not None:
                env_freq = self._cmd_freq[env_ids]                                         # [n, 1]
                d_freq = (env_freq - self._ref_freq_all.unsqueeze(0)).pow(2)                # [n, n_clips]
                d = d_cmd + d_freq
            else:
                d = d_cmd
            k = min(self._ref_cmd_topk, self._ref_num_clips)
            _, topk_idx = d.topk(k, dim=-1, largest=False)                                 # [n, k]
            pick = torch.randint(0, k, (n,), device=self.device)
            self._ref_clip_id[env_ids] = topk_idx.gather(1, pick.unsqueeze(1)).squeeze(1)
        else:
            self._ref_clip_id[env_ids] = torch.randint(
                0, self._ref_num_clips, (n,), device=self.device
            )
        lengths = self._ref_clip_lengths[self._ref_clip_id[env_ids]]
        # Phase-align start frame to env clock when ext_clock is available.
        # demo recorded at cmd_freq advances phase by step_phase = dt * cmd_freq per
        # frame (in [0, 1) fraction). frames_per_stride = 1/step_phase. Pick a random
        # stride index k ∈ [0, n_strides), then offset within that stride by
        # env_phase_frac * frames_per_stride so demo phase matches env clock at t=0.
        if self._ext_clock is not None:
            env_phase = self._ext_clock.phase_with_offset[env_ids, 0]                       # [n] rad
            env_phase_frac = (env_phase / (2 * torch.pi)) % 1.0                            # [n] in [0,1)
            clip_freq = self._ref_freq_all[self._ref_clip_id[env_ids]]                     # [n]
            dt = float(self.env.dt)
            step_phase = (dt * clip_freq).clamp(min=1e-4)                                  # [n]
            frames_per_stride = 1.0 / step_phase                                            # [n]
            n_strides = (lengths.float() / frames_per_stride).clamp(min=1.0)
            stride_k = (torch.rand(n, device=self.device) * n_strides).floor()             # [n]
            rand_frames = ((stride_k + env_phase_frac) * frames_per_stride).long() % lengths
        else:
            rand_frames = (torch.rand(n, device=self.device) * lengths.float()).long()
        self._ref_frame_idx[env_ids] = rand_frames
        self._ref_frame_frac[env_ids] = 0.0

    def get_reset_state(self, env_ids):
        """Hook called by legged_robot._reset_dofs / _reset_root_states.

        Returns (joint_pos, joint_vel, base_state[N,13]) loaded from clip frames
        (with optional small Gaussian noise) when full RSI is enabled, else None.
        Caller falls back to default reset behavior on None.
        """
        if not (self._has_ref and getattr(self, '_ref_rsi_state', False) and getattr(self, '_ref_has_base', False)):
            return None
        cid = self._ref_clip_id[env_ids]                                                   # [n]
        fid = self._ref_frame_idx[env_ids]                                                 # [n]
        jp = self._ref_jp_all[cid, fid].clone()                                            # [n, 10]
        jv = self._ref_jv_all[cid, fid].clone()
        bp = self._ref_bp_all[cid, fid].clone()                                            # [n, 3] world pos
        bq = self._ref_bq_all[cid, fid].clone()                                            # [n, 4] (w,x,y,z)
        blv = self._ref_blv_all[cid, fid].clone()                                          # [n, 3]
        bav = self._ref_bav_all[cid, fid].clone()                                          # [n, 3]
        # Add DR-friendly Gaussian perturbation
        n_jnt = jp.shape[-1]
        if self._ref_rsi_jnt_noise > 0:
            jp += self._ref_rsi_jnt_noise * (2 * torch.rand_like(jp) - 1)
            jv += self._ref_rsi_vel_noise * (2 * torch.rand_like(jv) - 1)
        if self._ref_rsi_vel_noise > 0:
            blv += self._ref_rsi_vel_noise * (2 * torch.rand_like(blv) - 1)
            bav += self._ref_rsi_vel_noise * (2 * torch.rand_like(bav) - 1)
        # Re-quat-normalize after potential noise (none added here, but cheap insurance)
        bq = bq / (bq.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        # base_state ordering matches Isaac Gym actor_root_state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        # Trace base_quat is (w,x,y,z); Isaac expects (x,y,z,w). Convert.
        bq_xyzw = torch.cat([bq[..., 1:], bq[..., :1]], dim=-1)
        base_state = torch.cat([bp, bq_xyzw, blv, bav], dim=-1)                            # [n, 13]
        return {'joint_pos': jp, 'joint_vel': jv, 'base_state': base_state}

    def w_imit_now(self):
        """Annealed imitation weight; always defined (returns w_imit_start when ref disabled)."""
        if not self._has_ref:
            return 0.0
        decay = max(int(getattr(self, '_w_imit_decay_iter', 5000)), 1)
        a = min(self.train_iter / decay, 1.0)
        return (1 - a) * self._w_imit_start + a * self._w_imit_end

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

        # Optional regime restriction — never sample certain (vx,vy,yaw) combinations.
        # Supported regimes:
        #   'vx_vy_or_vx_yaw': 50% mode A (vx+vy, yaw=0) / 50% mode B (vx+yaw, vy=0).
        #   'pure_and_pairs': 50% pure single-dim cmd, 25% vx+vy, 25% vx+yaw.
        #     Pure modes are split equally among (vx, vy, yaw) → ~16.7% each.
        #     Matches typical deploy distribution where single-dim cmds are most
        #     common, but still trains the useful pair combos.
        regime = getattr(self.cfg.command, 'regime', None)
        if regime == 'vx_vy_or_vx_yaw':
            is_vy_mode = torch.rand(n, device=self.device) < 0.5
            self.commands[env_ids[is_vy_mode], 2] = 0    # mode A: zero yaw
            self.commands[env_ids[~is_vy_mode], 1] = 0   # mode B: zero vy
        elif regime == 'pure_and_pairs':
            # mode 0: pure_vx, 1: pure_vy, 2: pure_yaw, 3: vx+vy, 4: vx+yaw
            # Default ratio biases toward singletons (matches typical deploy mix).
            # Override via cfg.command.regime_weights (5 floats, auto-normalized).
            _w = getattr(self.cfg.command, 'regime_weights', None)
            if _w is None:
                weights = torch.tensor([1/6, 1/6, 1/6, 1/4, 1/4], device=self.device)
            else:
                weights = torch.tensor(list(_w), dtype=torch.float, device=self.device)
                weights = weights / weights.sum()
            mode = torch.multinomial(weights, n, replacement=True)
            # Pure: zero the other two dims
            self.commands[env_ids[mode == 0], 1] = 0     # pure_vx → vy=0
            self.commands[env_ids[mode == 0], 2] = 0     # pure_vx → yaw=0
            self.commands[env_ids[mode == 1], 0] = 0     # pure_vy → vx=0
            self.commands[env_ids[mode == 1], 2] = 0     # pure_vy → yaw=0
            self.commands[env_ids[mode == 2], 0] = 0     # pure_yaw → vx=0
            self.commands[env_ids[mode == 2], 1] = 0     # pure_yaw → vy=0
            # Pairs: zero the third dim
            self.commands[env_ids[mode == 3], 2] = 0     # vx+vy → yaw=0
            self.commands[env_ids[mode == 4], 1] = 0     # vx+yaw → vy=0
        elif regime == 'pure_fwd_stand':
            # Discrete cmd: 50% pure stand (cmd_vx=0), 50% pure walk at vx_hi.
            # vy and yaw are forced to 0 (Phase B forward-only training).
            # Removes interpolation bias of uniform [0, vx_hi] sampling that
            # caused walk_clean_v2 to specialize at mean cmd 0.15 instead of
            # cmd 0.3 (release_eval @ cmd 0.3 showed only 33% survival).
            is_stand = torch.rand(n, device=self.device) < 0.5
            self.commands[env_ids[is_stand], 0] = 0.0
            self.commands[env_ids[~is_stand], 0] = vx_hi
            self.commands[env_ids, 1] = 0.0
            self.commands[env_ids, 2] = 0.0
        elif regime == 'pure_fwd_bwd_stand':
            # Discrete cmd: 1/3 stand (cmd_vx=0), 1/3 forward (cmd_vx=vx_hi),
            # 1/3 backward (cmd_vx=vx_lo). Extends pure_fwd_stand with backward.
            mode = torch.randint(0, 3, (n,), device=self.device)
            self.commands[env_ids[mode == 0], 0] = 0.0
            self.commands[env_ids[mode == 1], 0] = vx_hi
            self.commands[env_ids[mode == 2], 0] = vx_lo
            self.commands[env_ids, 1] = 0.0
            self.commands[env_ids, 2] = 0.0
        elif regime is not None:
            raise ValueError(f"Unknown command.regime: {regime!r}")

        # Phase frequency command (phase.mode == 'input' only): sample per env.
        # Training signal: policy must handle any freq in [freq_low, freq_high].
        if self._cmd_freq is not None:
            self._cmd_freq[env_ids] = torch_rand_float(
                self._freq_low, self._freq_high, (n, 1), device=self.device,
            )

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
        # Residual mode: _lp_target stores offset, reset to 0 (target = ref + 0).
        # Absolute mode: stores position target, reset to current_joint_act.
        if self._action_mode == 'residual':
            self._lp_target[env_ids] = 0.0
        else:
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
        # Seed measurement LP filters with current values to avoid step-1 transient
        self._lp_lin_vel[env_ids] = self.env.base_lin_vel[env_ids, :2]
        self._lp_yaw_rate[env_ids] = self.env.base_ang_vel[env_ids, 2:3]
        self.joint_act_for_pd[env_ids] = self.current_joint_act[env_ids]
        if self._use_act_filter:
            _alpha_range = getattr(self.cfg.action, 'actuator_filter_alpha_range', [0.3, 0.7])
            self.act_filter_alpha[env_ids] = torch.FloatTensor(len(env_ids), 1).uniform_(*_alpha_range).to(self.device)

        # Air time reset
        self.foot_air_time[env_ids] = 0.0
        self._foot_peak_z[env_ids] = 0.0
        self._held_air_delta[env_ids] = 0.0
        self._td_event[env_ids] = False
        self._td_swing_duration[env_ids] = 0.0

        # Stride state: seed last_td_xy to current foot xy so the first
        # touchdown after reset doesn't compute stride against stale data.
        _xy0 = self.env.foot_pos[:, [0, 1, 3, 4]].view(self.num_envs, self.num_legs, 2)
        self._last_td_foot_xy[env_ids] = _xy0[env_ids]

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
            # MIRL: cmd just changed mid-episode → re-match ref clip + RSI frame
            # (only frame reset, not full robot-state RSI which would teleport
            # mid-stride). Keeps imitation reward signal aligned with new cmd.
            if self._has_ref:
                self._assign_ref_clips(env_ids)
                self._update_ref_state()

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

        # External phase clock: advance using per-env commanded frequency
        if self._ext_clock is not None:
            self._ext_clock.update(self._cmd_freq)

        # Air time accumulation: reset to 0 on ground contact, else += dt.
        actual_in_air = (self.env.foot_frc < 1.0)
        # Detect touchdown event (was in air last step, on ground now).
        # Snapshot swing duration BEFORE the reset for cmd_freq-aware reward.
        was_in_air = (self.foot_air_time > 0)
        self._td_event = was_in_air & ~actual_in_air
        self._td_swing_duration = torch.where(self._td_event, self.foot_air_time, torch.zeros_like(self.foot_air_time))
        self.foot_air_time = (self.foot_air_time + self.env.dt) * actual_in_air.float()

        # Advance reference frame
        self._advance_ref_frames()


    def observation(self):
        self.obs_buf_pure = self.pure_observation()
        self.obs_history.append(self.obs_buf_pure)
        buf = list(self.obs_history)
        idx = [i * self._obs_skip for i in range(self._obs_history_n)]
        return torch.cat([buf[i] for i in idx], dim=-1)

    def critic_observation(self):
        pure_obs_buf = self.pure_critic_observation()
        self.cri_obs_history.append(pure_obs_buf)
        buf = list(self.cri_obs_history)
        idx = [i * self._obs_skip for i in range(self._obs_history_n)]
        return torch.cat([buf[i] for i in idx], dim=-1)

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

        # Phase signals — PRIVILEGED for critic (actor doesn't see them in
        # no-clock-in-obs setups). Critic needs the clock to predict value
        # of foot_phase / foot_clr / foot_supt rewards (which depend on
        # clock-derived swing/support masks). Without this, value loss
        # stays high because clock is hidden Markov state from critic's
        # perspective.
        if self._phase_mode == 'output':
            parts.extend([
                self.pm_phase * self.static_flag,
                (self.pm_f * 0.3 - 1.) * self.static_flag,
            ])
        elif self._phase_mode == 'input':
            # External clock sin/cos per leg [n, 4] + frequency normalized
            ext_phase = self._ext_clock.phase_with_offset    # [n, num_legs]
            freq_norm = (self._cmd_freq * 0.3 - 1.0)         # [n, 1]
            parts.extend([
                torch.sin(ext_phase) * self.static_flag,
                torch.cos(ext_phase) * self.static_flag,
                freq_norm * self.static_flag,
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
            elif self._action_mode == 'residual':
                # joint_out is the OFFSET from ref. Target = ref + offset.
                # Lowpass smooths the offset before adding.
                if self._lp_alpha < 1.0:
                    self._lp_target = self._lp_alpha * joint_out + (1.0 - self._lp_alpha) * self._lp_target
                else:
                    self._lp_target = joint_out
                self.current_joint_act = self.ref_joint_action + self._lp_target
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

    def terminate(self):
        time_out = torch.unsqueeze(self.env.time_out_buf, 1)
        # Tilt termination angle (radians). Default 0.7 (≈40°) matches the
        # legacy walk_v34 setting. BDX-R-MjLab uses 70° = 1.22 rad. Set via
        # `task.tilt_termination_angle` in config.
        _tilt_lim = float(getattr(self.cfg.task, 'tilt_termination_angle', 0.7) or 0.7)
        twist_over = torch.abs(self.env.base_euler[:, [0]]) > _tilt_lim
        twist_over |= torch.abs(self.env.base_euler[:, [1]]) > _tilt_lim

        pos_over = self.env.base_pos_hd[:, [2]] < 0.2

        jact_over = torch.sum(torch.logical_and(torch.abs(self.action_history[-1] - self.joint_action_limit_low_over) < 0.02,
                                                torch.abs(self.joint_pos - self.joint_action_limit_low_over) < 0.02), dim=1,
                              keepdim=True) >= 1
        jact_over |= torch.sum(torch.logical_and(torch.abs(self.action_history[-1] - self.joint_action_limit_high_over) < 0.02,
                                                 torch.abs(self.joint_pos - self.joint_action_limit_high_over) < 0.02), dim=1,
                               keepdim=True) >= 1
        con_over = torch.any(
            torch.norm(self.env.contact_forces[:, self.env.termination_contact_indices, :], dim=-1, keepdim=True) > 1.,
            dim=1)
        if self.env.render or self.env.epochs > 1 or self.env.tcn_name is not None:
            done = con_over | time_out | pos_over | twist_over
            if torch.sum(torch.any(done, dim=-1)) > 10:
                metrics = OrderedDict({"pos_over": pos_over, "twist_over": twist_over, 'time_out': time_out})
                print(f'==============================================')
                print(metrics)
                print(f'==============================================')
        else:
            done = con_over | pos_over | twist_over | time_out | jact_over
        return done

    def reward(self):
        constant_rew = to_torch([1.]).repeat(self.num_envs, 1)
        # `lin_vel_x_norm` historically scaled stability rewards (balance,
        # vertical_vel, ang_vel) and regulators in low-cmd modes. Norm of
        # [vx, vy] cmd only, clipped to [0.3, 2.0]. Stability rewards still
        # use this. Regulators bypass it via reg_use_norm_scaling=false
        # (walk_v34+) since the 1/norm amplification fights tracking.
        lin_vel_x_norm = torch.clip(torch.norm(self.commands[:, [0, 1]], dim=1, keepdim=True), min=0.3, max=2.) + 0.2
        # Regulator scaling factor: industry-standard repos (legged_gym, G1,
        # humanoid-gym) use constant reg coefficients. Our 1/lin_vel_x_norm
        # scaling amplifies regs in pure_yaw mode (norm=0.5 floor → factor 2),
        # fighting yaw tracking. Disable via reward.reg_use_norm_scaling=false.
        if getattr(self.cfg.reward, 'reg_use_norm_scaling', True):
            reg_norm_inv = reg_norm_inv
        else:
            reg_norm_inv = torch.ones_like(lin_vel_x_norm)
        _bh_slope = float(getattr(self.cfg.reward, 'base_heit_slope', 70.0) or 70.0)
        _bh_target = float(getattr(self.cfg.reward, 'base_heit_target', 0.45) or 0.45)
        base_heit_rew = torch.exp(-_bh_slope * (self.env.base_pos[:, [2]] - _bh_target) ** 2)

        # Unitree G1 / legged_gym style: strict L2 penalty on height deviation.
        # Unlike base_heit_rew (positive bell, saturates), this has unbounded
        # quadratic tail → policy keeps pulling toward target. Used by
        # walk_rl_v5+ to escape the 0.38 m local optimum.
        base_height_l2 = -((self.env.base_pos[:, [2]] - _bh_target) ** 2)   # ≤ 0

        # Unitree G1 style: strict L2 penalty on projected-gravity XY (i.e.
        # body tilt). Equivalent to penalizing roll² + pitch², but uses
        # projected gravity which is what the IMU actually reads. No saturation.
        flat_orient_l2 = -torch.sum(
            self.env.projected_gravity[:, :2] ** 2, dim=-1, keepdim=True
        )   # ≤ 0

        # BDX-R-MjLab style: positive bell on upright = exp(-||g_xy||² / σ²).
        # Softer than the L2 penalty above. Maxes at 1.0 (perfect upright),
        # decays to 0 as the body tilts. Good for stable training.
        _upright_std = float(getattr(self.cfg.reward, 'upright_std', 0.4) or 0.4)
        upright_rew = torch.exp(
            -torch.sum(self.env.projected_gravity[:, :2] ** 2, dim=-1, keepdim=True)
            / (_upright_std ** 2)
        )                                                              # [0, 1]

        # BDX-R-MjLab style: speed-conditional pose deviation reward.
        # std varies with commanded speed — TIGHT at stand (forces joint_pos
        # ≈ ref_joint_pos = upright pose), LOOSE during walk (allows legs to
        # swing without being penalized for deviation). Solves our recurring
        # "policy walks crouched because jnt_pos_err weight=0.02 too weak at
        # stand AND too restrictive during walk" issue.
        _ps_std_stand = float(getattr(self.cfg.reward, 'pose_std_standing', 0.15) or 0.15)
        _ps_std_walk  = float(getattr(self.cfg.reward, 'pose_std_walking',  0.45) or 0.45)
        _ps_std_run   = float(getattr(self.cfg.reward, 'pose_std_running',  0.80) or 0.80)
        _ps_walk_thr  = float(getattr(self.cfg.reward, 'pose_walking_threshold', 0.5) or 0.5)
        _ps_run_thr   = float(getattr(self.cfg.reward, 'pose_running_threshold', 1.5) or 1.5)
        # Total commanded speed (lin xy magnitude + |yaw_rate|).
        _cmd_speed = (torch.linalg.norm(self.commands[:, :2], dim=-1)
                      + torch.abs(self.commands[:, 2])).unsqueeze(-1)    # [n, 1]
        _ps_std = (
            _ps_std_stand * (_cmd_speed < _ps_walk_thr).float()
            + _ps_std_walk * ((_cmd_speed >= _ps_walk_thr) & (_cmd_speed < _ps_run_thr)).float()
            + _ps_std_run  * (_cmd_speed >= _ps_run_thr).float()
        )                                                                # [n, 1]
        _pose_err_sq = (self.env.joint_pos - self.ref_joint_action) ** 2   # [n, J]
        # Per-joint std support (BDX-R-MjLab uses dict {joint_pattern: std}).
        # When `pose_std_walking_per_joint` (and friends) is a list of J floats,
        # each joint gets its own σ — hip_pitch/knee can have loose σ while
        # hip_roll/hip_yaw stay tight. Same shape as joint dim.
        _ps_per_joint = getattr(self.cfg.reward, 'pose_std_per_joint_walking', None)
        if _ps_per_joint is not None:
            # Compile per-joint stds for stand / walk / run.
            _per_stand = to_torch(
                self.cfg.reward.pose_std_per_joint_standing, device=self.device
            )                                                            # [J]
            _per_walk  = to_torch(
                self.cfg.reward.pose_std_per_joint_walking, device=self.device
            )                                                            # [J]
            _per_run   = to_torch(
                self.cfg.reward.pose_std_per_joint_running, device=self.device
            )                                                            # [J]
            # Select per-env std vector based on cmd_speed.
            _stand_mask = (_cmd_speed < _ps_walk_thr).float()             # [n, 1]
            _walk_mask  = ((_cmd_speed >= _ps_walk_thr) & (_cmd_speed < _ps_run_thr)).float()
            _run_mask   = (_cmd_speed >= _ps_run_thr).float()
            _ps_std_pj = (
                _stand_mask * _per_stand.unsqueeze(0)
                + _walk_mask * _per_walk.unsqueeze(0)
                + _run_mask  * _per_run.unsqueeze(0)
            )                                                            # [n, J]
            pose_speed_rew = torch.exp(
                -torch.mean(_pose_err_sq / (_ps_std_pj ** 2), dim=-1, keepdim=True)
            )                                                            # [n, 1] in [0, 1]
        else:
            pose_speed_rew = torch.exp(
                -torch.mean(_pose_err_sq / (_ps_std ** 2), dim=-1, keepdim=True)
            )                                                            # [n, 1] in [0, 1]

        # v10 option: gate pose_speed by walking state. When walking, this
        # term unconditionally rewards joint_pos ≈ ref_joint_pos — exactly
        # what shuffle does. Setting pose_speed_walking_gate=0.0 disables
        # it during walking (only enforces upright pose at standstill).
        # NOTE: explicit None check; `getattr(...) or 1.0` would silently
        # rewrite the intended `0.0` to `1.0` (truthiness bug — v15 hit
        # this and ran with effective gate=1.0).
        _gate_raw = getattr(self.cfg.reward, 'pose_speed_walking_gate', 1.0)
        _ps_walking_gate = 1.0 if _gate_raw is None else float(_gate_raw)
        if _ps_walking_gate != 1.0:
            pose_speed_rew = pose_speed_rew * (
                (1.0 - self.static_flag) + _ps_walking_gate * self.static_flag
            )

        # BDX-R-MjLab style additional reward terms.
        #
        # `action_rate_l2`: ||action_t − action_{t-1}||²  — first-derivative
        # penalty on raw net_out. Our existing `act_smo` is 2nd-derivative
        # which is similar but not identical.  net_out_history holds the
        # raw policy outputs (joint-only part for non-output phase modes).
        if len(self.net_out_history) >= 2:
            _act_rate_l2 = -torch.sum(
                (self.net_out_history[-1] - self.net_out_history[-2]) ** 2,
                dim=-1, keepdim=True,
            )                                                            # ≤ 0
        else:
            _act_rate_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        action_rate_l2 = _act_rate_l2

        # `body_ang_vel`: penalise base xy angular velocity (roll/pitch rate).
        # Sum of squares. BDX-R weight −0.05 (full body) or −0.2 (legs).
        body_ang_vel = -torch.sum(
            self.env.base_ang_vel[:, :2] ** 2, dim=-1, keepdim=True
        )                                                                # ≤ 0

        # BDX-R `angular_momentum_penalty`: ||body angular momentum||².
        # We don't compute true whole-body angular momentum (mjlab uses
        # MuJoCo's built-in `root_angmom` sensor). As a proxy use ||ω_base||²
        # weighted by body mass — same scaling intent (penalise rotational
        # motion of the whole robot). For our use this is approximately
        # the same signal since most rotational momentum is base rotation
        # for a biped (legs cancel out at gait midpoint).
        angular_momentum = -torch.sum(
            self.env.base_ang_vel ** 2, dim=-1, keepdim=True
        )                                                                # ≤ 0

        # ── Shared helpers used by BDX-R-style foot rewards below ──────────
        # `cmd_total` is `||cmd_xy|| + |cmd_yaw|` per BDX-R; used by every
        # foot reward's "cmd_active" gate. `foot_xy_vel` is per-foot
        # horizontal speed.
        _cmd_total = (torch.linalg.norm(self.commands[:, :2], dim=-1)
                      + torch.abs(self.commands[:, 2])).unsqueeze(-1)    # [n, 1]
        _foot_xy_vel = torch.stack([
            torch.linalg.norm(self.env.foot_vel[:, 0:2], dim=-1),
            torch.linalg.norm(self.env.foot_vel[:, 3:5], dim=-1),
        ], dim=-1)                                                       # [n, 2]

        # BDX-R `feet_slip`: Σ ||v_xy(foot)||² × in_contact × cmd_active.
        # cmd threshold for slip is 0.01 (lower than other foot rewards' 0.05).
        _slip_cmd_threshold = float(getattr(self.cfg.reward, 'feet_slip_cmd_threshold', 0.01) or 0.01)
        _cmd_active_slip = (_cmd_total > _slip_cmd_threshold).float()
        _in_contact_f = (self.env.foot_frc >= 1.0).float()               # [n, num_legs]
        feet_slip_l2 = -torch.sum(
            (_foot_xy_vel ** 2) * _in_contact_f, dim=-1, keepdim=True
        ) * _cmd_active_slip                                             # ≤ 0

        # `dof_pos_limits`: hinge-style penalty for joint position near limits.
        # Standard legged_gym formulation: distance past a soft limit (95% of
        # range) gets penalised quadratically. Bounded to keep the penalty
        # well-behaved.
        if hasattr(self.env, 'dof_pos_limits'):
            _q = self.env.joint_pos                                       # [n, J]
            _q_low = self.env.dof_pos_limits[:, 0].unsqueeze(0)           # [1, J]
            _q_high = self.env.dof_pos_limits[:, 1].unsqueeze(0)
            _below = (_q_low - _q).clip(min=0.)
            _above = (_q - _q_high).clip(min=0.)
            dof_pos_limits = -torch.sum(_below + _above, dim=-1, keepdim=True)
        else:
            dof_pos_limits = torch.zeros(self.num_envs, 1, device=self.device)

        balance_rew = 0.5 * (base_heit_rew * torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2, max=8.) * torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)) + 1.)

        # Linear cmd-tracking reward: r = 1 - clip(α·|err|, 0, 1.5).
        # Peaks at +1, bottoms at -0.5. Constant gradient → never dies in tail
        # (Gaussian was giving ~50% reward at err=0.4 → no learning pressure).
        #
        # When _meas_lp_alpha > 0, error is computed against an EMA of the
        # measurement (not raw per-step) so natural gait oscillation isn't
        # penalized — policy is free to wobble per-step as long as the mean
        # tracks the command. α=0.9 ≈ 150ms time constant at 67Hz.
        if self._meas_lp_alpha > 0.0:
            a = self._meas_lp_alpha
            self._lp_lin_vel  = a * self._lp_lin_vel  + (1.0 - a) * self.env.base_lin_vel[:, :2]
            self._lp_yaw_rate = a * self._lp_yaw_rate + (1.0 - a) * self.env.base_ang_vel[:, 2:3]
            fwd_meas, lat_meas = self._lp_lin_vel[:, [0]], self._lp_lin_vel[:, [1]]
            yaw_meas = self._lp_yaw_rate
        else:
            fwd_meas = self.env.base_lin_vel[:, [0]]
            lat_meas = self.env.base_lin_vel[:, [1]]
            yaw_meas = self.env.base_ang_vel[:, [2]]

        fwd_err     = torch.abs(self.commands[:, [0]] - fwd_meas)
        lateral_err = torch.abs(self.commands[:, [1]] - lat_meas)
        yaw_err     = torch.abs(self.commands[:, [2]] - yaw_meas)

        # Deadzone: per-step gait oscillation imposes a noise floor (~0.1 m/s
        # for vx, ~0.1 rad/s for yaw). Subtract deadzone from |err| so the
        # policy isn't penalized for tracking within the noise floor — but
        # unlike LP filtering it can't game by alternating large oscillations.
        fwd_dz = float(getattr(self.cfg.reward, 'fwd_err_deadzone', 0.0) or 0.0)
        lat_dz = float(getattr(self.cfg.reward, 'lateral_err_deadzone', 0.0) or 0.0)
        yaw_dz = float(getattr(self.cfg.reward, 'yaw_err_deadzone', 0.0) or 0.0)
        if fwd_dz > 0.0: fwd_err     = torch.clamp(fwd_err     - fwd_dz, min=0.)
        if lat_dz > 0.0: lateral_err = torch.clamp(lateral_err - lat_dz, min=0.)
        if yaw_dz > 0.0: yaw_err     = torch.clamp(yaw_err     - yaw_dz, min=0.)

        # Per-component error slope: r = 1 - clip(slope·|err|, 0, 1.5).
        # Larger slope = steeper penalty = more pressure to track tightly.
        # Default values match the original Gaussian-replacement defaults.
        fwd_slope = float(getattr(self.cfg.reward, 'fwd_err_slope', 2.0) or 2.0)
        lat_slope = float(getattr(self.cfg.reward, 'lateral_err_slope', 3.0) or 3.0)
        yaw_slope = float(getattr(self.cfg.reward, 'yaw_err_slope', 2.0) or 2.0)
        forward_vel_rew = 1.0 - torch.clip(fwd_slope * fwd_err,     min=0., max=1.5)
        lateral_vel_rew = 1.0 - torch.clip(lat_slope * lateral_err, min=0., max=1.5)
        yaw_rate_rew    = 1.0 - torch.clip(yaw_slope * yaw_err,     min=0., max=1.5)

        ang_vel_rew = torch.exp(
            -torch.clip(2. / lin_vel_x_norm, min=0.7, max=6.) * torch.norm(self.env.base_ang_vel[:, :2], dim=1,
                                                                            keepdim=True) ** 2)
        base_acc_rew = -0.4 * reg_norm_inv * torch.norm((self.env.base_acc - to_torch([0, 0, 9.81], device=self.device)) * 0.1, dim=1, keepdim=True)
        base_acc_rew *= self.static_flag

        # v10 option: cmd-gated bell at target |v_z| oscillation.
        # Standing: minimize v_z (keep upright). Walking: peak around
        # `vertical_vel_target_walk` (≈ stride / 2·swing_time, ~0.15 m/s
        # at v=0.3). The legacy form (minimize v_z always) rewards
        # pogo-free standstill in walk mode → contributes to shuffle.
        # See docs/walking_physics_reference.md §5.
        _vv_target = float(getattr(self.cfg.reward, 'vertical_vel_target_walk', 0.0) or 0.0)
        if _vv_target > 0:
            _vv_std_walk   = float(getattr(self.cfg.reward, 'vertical_vel_std_walk',   0.10) or 0.10)
            _vv_std_static = float(getattr(self.cfg.reward, 'vertical_vel_std_static', 0.10) or 0.10)
            _vz_mag = torch.norm(self.env.base_lin_vel[:, [2]], dim=1, keepdim=True)
            _vv_walking = torch.exp(-((_vz_mag - _vv_target) / _vv_std_walk) ** 2)
            _vv_static  = torch.exp(-((_vz_mag - 0.0)        / _vv_std_static) ** 2)
            vertical_vel_rew = _vv_static * (1.0 - self.static_flag) + _vv_walking * self.static_flag
        else:
            vertical_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2., max=10.) * torch.norm(self.env.base_lin_vel[:, [2]], dim=1,
                                                                               keepdim=True) ** 2)
            vertical_vel_rew -= 0.2 * reg_norm_inv * torch.norm(self.env.base_lin_vel[:, [2]], dim=1, keepdim=True) * self.static_flag

        support_foot_index = torch.where(self.env.foot_frc >= 10., True, False)
        swing_foot_index = torch.where(self.env.foot_frc < 1., True, False)

        # Single-support gate — TRUE only when exactly ONE leg is in air and
        # the other on the ground (proper anti-phase walking state).
        # Removes the shuffle exploit: when both feet on ground (shuffle),
        # foot_supt was firing trivially because every grounded foot has 50%
        # chance of also matching its stance_mask. Gating by single-support
        # forces "right leg right time AND the other leg also doing its job".
        _l_air, _r_air = swing_foot_index[:, [0]], swing_foot_index[:, [1]]
        _l_grd, _r_grd = support_foot_index[:, [0]], support_foot_index[:, [1]]
        single_support_state = (
            (_l_air & _r_grd) | (_r_air & _l_grd)
        ).float()                                       # [n, 1], 1 iff single-support

        foot_clear_rew = (
            torch.sum(torch.logical_and(swing_foot_index, self.foot_swing_mask), dtype=torch.float, dim=1, keepdim=True)
            / self.num_legs
        ) * single_support_state

        foot_support_rew = (
            torch.sum(torch.logical_and(support_foot_index, self.foot_support_mask), dtype=torch.float, dim=1, keepdim=True)
            / self.num_legs
        ) * single_support_state * self.static_flag
        foot_clear_rew *= self.static_flag

        # Two formulations of foot height reward, selected by config:
        #
        # 'clip'  (legacy, default for walk_v34 lineage):
        #   reward = sum(swing_mask × 40 × clip(z, 0, 0.05)).clip(max=2)
        #   Caps reward at 5 cm — once foot is at 5 cm no extra reward to go
        #   higher. Combined with -20 × (z - 0.06)+ penalty above 6 cm,
        #   pins swing height to 5-6 cm. With big strides this is too flat
        #   an arc → foot scuffs (v4 duty_factor 0.53).
        #
        # 'l2'  (G1-style, new in v8):
        #   reward = -100 × sum(swing_mask × (z - target)²)
        #   L2 penalty around target. Swing-mask gate ensures support foot
        #   is NOT penalized for being at z=0. tanh-of-velocity gate (the
        #   G1 trick used in v7) was gameable: policy learned to keep
        #   feet stationary to disable the gate. Phase-clock swing_mask
        #   is hard-gated by the clock — policy can't escape via velocity.
        foot_heit_form = getattr(self.cfg.reward, 'foot_heit_form', 'clip') or 'clip'
        if foot_heit_form == 'l2':
            _fh_target = float(getattr(self.cfg.reward, 'foot_heit_target', 0.06) or 0.06)
            _fh_scale  = float(getattr(self.cfg.reward, 'foot_heit_l2_scale', 100.0) or 100.0)
            foot_heit_score = -_fh_scale * (self.foot_height - _fh_target) ** 2
            foot_height_rew = torch.sum(
                self.foot_swing_mask * foot_heit_score, dim=1, keepdim=True
            ) * self.static_flag
        else:
            foot_heit_score = 40. * torch.clip(self.foot_height, min=0.0, max=0.05)
            foot_height_rew = torch.sum(
                self.foot_swing_mask * foot_heit_score, dim=1, keepdim=True
            ).clip(max=2.) * self.static_flag

            foot_height_rew += -20. * torch.sum((self.foot_height - 0.06).clip(min=0.), dim=1, keepdim=True)
            foot_height_rew += -0.2 * torch.sum(self.foot_support_mask * foot_heit_score, dim=1, keepdim=True) * self.static_flag
            foot_height_rew += -0.2 * torch.sum(support_foot_index * foot_heit_score, dim=1, keepdim=True) * self.static_flag

        twist_rew = -torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)

        self.foot_frc_acc = (self.env.foot_frc - self.last_foot_frc).clone()
        foot_soft_rew = -0.1 * torch.clip(reg_norm_inv, min=0., max=1.5) * torch.norm(self.foot_frc_acc, dim=1, keepdim=True) / 100.

        self.last_foot_frc = self.env.foot_frc.clone().detach()

        feet_contact_frc_rew = -torch.norm(self.env.foot_frc * self.foot_swing_mask, dim=1, keepdim=True) * self.static_flag
        feet_contact_frc_rew += -torch.norm((torch.abs(self.env.foot_frc - 55.) * support_foot_index).clip(min=0.), dim=1, keepdim=True)

        clip_foot_h = torch.abs(self.foot_height) + 0.03

        foot_slip_rew = 2. * (lin_vel_x_norm * torch.sum(
            (self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, 0]) * self.commands[:, [0]].sign() * self.foot_swing_mask,
            dim=1, keepdim=True)).clip(min=-0., max=1.) * self.static_flag

        vy_walking = (torch.abs(self.commands[:, [1]]) > 0.1).float()
        # yaw_walking: turning command active. Used below (with vy_walking) to
        # gate hip_yaw/hip_roll-specific penalties — those legitimately need
        # large hip motion when turning OR strafing, so penalize ONLY when
        # neither is requested (i.e., pure forward / standstill).
        yaw_walking = (torch.abs(self.commands[:, [2]]) > 0.1).float()
        foot_slip_rew += -0.5 * torch.norm(torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [1]], dim=-1), dim=1,
                                           keepdim=True) * self.static_flag * (1. - vy_walking)

        foot_slip_rew += 0.3 * torch.norm(torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, :2], dim=-1), dim=1, keepdim=True) * (
                self.static_flag - 1.)

        foot_slip_rew += -0.3 * reg_norm_inv * torch.norm(
            0.1 * torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, :2], dim=-1) / clip_foot_h * self.foot_support_mask, dim=1,
            keepdim=True) * self.static_flag

        foot_vz_rew = -0.1 * torch.clip(reg_norm_inv, min=0., max=1.) * torch.norm(
            torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [2]].clip(max=0.), dim=-1) / clip_foot_h,
            dim=1, keepdim=True) * self.static_flag

        foot_vz_rew += 0.8 * torch.clip(reg_norm_inv, min=0., max=1.) * torch.norm(
            torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [2]].clip(max=0.), dim=-1),
            dim=1, keepdim=True) * (self.static_flag - 1.)

        foot_acc_rew = -0.4 * torch.clip(reg_norm_inv, min=0., max=2.) * torch.norm(self.env.foot_vel[:, [2, 5]], dim=1, keepdim=True)

        action_smooth_rew = -0.3 * torch.clip(reg_norm_inv, min=0., max=2.) * torch.norm(
            self.action_history[-3] - 2. * self.action_history[-2] + self.action_history[-1], dim=1, keepdim=True)
        # net_out smoothness: skip freq prefix when phase.mode == 'output'
        _jo = self.num_legs if self._phase_mode == 'output' else 0  # joint offset
        net_out_smooth_rew = -0.2 * torch.clip(reg_norm_inv, min=0., max=2.) * torch.norm(
            (self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1])[:, _jo:], dim=1, keepdim=True) ** 2

        action_constraint_rew = -0.1 * torch.clip(reg_norm_inv, 0, 1.) * torch.norm((self.current_joint_act - self.ref_joint_action), dim=1, keepdim=True)
        # hip_yaw/hip_roll deviation penalty — only when NEITHER strafing
        # nor turning. Otherwise turning policy needs hip_yaw motion, was
        # being penalized for it.
        action_constraint_rew += -3. * torch.norm(((self.current_joint_act - self.ref_joint_action)[:, [0, 1, 5, 6]]), dim=1, keepdim=True) * self.static_flag * (1. - vy_walking) * (1. - yaw_walking)

        sa_constraint_rew = -0.1 * torch.clip(reg_norm_inv, min=0., max=1.) * torch.norm(self.current_joint_act - self.ref_joint_action, dim=1,keepdim=True) ** 2 * self.static_flag

        sa_constraint_rew += -self.static_flag * torch.clip(reg_norm_inv, 0, 1) * torch.norm(
            ((self.env.joint_pos - self.ref_joint_action)[:, :5] * support_foot_index[:, [0]]), dim=1,
            keepdim=True) ** 2
        sa_constraint_rew += -self.static_flag * torch.clip(reg_norm_inv, 0, 1) * torch.norm(
            ((self.env.joint_pos - self.ref_joint_action)[:, 5:] * support_foot_index[:, [1]]), dim=1,
            keepdim=True) ** 2

        joint_pos_error_rew = - 0.4 * torch.clip(reg_norm_inv, min=0., max=1.) * torch.norm((self.current_joint_act - self.env.joint_pos), dim=1,keepdim=True) ** 2

        joint_velocity_rew = -0.4 * torch.clip(reg_norm_inv, min=0., max=1.) * torch.norm(self.env.joint_vel[:, :], dim=1,keepdim=True) ** 2
        # hip_yaw/hip_roll velocity penalty — same gating as act_const.
        # Yaw and strafe both legitimately need fast hip motion.
        joint_velocity_rew += -torch.clip(reg_norm_inv, 0, 1) * torch.norm(self.env.joint_vel[:, [0, 1, 5, 6]], dim=1,keepdim=True) ** 2 * (1. - vy_walking) * (1. - yaw_walking)

        joint_tor_rew = -0.4 * torch.clip(reg_norm_inv, min=0., max=2.) * torch.sum(
            (torch.abs(self.env.react_tau[:, :]) - self.env.torque_limits[:]).clip(min=0.), dim=1, keepdim=True)

        joint_tor_rew *= self.static_flag

        self.last_foot_vel = self.env.foot_vel.clone().detach()

        # PMF reward: only meaningful when phase.mode == 'output' (freq prefix exists)
        if self._phase_mode == 'output':
            pmf_rew = -0.02 * torch.clip(reg_norm_inv, min=0., max=1.) * torch.norm(
                (self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1])[:, :self.num_legs],
                dim=1, keepdim=True)
            pmf_rew += -0.5 * torch.clip(reg_norm_inv, 0, 1.) * torch.norm(self.net_out_history[-1][:, :self.num_legs] * self.foot_support_mask, dim=1, keepdim=True) ** 2
            pmf_rew *= self.static_flag
        else:
            pmf_rew = torch.zeros(self.num_envs, 1, device=self.device)

        net_out_val_rew = -0.4 * torch.clip(reg_norm_inv, min=0., max=1.) * torch.norm(self.net_out_history[-1][:, _jo:], dim=1, keepdim=True) ** 2
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
            # This drives CADENCE following (policy tries to step at clock rate).
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

        # single_support reward — XOR(left_contact, right_contact). Pure
        # clock-free anti-phase signal. Rewards EXACTLY ONE foot in air at
        # a time. Together with fwd_vel + air_time, replaces all clock-
        # dependent rewards in v12 → v13 minimal intrinsic setup.
        #
        # Walk @ duty 0.60 → single_support ~0.80 of cycle → reward +0.60
        # Shuffle @ duty 0.85 → single_support ~0.30 → reward -0.40
        # Stomp-in-place @ duty 0.50 → single_support ~1.0 → reward +1.0
        #   (STOMP avoided by fwd_vel pressure on body_vx)
        in_air_pair = (self.env.foot_frc < 1.0)
        on_ground_pair = (self.env.foot_frc >= 10.0)
        _l_air,  _r_air  = in_air_pair[:, [0]],  in_air_pair[:, [1]]
        _l_grd,  _r_grd  = on_ground_pair[:, [0]], on_ground_pair[:, [1]]
        single_support_state = (
            (_l_air & _r_grd) | (_r_air & _l_grd)
        ).float()
        single_support_rew = (2.0 * single_support_state - 1.0) * self.static_flag

        # Air time reward (cmd_freq-aware, held-value):
        # At each touchdown event, snapshot delta = (swing_dur - target_swing)
        # into _held_air_delta. Hold this value between touchdowns so it
        # contributes to per-step reward continuously (not just sparse spike).
        # target_swing = (1 - support_ratio) / cmd_freq, per env.
        # Mini-step has swing < target → delta < 0 → continuous penalty until
        # next swing is long enough.
        # Two flavors: cmd_freq-aware (when external clock supplies freq) or
        # fixed-target (legged-gym style, when there's no clock).
        if self._phase_mode == 'input' and self._cmd_freq is not None:
            support_ratio = float(getattr(self.cfg.phase, 'support_ratio', 0.6) or 0.6)
            target_swing = (1.0 - support_ratio) / self._cmd_freq        # [n_envs, 1]
        elif self._phase_mode == 'none':
            # No clock — target swing is a fixed scalar (legged-gym default ~0.2s).
            target_swing = float(getattr(self.cfg.phase, 'target_swing', 0.2) or 0.2)
        else:
            target_swing = None

        if target_swing is not None:
            new_delta = (self._td_swing_duration - target_swing).clip(min=-0.5, max=0.5)
            self._held_air_delta = torch.where(self._td_event, new_delta, self._held_air_delta)
            air_time_rew = self._held_air_delta.mean(dim=1, keepdim=True) * self.static_flag
        else:
            air_time_rew = torch.zeros(self.num_envs, 1, device=self.device)

        # G1-style velocity-gated foot clearance reward (unitree_rl_lab):
        #   penalty = Σ (foot_z - target)² × tanh(mult × ||foot_v_xy||)
        # The tanh-of-velocity factor automatically selects swing feet
        # (high horizontal velocity → tanh ≈ 1, penalty applies) while
        # leaving support feet unconstrained (v ≈ 0 → tanh ≈ 0). No
        # clip-saturation: deviation in either direction grows quadratically.
        # Replaces the clip-saturated `foot_heit` reward which capped at
        # 5 cm and prevented large strides from clearing the ground cleanly.
        # `foot_vel` is stored as [num_envs, 6] = [Lx, Ly, Lz, Rx, Ry, Rz].
        _fc_target = float(getattr(self.cfg.reward, 'foot_clearance_target', 0.06) or 0.06)
        _fc_mult   = float(getattr(self.cfg.reward, 'foot_clearance_tanh_mult', 2.0) or 2.0)
        _foot_err_sq = (self.foot_height - _fc_target) ** 2          # [num_envs, 2]
        foot_clearance_l2 = -torch.sum(
            _foot_err_sq * torch.tanh(_fc_mult * _foot_xy_vel),
            dim=-1, keepdim=True
        )                                                            # ≤ 0

        # BDX-R-MjLab L1 `feet_clearance` reward (legs override w=-4.0):
        #     cost = Σ |foot_z - target| × ||v_xy(foot)|| × cmd_active
        # The L1 form (vs L2 above) penalises any deviation from target,
        # weighted by foot horizontal velocity. The (commanded-magnitude
        # gate) means it's inactive at standstill so the policy isn't
        # punished for keeping feet still when stand is the goal.
        _fc_l1_target = float(getattr(self.cfg.reward, 'feet_clearance_target', 0.06) or 0.06)
        _fc_cmd_threshold = float(getattr(self.cfg.reward, 'feet_clearance_cmd_threshold', 0.05) or 0.05)
        _cmd_active_clr = (_cmd_total > _fc_cmd_threshold).float()
        _foot_z_err = torch.abs(self.foot_height - _fc_l1_target)    # [n, 2]
        # v10 option: replace foot_xy_vel multiplier (which silently weakens
        # the penalty when feet are slow → rewards shuffle) with `in_air`
        # mask (penalty active only during swing). Cleaner gradient.
        _fc_l1_mask_mode = str(getattr(self.cfg.reward, 'feet_clearance_l1_mask', 'foot_xy_vel') or 'foot_xy_vel')
        if _fc_l1_mask_mode == 'in_air':
            _fc_l1_mask = (self.env.foot_frc < 1.0).float()
        elif _fc_l1_mask_mode == 'none':
            _fc_l1_mask = torch.ones_like(_foot_z_err)
        else:  # 'foot_xy_vel' = legacy
            _fc_l1_mask = _foot_xy_vel
        feet_clearance_l1 = -torch.sum(
            _foot_z_err * _fc_l1_mask, dim=-1, keepdim=True
        ) * _cmd_active_clr                                          # ≤ 0

        # BDX-R-MjLab peak-tracking foot swing height + soft landing.
        # During each swing, _foot_peak_z accumulates the per-foot max
        # z; at first_contact (=`_td_event`) we charge
        # (peak / target − 1)² × event and immediately reset the peak.
        # This is far harder to game than continuous clearance rewards
        # because the policy must demonstrably reach `target` BEFORE
        # touching down — no credit for staying near the ground.
        _swing_peak_target = float(getattr(self.cfg.reward, 'feet_swing_target', 0.08) or 0.08)
        _in_air = (self.env.foot_frc < 1.0).float()                       # [n, num_legs]
        self._foot_peak_z = torch.maximum(
            self._foot_peak_z, self.foot_height * _in_air
        )                                                                 # update peak each step
        _swing_err = (self._foot_peak_z / _swing_peak_target - 1.0).clamp(min=-1.0, max=2.0)
        feet_swing_height_peak = -torch.sum(
            (_swing_err ** 2) * self._td_event.float(), dim=-1, keepdim=True
        )                                                                 # sparse, ≤ 0
        # Reset peak after using it
        self._foot_peak_z = torch.where(
            self._td_event,
            torch.zeros_like(self._foot_peak_z),
            self._foot_peak_z,
        )

        # Stride length reward (v11+): at each foot touchdown, world-frame
        # xy distance from previous TD of the same foot = stride length for
        # that leg. Bell-shaped reward centered at the LIPM target (~0.12 m
        # at v=0.3). Cmd-gated so a standing robot isn't penalised for not
        # stepping. Gives the FIRST direct gradient toward physical stride
        # — all prior foot rewards (phase, clearance, swing height, air
        # time) are indirect proxies that the policy gamed via 4-6 cm
        # micro-strides.
        _stride_target = float(getattr(self.cfg.reward, 'stride_length_target', 0.0) or 0.0)
        if _stride_target > 0:
            _stride_std = float(getattr(self.cfg.reward, 'stride_length_std', 0.04) or 0.04)
            _xy_now = self.env.foot_pos[:, [0, 1, 3, 4]].view(self.num_envs, self.num_legs, 2)
            _stride_per_foot = torch.norm(_xy_now - self._last_td_foot_xy, dim=-1)   # [n, num_legs]
            _stride_dev = (_stride_per_foot - _stride_target) / _stride_std
            _stride_bell = torch.exp(-(_stride_dev ** 2))                            # [n, num_legs]
            stride_length_rew = (
                torch.sum(_stride_bell * self._td_event.float(), dim=-1, keepdim=True)
                * self.static_flag
            )
            # Update last-TD position only at events (where mask is True)
            self._last_td_foot_xy = torch.where(
                self._td_event.unsqueeze(-1).expand_as(_xy_now), _xy_now, self._last_td_foot_xy
            )
        else:
            stride_length_rew = torch.zeros(self.num_envs, 1, device=self.device)

        # BDX-R-MjLab soft_landing: at first_contact, penalise contact-force
        # magnitude. Drives the policy toward soft, controlled touchdowns
        # rather than slamming the foot.
        soft_landing = -torch.sum(
            self.env.foot_frc * self._td_event.float(), dim=-1, keepdim=True
        )                                                                 # ≤ 0

        # Foot stand reward: ‖cmd‖<0.15 → reward both feet on ground.
        # Counters the "no positive incentive to keep feet planted" gap at
        # standstill (where all foot_* and air_time rewards are static_flag-gated).
        # Active ONLY when commanded to stand; range [0, 1].
        foot_stand_rew = (support_foot_index.float().mean(dim=1, keepdim=True)
                          * (1.0 - self.static_flag))

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
            foot_stand=foot_stand_rew * w.get('foot_stand', 0.0),
            base_height_l2=base_height_l2 * w.get('base_height_l2', 0.0),
            flat_orient_l2=flat_orient_l2 * w.get('flat_orient_l2', 0.0),
            foot_clearance_l2=foot_clearance_l2 * w.get('foot_clearance_l2', 0.0),
            upright=upright_rew * w.get('upright', 0.0),
            pose_speed=pose_speed_rew * w.get('pose_speed', 0.0),
            feet_swing_height_peak=feet_swing_height_peak * w.get('feet_swing_height_peak', 0.0),
            soft_landing=soft_landing * w.get('soft_landing', 0.0),
            stride_length=stride_length_rew * w.get('stride_length', 0.0),
            single_support=single_support_rew * w.get('single_support', 0.0),
            action_rate_l2=action_rate_l2 * w.get('action_rate_l2', 0.0),
            body_ang_vel=body_ang_vel * w.get('body_ang_vel', 0.0),
            dof_pos_limits=dof_pos_limits * w.get('dof_pos_limits', 0.0),
            angular_momentum=angular_momentum * w.get('angular_momentum', 0.0),
            feet_clearance_l1=feet_clearance_l1 * w.get('feet_clearance_l1', 0.0),
            feet_slip_l2=feet_slip_l2 * w.get('feet_slip_l2', 0.0),
        )

        # Imitation rewards (only active when reference clips are loaded).
        # w_imit is annealed via self.train_iter (set by train.py each iter).
        # When w_imit_start == w_imit_end, behaves like a constant weight.
        if self._has_ref:
            w_imit = self.w_imit_now()
            w_task = float(getattr(self.cfg.task, 'w_task', 1.0) or 1.0)

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
        # Per-component reward clip — bounds each weighted reward term per step.
        # Default [-4, +5] historically capped yaw_rat (weight 8 × max +1 = +8 →
        # clipped to +5, losing 38% of perfect-tracking incentive) and foot_phase
        # (weight 8 × min -1 = -8 → clipped to -4, hiding "fully out of phase" cost).
        # Widen via config for runs where reward weights exceed 4-5 in magnitude.
        _rc_min = float(getattr(self.cfg.reward, 'component_clip_min', -4.0) or -4.0)
        _rc_max = float(getattr(self.cfg.reward, 'component_clip_max', 5.0) or 5.0)
        rewards = torch.cat(
            [torch.clip(value.to(self.device), min=_rc_min, max=_rc_max) * self.env.dt for value in rew_dict.values()], dim=1)
        self._last_rew_components = rewards.detach()
        eval_rew = torch.cat([rew_dict[key] * self.env.dt for key in
                              ['fwd_vel', 'yaw_rat', 'ang_vel', 'lateral_vel', 'vertical_vel', 'twist']],
                             dim=1).sum(dim=1)
        return rewards, eval_rew
