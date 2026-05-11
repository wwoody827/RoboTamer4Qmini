"""
Policy evaluation across multiple conditions.

Runs sim2sim headlessly over a test matrix (friction × cmd_vx × cmd_yaw),
each condition repeated N times, outputs a CSV summary, and prints a
breakdown report with optional matplotlib plots.

Usage:
    python deploy/sim2sim/evaluate.py --policy experiments/<name>/deploy/policy_<iter>.onnx
                                      [--runs 10]
                                      [--duration 10]
                                      [--out experiments/my_run/eval.csv]
                                      [--no-plots]
"""

import os
import sys
import argparse
import itertools
import csv
import math
from collections import deque

import numpy as np
import mujoco
import yaml

# reuse sim2sim helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim import (
    build_mujoco_model, PhaseModulator, ExternalPhaseClock,
    quat_to_euler_xyz, quat_rotate_inverse, scale_transform,
    load_manifest, manifest_to_sim2sim_cfg,
)

import onnxruntime as ort

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(cfg, cmd_vx, cmd_yaw, floor_friction, duration, seed=None, cmd_vy=0.0, cmd_freq=None):
    """Run one episode. Returns a dict of scalar metrics."""
    if seed is not None:
        np.random.seed(seed)

    sim_dt     = cfg['simulation_dt']
    decimation = cfg['control_decimation']
    policy_dt  = sim_dt * decimation

    ref_joint  = np.array(cfg['ref_joint_pos'],   dtype=np.float32)
    kps        = np.array(cfg['kps'],              dtype=np.float32)
    kds        = np.array(cfg['kds'],              dtype=np.float32)
    tor_offset = np.array(cfg['joint_tor_offset'], dtype=np.float32)
    vel_sign   = np.array(cfg['joint_vel_sign'],   dtype=np.float32)
    _act_low_cfg  = cfg['action_inc_low']
    _act_high_cfg = cfg['action_inc_high']
    jlim_low_cfg  = cfg['joint_limit_low']
    jlim_high_cfg = cfg['joint_limit_high']
    num_legs   = cfg['num_legs']
    obs_hist   = cfg['obs_history']
    obs_dim    = cfg['num_obs_per_step']
    static_thr = cfg['static_cmd_threshold']

    phase_mode  = cfg.get('phase_mode', 'output')
    action_mode = cfg.get('action_mode', 'increment')
    lp_alpha    = cfg.get('action_lowpass_alpha', 1.0)
    # obs_per_step=38 = "no-clock-in-obs" layout: 6 plain slots, no phase, no ref.
    # Used by walk_noclock (phase.mode=none) AND walk_noclock_v3 (phase.mode=input
    # but the phase slots are intentionally absent from obs — Cassie-style).
    is_noclock  = (obs_dim == 38)
    is_mirl     = (phase_mode == 'none' and not is_noclock)
    is_bdx      = (phase_mode == 'input' and not is_noclock)

    commands    = np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float32)
    static_flag = float(np.linalg.norm(commands) >= static_thr)

    session    = ort.InferenceSession(cfg['policy_path'])
    input_name = session.get_inputs()[0].name

    model = build_mujoco_model(cfg['urdf_path'], sim_dt, cfg['init_height'],
                               floor_friction=floor_friction)
    data  = mujoco.MjData(model)

    NUM_JOINTS = 10
    QPOS_START = 7
    QVEL_START = 6
    imu_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'imu_in_torso')
    foot_l_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ankle_pitch_l')
    foot_r_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ankle_pitch_r')
    target_h    = float(cfg.get('init_height', 0.45))

    # Resolve action scaling — absolute mode falls back to URDF joint limits
    if _act_low_cfg is not None:
        act_low  = np.array(_act_low_cfg, dtype=np.float32)
        act_high = np.array(_act_high_cfg, dtype=np.float32)
    elif action_mode == 'absolute':
        jnt_start = 1  # skip root free joint
        act_low  = np.array([model.jnt_range[jnt_start + i, 0] for i in range(NUM_JOINTS)], dtype=np.float32)
        act_high = np.array([model.jnt_range[jnt_start + i, 1] for i in range(NUM_JOINTS)], dtype=np.float32)
    else:
        raise ValueError("action_mode='increment' requires action scaling ranges in manifest")

    if jlim_low_cfg is not None:
        jlim_low  = np.array(jlim_low_cfg, dtype=np.float32)
        jlim_high = np.array(jlim_high_cfg, dtype=np.float32)
    else:
        jnt_start = 1
        jlim_low  = np.array([model.jnt_range[jnt_start + i, 0] for i in range(NUM_JOINTS)], dtype=np.float32)
        jlim_high = np.array([model.jnt_range[jnt_start + i, 1] for i in range(NUM_JOINTS)], dtype=np.float32)

    mujoco.mj_resetData(model, data)
    data.qpos[QPOS_START:QPOS_START + NUM_JOINTS] = ref_joint
    mujoco.mj_forward(model, data)

    pm = PhaseModulator(dt=policy_dt, num_legs=num_legs)
    pm.reset()
    ext_clock = None
    freq_mid = None
    freq_scale = None
    if is_bdx:
        freq_default = float(cfg.get('phase_freq_default', 2.5))
        freq_lo = float(cfg.get('phase_freq_low', freq_default - 0.5))
        freq_hi = float(cfg.get('phase_freq_high', freq_default + 0.5))
        freq_mid = 0.5 * (freq_lo + freq_hi)
        freq_scale = max(0.5 * (freq_hi - freq_lo), 1e-6)
        if cmd_freq is None:
            cmd_freq = freq_default
        ext_clock = ExternalPhaseClock(
            dt=policy_dt, num_legs=num_legs, default_freq=freq_default,
        )
        ext_clock.reset()
    lp_target = ref_joint.copy()
    current_joint_act = ref_joint.copy()
    # Framestack: training uses obs_history × obs_skip — buffer holds enough
    # raw frames so we can index back by `i * obs_skip` for i in [0, obs_hist).
    # The earlier `maxlen=obs_hist` was WRONG — it ignored obs_skip and fed
    # the policy 5 consecutive frames (~75 ms window) instead of the trained
    # 5 frames × skip 2 (~150 ms window). The bug silently inflated all
    # survival numbers because the policy saw fresher obs than training.
    obs_skip = int(cfg.get('obs_skip', 1) or 1)
    _buf_len = (obs_hist - 1) * obs_skip + 1
    obs_history = deque(maxlen=_buf_len)
    for _ in range(_buf_len):
        obs_history.append(np.zeros(obs_dim, dtype=np.float32))

    def _imu_state():
        if imu_body_id >= 0:
            quat         = data.xquat[imu_body_id]
            world_angvel = data.cvel[imu_body_id][0:3]
        else:
            quat         = data.qpos[3:7]
            world_angvel = data.qvel[3:6]
        return quat, quat_rotate_inverse(quat, world_angvel)

    # ─── Obs-delay DR (mirrors training, was missing in sim2sim) ────────────
    # Training side: env/legged_robot.py:step_torques runs once per PHYSICS
    # step (1000 Hz). joint_pos_his.append(...) happens there. delay(N) in
    # birl_task.step() then reads N PHYSICS-step-old values. So config
    # delay_joint_ranges=[10, 40] means 10-40 ms at 1 ms phys dt.
    #
    # We must push at PHYSICS rate (inside the decimation loop), not policy
    # rate. An earlier version pushed once per policy step (67 Hz), which
    # made the effective delay 15× too long (150-600 ms vs the 10-40 ms the
    # policy was actually trained against). That broke walk_v34 from 100 %
    # to 7 % sim2sim survival — root cause of the "no-clock failure" panic.
    rng = np.random.default_rng(seed)
    _dj = cfg.get('delay_joint_ranges', [10, 40])
    _da = cfg.get('delay_angle_ranges', [20, 50])
    _dr = cfg.get('delay_rate_ranges',  [20, 50])
    delay_jnt   = int(rng.integers(_dj[0], _dj[1] + 1))   # physics steps
    delay_angle = int(rng.integers(_da[0], _da[1] + 1))   # physics steps
    delay_rate  = int(rng.integers(_dr[0], _dr[1] + 1))   # physics steps
    delay_max   = max(delay_jnt, delay_angle, delay_rate, 1)
    # History deques — newest at right (index -1), oldest at left.
    _q_buf   = deque([ref_joint.copy() for _ in range(delay_max + 1)], maxlen=delay_max + 1)
    _dq_buf  = deque([np.zeros(NUM_JOINTS, dtype=np.float32) for _ in range(delay_max + 1)], maxlen=delay_max + 1)
    _eul_buf = deque([np.zeros(2, dtype=np.float32) for _ in range(delay_max + 1)], maxlen=delay_max + 1)
    _av_buf  = deque([np.zeros(3, dtype=np.float32) for _ in range(delay_max + 1)], maxlen=delay_max + 1)

    def _push_obs_history():
        """Push current rigid-body state into the per-quantity delay buffers.
        Call once per PHYSICS step (matches training's step_torques cadence)."""
        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS].astype(np.float32).copy()
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS].astype(np.float32).copy()
        quat, base_ang_vel = _imu_state()
        base_euler = quat_to_euler_xyz(quat)[:2].astype(np.float32).copy()
        _q_buf.append(q)
        _dq_buf.append(dq)
        _eul_buf.append(base_euler)
        _av_buf.append(base_ang_vel.astype(np.float32).copy())

    def _delayed():
        """Return (q, dq, base_euler, base_ang_vel) at the per-episode delays."""
        return (_q_buf[-1 - delay_jnt], _dq_buf[-1 - delay_jnt],
                _eul_buf[-1 - delay_angle], _av_buf[-1 - delay_rate])

    def get_obs():
        """BIRL obs: 44-dim standard, 47-dim teacher (+ base_lin_vel)."""
        q, dq, base_euler, base_ang_vel = _delayed()
        pm_phase_val = np.concatenate([np.sin(pm.phase), np.cos(pm.phase)]) * static_flag
        pm_f_val     = (pm.frequency * 0.3 - 1.0) * static_flag
        parts = [
            commands, base_euler, base_ang_vel * 0.5,
            q - ref_joint, dq * 0.1, current_joint_act - q,
            pm_phase_val, pm_f_val,
        ]
        if obs_dim == 47:
            # Teacher uses current (un-delayed) base_lin_vel — privileged obs.
            quat_now, _ = _imu_state()
            base_lin_vel = quat_rotate_inverse(quat_now, data.qvel[:3]).astype(np.float32)
            parts.append(base_lin_vel)
        obs = np.concatenate(parts).astype(np.float32)
        return np.clip(obs, -3.0, 3.0)

    def get_obs_mirl():
        """MIRL obs: 64-dim with 8 command slots, no phase modulator."""
        q, dq, base_euler, base_ang_vel = _delayed()
        commands_8    = np.array([commands[0], commands[1], commands[2],
                                   0., 0., 0., 0., 0.], dtype=np.float32)
        obs = np.concatenate([
            commands_8,          # 8
            base_euler,          # 2
            base_ang_vel * 0.5,  # 3
            q - ref_joint,       # 10
            dq * 0.1,            # 10
            current_joint_act - q,  # 10
            np.zeros(21, dtype=np.float32),  # ref slots + phase_progress
        ]).astype(np.float32)
        return np.clip(obs, -3.0, 3.0)

    def get_obs_noclock():
        """walk_noclock obs: 38-dim, 6 slots, no phase, no ref."""
        q, dq, base_euler, base_ang_vel = _delayed()
        obs = np.concatenate([
            commands,                # 3
            base_euler,              # 2
            base_ang_vel * 0.5,      # 3
            q - ref_joint,           # 10
            dq * 0.1,                # 10
            current_joint_act - q,   # 10
        ]).astype(np.float32)
        return np.clip(obs, -3.0, 3.0)

    def get_obs_bdx():
        """BD_X obs: 43-dim with external phase clock + normalized freq cmd."""
        q, dq, base_euler, base_ang_vel = _delayed()
        phase_clock   = ext_clock.sin_cos() * static_flag
        freq_cmd_norm = np.array(
            [((cmd_freq - freq_mid) / freq_scale) * static_flag],
            dtype=np.float32,
        )
        obs = np.concatenate([
            commands,            # 3
            base_euler,          # 2
            base_ang_vel * 0.5,  # 3
            q - ref_joint,       # 10
            dq * 0.1,            # 10
            current_joint_act - q,  # 10
            phase_clock,         # 4
            freq_cmd_norm,       # 1
        ]).astype(np.float32)
        return np.clip(obs, -3.0, 3.0)

    def compute_torques(target_q, q, dq):
        error = target_q - q
        return kps * error + kds - dq + tor_offset - 3.5 * np.sign(dq) * vel_sign

    total_steps = int(duration / sim_dt)
    fall_thresh = 0.25

    vx_errors    = []
    vx_body_samples = []    # (elapsed_sec, vx_body) for trajectory-level metrics
    vy_body_samples = []    # (elapsed_sec, vy_body) — same purpose for lateral cmds
    yaw_rate_samples = []   # (elapsed_sec, yaw_rate) — passive yaw drift detection
    yaw_errors   = []
    vy_abs       = []
    roll_rms_acc = []
    pitch_rms_acc= []
    torque_acc   = []
    # Gait-quality accumulators (policy-step rate)
    base_z_samples   = []   # (t, z)
    yaw_samples      = []   # (t, yaw_world)
    contact_samples  = []   # (t, in_l, in_r)
    touchdown_l      = []   # list of (t, |foot_vz|) — rising edges for left foot
    touchdown_r      = []
    prev_in_l = False
    prev_in_r = False
    CONTACT_THR = 20.0      # N — foot-in-contact threshold (standing ≈ 35N/foot)
    survived     = True
    survive_steps= total_steps
    # Torque sim2real diagnostics: peak |τ_i| / motor_effort_i across the run.
    # >1 means policy commands more than the motor can deliver — sim silently
    # clips, real robot saturates. Same metric as recovery eval.
    effort_limits = np.abs(model.actuator_forcerange[:NUM_JOINTS, 1]).astype(np.float32)
    peak_tau   = 0.0
    peak_ratio = 0.0

    for step in range(total_steps):
        # Push to delay buffers every PHYSICS step (matches training's
        # step_torques cadence). Policy reads delayed values at the policy
        # boundary below.
        _push_obs_history()
        if step % decimation == 0:
            if is_bdx:
                ext_clock.update(cmd_freq)
                obs_now = get_obs_bdx()
            elif is_noclock:
                obs_now = get_obs_noclock()
            elif is_mirl:
                obs_now = get_obs_mirl()
            else:
                obs_now = get_obs()
            obs_history.append(obs_now)
            _buf = list(obs_history)
            _idx = [i * obs_skip for i in range(obs_hist)]
            obs_stacked = np.concatenate([_buf[i] for i in _idx])[np.newaxis, :]
            net_out = session.run(None, {input_name: obs_stacked})[0][0]
            scaled  = scale_transform(net_out, act_low, act_high)

            if phase_mode == 'output':
                pm.compute(scaled[:num_legs])
                joint_out = scaled[num_legs:]
            else:
                joint_out = scaled

            if action_mode == 'increment':
                current_joint_act[:] += joint_out * policy_dt
            else:
                if lp_alpha < 1.0:
                    lp_target[:] = lp_alpha * joint_out + (1.0 - lp_alpha) * lp_target
                else:
                    lp_target[:] = joint_out
                current_joint_act[:] = lp_target
            current_joint_act[:] = np.clip(current_joint_act, jlim_low, jlim_high)
            static_flag = float(np.linalg.norm(commands) >= static_thr)

        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        torques = compute_torques(current_joint_act, q, dq)
        data.ctrl[:NUM_JOINTS] = torques
        mujoco.mj_step(model, data)
        abs_tau = np.abs(torques)
        peak_tau   = max(peak_tau,   float(abs_tau.max()))
        peak_ratio = max(peak_ratio, float((abs_tau / effort_limits).max()))

        z = data.qpos[2]
        if z < fall_thresh:
            survived = False
            survive_steps = step
            break

        if step % decimation == 0:
            quat  = data.xquat[imu_body_id] if imu_body_id >= 0 else data.qpos[3:7]
            euler = quat_to_euler_xyz(quat)
            roll_rms_acc.append(euler[0] ** 2)
            pitch_rms_acc.append(euler[1] ** 2)
            torque_acc.append(np.sum(np.abs(torques * dq)))
            # Body-frame velocities — matches training reward (which uses
            # quat_rotate_inverse(base_quat, base_lvel)). World-frame qvel[0]
            # would diverge from cmd_vx when robot yaw drifts during episode.
            yaw_world = euler[2]
            cy, sy = math.cos(yaw_world), math.sin(yaw_world)
            vx_body =  data.qvel[0] * cy + data.qvel[1] * sy
            vy_body = -data.qvel[0] * sy + data.qvel[1] * cy
            # vx_errors / yaw_errors / vy_abs now in body frame
            vx_errors.append(abs(vx_body - cmd_vx))
            yaw_errors.append(abs(data.qvel[5] - cmd_yaw))   # qvel[5] = world yaw rate ≈ body yaw rate when roll/pitch small (flat walking)
            vy_abs.append(abs(vy_body))
            t_now   = step * sim_dt
            vx_body_samples.append((t_now, vx_body))
            vy_body_samples.append((t_now, vy_body))
            yaw_rate_samples.append((t_now, float(data.qvel[5])))
            base_z_samples.append((t_now, float(data.qpos[2])))
            yaw_samples.append((t_now, float(yaw_world)))
            # Contact forces: iterate contact list (cfrc_ext isn't auto-populated
            # in MuJoCo 3). Aggregate per foot body.
            force_l = 0.0
            force_r = 0.0
            cf_buf = np.zeros(6)
            for c_idx in range(data.ncon):
                con = data.contact[c_idx]
                b1 = model.geom_bodyid[con.geom1]
                b2 = model.geom_bodyid[con.geom2]
                other = b2 if b1 == 0 else (b1 if b2 == 0 else -1)
                if other < 0:
                    continue
                mujoco.mj_contactForce(model, data, c_idx, cf_buf)
                fmag = float(np.linalg.norm(cf_buf[:3]))
                if other == foot_l_id:
                    force_l += fmag
                elif other == foot_r_id:
                    force_r += fmag
            in_l = force_l > CONTACT_THR
            in_r = force_r > CONTACT_THR
            contact_samples.append((t_now, in_l, in_r))
            # rising-edge touchdown detection — record |foot_vz| at the touchdown step
            if foot_l_id >= 0 and in_l and not prev_in_l:
                vz_l = abs(float(data.cvel[foot_l_id, 5]))
                touchdown_l.append((t_now, vz_l))
            if foot_r_id >= 0 and in_r and not prev_in_r:
                vz_r = abs(float(data.cvel[foot_r_id, 5]))
                touchdown_r.append((t_now, vz_r))
            prev_in_l, prev_in_r = in_l, in_r

    x_final = data.qpos[0]
    y_final = data.qpos[1]
    elapsed = survive_steps * sim_dt

    total_mass = 7.0
    g = 9.81
    dx = abs(x_final)
    cot = (np.sum(torque_acc) * policy_dt) / (total_mass * g * dx) if dx > 0.05 else float('nan')

    # Trajectory-level metrics (more meaningful than per-step |qvel[0] - cmd_vx|):
    # - displacement_err: |actual distance − commanded distance|. Insensitive to gait
    #   oscillation. Only defined for cmd_yaw == 0 (else heading rotates and x isn't
    #   the right axis).
    # - vx_bias_body: signed mean of (vx_body − cmd_vx) over the steady-state window.
    #   Tells whether policy is consistently slow (+) or fast (−). Body-frame so it's
    #   fair under turning.
    # - vx_err_steady: |vx_body − cmd_vx| averaged over steady state (skip first 1s
    #   to drop startup transient).
    STARTUP = 1.0  # seconds to skip
    steady = [v for (t, v) in vx_body_samples if t >= STARTUP]

    displacement_err = (abs(x_final - cmd_vx * elapsed)
                        if (cmd_yaw == 0.0 and elapsed > 0) else float('nan'))
    vx_bias_body  = float(np.mean([v - cmd_vx for v in steady]))        if steady else float('nan')
    vx_err_steady = float(np.mean([abs(v - cmd_vx) for v in steady]))   if steady else float('nan')

    # Lateral (vy) trajectory metrics — same idea as vx_bias_body / displacement_err.
    # vy_bias_body detects "gait-sway-impersonating-strafe" failure mode where
    # body_vy oscillates ±0.3 around 0 but per-step |err| looks small. Signed mean
    # exposes whether net lateral motion matches cmd.
    steady_vy = [v for (t, v) in vy_body_samples if t >= STARTUP]
    vy_bias_body  = float(np.mean([v - cmd_vy for v in steady_vy]))      if steady_vy else float('nan')
    vy_err_steady = float(np.mean([abs(v - cmd_vy) for v in steady_vy])) if steady_vy else float('nan')
    displacement_err_y = (abs(y_final - cmd_vy * elapsed)
                          if (cmd_yaw == 0.0 and elapsed > 0) else float('nan'))

    # Passive yaw drift — net yaw_rate when cmd_yaw=0. Detects "spinning while
    # strafing" pathology (body_vy may track but body keeps rotating). Big mean
    # |yaw_rate| with cmd_yaw=0 = bad.
    steady_yaw_rate = [v for (t, v) in yaw_rate_samples if t >= STARTUP]
    yaw_rate_bias = float(np.mean([v - cmd_yaw for v in steady_yaw_rate])) if steady_yaw_rate else float('nan')
    yaw_drift_passive = (abs(yaw_rate_bias) if cmd_yaw == 0.0 else float('nan'))

    # ── Gait-quality metrics (all over steady-state window t >= STARTUP) ────
    # com_z_rms: body vertical oscillation — big = big steps / bouncy gait
    z_steady = [z for (t, z) in base_z_samples if t >= STARTUP]
    com_z_rms = (float(np.sqrt(np.mean([(z - target_h) ** 2 for z in z_steady])))
                 if z_steady else float('nan'))
    com_z_mean = float(np.mean(z_steady)) if z_steady else float('nan')

    # yaw_osc_rms: residual yaw after removing cmd_yaw linear trend + mean offset —
    # measures "hip-swinging". Only defined after unwrap; for cmd_yaw==0 this is
    # simply demeaned yaw RMS.
    yaw_steady_pairs = [(t, y) for (t, y) in yaw_samples if t >= STARTUP]
    if len(yaw_steady_pairs) >= 10:
        t_arr   = np.array([t for (t, _) in yaw_steady_pairs])
        yaw_arr = np.unwrap(np.array([y for (_, y) in yaw_steady_pairs]))
        residual = yaw_arr - cmd_yaw * t_arr
        residual -= residual.mean()
        yaw_osc_rms = float(np.sqrt(np.mean(residual ** 2)))
    else:
        yaw_osc_rms = float('nan')

    # measured_freq: mean step freq per leg from touchdown events.
    # Need >=2 touchdowns per leg (1 full stride period) to be meaningful.
    def _leg_period(events):
        pts = [t for (t, _) in events if t >= STARTUP]
        if len(pts) < 2:
            return float('nan')
        return float(np.mean(np.diff(pts)))

    period_l = _leg_period(touchdown_l)
    period_r = _leg_period(touchdown_r)
    periods  = [p for p in (period_l, period_r) if not math.isnan(p)]
    measured_freq = float(1.0 / np.mean(periods)) if periods else float('nan')

    # duty_factor: fraction of steady-state time each foot is in contact, averaged.
    steady_contact = [(l, r) for (t, l, r) in contact_samples if t >= STARTUP]
    if steady_contact:
        df_l = np.mean([1.0 if l else 0.0 for (l, _) in steady_contact])
        df_r = np.mean([1.0 if r else 0.0 for (_, r) in steady_contact])
        duty_factor = float(0.5 * (df_l + df_r))
    else:
        duty_factor = float('nan')

    # landing_vz_peak: max |foot_vz| across touchdown events in steady state.
    td_vz = ([vz for (t, vz) in touchdown_l if t >= STARTUP]
             + [vz for (t, vz) in touchdown_r if t >= STARTUP])
    landing_vz_peak = float(max(td_vz)) if td_vz else float('nan')

    # stride_length: |vx_body| × leg_period (distance the CoM travels in one full
    # ipsilateral stride). Use abs(mean_vx_body) so backward walking is positive.
    mean_vx_body = float(np.mean(steady)) if steady else float('nan')
    if steady and periods:
        stride_length = abs(mean_vx_body) * float(np.mean(periods))
    else:
        stride_length = float('nan')

    # stride_asymm: |T_L - T_R| / mean — left/right timing asymmetry.
    if not math.isnan(period_l) and not math.isnan(period_r):
        mean_p = 0.5 * (period_l + period_r)
        stride_asymm = abs(period_l - period_r) / mean_p if mean_p > 1e-6 else float('nan')
    else:
        stride_asymm = float('nan')

    return {
        'survived':       int(survived),
        'survive_time':   elapsed,
        'x_final':        x_final,
        'y_final':        y_final,
        'vx_error_mean':  float(np.mean(vx_errors))  if vx_errors  else float('nan'),
        'vx_err_steady':  vx_err_steady,
        'vx_bias_body':   vx_bias_body,
        'displacement_err': float(displacement_err),
        'vy_bias_body':   vy_bias_body,
        'vy_err_steady':  vy_err_steady,
        'displacement_err_y': float(displacement_err_y),
        'yaw_drift_passive': float(yaw_drift_passive),
        'yaw_error_mean': float(np.mean(yaw_errors)) if yaw_errors else float('nan'),
        'vy_abs_mean':    float(np.mean(vy_abs))     if vy_abs     else float('nan'),
        'roll_rms':       float(np.sqrt(np.mean(roll_rms_acc)))  if roll_rms_acc  else float('nan'),
        'pitch_rms':      float(np.sqrt(np.mean(pitch_rms_acc))) if pitch_rms_acc else float('nan'),
        'cot':            cot,
        'com_z_rms':      com_z_rms,
        'com_z_mean':     com_z_mean,
        'yaw_osc_rms':    yaw_osc_rms,
        'measured_freq':  measured_freq,
        'duty_factor':    duty_factor,
        'landing_vz_peak': landing_vz_peak,
        'stride_length':  stride_length,
        'stride_asymm':   stride_asymm,
        'peak_tau_nm':    float(peak_tau),
        'peak_tau_ratio': float(peak_ratio),
    }


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(cfg, runs, duration, frictions, vx_list, yaw_list, out_path, vy_list=(0.0,)):
    conditions = list(itertools.product(frictions, vx_list, vy_list, yaw_list))
    total = len(conditions) * runs
    print(f"Evaluating {len(conditions)} conditions × {runs} runs = {total} episodes")
    print(f"Policy: {cfg['policy_path']}\n")

    fieldnames = [
        'friction', 'cmd_vx', 'cmd_vy', 'cmd_yaw', 'run',
        'survived', 'survive_time', 'x_final', 'y_final',
        'vx_error_mean', 'vx_err_steady', 'vx_bias_body', 'displacement_err',
        'vy_bias_body', 'vy_err_steady', 'displacement_err_y',
        'yaw_drift_passive',
        'yaw_error_mean', 'vy_abs_mean', 'roll_rms', 'pitch_rms', 'cot',
        'com_z_rms', 'yaw_osc_rms', 'measured_freq', 'duty_factor',
        'landing_vz_peak', 'stride_length', 'stride_asymm',
        'peak_tau_nm', 'peak_tau_ratio',
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        done = 0
        for friction, cmd_vx, cmd_vy, cmd_yaw in conditions:
            results = []
            for run_i in range(runs):
                metrics = run_episode(cfg, cmd_vx, cmd_yaw, friction, duration,
                                      seed=run_i, cmd_vy=cmd_vy)
                row = {'friction': friction, 'cmd_vx': cmd_vx, 'cmd_vy': cmd_vy,
                       'cmd_yaw': cmd_yaw, 'run': run_i, **metrics}
                writer.writerow(row)
                f.flush()
                results.append(metrics)
                done += 1

            surv_rate   = np.mean([r['survived'] for r in results])
            vx_err      = np.nanmean([r['vx_error_mean'] for r in results])
            vx_disp     = np.nanmean([r['displacement_err'] for r in results])
            vy_bias     = np.nanmean([r['vy_bias_body'] for r in results])
            vy_disp_y   = np.nanmean([r['displacement_err_y'] for r in results])
            yaw_drift_p = np.nanmean([r['yaw_drift_passive'] for r in results])
            mean_pkr    = np.mean([r['peak_tau_ratio'] for r in results])
            print(f"f={friction:.1f} vx={cmd_vx:+.1f} vy={cmd_vy:+.1f} yaw={cmd_yaw:+.1f} | "
                  f"surv={surv_rate*100:3.0f}%  "
                  f"vx_err={vx_err:.3f} disp_x={vx_disp:.3f}  "
                  f"vy_bias={vy_bias:+.3f} disp_y={vy_disp_y:.3f}  "
                  f"yaw_drift={yaw_drift_p:.3f}  "
                  f"τ/eff={mean_pkr:.2f}× [{done}/{total}]")

    print(f"\nResults saved to: {out_path}")
    return out_path


# ── report ────────────────────────────────────────────────────────────────────

def _bar(value, lo, hi, width=20):
    frac = max(0., min(1., (value - lo) / (hi - lo + 1e-9)))
    filled = round(frac * width)
    return '[' + '#' * filled + '.' * (width - filled) + ']'


def print_report(csv_path):
    try:
        import pandas as pd
    except ImportError:
        print('[report] pandas not available — skipping breakdown report')
        return

    df = pd.read_csv(csv_path)
    frictions = sorted(df['friction'].unique())
    vx_vals   = sorted(df['cmd_vx'].unique())

    sep = '+----------+---------+-----+---------+----------+-----------+-----------+------------+----------+'
    hdr = ('| friction | cmd_vx  |  N  |  Surv%  |  vx_err  |  vy_drift |  roll°rms |  pitch°rms |   CoT    |')

    print('\n' + '=' * len(sep))
    print(' Breakdown Report')
    print('=' * len(sep))
    print(sep)
    print(hdr)
    print(sep)

    for fr in frictions:
        for vx in vx_vals:
            sub = df[(df['friction'] == fr) & (df['cmd_vx'] == vx)]
            if sub.empty:
                continue
            n     = len(sub)
            surv  = sub['survived'].mean() * 100
            vxe   = sub['vx_error_mean'].mean()
            vy    = sub['vy_abs_mean'].mean()
            roll  = math.degrees(sub['roll_rms'].mean())
            pitch = math.degrees(sub['pitch_rms'].mean())
            cot   = sub['cot'].mean()
            cot_s = f'{cot:.2f}' if not math.isnan(cot) else ' nan'
            flag  = ' ' if surv == 100 else ('!' if surv >= 50 else 'X')
            print(f'| {fr:^8.1f} | {vx:^+7.1f} | {n:^3} |'
                  f' {flag}{surv:5.0f}% |'
                  f'  {vxe:6.3f}  |'
                  f'   {vy:6.3f}  |'
                  f'   {roll:6.1f}   |'
                  f'   {pitch:7.1f}   |'
                  f' {cot_s:^8} |')
        print(sep)

    # ASCII bar summaries
    print('\n── Survival rate by friction ──')
    for fr in frictions:
        rate = df[df['friction'] == fr]['survived'].mean()
        print(f'  {fr:.1f}  {_bar(rate, 0, 1)}  {rate*100:5.1f}%')

    sub_ok = df[(df['friction'] <= 1.5) & (df['survived'] == 1)]

    print('\n── vx tracking error (friction≤1.5, survived) ──')
    for vx in vx_vals:
        s = sub_ok[sub_ok['cmd_vx'] == vx]
        if s.empty:
            continue
        mu, std = s['vx_error_mean'].mean(), s['vx_error_mean'].std()
        print(f'  {vx:+.1f}  {_bar(mu, 0, 0.7)}  {mu:.3f} ± {std:.3f} m/s')

    print('\n── lateral drift vy (friction≤1.5, survived) ──')
    for vx in vx_vals:
        s = sub_ok[sub_ok['cmd_vx'] == vx]
        if s.empty:
            continue
        mu, std = s['vy_abs_mean'].mean(), s['vy_abs_mean'].std()
        print(f'  {vx:+.1f}  {_bar(mu, 0, 0.5)}  {mu:.3f} ± {std:.3f} m/s')

    # ── forward vs backward symmetry table ──────────────────────────────────
    fwd_speeds = sorted([v for v in vx_vals if v > 0])
    bwd_speeds = sorted([-v for v in vx_vals if v < 0])
    paired = sorted(set(fwd_speeds) & set(bwd_speeds))
    if paired:
        print('\n── Forward vs Backward symmetry (friction≤1.5, survived) ──')
        sep2 = '+-------+----------+----------+----------+----------+----------+----------+'
        print(sep2)
        print('| speed | fwd_err  | bwd_err  | Δerr     | fwd_vy   | bwd_vy   | Δvy      |')
        print(sep2)
        for spd in paired:
            sf = sub_ok[sub_ok['cmd_vx'] ==  spd]
            sb = sub_ok[sub_ok['cmd_vx'] == -spd]
            if sf.empty or sb.empty:
                continue
            fe = sf['vx_error_mean'].mean()
            be = sb['vx_error_mean'].mean()
            fv = sf['vy_abs_mean'].mean()
            bv = sb['vy_abs_mean'].mean()
            de = be - fe
            dv = bv - fv
            de_s = f'{de:+.3f}'
            dv_s = f'{dv:+.3f}'
            print(f'| {spd:.1f}   | {fe:8.3f} | {be:8.3f} | {de_s:8} | {fv:8.3f} | {bv:8.3f} | {dv_s:8} |')
        print(sep2)
        print('  Δ = backward − forward  (positive = backward is worse)')


def make_plots(csv_path):
    if not HAS_MPL:
        print('[report] matplotlib not available — skipping plots')
        return
    try:
        import pandas as pd
    except ImportError:
        print('[report] pandas not available — skipping plots')
        return

    df = pd.read_csv(csv_path)
    frictions = sorted(df['friction'].unique())
    vx_vals   = sorted(df['cmd_vx'].unique())
    out_dir   = os.path.dirname(os.path.abspath(csv_path))

    # 1. Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Policy Evaluation Heatmaps  (rows=friction, cols=cmd_vx)', fontsize=13)

    specs = [
        ('survived',      'Survival rate (%)',  True,  plt.cm.RdYlGn),
        ('vx_error_mean', 'vx tracking err',    False, plt.cm.RdYlGn_r),
        ('vy_abs_mean',   'lateral drift (m/s)',False, plt.cm.RdYlGn_r),
        ('roll_rms',      'roll RMS (deg)',      False, plt.cm.RdYlGn_r),
    ]
    for ax, (col, title, pct, cmap) in zip(axes.flat, specs):
        mat = np.full((len(frictions), len(vx_vals)), np.nan)
        for i, fr in enumerate(frictions):
            for j, vx in enumerate(vx_vals):
                sub = df[(df['friction'] == fr) & (df['cmd_vx'] == vx)]
                if sub.empty:
                    continue
                v = sub[col].mean()
                if col in ('roll_rms', 'pitch_rms'):
                    v = math.degrees(v)
                if pct:
                    v *= 100
                mat[i, j] = v
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0,
                       vmax=(100 if pct else None))
        ax.set_xticks(range(len(vx_vals)))
        ax.set_xticklabels([f'{v:+.1f}' for v in vx_vals])
        ax.set_yticks(range(len(frictions)))
        ax.set_yticklabels([f'{f:.1f}' for f in frictions])
        ax.set_xlabel('cmd_vx (m/s)')
        ax.set_ylabel('friction')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i in range(len(frictions)):
            for j in range(len(vx_vals)):
                v = mat[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                            fontsize=8, color='black')
    plt.tight_layout()
    p = os.path.join(out_dir, 'eval_heatmaps.png')
    plt.savefig(p, dpi=120); plt.close()
    print(f'  saved: {p}')

    # 2. Grouped bar: vx_err and vy_drift
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Velocity tracking & lateral drift (survived)', fontsize=12)
    for ax, (col, ylabel) in zip(axes, [('vx_error_mean', 'vx error (m/s)'),
                                         ('vy_abs_mean',   'vy drift (m/s)')]):
        x = np.arange(len(vx_vals))
        width = 0.8 / len(frictions)
        for k, fr in enumerate(frictions):
            means, stds = [], []
            for vx in vx_vals:
                s = df[(df['friction'] == fr) & (df['cmd_vx'] == vx) & (df['survived'] == 1)]
                means.append(s[col].mean() if not s.empty else np.nan)
                stds.append(s[col].std()   if not s.empty else np.nan)
            off = (k - len(frictions) / 2 + 0.5) * width
            ax.bar(x + off, means, width, yerr=stds, label=f'fr={fr:.1f}',
                   capsize=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:+.1f}' for v in vx_vals])
        ax.set_xlabel('cmd_vx (m/s)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    p = os.path.join(out_dir, 'eval_tracking_bars.png')
    plt.savefig(p, dpi=120); plt.close()
    print(f'  saved: {p}')

    # 3. Histograms (friction≤1.5, forward, survived)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Metric distributions (friction≤1.5, vx>0, survived)', fontsize=11)
    sub_ok = df[(df['friction'] <= 1.5) & (df['cmd_vx'] > 0) & (df['survived'] == 1)]
    for ax, (col, xlabel) in zip(axes, [('vx_error_mean', 'vx error (m/s)'),
                                         ('vy_abs_mean',   'vy drift (m/s)'),
                                         ('roll_rms',      'roll RMS (rad)')]):
        data = sub_ok[col].dropna()
        ax.hist(data, bins=20, edgecolor='white', alpha=0.85, color='steelblue')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'mean={data.mean():.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'eval_histograms.png')
    plt.savefig(p, dpi=120); plt.close()
    print(f'  saved: {p}')


# ── quick eval for tensorboard (called from train.py) ────────────────────────

def quick_eval(onnx_path, sim_cfg,
               frictions=(1.0, 1.5),
               vx_list=(-0.5, 0.0, 0.5),
               vy_list=(-0.3, 0.3),
               yaw_list=(0.0, 0.5, -0.5),
               runs=10,
               duration=15.0):
    """
    Run a small evaluation matrix and return a flat dict of scalars for
    TensorBoard logging.  Called periodically from train.py.

    Returns dict with keys:
        sim2sim/survive_time_fr{f}   — mean survive time per friction level
        sim2sim/vx_err_fwd/bwd       — mean per-step |qvel[0]-cmd_vx| (world frame, legacy)
        sim2sim/vx_err_steady_fwd/bwd — body-frame |vx-cmd_vx| averaged over steady state (t>=1s)
        sim2sim/vx_bias_fwd/bwd      — signed body-frame bias (+=slow, -=fast), steady state
        sim2sim/displacement_err_fwd/bwd — |x_final - cmd_vx * duration|, trajectory-level
        sim2sim/yaw_err              — mean yaw rate error, non-zero yaw commands
        sim2sim/vy_bias_strafe       — signed mean (body_vy - cmd_vy) on cmd_vy != 0 runs;
                                       big magnitude = "gait sway impersonating strafe"
        sim2sim/displacement_err_y   — |y_final - cmd_vy * duration|, real lateral tracking
        sim2sim/yaw_drift_passive    — |mean yaw_rate| when cmd_yaw=0, includes strafe runs;
                                       big = robot spinning while moving (sim2sim findings)
        sim2sim/vy_drift             — mean lateral drift (legacy: |vy| on cmd_yaw=0 fwd-only)
        sim2sim/pitch_rms_fwd/bwd    — mean pitch RMS forward/backward
        sim2sim/roll_rms             — mean roll RMS

        Gait-quality (forward, survived):
        sim2sim/com_z_rms            — vertical CoM oscillation (big = bouncy/big-step)
        sim2sim/yaw_osc_rms          — body yaw oscillation in deg (hip-swinging)
        sim2sim/measured_freq        — actual step freq per leg (Hz) from foot touchdown
        sim2sim/duty_factor          — fraction of time feet in contact (~0.6 walk, <0.5 run)
        sim2sim/landing_vz_peak      — max |foot_vz| at touchdown (hard-slam indicator)
        sim2sim/stride_length        — |vx_body| × leg_period (distance per stride)
        sim2sim/stride_asymm         — |T_L - T_R| / mean (L/R timing asymmetry)
        sim2sim/cot_fr{f}            — per-friction cost of transport
    """
    cfg = dict(sim_cfg)
    cfg['policy_path'] = onnx_path

    rows = []
    # vx sweep (cmd_yaw=0): covers survival, vx tracking, pitch/roll
    for friction, cmd_vx in itertools.product(frictions, vx_list):
        for run_i in range(runs):
            m = run_episode(cfg, cmd_vx, 0.0, friction, duration, seed=run_i)
            rows.append({'friction': friction, 'cmd_vx': cmd_vx, 'cmd_vy': 0.0,
                         'cmd_yaw': 0.0, **m})

    # yaw sweep (vx=0, fr=1.0 only): covers yaw tracking
    for cmd_yaw in yaw_list:
        if cmd_yaw == 0.0:
            continue
        for run_i in range(runs):
            m = run_episode(cfg, 0.0, cmd_yaw, 1.0, duration, seed=run_i)
            rows.append({'friction': 1.0, 'cmd_vx': 0.0, 'cmd_vy': 0.0, 'cmd_yaw': cmd_yaw, **m})

    # vy sweep (vx=0, cmd_yaw=0, fr=1.0 only): covers strafe + passive yaw drift
    for cmd_vy in vy_list:
        if cmd_vy == 0.0:
            continue
        for run_i in range(runs):
            m = run_episode(cfg, 0.0, 0.0, 1.0, duration, seed=run_i, cmd_vy=cmd_vy)
            rows.append({'friction': 1.0, 'cmd_vx': 0.0, 'cmd_vy': cmd_vy, 'cmd_yaw': 0.0, **m})

    if not rows:
        return {}

    metrics = {}

    # mean survive_time per friction (continuous, 0–duration seconds)
    vx_rows = [r for r in rows if r['cmd_yaw'] == 0.0]
    for fr in frictions:
        sub = [r for r in vx_rows if r['friction'] == fr]
        metrics[f'sim2sim/survive_time_fr{fr:.1f}'] = float(np.mean([r['survive_time'] for r in sub]))

    # vx tracking / lateral drift / pitch (survived, cmd_yaw=0 only)
    survived = [r for r in vx_rows if r['survived']]
    fwd = [r for r in survived if r['cmd_vx'] > 0]
    bwd = [r for r in survived if r['cmd_vx'] < 0]

    metrics['sim2sim/vx_err_fwd']    = float(np.nanmean([r['vx_error_mean'] for r in fwd]))     if fwd      else float('nan')
    metrics['sim2sim/vx_err_bwd']    = float(np.nanmean([r['vx_error_mean'] for r in bwd]))     if bwd      else float('nan')
    metrics['sim2sim/vx_err_steady_fwd']   = float(np.nanmean([r['vx_err_steady']    for r in fwd])) if fwd else float('nan')
    metrics['sim2sim/vx_err_steady_bwd']   = float(np.nanmean([r['vx_err_steady']    for r in bwd])) if bwd else float('nan')
    metrics['sim2sim/vx_bias_fwd']         = float(np.nanmean([r['vx_bias_body']     for r in fwd])) if fwd else float('nan')
    metrics['sim2sim/vx_bias_bwd']         = float(np.nanmean([r['vx_bias_body']     for r in bwd])) if bwd else float('nan')
    metrics['sim2sim/displacement_err_fwd'] = float(np.nanmean([r['displacement_err'] for r in fwd])) if fwd else float('nan')
    metrics['sim2sim/displacement_err_bwd'] = float(np.nanmean([r['displacement_err'] for r in bwd])) if bwd else float('nan')
    metrics['sim2sim/vy_drift']      = float(np.nanmean([r['vy_abs_mean']   for r in survived])) if survived else float('nan')
    metrics['sim2sim/roll_rms']      = float(np.nanmean([math.degrees(r['roll_rms'])   for r in survived])) if survived else float('nan')
    metrics['sim2sim/pitch_rms_fwd'] = float(np.nanmean([math.degrees(r['pitch_rms']) for r in fwd]))       if fwd      else float('nan')
    metrics['sim2sim/pitch_rms_bwd'] = float(np.nanmean([math.degrees(r['pitch_rms']) for r in bwd]))       if bwd      else float('nan')

    # Gait-quality (survived forward episodes — clearest signal; bwd is mirror)
    if fwd:
        metrics['sim2sim/com_z_rms']        = float(np.nanmean([r['com_z_rms']        for r in fwd]))
        metrics['sim2sim/yaw_osc_rms']      = float(np.nanmean([math.degrees(r['yaw_osc_rms']) for r in fwd]))
        metrics['sim2sim/measured_freq']    = float(np.nanmean([r['measured_freq']    for r in fwd]))
        metrics['sim2sim/duty_factor']      = float(np.nanmean([r['duty_factor']      for r in fwd]))
        metrics['sim2sim/landing_vz_peak']  = float(np.nanmean([r['landing_vz_peak']  for r in fwd]))
        metrics['sim2sim/stride_length']    = float(np.nanmean([r['stride_length']    for r in fwd]))
        metrics['sim2sim/stride_asymm']     = float(np.nanmean([r['stride_asymm']     for r in fwd]))

    # Per-friction CoT (forward, survived) — energy efficiency vs surface
    for fr in frictions:
        sub = [r for r in fwd if r['friction'] == fr]
        if sub:
            metrics[f'sim2sim/cot_fr{fr:.1f}'] = float(np.nanmean([r['cot'] for r in sub]))

    # yaw tracking (survived yaw-sweep episodes)
    yaw_rows = [r for r in rows if r['cmd_yaw'] != 0.0 and r['survived']]
    metrics['sim2sim/yaw_err'] = float(np.nanmean([r['yaw_error_mean'] for r in yaw_rows])) if yaw_rows else float('nan')

    # Strafe tracking + passive yaw drift (the metrics that exposed the
    # gait-sway-impersonating-strafe pathology in walk_v30 sim2sim eval).
    strafe_rows = [r for r in rows if r.get('cmd_vy', 0.0) != 0.0 and r['survived']]
    if strafe_rows:
        metrics['sim2sim/vy_bias_strafe']     = float(np.nanmean([abs(r['vy_bias_body'])   for r in strafe_rows]))
        metrics['sim2sim/displacement_err_y'] = float(np.nanmean([r['displacement_err_y']   for r in strafe_rows]))
    else:
        metrics['sim2sim/vy_bias_strafe']     = float('nan')
        metrics['sim2sim/displacement_err_y'] = float('nan')

    # Passive yaw drift on cmd_yaw=0 runs (covers fwd, bwd, strafe). Big magnitude
    # = robot rotates without being asked to — the dominant deployment failure.
    passive_yaw_rows = [r for r in rows if r['cmd_yaw'] == 0.0 and r['survived']]
    if passive_yaw_rows:
        metrics['sim2sim/yaw_drift_passive'] = float(np.nanmean([r['yaw_drift_passive']
                                                                  for r in passive_yaw_rows
                                                                  if not math.isnan(r['yaw_drift_passive'])]))
    else:
        metrics['sim2sim/yaw_drift_passive'] = float('nan')

    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default=None,
                        help='Sim2sim config YAML (optional — auto-reads manifest from --policy if omitted)')
    parser.add_argument('--policy',   default=None, help='Path to .onnx policy (auto-discovers manifest)')
    parser.add_argument('--runs',     type=int,   default=10,   help='Runs per condition')
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds per episode')
    parser.add_argument('--out',      default=None, help='Output CSV path')
    parser.add_argument('--no-plots', action='store_true', help='Skip matplotlib plots')
    parser.add_argument('--grid', default='vx', choices=['vx', 'vy', 'yaw', 'omni'],
                        help='Cmd grid: vx (default, 7 vx × 0 yaw), vy (3 vy), '
                             'yaw (5 yaw), omni (4×3×3 = 36 combos)')
    parser.add_argument('--report-only', default=None, metavar='CSV',
                        help='Skip evaluation; print report for existing CSV')
    args = parser.parse_args()

    if args.report_only:
        print_report(args.report_only)
        if not args.no_plots:
            make_plots(args.report_only)
        return

    if args.config is not None:
        # Explicit config file (legacy path — still supported)
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if args.policy is not None:
            cfg['policy_path'] = args.policy
    elif args.policy is not None:
        # Auto-discover manifest from policy path
        manifest, manifest_path = load_manifest(args.policy)
        if manifest is not None:
            print(f'[evaluate] Using manifest: {manifest_path}')
            cfg = manifest_to_sim2sim_cfg(manifest, args.policy)
        else:
            parser.error(f'No manifest found next to {args.policy}. '
                         f'Either export with export_pt2onnx.py first, or pass --config explicitly.')
    else:
        parser.error('Provide --policy (auto-discovers manifest) or --config (legacy sim2sim YAML).')

    if args.out is None:
        policy_dir = os.path.dirname(cfg['policy_path'])
        args.out = os.path.join(policy_dir, 'eval.csv')

    frictions = [0.5, 1.0, 1.5]
    if args.grid == 'vx':
        vx_list, vy_list, yaw_list = [-0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7], [0.0], [0.0]
    elif args.grid == 'vy':
        vx_list, vy_list, yaw_list = [0.0], [-0.3, 0.0, 0.3], [0.0]
    elif args.grid == 'yaw':
        vx_list, vy_list, yaw_list = [0.0], [0.0], [-1.0, -0.5, 0.0, 0.5, 1.0]
    elif args.grid == 'omni':
        # In-distribution grid matching pure_and_pairs training regime:
        # pure_vx / pure_vy / pure_yaw / vx+vy / vx+yaw. No 3-axis or vy+yaw
        # (env never trains those — testing them yields meaningless results).
        # The full Cartesian product below INCLUDES OOD combos; report_only
        # / downstream code is expected to filter to ID cmds before scoring.
        vx_list, vy_list, yaw_list = [-0.3, 0.0, 0.3, 0.5], [-0.3, 0.0, 0.3], [-0.5, 0.0, 0.5]
        print('[evaluate] WARNING: omni grid includes OOD 3-axis combos; filter '
              'downstream (vy+yaw and 3-axis are not in training distribution).')
    else:
        parser.error(f"Unknown --grid {args.grid!r}; pick from vx|vy|yaw|omni")

    csv_path = evaluate(cfg, args.runs, args.duration, frictions, vx_list, yaw_list,
                        args.out, vy_list=vy_list)

    print_report(csv_path)
    if not args.no_plots:
        print(f'\nGenerating plots → {os.path.dirname(os.path.abspath(csv_path))}')
        make_plots(csv_path)


if __name__ == '__main__':
    main()
