#!/usr/bin/env python
"""
reward_budget.py — Principled reward analysis for no-clock-in-obs walking.

Computes per-step reward contribution at TWO hand-built trajectory states:
  - WALK   = nominal walking at cmd_vx = 0.3 m/s, per docs/walking_physics_reference.md §2
  - SHUFFLE = v14 measured shuffle (latest TB iter 2000)

For each reward term in `configs/walk_noclock_v14.yaml`, calculate:
  - value at WALK
  - value at SHUFFLE
  - weighted contribution (= value × config weight)
  - gap Δ = walk_contrib − shuffle_contrib  (positive = favors walking)

Outputs a sorted table identifying reward terms that:
  1. Reward shuffle (Δ < 0 — backward signals; flip or remove)
  2. Don't discriminate (|Δ| small — wasted weight)
  3. Strongly favor walking (Δ > 0, large — keep / boost)

Total reward gap must be ≥ 5× larger than the SHUFFLE base reward for PPO
gradient to reliably converge to walking. If not, weights need redesign.

Trajectory values are HAND-COMPUTED from LIPM physics + URDF dimensions, NOT
empirical RL results. The goal is to know what we WANT before training.
"""

from __future__ import annotations
import math
import sys
import os
from dataclasses import dataclass, field
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_config


# ── Trajectory state definitions ─────────────────────────────────────────────

@dataclass
class State:
    """Hand-computed quantities describing the robot's instantaneous behaviour.

    All averages are over a steady-state walking window, not transient.
    See docs/walking_physics_reference.md §2 for the WALK derivation.
    """
    name: str

    # Command (we assume the steady-state cmd we care about)
    cmd_vx: float = 0.30
    cmd_vy: float = 0.0
    cmd_yaw: float = 0.0

    # Body motion
    body_vx: float = 0.30
    body_vy: float = 0.0
    body_yaw_rate: float = 0.0
    body_height: float = 0.40              # walking mean height (m)
    body_vz_rms: float = 0.15              # vertical osc RMS (m/s)
    base_ang_vel_xy_rms: float = 0.5       # roll+pitch rates (rad/s)
    base_acc_norm: float = 1.0             # |a - g| (m/s², small in steady walk)

    # Body orientation
    roll_rms: float = math.radians(1.5)    # rad
    pitch_rms: float = math.radians(3.0)   # rad
    projected_gravity_xy_norm: float = math.radians(3.0)  # ≈ sin(pitch) for small angles

    # Joint state
    # mean(|q-ref|²) per joint, used by pose_speed
    # Order: hip_yaw, hip_roll, hip_pitch, knee, ankle (×2 legs symmetric)
    joint_err_rms: tuple = (0.05, 0.03, 0.35, 0.30, 0.10)  # per joint
    joint_vel_rms: tuple = (1.0, 0.5, 3.0, 4.0, 1.0)        # peak ~5 rad/s
    joint_torque_rms: tuple = (3.0, 2.0, 8.0, 8.0, 3.0)     # Nm-ish

    # Foot kinematics
    duty_factor: float = 0.60              # per-foot ground-contact fraction
    swing_time_per_leg: float = 0.16       # s per swing
    stance_time_per_leg: float = 0.24      # s per stance
    leg_cycle_time: float = 0.40           # s
    measured_freq_per_leg: float = 2.5     # Hz
    stride_length: float = 0.12            # m per TD
    foot_swing_peak_z: float = 0.07        # m above ground at apex
    foot_clearance_mean_in_air: float = 0.035  # avg |foot_z - 0.06| over swing
    foot_xy_vel_avg_in_air: float = 0.75   # m/s horizontal during swing
    foot_contact_force_at_td: float = 50.  # N (soft landing target)

    # Phase alignment (used by foot_phase/foot_clr/foot_supt)
    # These three reward "fraction of feet correctly in their phase regime"
    foot_clr_phase_match_rate: float = 0.45 # fraction of cycle one foot is "correctly swinging"
    foot_supt_phase_match_rate: float = 0.50 # fraction of cycle one foot is "correctly in stance"
    foot_phase_alignment: float = 0.95     # foot_phase reward score (0-1)

    # Power
    mech_power_W: float = 8.0              # m·g·v·CoT

    # Action / network signals
    action_2nd_deriv_norm: float = 0.05    # L1 norm of finite-diff 2nd deriv
    net_out_norm: float = 0.5              # ‖net_out‖_L2

    # Constants
    static_flag: float = 1.0               # 1 = walking, 0 = standing
    dt: float = 0.015                       # policy step time (s)


WALK = State(
    name='WALK (IDEAL: LIPM nominal physics, cmd_vx=0.3)',
    # All defaults set for nominal walking per physics ref §2.
    # Phase alignment values (foot_clr_phase_match_rate, foot_supt_phase_match_rate,
    # foot_phase_alignment) are IDEAL — assume policy syncs perfectly to phase
    # clock. In practice (no-clock-in-obs), policy doesn't see phase so alignment
    # is partial. See WALK_ACHIEVABLE for empirical-calibrated values.
)

# Best empirical state from walk_clean_v3 iter 1200 (TB measurements).
# This is what current reward design ACTUALLY achieves, not ideal.
# Use this to compute realistic budget ratios.
WALK_ACHIEVABLE = State(
    name='WALK_ACHIEVABLE (v3 iter 1200 empirical: stride 8.8cm, duty 0.63)',
    body_vx=0.20,                          # bias +0.10 → body moves 0.20 at cmd 0.30
    body_height=0.41,
    body_vz_rms=0.08,
    base_ang_vel_xy_rms=0.7,
    pitch_rms=math.radians(5.0),
    projected_gravity_xy_norm=math.radians(5.0),
    roll_rms=math.radians(2.0),
    joint_err_rms=(0.05, 0.03, 0.30, 0.25, 0.08),
    joint_vel_rms=(0.8, 0.5, 2.5, 3.0, 0.9),
    joint_torque_rms=(2.5, 2.0, 6.0, 6.0, 2.5),
    duty_factor=0.63,                       # measured
    swing_time_per_leg=0.11,                # measured (-0.05 from 0.16 target)
    stance_time_per_leg=0.19,
    leg_cycle_time=0.30,
    measured_freq_per_leg=3.3,
    stride_length=0.088,                    # measured
    foot_swing_peak_z=0.05,
    foot_clearance_mean_in_air=0.025,
    foot_xy_vel_avg_in_air=0.55,
    foot_contact_force_at_td=60.,
    # Phase alignment empirically MUCH lower than ideal (policy can't see clock):
    foot_clr_phase_match_rate=0.17,         # measured (vs 0.45 ideal)
    foot_supt_phase_match_rate=0.29,        # measured (vs 0.50 ideal)
    foot_phase_alignment=0.55,
    mech_power_W=6.0,
    action_2nd_deriv_norm=0.06,
    net_out_norm=0.45,
)

# Phase B target: stable forward walking at cmd_vx=0.3.
# Anti-pattern at cmd_vx=0.3: shuffle (high duty, short stride).
WALK_PHASE_B = State(
    name='WALK_PHASE_B (forward 0.3, target gait — duty 0.6, stride 12cm)',
    # Same as WALK (LIPM nominal at v=0.3). Renamed for clarity.
)

# Phase B failure mode: policy converges to shuffle when commanded forward.
SHUFFLE_AT_FWD = State(
    name='SHUFFLE_AT_FWD (cmd_vx=0.3, policy shuffles)',
    body_vx=0.20,                          # undershoots cmd
    body_height=0.38,
    body_vz_rms=0.05,                       # crushed by minimize vertical_vel
    base_ang_vel_xy_rms=0.7,
    pitch_rms=math.radians(5.0),
    projected_gravity_xy_norm=math.radians(5.0),
    roll_rms=math.radians(2.0),
    joint_err_rms=(0.04, 0.03, 0.15, 0.13, 0.06),
    joint_vel_rms=(0.6, 0.4, 2.0, 2.0, 0.7),
    joint_torque_rms=(2.8, 1.8, 6.0, 6.0, 2.8),
    duty_factor=0.75,
    swing_time_per_leg=0.07,
    stance_time_per_leg=0.21,
    leg_cycle_time=0.28,
    measured_freq_per_leg=3.6,
    stride_length=0.06,
    foot_swing_peak_z=0.035,
    foot_clearance_mean_in_air=0.025,
    foot_xy_vel_avg_in_air=0.55,
    foot_contact_force_at_td=70.,
    foot_clr_phase_match_rate=0.10,
    foot_supt_phase_match_rate=0.50,
    foot_phase_alignment=0.80,
    mech_power_W=5.0,
    action_2nd_deriv_norm=0.07,
    net_out_norm=0.45,
)

# Phase B failure mode: policy keeps standing when commanded forward (cmd=0.3
# but body doesn't move). This is what walk_clean_v1 does → OK for Phase A
# but a failure for Phase B.
STAND_AT_FWD = State(
    name='STAND_AT_FWD (cmd_vx=0.3, policy stands still — wrong)',
    body_vx=0.0,                            # NOT moving
    body_height=0.45,
    body_vz_rms=0.001,
    base_ang_vel_xy_rms=0.1,
    pitch_rms=math.radians(0.5),
    projected_gravity_xy_norm=math.radians(0.5),
    roll_rms=math.radians(0.5),
    joint_err_rms=(0.01, 0.01, 0.02, 0.02, 0.01),  # near-perfect ref pose
    joint_vel_rms=(0.02, 0.01, 0.05, 0.05, 0.02),
    joint_torque_rms=(2.0, 1.5, 4.0, 4.0, 2.0),
    duty_factor=1.0,                        # both feet always down
    swing_time_per_leg=0.0,
    stance_time_per_leg=0.5,
    leg_cycle_time=0.5,
    measured_freq_per_leg=0.0,
    stride_length=0.0,
    foot_swing_peak_z=0.0,
    foot_clearance_mean_in_air=0.0,
    foot_xy_vel_avg_in_air=0.0,
    foot_contact_force_at_td=0.0,
    foot_clr_phase_match_rate=0.0,         # never matches swing (foot never in air)
    foot_supt_phase_match_rate=0.50,       # half-time correctly in stance
    foot_phase_alignment=0.50,             # phase ticks but feet don't match swing half
    mech_power_W=2.0,                       # idle power
    action_2nd_deriv_norm=0.005,
    net_out_norm=0.15,
)

SHORT_STEP = State(
    name='SHORT_STEP (v15 measured @ iter 1400, q=0.75, NEW failure mode)',
    body_vx=0.287,                         # bias -0.013, near-perfect tracking
    body_height=0.40,                       # didn't drop
    body_vz_rms=0.10,                       # less osc with short steps
    base_ang_vel_xy_rms=1.0,                # high — fast cadence creates angular bursts
    pitch_rms=math.radians(10.0),           # still too high
    projected_gravity_xy_norm=math.radians(10.0),
    roll_rms=math.radians(3.0),
    joint_err_rms=(0.06, 0.04, 0.18, 0.15, 0.08),  # mid-range — some swing but not full
    joint_vel_rms=(0.8, 0.5, 2.5, 2.5, 1.0),       # higher cadence → bigger avg
    joint_torque_rms=(3.0, 2.0, 6.0, 6.0, 2.5),
    duty_factor=0.51,                       # ⭐ nearly perfect alternation
    swing_time_per_leg=0.075,               # short cycle, brief swing
    stance_time_per_leg=0.078,
    leg_cycle_time=0.153,                   # 6.5 Hz/leg, FAST
    measured_freq_per_leg=6.5,
    stride_length=0.047,                    # short!
    foot_swing_peak_z=0.035,                # low (short steps lift less)
    foot_clearance_mean_in_air=0.025,
    foot_xy_vel_avg_in_air=0.60,
    foot_contact_force_at_td=60.,
    foot_clr_phase_match_rate=0.45,         # GOOD alternation w/ phase
    foot_supt_phase_match_rate=0.50,
    foot_phase_alignment=0.95,              # high — phase clock matches
    mech_power_W=6.0,                       # rapid steps cost moderate power
    action_2nd_deriv_norm=0.07,             # rapid actions
    net_out_norm=0.5,
)

# Hypothetical: perfect anti-phase alternation BUT no forward translation.
# Tests whether phase rewards alone discriminate "real walking" from
# "stomping in place".
STOMP_IN_PLACE = State(
    name='STOMP_IN_PLACE (cmd=0.3, feet alternate but body_vx=0)',
    body_vx=0.0,                            # NOT moving despite cmd 0.3
    body_height=0.40,
    body_vz_rms=0.12,
    base_ang_vel_xy_rms=0.4,
    pitch_rms=math.radians(2.0),
    projected_gravity_xy_norm=math.radians(2.0),
    roll_rms=math.radians(1.5),
    joint_err_rms=(0.04, 0.03, 0.20, 0.20, 0.06),
    joint_vel_rms=(0.6, 0.4, 2.5, 2.5, 0.8),
    joint_torque_rms=(2.5, 1.8, 6.0, 6.0, 2.5),
    duty_factor=0.50,                       # perfect alternation
    swing_time_per_leg=0.15,                # near-target
    stance_time_per_leg=0.15,
    leg_cycle_time=0.30,
    measured_freq_per_leg=3.3,
    stride_length=0.0,                      # ZERO — no translation
    foot_swing_peak_z=0.06,                 # proper swing height
    foot_clearance_mean_in_air=0.020,
    foot_xy_vel_avg_in_air=0.0,             # foot moves up/down, not forward
    foot_contact_force_at_td=50.,
    foot_clr_phase_match_rate=0.45,         # perfect phase alignment
    foot_supt_phase_match_rate=0.50,
    foot_phase_alignment=0.95,
    mech_power_W=5.0,
    action_2nd_deriv_norm=0.05,
    net_out_norm=0.4,
)

SHUFFLE = State(
    name='SHUFFLE (v14 measured @ iter 2000)',
    body_vx=0.25,                          # vx_bias -0.05 → body moves 0.25 vs cmd 0.30
    body_height=0.37,
    body_vz_rms=0.02,                      # crushed by vertical_vel reward
    base_ang_vel_xy_rms=0.8,
    pitch_rms=math.radians(10.0),          # measured!
    projected_gravity_xy_norm=math.radians(10.0),
    roll_rms=math.radians(3.0),
    joint_err_rms=(0.03, 0.02, 0.12, 0.10, 0.05),  # close to ref, shuffle-like
    joint_vel_rms=(0.5, 0.3, 1.5, 1.5, 0.5),
    joint_torque_rms=(2.5, 1.5, 5.0, 5.0, 2.5),
    duty_factor=0.75,                      # measured!
    swing_time_per_leg=0.06,               # brief lift
    stance_time_per_leg=0.20,
    leg_cycle_time=0.26,                   # ~3.8 Hz
    measured_freq_per_leg=3.8,
    stride_length=0.066,                   # measured!
    foot_swing_peak_z=0.04,                # lower lifts
    foot_clearance_mean_in_air=0.025,
    foot_xy_vel_avg_in_air=0.50,
    foot_contact_force_at_td=80.,
    foot_clr_phase_match_rate=0.10,        # phase mostly misaligned with brief lifts
    foot_supt_phase_match_rate=0.45,       # support side roughly right since both feet on ground
    foot_phase_alignment=0.75,
    mech_power_W=4.0,                      # less effort = less power
    action_2nd_deriv_norm=0.08,            # micro-tremor = bigger 2nd deriv
    net_out_norm=0.4,
)


# ── Reward formulas ──────────────────────────────────────────────────────────
# Each function returns the UNWEIGHTED per-step reward for the given state.
# Sign matters: negative values are penalties.

def _lvxn(s):
    """lin_vel_x_norm = clip(||cmd||,0.3,2)+0.2, used as denom in many rewards."""
    cmd_mag = math.sqrt(s.cmd_vx**2 + s.cmd_vy**2)
    return max(0.3, min(2.0, cmd_mag)) + 0.2


def _reg_norm_inv(s):
    return 1.0 / _lvxn(s)


def fwd_vel(s, cfg):
    slope = cfg.reward.fwd_err_slope
    err = abs(s.body_vx - s.cmd_vx)
    return 1.0 - max(0.0, min(1.5, slope * err))


def lateral_vel(s, cfg):
    slope = cfg.reward.lateral_err_slope
    err = abs(s.body_vy - s.cmd_vy)
    return 1.0 - max(0.0, min(1.5, slope * err))


def yaw_rat(s, cfg):
    slope = cfg.reward.yaw_err_slope
    err = abs(s.body_yaw_rate - s.cmd_yaw)
    return 1.0 - max(0.0, min(1.5, slope * err))


def base_heit(s, cfg):
    slope = cfg.reward.base_heit_slope
    target = cfg.reward.base_heit_target
    return math.exp(-slope * (s.body_height - target)**2)


def balance(s, cfg):
    bh = base_heit(s, cfg)
    tilt_alpha = max(2.0, min(8.0, 5.0 / _lvxn(s)))
    return 0.5 * (bh * math.exp(-tilt_alpha * math.sqrt(s.roll_rms**2 + s.pitch_rms**2)) + 1.0)


def upright(s, cfg):
    std = cfg.reward.upright_std
    return math.exp(-(s.projected_gravity_xy_norm**2) / (std**2))


def pose_speed(s, cfg):
    # bell over per-joint err² / σ²
    std_walk = getattr(cfg.reward, 'pose_std_per_joint_walking', None) or [0.15]*5
    # First 5 joints (one leg) — mirror for other leg
    std_one = std_walk[:5]
    err_sq_per_joint = [e**2 / max(1e-6, sw**2) for e, sw in zip(s.joint_err_rms, std_one)]
    # Bell uses MEAN over joints (line 1048)
    mean_err_sq_over_sigma_sq = sum(err_sq_per_joint) / len(err_sq_per_joint)
    pose_rew = math.exp(-mean_err_sq_over_sigma_sq)

    # Walking gate: pose_speed_walking_gate=0.3 means reward scaled by 0.3 during walk
    gate = float(getattr(cfg.reward, 'pose_speed_walking_gate', 1.0) or 1.0)
    return pose_rew * ((1.0 - s.static_flag) + gate * s.static_flag)


def twist(s, cfg):
    return -math.sqrt(s.roll_rms**2 + s.pitch_rms**2)


def vertical_vel(s, cfg):
    target = float(getattr(cfg.reward, 'vertical_vel_target_walk', 0.0) or 0.0)
    if target > 0:
        std_walk = float(getattr(cfg.reward, 'vertical_vel_std_walk', 0.10) or 0.10)
        std_static = float(getattr(cfg.reward, 'vertical_vel_std_static', 0.10) or 0.10)
        vz_mag = s.body_vz_rms
        bw = math.exp(-((vz_mag - target)/std_walk)**2)
        bs = math.exp(-((vz_mag - 0.0)/std_static)**2)
        return bs * (1.0 - s.static_flag) + bw * s.static_flag
    else:
        alpha = max(2.0, min(10.0, 5.0 / _lvxn(s)))
        return math.exp(-alpha * s.body_vz_rms**2) - 0.2 * _reg_norm_inv(s) * s.body_vz_rms * s.static_flag


def ang_vel(s, cfg):
    alpha = max(0.7, min(6.0, 2.0 / _lvxn(s)))
    return math.exp(-alpha * s.base_ang_vel_xy_rms**2)


def base_acc(s, cfg):
    return -0.4 * _reg_norm_inv(s) * s.base_acc_norm * 0.1 * s.static_flag


def foot_phase(s, cfg):
    """PENALTY form (phase.mode=input): (swing_match + support_match)/2 - 1.0.
    Range [-1, 0]: 0=perfect alignment, -1=fully wrong, -0.5=half (e.g. shuffle).
    See birl_task.py:1369-1373."""
    avg_match = 0.5 * (s.foot_clr_phase_match_rate + s.foot_supt_phase_match_rate)
    return (2.0 * avg_match - 1.0) * s.static_flag


def _single_support_fraction(duty):
    """Fraction of time exactly one foot on ground. p_single = 2 - 2·duty
    (for duty ≥ 0.5). At duty 0.6 = 0.80, at duty 0.75 = 0.50, at duty 1.0 = 0."""
    return max(0.0, min(1.0, 2.0 - 2.0 * duty))


def foot_clr(s, cfg):
    """Gated by single-support state (NEW): reward only when exactly
    one foot in air AND that foot matches swing_mask. Removes shuffle
    exploit where both-grounded trivially matched stance half the time."""
    return s.foot_clr_phase_match_rate * _single_support_fraction(s.duty_factor) * s.static_flag


def foot_supt(s, cfg):
    """Gated by single-support state (NEW): reward only when exactly
    one foot on ground AND it matches support_mask."""
    return s.foot_supt_phase_match_rate * _single_support_fraction(s.duty_factor) * s.static_flag


def foot_heit(s, cfg):
    """Foot height target ~0.05m during swing. Approximated by inverse |peak - 0.05|."""
    err = abs(s.foot_swing_peak_z - 0.05)
    return math.exp(-50 * err**2) * s.static_flag


def foot_stand(s, cfg):
    """Reward both feet on ground when STANDING (cmd ≈ 0).
    During walking, static_flag=1, so foot_stand × (1-static_flag) = 0."""
    return 0.0  # walking state always


def feet_clearance_l1(s, cfg):
    """v10 form: -|foot_z - target| × in_air mask × cmd_active.
    Per step, summed over 2 feet. In walking, ONE foot is typically in air ~40% of cycle."""
    target = float(getattr(cfg.reward, 'feet_clearance_target', 0.06) or 0.06)
    cmd_threshold = float(getattr(cfg.reward, 'feet_clearance_cmd_threshold', 0.05) or 0.05)
    cmd_total = math.sqrt(s.cmd_vx**2 + s.cmd_vy**2) + abs(s.cmd_yaw)
    cmd_active = 1.0 if cmd_total > cmd_threshold else 0.0
    # avg |foot_z - target| over in-air time × fraction-of-time in air, per leg
    in_air_fraction_per_leg = 1.0 - s.duty_factor
    # Sum over 2 legs:
    return -2 * s.foot_clearance_mean_in_air * in_air_fraction_per_leg * cmd_active


def feet_swing_height_peak(s, cfg):
    """Sparse at TD events: -(peak/target - 1)² × td_event."""
    target = float(getattr(cfg.reward, 'feet_swing_target', 0.06) or 0.06)
    swing_err = max(-1.0, min(2.0, s.foot_swing_peak_z / target - 1.0))
    # TD events: 2 feet × measured_freq per leg per step
    tds_per_step = 2 * s.measured_freq_per_leg * s.dt
    return -(swing_err**2) * tds_per_step


def soft_landing(s, cfg):
    tds_per_step = 2 * s.measured_freq_per_leg * s.dt
    return -s.foot_contact_force_at_td * tds_per_step


def air_time(s, cfg):
    """PENALTY form: held delta = swing_duration - target_swing_time (≈0.16s).
    Negative when swing too short, zero when matching target. Each TD event
    snapshots this delta, then it's held between TDs (constant per-step).
    See birl_task.py:1377-1380."""
    target_swing = 0.16     # physics nominal swing time for cmd_vx=0.3
    delta = s.swing_time_per_leg - target_swing
    return delta * s.static_flag


def act_smo(s, cfg):
    return -max(0.0, min(2.0, _reg_norm_inv(s))) * s.action_2nd_deriv_norm


def jnt_vel(s, cfg):
    # ||dq||² summed across 10 joints (2 legs × 5 joints, mirror)
    sum_sq = 2 * sum(v**2 for v in s.joint_vel_rms)
    return -sum_sq * 0.1   # legacy scaling


def joint_tor(s, cfg):
    sum_sq = 2 * sum(t**2 for t in s.joint_torque_rms)
    return -0.4 * max(0.0, min(2.0, _reg_norm_inv(s))) * sum_sq * 0.001


def power(s, cfg):
    return -s.mech_power_W / 100.


def stride_length(s, cfg):
    """Bell on stride per TD, fired at TD events."""
    target = float(getattr(cfg.reward, 'stride_length_target', 0.0) or 0.0)
    if target <= 0:
        return 0.0
    std = float(getattr(cfg.reward, 'stride_length_std', 0.05) or 0.05)
    bell = math.exp(-((s.stride_length - target)/std)**2)
    tds_per_step = 2 * s.measured_freq_per_leg * s.dt
    return bell * tds_per_step * s.static_flag


def constant(s, cfg):
    return 1.0


def single_support(s, cfg):
    """XOR(left_in_air, right_on_ground) — rewards exactly one foot up.
    Clock-free. Time average ≈ single_support_fraction.
    Output: 2 × single_support_fraction - 1 (range [-1, +1])."""
    sf = _single_support_fraction(s.duty_factor)
    return (2.0 * sf - 1.0) * s.static_flag


# Minor terms — approximate (won't dominate the budget)
def foot_sft(s, cfg):
    """Soft landing — penalises hard touchdown velocity. Approx -peak_vz * TD."""
    peak_vz = 1.0 if s.duty_factor < 0.7 else 1.5  # walking lands harder
    tds_per_step = 2 * s.measured_freq_per_leg * s.dt
    return -peak_vz * tds_per_step * 0.5


def foot_slip(s, cfg):
    return -s.foot_xy_vel_avg_in_air * (1.0 - s.duty_factor) * 0.5


def feet_py(s, cfg):
    return -0.05   # crude — foot lateral position stays small


def leg_width_rew(s, cfg):
    return 0.8     # crude — both runs maintain ok leg width


# ── Reward registry ──────────────────────────────────────────────────────────

REWARDS = [
    # (name, fn, [weight_attr — defaults to name])
    ('fwd_vel',                fwd_vel),
    ('lateral_vel',            lateral_vel),
    ('yaw_rat',                yaw_rat),
    ('base_heit',              base_heit),
    ('balance',                balance),
    ('upright',                upright),
    ('pose_speed',             pose_speed),
    ('twist',                  twist),
    ('vertical_vel',           vertical_vel),
    ('ang_vel',                ang_vel),
    ('base_acc',               base_acc),
    ('foot_phase',             foot_phase),
    ('foot_clr',               foot_clr),
    ('foot_supt',              foot_supt),
    ('foot_heit',              foot_heit),
    ('foot_stand',             foot_stand),
    ('feet_clearance_l1',      feet_clearance_l1),
    ('feet_swing_height_peak', feet_swing_height_peak),
    ('soft_landing',           soft_landing),
    ('air_time',               air_time),
    ('act_smo',                act_smo),
    ('jnt_vel',                jnt_vel),
    ('joint_tor',              joint_tor),
    ('power',                  power),
    ('stride_length',          stride_length),
    ('foot_sft',               foot_sft),
    ('foot_slip',              foot_slip),
    ('feet_py',                feet_py),
    ('leg_width_rew',          leg_width_rew),
    ('constant',               constant),
    ('single_support',         single_support),
]


# ── Main ─────────────────────────────────────────────────────────────────────

def analyse(cfg_path, state_b=None):
    cfg = load_config(cfg_path)
    if state_b is None:
        state_b = SHUFFLE
    print(f'\nConfig: {cfg_path}')
    print(f'State A: {WALK.name}')
    print(f'State B: {state_b.name}\n')

    # Build table
    rows = []
    walk_total = 0.0
    other_total = 0.0
    for name, fn in REWARDS:
        w = float(getattr(cfg.reward, name, 0.0) or 0.0)
        if w == 0:
            continue
        wv = fn(WALK,    cfg)
        sv = fn(state_b, cfg)
        ww = w * wv
        sw = w * sv
        gap = ww - sw
        rows.append((name, w, wv, sv, ww, sw, gap))
        walk_total += ww
        other_total += sw

    # Sort by absolute gap descending
    rows.sort(key=lambda r: -abs(r[6]))

    print(f"{'name':<28} {'w':>6} {'walk_v':>8} {'other_v':>8} {'walk×w':>8} {'other×w':>8} {'Δ/step':>8}")
    print('-' * 95)
    for name, w, wv, sv, ww, sw, gap in rows:
        flag = ''
        if gap < -0.01:    flag = ' ✗ (favors OTHER)'
        elif abs(gap) < 0.01: flag = ' ~ (no discr.)'
        elif gap > 0.5:    flag = ' ⭐ (strong walk)'
        print(f"{name:<28} {w:>6.3g} {wv:>+8.3f} {sv:>+8.3f} {ww:>+8.3f} {sw:>+8.3f} {gap:>+8.3f}{flag}")

    print('-' * 95)
    print(f"{'TOTAL':<28} {'':>6} {'':>8} {'':>8} {walk_total:>+8.3f} {other_total:>+8.3f} {walk_total - other_total:>+8.3f}")
    print()
    ratio = (walk_total / other_total) if other_total > 0 else float('inf')
    print(f'Total walk_reward    = {walk_total:.3f}')
    print(f'Total other_reward   = {other_total:.3f}')
    print(f'Walk/Other ratio     = {ratio:.2f}× ', end='')
    if ratio >= 5.0:
        print('  ⭐ (rule of thumb: PPO converges to walk)')
    elif ratio >= 2.0:
        print('  ⚠ (marginal — may oscillate)')
    else:
        print('  ✗ (too close — competing optimum)')


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/walk_noclock_v14.yaml'
    state_arg = sys.argv[2] if len(sys.argv) > 2 else 'shuffle'
    state_b = {
        'shuffle': SHUFFLE,
        'short': SHORT_STEP,
        'shuffle_fwd': SHUFFLE_AT_FWD,
        'stand_fwd':   STAND_AT_FWD,
        'achievable':  WALK_ACHIEVABLE,
        'stomp':       STOMP_IN_PLACE,
    }[state_arg.lower()]
    analyse(cfg_path, state_b)
