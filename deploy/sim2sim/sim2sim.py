"""
Sim2Sim: Validate a trained policy in MuJoCo before real deployment.

Train in Isaac Gym → validate here → deploy to real robot.

Usage:
    cd ~/code/RoboTamer4Qmini
    python deploy/sim2sim/sim2sim.py --policy experiments/<name>/deploy/policy_<iter>.onnx
                                     [--cmd_vx 0.5] [--cmd_yaw 0.0]
                                     [--duration 30]
                                     [--interactive]   # keyboard / joystick control
"""

import os
import argparse
import time
import threading
from collections import deque
from math import tau

import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import yaml

# Optional pygame for joystick support
try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

# ---------------------------------------------------------------------------
# Manifest loader — build sim2sim cfg from self-describing export manifest
# ---------------------------------------------------------------------------

# Torque correction terms (hardcoded in legged_robot.py — robot-specific constants)
_JOINT_TOR_OFFSET = [0.6, 1.0, 0.0, 0.7, 0.0, -0.6, -1.0, 0.0, -0.7, 0.0]
_JOINT_VEL_SIGN   = [0.0, 1.0, 0.0, 0.0, 0.0,  0.0,  1.0, 0.0,  0.0, 0.0]


def load_manifest(policy_path):
    """Auto-discover and load the manifest YAML next to an ONNX policy file.

    Looks for:
      1. <policy_name>_manifest.yaml  (e.g. policy_2000_manifest.yaml)
      2. manifest.yaml                (generic fallback)

    Returns (manifest_dict, manifest_path) or (None, None) if not found.
    """
    policy_dir = os.path.dirname(os.path.abspath(policy_path))
    stem = os.path.splitext(os.path.basename(policy_path))[0]

    candidates = [
        os.path.join(policy_dir, f'{stem}_manifest.yaml'),
        os.path.join(policy_dir, 'manifest.yaml'),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return yaml.safe_load(f), p
    return None, None


def manifest_to_sim2sim_cfg(manifest, policy_path):
    """Convert a deploy manifest dict into the flat cfg dict used by run() / run_episode().

    This replaces the hand-maintained deploy/sim2sim/configs/*.yaml files.
    """
    action_mode = manifest['action_mode']
    scaling = manifest['action_scaling']
    if action_mode == 'absolute':
        act_low = scaling['abs_low']
        act_high = scaling['abs_high']
    else:
        act_low = scaling['inc_low']
        act_high = scaling['inc_high']

    phase_mode = manifest['phase_modulator'].get('mode', 'output')
    pm_block = manifest['phase_modulator']
    freq_default = float(pm_block.get('freq_default', pm_block.get('base_freq', 2.5)))
    freq_low = float(pm_block.get('freq_low', freq_default - 0.5))
    freq_high = float(pm_block.get('freq_high', freq_default + 0.5))

    cfg = {
        'policy_path':        policy_path,
        'urdf_path':          manifest.get('urdf_path', 'assets/q1/urdf/q1.urdf'),
        'simulation_duration': 30.0,
        'simulation_dt':      manifest.get('simulation_dt', 0.001),
        'control_decimation': manifest['pd_gains']['decimation'],
        'init_height':        manifest.get('init_height', 0.5),
        'cmd_vx':             0.5,
        'cmd_vy':             0.0,
        'cmd_yaw':            0.0,
        'cmd_freq':           freq_default,
        'ref_joint_pos':      manifest['ref_joint_pos'],
        'kps':                manifest['pd_gains']['kps'],
        'kds':                manifest['pd_gains']['kds'],
        'joint_tor_offset':   _JOINT_TOR_OFFSET,
        'joint_vel_sign':     _JOINT_VEL_SIGN,
        'action_inc_low':     act_low,
        'action_inc_high':    act_high,
        'joint_limit_low':    manifest['joint_limits']['low'],
        'joint_limit_high':   manifest['joint_limits']['high'],
        'num_obs_per_step':   manifest['obs_per_step'],
        'obs_history':        manifest['obs_history'],
        'num_legs':           manifest['phase_modulator']['num_legs'],
        'static_cmd_threshold': manifest['phase_modulator'].get('static_cmd_threshold', 0.15),
        'action_mode':        manifest['action_mode'],
        'action_lowpass_alpha': manifest['action_lowpass_alpha'],
        'phase_mode':         phase_mode,
        'phase_freq_low':     freq_low,
        'phase_freq_high':    freq_high,
        'phase_freq_default': freq_default,
    }
    return cfg


# ---------------------------------------------------------------------------
# Phase modulator (mirrors env/utils/phase_modulator.py)
# ---------------------------------------------------------------------------

class PhaseModulator:
    def __init__(self, dt, num_legs=2):
        self.dt = dt
        self.num_legs = num_legs
        self.phase = np.zeros(num_legs, dtype=np.float32)
        self.frequency = np.ones(num_legs, dtype=np.float32) * 0.5

    def reset(self):
        self.phase[:] = 0.0  # start at zero for reproducibility
        self.frequency[:] = 0.5

    def compute(self, frequency):
        self.frequency = np.array(frequency, dtype=np.float32)
        self.phase = (self.phase + tau * self.frequency * self.dt) % tau
        return self.phase


class ExternalPhaseClock:
    """Numpy mirror of env/utils/phase_modulator.ExternalPhaseClock (BD_X style).

    Frequency is an externally-supplied command (deploy-time adjustable),
    not a function of velocity command magnitude.
    """

    def __init__(self, dt, num_legs=2, default_freq=2.5):
        self.dt = dt
        self.num_legs = num_legs
        self.default_freq = default_freq
        self.phase = np.zeros(num_legs, dtype=np.float32)
        self.frequency = np.ones(num_legs, dtype=np.float32) * default_freq
        self._leg_offset = np.zeros(num_legs, dtype=np.float32)
        self._leg_offset[1] = np.pi  # anti-phase

    def reset(self):
        self.phase[:] = 0.0
        self.frequency[:] = self.default_freq

    def update(self, freq):
        """Advance phase using an externally-supplied frequency (Hz scalar)."""
        freq = float(freq)
        self.frequency[:] = freq
        self.phase = (self.phase + tau * freq * self.dt) % tau

    def sin_cos(self):
        """Return [sin(L), sin(R), cos(L), cos(R)] — 4-dim."""
        p = (self.phase + self._leg_offset) % tau
        return np.concatenate([np.sin(p), np.cos(p)]).astype(np.float32)


# ---------------------------------------------------------------------------
# Math helpers (mirror training code conventions)
# ---------------------------------------------------------------------------

def quat_to_euler_xyz(q_wxyz):
    """
    MuJoCo quaternion [w, x, y, z] → euler [roll, pitch, yaw] in [-pi, pi].
    Matches Isaac Gym's get_euler_xyz + _from_2pi_to_pi.
    """
    w, x, y, z = q_wxyz
    # roll
    sinr = 2.0 * (w * x + y * z)
    cosr = w*w - x*x - y*y + z*z
    roll = np.arctan2(sinr, cosr)
    # pitch
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw
    siny = 2.0 * (w * z + x * y)
    cosy = w*w + x*x - y*y - z*z
    yaw = np.arctan2(siny, cosy)
    # wrap to [-pi, pi]
    roll  = roll  - 2 * np.pi * np.floor((roll  + np.pi) / (2 * np.pi))
    pitch = pitch - 2 * np.pi * np.floor((pitch + np.pi) / (2 * np.pi))
    yaw   = yaw   - 2 * np.pi * np.floor((yaw   + np.pi) / (2 * np.pi))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def quat_rotate_inverse(q_wxyz, vec):
    """
    Rotate vector from world frame to body frame.
    Matches Isaac Gym's quat_rotate_inverse(base_quat, world_vel).
    MuJoCo quaternion convention: [w, x, y, z].
    """
    w, x, y, z = q_wxyz
    # rotation matrix (body-to-world), then transpose for world-to-body
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [    2*(x*y + w*z),   1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [    2*(x*z - w*y),       2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return (R.T @ np.array(vec)).astype(np.float32)


def scale_transform(action, low, high, clip_val=1.0):
    """Matches env/utils/math.py scale_transform."""
    action = np.clip(action, -clip_val, clip_val)
    return (action + 1.0) / 2.0 * (high - low) + low


# ---------------------------------------------------------------------------
# Interactive command input (keyboard via MuJoCo key_callback + joystick via pygame)
# ---------------------------------------------------------------------------

class CommandInput:
    """
    Live command input — no conflict with MuJoCo hotkeys.

    Two input modes (automatically selected):
      1. Joystick via pygame (if connected):
            Left stick Y  → cmd_vx   (push forward = positive)
            Left stick X  → cmd_vy   (push left    = positive)
            Right stick X → cmd_yaw
            Button 0      → reset robot

      2. Arrow keys via MuJoCo key_callback (fallback — safe, no MuJoCo conflicts):
            ↑ / ↓         cmd_vx  +/- 0.1 m/s
            ← / →         cmd_vy  +/- 0.1 m/s  (← = left/+vy, → = right/-vy)
            PgUp / PgDn   cmd_yaw +/- 0.2 rad/s
            Delete        stop (zero all)
            End           reset robot
    """
    # GLFW key codes — arrow/nav keys not bound by MuJoCo viewer
    _KEY_UP    = 265
    _KEY_DOWN  = 264
    _KEY_LEFT  = 263
    _KEY_RIGHT = 262
    _KEY_PGUP  = 266
    _KEY_PGDN  = 267
    _KEY_DEL   = 261   # stop
    _KEY_END   = 269   # reset

    _VX_STEP   = 0.1
    _VY_STEP   = 0.1
    _YAW_STEP  = 0.2
    _VX_RANGE  = (-0.5, 0.7)
    _VY_RANGE  = (-0.3, 0.3)
    _YAW_RANGE = (-1.0, 1.0)

    def __init__(self, cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=0.0):
        self.cmd_vx  = cmd_vx
        self.cmd_vy  = cmd_vy
        self.cmd_yaw = cmd_yaw
        self._reset_requested = False
        self._lock = threading.Lock()

        # Try joystick first
        self._joystick = None
        if _PYGAME_OK:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                print(f"[input] Joystick: {self._joystick.get_name()} — using joystick mode")
                self._print_joystick_help()
                return

        # Fallback: arrow keys via MuJoCo key_callback
        self._print_keyboard_help()

    def _print_joystick_help(self):
        print("\n=== Joystick Control ===")
        print("  Left stick Y   cmd_vx  (forward/back)")
        print("  Left stick X   cmd_vy  (left/right strafe)")
        print("  Right stick X  cmd_yaw (turn)")
        print("  Button 0       reset robot\n")

    def _print_keyboard_help(self):
        print("\n=== Keyboard Control (MuJoCo window must have focus) ===")
        print("  ↑ / ↓          cmd_vx  +/-0.1 m/s  (forward/back)")
        print("  ← / →          cmd_vy  +/-0.1 m/s  (strafe left/right)")
        print("  PgUp / PgDn    cmd_yaw +/-0.2 rad/s (turn)")
        print("  Delete         stop (zero all commands)")
        print("  End            reset robot to start\n")

    def key_callback(self, keycode):
        """Called by MuJoCo viewer — arrow/nav keys only, no conflicts."""
        changed = True
        with self._lock:
            if   keycode == self._KEY_UP:    self.cmd_vx  = float(np.clip(self.cmd_vx  + self._VX_STEP,  *self._VX_RANGE))
            elif keycode == self._KEY_DOWN:  self.cmd_vx  = float(np.clip(self.cmd_vx  - self._VX_STEP,  *self._VX_RANGE))
            elif keycode == self._KEY_LEFT:  self.cmd_vy  = float(np.clip(self.cmd_vy  + self._VY_STEP,  *self._VY_RANGE))
            elif keycode == self._KEY_RIGHT: self.cmd_vy  = float(np.clip(self.cmd_vy  - self._VY_STEP,  *self._VY_RANGE))
            elif keycode == self._KEY_PGUP:  self.cmd_yaw = float(np.clip(self.cmd_yaw + self._YAW_STEP, *self._YAW_RANGE))
            elif keycode == self._KEY_PGDN:  self.cmd_yaw = float(np.clip(self.cmd_yaw - self._YAW_STEP, *self._YAW_RANGE))
            elif keycode == self._KEY_DEL:   self.cmd_vx = self.cmd_vy = self.cmd_yaw = 0.0
            elif keycode == self._KEY_END:   self._reset_requested = True
            else: changed = False
            vx, vy, yaw = self.cmd_vx, self.cmd_vy, self.cmd_yaw
        if changed:
            print(f"\r[cmd] vx={vx:+.1f}  vy={vy:+.1f}  yaw={yaw:+.1f}    ", end='', flush=True)

    def _apply_deadzone(self, value, zone):
        """Zero out small stick drift; rescale remaining range to [0, 1]."""
        if abs(value) < zone:
            return 0.0
        # rescale so edge of deadzone maps to 0, not zone
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - zone) / (1.0 - zone)

    def update_joystick(self):
        """Poll joystick axes each physics step (no-op if no joystick)."""
        if self._joystick is None:
            return
        pygame.event.pump()
        vx  = -self._joystick.get_axis(1)   # left stick Y  (push fwd = negative → negate)
        vy  =  self._joystick.get_axis(0)   # left stick X  (push left = positive, no negate)
        yaw = -self._joystick.get_axis(3)   # right stick X (ax2 is LT trigger, not a stick)
        vx  = self._apply_deadzone(vx,  0.10)
        vy  = self._apply_deadzone(vy,  0.10)
        yaw = self._apply_deadzone(yaw, 0.15)   # yaw axis often drifts more
        reset = any(self._joystick.get_button(i) for i in range(min(self._joystick.get_numbuttons(), 1)))
        with self._lock:
            self.cmd_vx  = float(np.clip(vx  * 0.7, *self._VX_RANGE))
            self.cmd_vy  = float(np.clip(vy  * 0.3, *self._VY_RANGE))
            self.cmd_yaw = float(np.clip(yaw * 1.0, *self._YAW_RANGE))
            if reset:
                self._reset_requested = True

    def get(self):
        with self._lock:
            return self.cmd_vx, self.cmd_vy, self.cmd_yaw

    def pop_reset(self):
        with self._lock:
            r = self._reset_requested
            self._reset_requested = False
        return r


# ---------------------------------------------------------------------------
# MuJoCo scene builder
# ---------------------------------------------------------------------------

def build_mujoco_model(urdf_path, sim_dt, init_height=0.5, floor_friction=1.0, floor_aniso=False,
                       fix_base=False):
    """
    Load robot from URDF via MjSpec, add a floor and actuators, return compiled model.
    Requires MuJoCo >= 3.0.
    """
    urdf_path = os.path.abspath(urdf_path)
    mesh_dir  = os.path.join(os.path.dirname(os.path.dirname(urdf_path)), 'meshes')

    with open(urdf_path) as f:
        urdf_str = f.read()

    spec = mujoco.MjSpec()
    spec.discardvisual = False  # keep visual meshes
    spec.modelfiledir  = os.path.dirname(urdf_path)
    spec.from_string(urdf_str)
    # from_string strips mesh path prefixes — override to the actual meshes directory
    spec.meshdir = mesh_dir

    # Floor
    floor          = spec.worldbody.add_geom()
    floor.name     = "floor"
    floor.type     = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size     = np.array([100.0, 100.0, 0.01])
    floor.rgba     = np.array([0.6, 0.6, 0.6, 1.0])
    if floor_aniso:
        # Anisotropic friction (carpet): high resistance backward (X-), low forward (X+)
        # condim=4 enables separate friction in two sliding directions
        floor.friction = np.array([floor_friction, floor_friction * 0.2, 0.001])
        floor.condim = 4
    else:
        floor.friction = np.array([floor_friction, 0.005, 0.001])

    # Lighting
    light     = spec.worldbody.add_light()
    light.pos = np.array([0.0, 0.0, 4.0])
    light.dir = np.array([0.0, -0.5, -1.0])

    # Place robot at init height and give it a floating base joint
    base     = spec.find_body('base_link')
    base.pos = np.array([0.0, 0.0, init_height])
    if not fix_base:
        fj       = base.add_freejoint()
        fj.name  = "root"

    # Add torque actuators for each revolute joint
    joint_names = [
        'hip_yaw_l', 'hip_roll_l', 'hip_pitch_l', 'knee_pitch_l', 'ankle_pitch_l',
        'hip_yaw_r', 'hip_roll_r', 'hip_pitch_r', 'knee_pitch_r', 'ankle_pitch_r',
    ]
    effort = [20., 60., 20., 20., 20., 20., 60., 20., 20., 20.]
    for jname, eff in zip(joint_names, effort):
        act              = spec.add_actuator()
        act.name         = jname
        act.target       = jname
        act.trntype      = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype      = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype     = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype     = mujoco.mjtBias.mjBIAS_NONE
        gainprm          = np.zeros(10); gainprm[0] = 1.0
        act.gainprm      = gainprm
        act.forcelimited = True
        act.forcerange   = np.array([-eff, eff])

    model = spec.compile()
    # spec.option is not accessible via Python bindings — set after compilation
    model.opt.timestep = sim_dt
    model.opt.gravity  = np.array([0.0, 0.0, -9.81])

    return model


# ---------------------------------------------------------------------------
# Main sim2sim loop
# ---------------------------------------------------------------------------

def run(cfg, cmd_vx=None, cmd_vy=None, cmd_yaw=None, cmd_freq=None, duration=None, headless=False,
        stand_only=False, interactive=False,
        record_path=None, record_skill="walk", record_loop=True, record_skip=20,
        video_path=None, video_fps=30, video_size=(1280, 720)):
    # Override commands if provided
    cmd_vx  = cmd_vx  if cmd_vx  is not None else cfg['cmd_vx']
    cmd_vy  = cmd_vy  if cmd_vy  is not None else cfg.get('cmd_vy', 0.0)
    cmd_yaw = cmd_yaw if cmd_yaw is not None else cfg['cmd_yaw']
    cmd_freq = cmd_freq if cmd_freq is not None else cfg.get('cmd_freq', cfg.get('phase_freq_default', 2.5))
    duration = duration if duration is not None else cfg['simulation_duration']

    sim_dt      = cfg['simulation_dt']
    decimation  = cfg['control_decimation']
    policy_dt   = sim_dt * decimation

    ref_joint   = np.array(cfg['ref_joint_pos'],    dtype=np.float32)
    kps         = np.array(cfg['kps'],               dtype=np.float32)
    kds         = np.array(cfg['kds'],               dtype=np.float32)
    tor_offset  = np.array(cfg['joint_tor_offset'],  dtype=np.float32)
    vel_sign    = np.array(cfg['joint_vel_sign'],    dtype=np.float32)
    # act_low/act_high resolved after model load (absolute mode may need URDF limits)
    _act_low_cfg  = cfg['action_inc_low']
    _act_high_cfg = cfg['action_inc_high']
    jlim_low    = cfg['joint_limit_low']
    jlim_high   = cfg['joint_limit_high']
    num_legs    = cfg['num_legs']
    obs_hist    = cfg['obs_history']
    obs_dim     = cfg['num_obs_per_step']
    static_thr  = cfg['static_cmd_threshold']

    # Mode detection from manifest (explicit config, not dimension inference)
    phase_mode  = cfg.get('phase_mode', 'output')
    action_mode = cfg.get('action_mode', 'increment')
    lp_alpha    = cfg.get('action_lowpass_alpha', 1.0)
    is_mirl     = (phase_mode == 'none')
    is_bdx      = (phase_mode == 'input')
    print(f"[sim2sim] phase.mode={phase_mode}, action_mode={action_mode}, lowpass_alpha={lp_alpha}")
    if is_mirl:
        print("[sim2sim] MIRL mode (10-dim action, 64-dim obs, no phase modulator)")
    elif is_bdx:
        print("[sim2sim] BD_X mode (10-dim action, external phase clock, absolute targets)")

    # Interactive mode: always start from zero (joystick/keyboard is authoritative)
    cin = CommandInput(0.0, 0.0, 0.0) if (interactive and not headless) else None
    _init_vx  = 0.0 if cin else cmd_vx
    _init_vy  = 0.0 if cin else cmd_vy
    _init_yaw = 0.0 if cin else cmd_yaw
    commands = np.array([_init_vx, _init_vy, _init_yaw], dtype=np.float32)
    static_flag = float(np.linalg.norm(commands) >= static_thr)

    # Load ONNX policy (skip if stand_only)
    if stand_only:
        session = None
        input_name = None
        print("Stand-only mode: policy inference disabled, holding ref joint positions.")
    else:
        policy_path = cfg['policy_path']
        print(f"Loading policy: {policy_path}")
        session = ort.InferenceSession(policy_path)
        input_name = session.get_inputs()[0].name

    # Build MuJoCo model
    print(f"Building MuJoCo scene from: {cfg['urdf_path']}")
    model = build_mujoco_model(cfg['urdf_path'], sim_dt, cfg['init_height'],
                               floor_friction=cfg.get('floor_friction', 1.0),
                               floor_aniso=cfg.get('floor_aniso', False),
                               fix_base=stand_only)
    data  = mujoco.MjData(model)

    # Print joint order (useful for debugging)
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    print(f"MuJoCo joints: {joint_names}")

    NUM_JOINTS = 10

    # Resolve action scaling — absolute mode falls back to URDF joint limits
    if _act_low_cfg is not None:
        act_low  = np.array(_act_low_cfg, dtype=np.float32)
        act_high = np.array(_act_high_cfg, dtype=np.float32)
    elif action_mode == 'absolute':
        # Read joint limits from MuJoCo model (skip free joint if present)
        jnt_start = 1 if not stand_only else 0  # skip root free joint
        act_low  = np.array([model.jnt_range[jnt_start + i, 0] for i in range(NUM_JOINTS)], dtype=np.float32)
        act_high = np.array([model.jnt_range[jnt_start + i, 1] for i in range(NUM_JOINTS)], dtype=np.float32)
        print(f"[sim2sim] Absolute mode: using URDF joint limits as action range")
    else:
        raise ValueError("action_mode='increment' requires action scaling ranges in manifest")

    # Resolve joint limits for clipping (from manifest or model)
    if jlim_low is not None:
        jlim_low  = np.array(jlim_low, dtype=np.float32)
        jlim_high = np.array(jlim_high, dtype=np.float32)
    else:
        jnt_start = 1 if not stand_only else 0
        jlim_low  = np.array([model.jnt_range[jnt_start + i, 0] for i in range(NUM_JOINTS)], dtype=np.float32)
        jlim_high = np.array([model.jnt_range[jnt_start + i, 1] for i in range(NUM_JOINTS)], dtype=np.float32)

    # qpos/qvel layout depends on whether base is free or fixed
    # free:  qpos = [7 (pos+quat)] + [10 joints],  qvel = [6 (lin+ang)] + [10]
    # fixed: qpos = [10 joints],                   qvel = [10]
    QPOS_START = 0 if stand_only else 7
    QVEL_START = 0 if stand_only else 6

    # Find imu_in_torso body id (training reads quat/angvel from IMU, not base_link)
    imu_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'imu_in_torso')
    print(f"imu_in_torso body id: {imu_body_id}")

    # Initialize state
    mujoco.mj_resetData(model, data)
    data.qpos[QPOS_START:QPOS_START + NUM_JOINTS] = ref_joint
    mujoco.mj_forward(model, data)

    # Policy state
    pm = PhaseModulator(dt=policy_dt, num_legs=num_legs)
    pm.reset()
    ext_clock = None
    freq_mid = None
    freq_scale = None
    if is_bdx:
        freq_default = cfg.get('phase_freq_default', cmd_freq)
        freq_lo = cfg.get('phase_freq_low', freq_default - 0.5)
        freq_hi = cfg.get('phase_freq_high', freq_default + 0.5)
        freq_mid = 0.5 * (freq_lo + freq_hi)
        freq_scale = max(0.5 * (freq_hi - freq_lo), 1e-6)
        ext_clock = ExternalPhaseClock(
            dt=policy_dt, num_legs=num_legs, default_freq=freq_default,
        )
        ext_clock.reset()
        print(f"[sim2sim] cmd_freq={cmd_freq:.3f} Hz  (train range [{freq_lo:.2f}, {freq_hi:.2f}])")
    lp_target = ref_joint.copy()  # lowpass state for absolute mode

    current_joint_act = ref_joint.copy()
    obs_history = deque(maxlen=obs_hist)
    for _ in range(obs_hist):
        obs_history.append(np.zeros(obs_dim, dtype=np.float32))

    def _get_imu_state():
        """Return (quat [w,x,y,z], base_ang_vel body-frame) from MuJoCo state."""
        if imu_body_id >= 0:
            quat         = data.xquat[imu_body_id]      # [w, x, y, z]
            world_angvel = data.cvel[imu_body_id][0:3]  # cvel: [ang(3), lin(3)]
        else:
            quat         = data.qpos[3:7]
            world_angvel = data.qvel[3:6]
        base_ang_vel = quat_rotate_inverse(quat, world_angvel)
        return quat, base_ang_vel

    def get_obs():
        """Build BIRL observation vector.
        44-dim (current):  [cmd×3, roll, pitch, ang_vel×3, jp×10, jv×10, jerr×10, phase×4, freq×2]
        47-dim (teacher):  same + base_lin_vel×3 (privileged — body-frame linear velocity)
        """
        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        quat, base_ang_vel = _get_imu_state()
        euler        = quat_to_euler_xyz(quat)
        base_euler   = euler[:2]
        joint_pos_rel = q - ref_joint
        joint_vel_sc  = dq * 0.1
        joint_pos_err = current_joint_act - q
        pm_phase_val  = np.concatenate([np.sin(pm.phase), np.cos(pm.phase)]) * static_flag
        pm_f_val      = (pm.frequency * 0.3 - 1.0) * static_flag
        obs_parts = [
            commands,           # 3
            base_euler,         # 2: roll, pitch
            base_ang_vel * 0.5, # 3: ang vel
            joint_pos_rel,      # 10
            joint_vel_sc,       # 10
            joint_pos_err,      # 10
            pm_phase_val,       # 4
            pm_f_val,           # 2
        ]
        if obs_dim == 47:
            # Teacher: append body-frame linear velocity (privileged obs)
            world_lin_vel = data.qvel[:3]
            base_lin_vel  = quat_rotate_inverse(quat, world_lin_vel)
            obs_parts.append(base_lin_vel.astype(np.float32))  # 3
        obs = np.concatenate(obs_parts).astype(np.float32)
        obs = np.clip(obs, -3.0, 3.0)
        return obs

    def get_obs_mirl():
        """Build observation vector matching MIRLTask.pure_observation() (64-dim).

        Layout:
          [0-7]   8 command slots: [vx, vy, yaw, 0, 0, 0, 0, 0]
          [8-9]   roll, pitch
          [10-12] angular velocity × 0.5
          [13-22] joint_pos − ref_joint_pos
          [23-32] joint_vel × 0.1
          [33-42] joint_act − joint_pos  (tracking error)
          [43-63] ref slots + phase_progress (zeros — no reference clip in sim2sim yet)
        """
        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        quat, base_ang_vel = _get_imu_state()
        euler         = quat_to_euler_xyz(quat)
        base_euler    = euler[:2]
        joint_pos_rel = q - ref_joint
        joint_vel_sc  = dq * 0.1
        joint_pos_err = current_joint_act - q
        commands_8    = np.array([commands[0], commands[1], commands[2],
                                   0., 0., 0., 0., 0.], dtype=np.float32)
        obs = np.concatenate([
            commands_8,          # 8
            base_euler,          # 2
            base_ang_vel * 0.5,  # 3
            joint_pos_rel,       # 10
            joint_vel_sc,        # 10
            joint_pos_err,       # 10
            np.zeros(21, dtype=np.float32),  # ref slots + phase_progress
        ]).astype(np.float32)
        obs = np.clip(obs, -3.0, 3.0)
        return obs

    def get_obs_bdx():
        """Build BD_X-style observation vector (phase.mode=input, 43-dim).

        Layout matches bdx.yaml obs_slots:
          [0-2]   commands_3: vx, vy, yaw
          [3-4]   base_euler: roll, pitch
          [5-7]   base_ang_vel × 0.5
          [8-17]  joint_pos − ref_joint_pos
          [18-27] joint_vel × 0.1
          [28-37] joint_act − joint_pos (tracking error)
          [38-41] phase_clock: sin/cos of external phase × static_flag
          [42]    phase_freq_cmd: normalized commanded frequency × static_flag
        """
        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        quat, base_ang_vel = _get_imu_state()
        euler         = quat_to_euler_xyz(quat)
        base_euler    = euler[:2]
        joint_pos_rel = q - ref_joint
        joint_vel_sc  = dq * 0.1
        joint_pos_err = current_joint_act - q
        phase_clock   = ext_clock.sin_cos() * static_flag
        freq_cmd_norm = np.array(
            [((cmd_freq - freq_mid) / freq_scale) * static_flag],
            dtype=np.float32,
        )
        obs = np.concatenate([
            commands,            # 3
            base_euler,          # 2
            base_ang_vel * 0.5,  # 3
            joint_pos_rel,       # 10
            joint_vel_sc,        # 10
            joint_pos_err,       # 10
            phase_clock,         # 4
            freq_cmd_norm,       # 1
        ]).astype(np.float32)
        obs = np.clip(obs, -3.0, 3.0)
        return obs

    def compute_torques(target_q, q, dq):
        """
        Matches legged_robot.py line 414:
          kp * kp_rand * error + kd_const - kd_rand * vel + offset - friction
        With rand=1: kp*error + kd - vel + offset - 3.5*sign(vel)*vel_sign
        """
        error = target_q - q
        torques = (kps * error
                   + kds
                   - dq
                   + tor_offset
                   - 3.5 * np.sign(dq) * vel_sign)
        return torques

    step = 0
    total_steps = int(duration / sim_dt)
    log_interval = int(1.0 / sim_dt)  # print every 1 simulated second

    # Reference recording state (populated if record_path is set)
    _rec = {"joint_pos": [], "joint_vel": [], "base_pos": [],
            "base_quat": [], "base_lin_vel": [], "base_ang_vel": []}
    _rec_skip_steps = record_skip  # skip first N policy steps to let robot settle

    print(f"\nRunning sim2sim: cmd_vx={cmd_vx:.2f} m/s, cmd_vy={cmd_vy:.2f} m/s, cmd_yaw={cmd_yaw:.2f} rad/s")
    print(f"Duration: {duration:.1f}s  |  Policy @ {1/policy_dt:.0f}Hz  |  Physics @ {1/sim_dt:.0f}Hz")
    print(f"{'time':>6}  {'x':>7}  {'y':>7}  {'z':>7}  {'roll':>7}  {'pitch':>7}  {'yaw':>7}  {'vx':>7}  {'vy':>7}")

    def _physics_step():
        nonlocal step, static_flag
        # Update commands from interactive input
        if cin is not None:
            cin.update_joystick()
            if cin.pop_reset():
                mujoco.mj_resetData(model, data)
                data.qpos[QPOS_START:QPOS_START + NUM_JOINTS] = ref_joint
                current_joint_act[:] = ref_joint.copy()
                lp_target[:] = ref_joint.copy()
                pm.reset()
                if ext_clock is not None:
                    ext_clock.reset()
                mujoco.mj_forward(model, data)
            vx, vy, yaw = cin.get()
            commands[0] = vx; commands[1] = vy; commands[2] = yaw

        if step % decimation == 0:
            if not stand_only:
                # Select obs function based on phase mode
                if is_bdx:
                    ext_clock.update(cmd_freq)
                    obs_now = get_obs_bdx()
                elif is_mirl:
                    obs_now = get_obs_mirl()
                else:
                    obs_now = get_obs()
                obs_history.append(obs_now)
                obs_stacked = np.concatenate(list(obs_history))[np.newaxis, :]

                net_out = session.run(None, {input_name: obs_stacked})[0][0]
                scaled = scale_transform(net_out, act_low, act_high)

                if phase_mode == 'output':
                    # BIRL: first num_legs outputs drive phase, rest are joints
                    pm.compute(scaled[:num_legs])
                    joint_out = scaled[num_legs:]
                else:
                    # MIRL / BD_X: all outputs are joint targets
                    joint_out = scaled

                if action_mode == 'increment':
                    current_joint_act[:] += joint_out * policy_dt
                else:
                    # Absolute mode with optional lowpass
                    if lp_alpha < 1.0:
                        lp_target[:] = lp_alpha * joint_out + (1.0 - lp_alpha) * lp_target
                    else:
                        lp_target[:] = joint_out
                    current_joint_act[:] = lp_target
                current_joint_act[:] = np.clip(current_joint_act, jlim_low, jlim_high)
            static_flag = float(np.linalg.norm(commands) >= static_thr)

            # Record reference frame at policy rate
            if record_path is not None and not stand_only:
                nonlocal _rec_skip_steps
                if _rec_skip_steps > 0:
                    _rec_skip_steps -= 1
                else:
                    _rec["joint_pos"].append(data.qpos[QPOS_START:QPOS_START + NUM_JOINTS].copy())
                    _rec["joint_vel"].append(data.qvel[QVEL_START:QVEL_START + NUM_JOINTS].copy())
                    _rec["base_pos"].append(data.qpos[0:3].copy())
                    _rec["base_quat"].append(data.xquat[imu_body_id].copy())       # [w,x,y,z]
                    _rec["base_lin_vel"].append(data.qvel[0:3].copy())              # world frame
                    _rec["base_ang_vel"].append(data.cvel[imu_body_id][0:3].copy()) # body frame ang vel

        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        torques = compute_torques(current_joint_act, q, dq)
        data.ctrl[:NUM_JOINTS] = torques
        mujoco.mj_step(model, data)
        step += 1

        if step % log_interval == 0:
            t = step * sim_dt
            if stand_only:
                print(f"{t:6.1f}  (fixed base — standing pose)")
            else:
                x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
                quat = data.xquat[imu_body_id]  # [w,x,y,z]
                euler = quat_to_euler_xyz(quat)
                vx_act = data.qvel[0]
                vy_act = data.qvel[1]
                cmd_str = f"cmd=[{commands[0]:+.1f},{commands[1]:+.1f},{commands[2]:+.1f}]" if cin else ""
                print(f"\n{t:6.1f}  {x:7.3f}  {y:7.3f}  {z:7.3f}  "
                      f"{np.degrees(euler[0]):7.2f}  {np.degrees(euler[1]):7.2f}  "
                      f"{np.degrees(euler[2]):7.2f}  "
                      f"{vx_act:7.3f}  {vy_act:7.3f}  {cmd_str}")

    if headless:
        # Optional offscreen video recording (works over SSH with MUJOCO_GL=egl).
        # Renders the tracking camera at `video_fps` by sampling every N sim steps.
        vwriter = None
        renderer = None
        cam = None
        frame_stride = None
        if video_path is not None:
            import cv2
            vw, vh = int(video_size[0]), int(video_size[1])
            # MuJoCo's default offscreen framebuffer is 640x480 — bump it so the
            # renderer can produce HD frames.
            model.vis.global_.offwidth  = vw
            model.vis.global_.offheight = vh
            renderer = mujoco.Renderer(model, height=vh, width=vw)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -20
            # cam.lookat is updated each frame to track the base.
            frame_stride = max(1, int(round(1.0 / (video_fps * sim_dt))))
            os.makedirs(os.path.dirname(os.path.abspath(video_path)) or '.', exist_ok=True)
            vwriter = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (vw, vh))
            print(f"[sim2sim] Recording video → {video_path}  ({vw}x{vh} @ {video_fps}fps, stride={frame_stride})")

        while step < total_steps:
            _physics_step()
            if vwriter is not None and step % frame_stride == 0:
                cam.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                vwriter.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if vwriter is not None:
            vwriter.release()
            renderer.close()
            print(f"[sim2sim] Video saved: {video_path}")
        print(f"\nSim2Sim finished: {step * sim_dt:.1f}s simulated.")
    else:
        key_cb = cin.key_callback if cin is not None else None
        with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
            viewer.cam.azimuth   = 90
            viewer.cam.elevation = -20
            viewer.cam.distance  = 3.0
            wall_start = time.time()
            while viewer.is_running() and step < total_steps:
                _physics_step()
                viewer.sync()
                elapsed = time.time() - wall_start
                target  = step * sim_dt
                if target > elapsed:
                    time.sleep(target - elapsed)
        print(f"Sim2Sim finished: {step * sim_dt:.1f}s simulated.")

    # Save reference clip if requested
    if record_path is not None and len(_rec["joint_pos"]) > 0:
        _save_reference_clip(_rec, record_path, policy_dt, record_skill, record_loop)


def _save_reference_clip(rec, path, dt, skill, loop):
    """Save recorded frames as a reference clip .npz (MIRL format)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {k: np.array(v, dtype=np.float32) for k, v in rec.items()}
    T = len(arrays["joint_pos"])
    np.savez(
        path,
        joint_pos    = arrays["joint_pos"],    # [T, 10]
        joint_vel    = arrays["joint_vel"],    # [T, 10]
        base_pos     = arrays["base_pos"],     # [T, 3]
        base_quat    = arrays["base_quat"],    # [T, 4]  [w,x,y,z]
        base_lin_vel = arrays["base_lin_vel"], # [T, 3]
        base_ang_vel = arrays["base_ang_vel"], # [T, 3]
        dt           = np.float32(dt),
        source       = np.bytes_("rollout"),
        skill        = np.bytes_(skill),
        loop         = np.bool_(loop),
    )
    duration = T * dt
    print(f"\nReference clip saved: {path}")
    print(f"  {T} frames  |  {duration:.1f}s  |  {1/dt:.0f}Hz  |  skill={skill}  loop={loop}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Sim2Sim validation in MuJoCo')
    parser.add_argument('--config',   type=str,   default=None,
                        help='Sim2sim config YAML (optional — auto-reads manifest from --policy if omitted)')
    parser.add_argument('--cmd_vx',   type=float, default=None, help='Forward velocity (m/s)')
    parser.add_argument('--cmd_vy',   type=float, default=None, help='Lateral velocity (m/s)')
    parser.add_argument('--cmd_yaw',  type=float, default=None, help='Yaw rate (rad/s)')
    parser.add_argument('--cmd_freq', type=float, default=None,
                        help='Phase frequency command in Hz (BD_X / phase.mode=input only). '
                             'Defaults to manifest phase.freq_default.')
    parser.add_argument('--duration', type=float, default=None, help='Duration (s)')
    parser.add_argument('--headless',    action='store_true', help='Run without viewer, print state to stdout')
    parser.add_argument('--stand_only', action='store_true', help='No policy — hold ref joint positions (standing pose only)')
    parser.add_argument('--policy',     type=str,   default=None, help='Path to .onnx policy (auto-discovers manifest)')
    parser.add_argument('--floor_friction', type=float, default=None, help='Floor sliding friction (default 1.0, carpet ~3.0)')
    parser.add_argument('--floor_aniso', action='store_true', help='Anisotropic floor friction (carpet-like)')
    parser.add_argument('--interactive', action='store_true', help='Live keyboard/joystick command input')
    parser.add_argument('--record',       type=str,   default=None,   help='Save reference clip to this .npz path')
    parser.add_argument('--record_skill', type=str,   default='walk', help='Skill label written into the clip (default: walk)')
    parser.add_argument('--record_loop',  action='store_true',        help='Mark clip as cyclic/looping (default: False)')
    parser.add_argument('--record_skip',  type=int,   default=20,     help='Skip first N policy steps before recording (default: 20 ≈ 0.3s)')
    parser.add_argument('--video',        type=str,   default=None,   help='Offscreen-render video to this .mp4 path (forces --headless; SSH-friendly)')
    parser.add_argument('--video_fps',    type=int,   default=30,     help='Output video framerate (default: 30)')
    parser.add_argument('--video_size',   type=str,   default='1280x720', help='Video resolution WxH (default: 1280x720)')
    args = parser.parse_args()

    # Offscreen rendering needs the EGL backend when there is no display (SSH).
    if args.video is not None:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        args.headless = True
    try:
        vw, vh = map(int, args.video_size.lower().split('x'))
    except Exception:
        parser.error(f"--video_size must be WxH, got {args.video_size!r}")

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
            print(f'[sim2sim] Using manifest: {manifest_path}')
            cfg = manifest_to_sim2sim_cfg(manifest, args.policy)
        else:
            parser.error(f'No manifest found next to {args.policy}. '
                         f'Either export with export_pt2onnx.py first, or pass --config explicitly.')
    else:
        parser.error('Provide --policy (auto-discovers manifest) or --config (legacy sim2sim YAML).')

    if args.floor_friction is not None:
        cfg['floor_friction'] = args.floor_friction
    if args.floor_aniso:
        cfg['floor_aniso'] = True
    run(cfg, cmd_vx=args.cmd_vx, cmd_vy=args.cmd_vy, cmd_yaw=args.cmd_yaw,
        cmd_freq=args.cmd_freq,
        duration=args.duration, headless=args.headless, stand_only=args.stand_only,
        interactive=args.interactive,
        record_path=args.record, record_skill=args.record_skill,
        record_loop=args.record_loop, record_skip=args.record_skip,
        video_path=args.video, video_fps=args.video_fps, video_size=(vw, vh))


if __name__ == '__main__':
    main()
