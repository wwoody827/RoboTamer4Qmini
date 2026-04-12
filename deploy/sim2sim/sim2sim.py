"""
Sim2Sim: Validate a trained policy in MuJoCo before real deployment.

Train in Isaac Gym → validate here → deploy to real robot.

Usage:
    cd ~/code/RoboTamer4Qmini
    python deploy/sim2sim/sim2sim.py [--config deploy/sim2sim/configs/qmini_birl.yaml]
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

    def update_joystick(self):
        """Poll joystick axes each physics step (no-op if no joystick)."""
        if self._joystick is None:
            return
        pygame.event.pump()
        vx  = -self._joystick.get_axis(1)
        vy  = -self._joystick.get_axis(0)
        yaw = -self._joystick.get_axis(2)
        vx  = 0.0 if abs(vx)  < 0.05 else vx
        vy  = 0.0 if abs(vy)  < 0.05 else vy
        yaw = 0.0 if abs(yaw) < 0.05 else yaw
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

def build_mujoco_model(urdf_path, sim_dt, init_height=0.5, floor_friction=1.0, floor_aniso=False, fix_base=False):
    """
    Load robot from URDF via MjSpec, add a floor and actuators, return compiled model.
    Requires MuJoCo >= 3.0.
    """
    urdf_path = os.path.abspath(urdf_path)
    mesh_dir  = os.path.join(os.path.dirname(os.path.dirname(urdf_path)), 'meshes')

    # base_link has visual but no collision in URDF — patch it in before parsing
    # so MuJoCo loads the mesh for both physics and rendering
    with open(urdf_path) as f:
        urdf_str = f.read()
    base_link_collision = (
        '  <collision>\n'
        '    <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        '    <geometry><mesh filename="../meshes/base_link.STL"/></geometry>\n'
        '  </collision>\n'
        '</link>'
    )
    urdf_str = urdf_str.replace(
        '  </link>\n\n <!-- IMU -->',
        base_link_collision + '\n\n <!-- IMU -->',
        1  # only first occurrence (base_link)
    )

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

    # Disable collision on base_link mesh (visual only — it would hit the floor)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == base_body_id and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_contype[i]    = 0
            model.geom_conaffinity[i] = 0

    return model


# ---------------------------------------------------------------------------
# Main sim2sim loop
# ---------------------------------------------------------------------------

def run(cfg, cmd_vx=None, cmd_vy=None, cmd_yaw=None, duration=None, headless=False, stand_only=False, interactive=False):
    # Override commands if provided
    cmd_vx  = cmd_vx  if cmd_vx  is not None else cfg['cmd_vx']
    cmd_vy  = cmd_vy  if cmd_vy  is not None else cfg.get('cmd_vy', 0.0)
    cmd_yaw = cmd_yaw if cmd_yaw is not None else cfg['cmd_yaw']
    duration = duration if duration is not None else cfg['simulation_duration']

    sim_dt      = cfg['simulation_dt']
    decimation  = cfg['control_decimation']
    policy_dt   = sim_dt * decimation

    ref_joint   = np.array(cfg['ref_joint_pos'],    dtype=np.float32)
    kps         = np.array(cfg['kps'],               dtype=np.float32)
    kds         = np.array(cfg['kds'],               dtype=np.float32)
    tor_offset  = np.array(cfg['joint_tor_offset'],  dtype=np.float32)
    vel_sign    = np.array(cfg['joint_vel_sign'],    dtype=np.float32)
    act_low     = np.array(cfg['action_inc_low'],    dtype=np.float32)
    act_high    = np.array(cfg['action_inc_high'],   dtype=np.float32)
    jlim_low    = np.array(cfg['joint_limit_low'],   dtype=np.float32)
    jlim_high   = np.array(cfg['joint_limit_high'],  dtype=np.float32)
    num_legs    = cfg['num_legs']
    obs_hist    = cfg['obs_history']
    obs_dim     = cfg['num_obs_per_step']
    static_thr  = cfg['static_cmd_threshold']

    commands = np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float32)
    static_flag = float(np.linalg.norm(commands) >= static_thr)

    # Interactive input controller
    cin = CommandInput(cmd_vx, cmd_vy, cmd_yaw) if (interactive and not headless) else None

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

    # qpos/qvel layout depends on whether base is free or fixed
    # free:  qpos = [7 (pos+quat)] + [10 joints],  qvel = [6 (lin+ang)] + [10]
    # fixed: qpos = [10 joints],                   qvel = [10]
    NUM_JOINTS = 10
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

    current_joint_act = ref_joint.copy()
    obs_history = deque(maxlen=obs_hist)
    for _ in range(obs_hist):
        obs_history.append(np.zeros(obs_dim, dtype=np.float32))

    def get_obs():
        """Build observation vector matching BIRLTask.pure_observation()."""
        q     = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq    = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]

        # Training uses imu_in_torso body (fixed joint offset from base_link)
        # xquat is [w, x, y, z] in MuJoCo
        if imu_body_id >= 0:
            quat         = data.xquat[imu_body_id]       # [w, x, y, z]
            world_angvel = data.cvel[imu_body_id][0:3]   # cvel: [ang(3), lin(3)]
        else:
            quat         = data.qpos[3:7]
            world_angvel = data.qvel[3:6]

        euler        = quat_to_euler_xyz(quat)   # [roll, pitch, yaw]
        base_euler   = euler[:2]               # obs only uses roll, pitch
        base_ang_vel = quat_rotate_inverse(quat, world_angvel)

        joint_pos_rel = q - ref_joint                              # relative to ref
        joint_vel_sc  = dq * 0.1
        joint_pos_err = current_joint_act - q                      # action - pos

        pm_phase_val  = np.concatenate([
            np.sin(pm.phase),
            np.cos(pm.phase),
        ]) * static_flag

        pm_f_val = (pm.frequency * 0.3 - 1.0) * static_flag

        obs = np.concatenate([
            commands,          # 3: vx, vy, yaw
            base_euler,        # 2: roll, pitch
            base_ang_vel * 0.5,# 3: ang vel
            joint_pos_rel,     # 10
            joint_vel_sc,      # 10
            joint_pos_err,     # 10
            pm_phase_val,      # 4
            pm_f_val,          # 2
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
                pm.reset()
                mujoco.mj_forward(model, data)
            vx, vy, yaw = cin.get()
            commands[0] = vx; commands[1] = vy; commands[2] = yaw

        if step % decimation == 0:
            if not stand_only:
                obs_now = get_obs()
                obs_history.append(obs_now)
                obs_stacked = np.concatenate(list(obs_history))[np.newaxis, :]

                net_out = session.run(None, {input_name: obs_stacked})[0][0]
                scaled = scale_transform(net_out, act_low, act_high)
                pm.compute(scaled[:num_legs])
                current_joint_act[:] += scaled[num_legs:] * policy_dt
                current_joint_act[:] = np.clip(current_joint_act, jlim_low, jlim_high)
            static_flag = float(np.linalg.norm(commands) >= static_thr)

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
        while step < total_steps:
            _physics_step()
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Sim2Sim validation in MuJoCo')
    parser.add_argument('--config',   type=str,   default='deploy/sim2sim/configs/qmini_birl.yaml')
    parser.add_argument('--cmd_vx',   type=float, default=None, help='Forward velocity (m/s)')
    parser.add_argument('--cmd_vy',   type=float, default=None, help='Lateral velocity (m/s)')
    parser.add_argument('--cmd_yaw',  type=float, default=None, help='Yaw rate (rad/s)')
    parser.add_argument('--duration', type=float, default=None, help='Duration (s)')
    parser.add_argument('--headless',    action='store_true', help='Run without viewer, print state to stdout')
    parser.add_argument('--stand_only', action='store_true', help='No policy — hold ref joint positions (standing pose only)')
    parser.add_argument('--policy',     type=str,   default=None, help='Path to .onnx policy (overrides config)')
    parser.add_argument('--floor_friction', type=float, default=None, help='Floor sliding friction (default 1.0, carpet ~3.0)')
    parser.add_argument('--floor_aniso', action='store_true', help='Anisotropic floor friction (carpet-like)')
    parser.add_argument('--interactive', action='store_true', help='Live keyboard/joystick command input (W/S/A/D/Q/E/Space/R)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.policy is not None:
        cfg['policy_path'] = args.policy
    if args.floor_friction is not None:
        cfg['floor_friction'] = args.floor_friction
    if args.floor_aniso:
        cfg['floor_aniso'] = True
    run(cfg, cmd_vx=args.cmd_vx, cmd_vy=args.cmd_vy, cmd_yaw=args.cmd_yaw,
        duration=args.duration, headless=args.headless, stand_only=args.stand_only,
        interactive=args.interactive)


if __name__ == '__main__':
    main()
