"""
Sim2Sim: Validate a trained policy in MuJoCo before real deployment.

Train in Isaac Gym → validate here → deploy to real robot.

Usage:
    cd ~/code/RoboTamer4Qmini
    python deploy/sim2sim/sim2sim.py [--config deploy/sim2sim/configs/qmini_birl.yaml]
                                     [--cmd_vx 0.5] [--cmd_yaw 0.0]
                                     [--duration 30]
"""

import os
import argparse
import time
from collections import deque
from math import tau

import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import yaml

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
# MuJoCo scene builder
# ---------------------------------------------------------------------------

def build_mujoco_model(urdf_path, sim_dt, init_height=0.5, floor_friction=1.0, floor_aniso=False):
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

def run(cfg, cmd_vx=None, cmd_yaw=None, duration=None, headless=False):
    # Override commands if provided
    cmd_vx  = cmd_vx  if cmd_vx  is not None else cfg['cmd_vx']
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

    commands = np.array([cmd_vx, cmd_yaw], dtype=np.float32)
    static_flag = float(np.linalg.norm(commands) >= static_thr)

    # Load ONNX policy
    policy_path = cfg['policy_path']
    print(f"Loading policy: {policy_path}")
    session = ort.InferenceSession(policy_path)
    input_name = session.get_inputs()[0].name

    # Build MuJoCo model
    print(f"Building MuJoCo scene from: {cfg['urdf_path']}")
    model = build_mujoco_model(cfg['urdf_path'], sim_dt, cfg['init_height'],
                               floor_friction=cfg.get('floor_friction', 1.0),
                               floor_aniso=cfg.get('floor_aniso', False))
    data  = mujoco.MjData(model)

    # Print joint order (useful for debugging)
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    print(f"MuJoCo joints: {joint_names}")

    # Find freejoint index (root body)
    # qpos layout: [7 freejoint (pos+quat)] + [10 revolute joints]
    # qvel layout: [6 freejoint (lin+ang vel)] + [10 joint vels]
    NUM_JOINTS = 10
    QPOS_START = 7   # after freejoint pos/quat
    QVEL_START = 6   # after freejoint lin/ang vel

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
            commands,          # 2: vx, yaw
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

    print(f"\nRunning sim2sim: cmd_vx={cmd_vx:.2f} m/s, cmd_yaw={cmd_yaw:.2f} rad/s")
    print(f"Duration: {duration:.1f}s  |  Policy @ {1/policy_dt:.0f}Hz  |  Physics @ {1/sim_dt:.0f}Hz")
    print(f"{'time':>6}  {'x':>7}  {'y':>7}  {'z':>7}  {'roll':>7}  {'pitch':>7}  {'yaw':>7}  {'vx':>7}  {'vy':>7}")

    def _physics_step():
        nonlocal step, static_flag
        if step % decimation == 0:
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
            x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
            # freejoint qpos: [x,y,z, w,x,y,z] — use xquat which is always [w,x,y,z]
            quat = data.xquat[imu_body_id]  # [w,x,y,z]
            euler = quat_to_euler_xyz(quat)
            vx = data.qvel[0]
            vy = data.qvel[1]
            print(f"{t:6.1f}  {x:7.3f}  {y:7.3f}  {z:7.3f}  "
                  f"{np.degrees(euler[0]):7.2f}  {np.degrees(euler[1]):7.2f}  "
                  f"{np.degrees(euler[2]):7.2f}  "
                  f"{vx:7.3f}  {vy:7.3f}")

    if headless:
        while step < total_steps:
            _physics_step()
        print(f"\nSim2Sim finished: {step * sim_dt:.1f}s simulated.")
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
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
    parser.add_argument('--cmd_yaw',  type=float, default=None, help='Yaw rate (rad/s)')
    parser.add_argument('--duration', type=float, default=None, help='Duration (s)')
    parser.add_argument('--headless', action='store_true', help='Run without viewer, print state to stdout')
    parser.add_argument('--floor_friction', type=float, default=None, help='Floor sliding friction (default 1.0, carpet ~3.0)')
    parser.add_argument('--floor_aniso', action='store_true', help='Anisotropic floor friction (carpet-like)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.floor_friction is not None:
        cfg['floor_friction'] = args.floor_friction
    if args.floor_aniso:
        cfg['floor_aniso'] = True
    run(cfg, cmd_vx=args.cmd_vx, cmd_yaw=args.cmd_yaw, duration=args.duration, headless=args.headless)


if __name__ == '__main__':
    main()
