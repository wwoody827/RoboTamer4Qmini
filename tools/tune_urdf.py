# This file is used to tune PID without RL learning, run it to see how well it will behave
# Parameters 'p' & 'd' can be changed in line 118 & 119
# Data will be saved in foldr 'excel/tune_PID'
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from os.path import join
from collections import deque
from env.legged_robot import get_euler_xyz, to_torch
import torch

warnings.filterwarnings('ignore')


def _from_2pi_to_pi(rpy):
    return rpy.cpu() - 2 * pi * np.floor((rpy.cpu() + pi) / (2 * pi))


def get_euler_from_quat(quat):
    base_rpy = get_euler_xyz(quat)
    r = to_torch(_from_2pi_to_pi(base_rpy[0]))
    p = to_torch(_from_2pi_to_pi(base_rpy[1]))
    y = to_torch(_from_2pi_to_pi(base_rpy[2]))
    return torch.t(torch.vstack((r, p, y)))


# Z-up axis in this file, (x,y,z)

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def exp_filter(history, present, weight):
    """
    exponential filter
    result = history * weight + present * (1. - weight)
    """
    return history * weight + present * (1. - weight)


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.001  # dt*action_repeat=0.01
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # -9.81
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -0)  # -9.81

sim_params.substeps = 1
sim_params.use_gpu_pipeline = False
print("WARNING: Forcing CPU pipeline.")

# physx parameters
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = 10
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.5
sim_params.physx.max_depenetration_velocity = 1.0
sim_params.physx.max_gpu_contact_pairs = 2 ** 23
sim_params.physx.default_buffer_size_multiplier = 5
sim_params.physx.use_gpu = True

asset_options = gymapi.AssetOptions()
asset_options.angular_damping = 0.
asset_options.collapse_fixed_joints = True
asset_options.default_dof_drive_mode = 3
asset_options.density = 0.001
asset_options.max_angular_velocity = 100.
asset_options.replace_cylinder_with_capsule = True
asset_options.thickness = 0.01
asset_options.flip_visual_attachments = False

render = True
asset_options.fix_base_link = True  # todo "on rack  if True"

asset_options.use_mesh_materials = False  # color!!! False have color
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# create viewer
if render:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

# load asset
asset_path = "assets/q1/urdf/q1.urdf"

asset_root = os.path.dirname(asset_path)
asset_file = os.path.basename(asset_path)

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# some parameters from 'rhloc_v1/tune_PID'
action_repeat = 15
dt = action_repeat * sim_params.dt
action, motor_position, motor_torque, motor_velocity = [], [], [], []

pose = gymapi.Transform()
pose.p = gymapi.Vec3(-0., 0., 0.6)  # todo base init position

pose.r = gymapi.Quat(0, -0, -0, 1)  # actor init orientation

# set up the env grid
num_envs = 1
num_per_row = 1
spacing = 3.
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
if render:
    cam_pos = gymapi.Vec3(1, -1., 0.55)
    cam_target = gymapi.Vec3(0, 0, 0.55)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# get array of DOF properties
init_dof_props = gym.get_asset_dof_properties(asset)

init_dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)

joint_pos_lower_limit = init_dof_props['lower']
joint_pos_upper_limit = init_dof_props['upper']
torque_lower_limits = -init_dof_props["effort"]
torque_upper_limits = init_dof_props["effort"]

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
names_dofs = gym.get_asset_dof_names(asset)

kp = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)
kd = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)
kp_real = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)
kd_real = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)

#
stiffness = {'hip_yaw': 50., 'hip_roll': 100., 'hip_pitch': 70., 'knee': 40., 'ankle': 25.}  # [N*m/rad]
damping = {'hip_yaw': 0.3, 'hip_roll': 2.5, 'hip_pitch': 0.3, 'knee': 0.5, 'ankle': 0.25}  # [N*m*s/rad]

plt_id = list(range(0, 10))

joint_tor_offset = torch.tensor([0.6, 1, 0., 0.7, 0.] + [-0.6, -1, -0., -0.7, 0.], dtype=torch.float)
joint_vel_sign = torch.tensor([0., 1, 0., 0., 0.] * 2, dtype=torch.float)

s_nsteps_ago = -30
enable_d_ctrl, a_nsteps_agp = 0., -1

for i in range(num_dofs):
    for gain_name in stiffness:
        if gain_name in names_dofs[i]:
            kp[i] = stiffness[gain_name]
            kd[i] = damping[gain_name]
            kp_real[i] = stiffness[gain_name]
            kd_real[i] = damping[gain_name]
debug_dir = join('experiments', 'tune_urdf')  # sin_test birl10_on_rack
ref_act = torch.tensor([0.4, 0.02, -2.3, -0.1, -1.4,
                             -0.4, -0.02, 2.3, 0.1, 1.4], dtype=torch.float)

os.makedirs(debug_dir, exist_ok=True)
torque_limits = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)
for i in range(num_dofs):
    torque_limits[i] = init_dof_props["effort"][i].item()

env = gym.create_env(sim, env_lower, env_upper, num_per_row)
actor_handle = gym.create_actor(env, asset, pose, "actor", 0, 1)
# reload parameters
gym.set_actor_dof_properties(env, actor_handle, init_dof_props)
dof_props = gym.get_actor_dof_properties(env, actor_handle)
gym.enable_actor_dof_force_sensors(env, actor_handle)


def compute_torques(actions, joint_pos, joint_vel, joint_pos_inc):
    error = actions - joint_pos
    torques = kp * error + kd * (joint_pos_inc * enable_d_ctrl - joint_vel) - 3.5 * torch.sign(joint_vel) * joint_vel_sign + joint_tor_offset
    return torch.clip(torques, torch.tensor(torque_lower_limits), torch.tensor(torque_upper_limits))


reset_act = ref_act.clone()  # torch.tensor([0.] * num_dofs, dtype=torch.float)
init_dof_states = np.array(reset_act, dtype=gymapi.DofState.dtype)
init_dof_vel = np.array([0] * num_dofs, dtype=gymapi.DofState.dtype)

gym.set_actor_dof_states(env, actor_handle, init_dof_states, gymapi.STATE_POS)
# set init dof velocity
gym.set_actor_dof_states(env, actor_handle, init_dof_vel, gymapi.STATE_VEL)

dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(dof_state_tensor)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(actor_root_state)

gym.simulate(sim)
gym.fetch_results(sim, True)
gym.refresh_dof_force_tensor(sim)
gym.refresh_dof_state_tensor(sim)

joint_pos = dof_states.view(num_dofs, 2)[..., 0]
joint_vel = dof_states.view(num_dofs, 2)[..., 1]
base_quat = root_states[:, 3:7]

noisy_joint_pos = joint_pos.clone()
noisy_joint_vel = joint_vel.clone()
joint_pos_his, joint_vel_his, joint_tau_his, joint_act_his = deque(maxlen=150), deque(maxlen=150), deque(maxlen=150), deque(maxlen=150)
torques = compute_torques(torch.tensor(reset_act), joint_pos, joint_vel, joint_pos * 0.)

start = time.time()
act = reset_act

act = reset_act.clone()
act_inc = torch.tensor([0.] * num_dofs, dtype=torch.float)

for _ in range(joint_pos_his.maxlen):
    joint_pos_his.append(joint_pos.clone())
    joint_vel_his.append(joint_vel.clone())
    joint_tau_his.append(torques.clone())
    joint_act_his.append(joint_pos.clone())
joint_pos_noise = (2. * torch.rand_like(joint_pos) - 1.) * 0.1
joint_vel_noise = (2. * torch.rand_like(joint_pos) - 1.) * 1
joint_tau_noise = (2. * torch.rand_like(joint_pos) - 1.) * 1
while True:
    act = ref_act
    joint_act_his.append(act.clone())
    for i in range(action_repeat):
        joint_pos_noise = (2. * torch.rand_like(joint_pos) - 1.) * 0.
        joint_vel_noise = (2. * torch.rand_like(joint_pos) - 1.) * 0
        joint_tau_noise = (2. * torch.rand_like(joint_pos) - 1.) * 0
        selected_actions = joint_act_his[a_nsteps_agp]

        torques = compute_torques(selected_actions,
                                  joint_pos_his[a_nsteps_agp].clone(),
                                  joint_vel_his[a_nsteps_agp].clone(),
                                  act_inc.clone())
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_dof_force_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        mt = gym.get_actor_dof_forces(env, actor_handle)
        joint_pos_filtered = exp_filter(joint_pos_his[-1], joint_pos + joint_pos_noise, 0.)
        joint_vel_filtered = exp_filter(joint_vel_his[-1], joint_vel + joint_vel_noise, 0.)
        joint_tau_filtered = exp_filter(joint_tau_his[-1], torques + joint_tau_noise, 0.)

        joint_pos_his.append(joint_pos_filtered.clone())
        joint_vel_his.append(joint_vel_filtered.clone())
        joint_tau_his.append(joint_tau_filtered.clone())
    base_quat = root_states[:, 3:7]
    base_rpy = get_euler_from_quat(base_quat)
    # print("base_quat", base_quat)
    # print("base_rpy:", base_rpy)

    noisy_joint_pos = joint_pos_his[s_nsteps_ago]
    noisy_joint_vel = joint_vel_his[s_nsteps_ago]
    noisy_joint_tau = joint_tau_his[s_nsteps_ago]
    action.append(act.tolist().copy())
    mp = noisy_joint_pos.tolist().copy()
    motor_position.append(mp)
    mv = noisy_joint_vel.tolist().copy()
    motor_velocity.append(mv)
    motor_torque.append(noisy_joint_tau.tolist().copy())
    if render:
        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

end = time.time()
print(end - start)
print("Done")

# convert them from 1 dimension to 2 dimension
joint_action = np.stack(action.copy())
joint_position = np.stack(motor_position.copy())
joint_velocity = np.stack(motor_velocity.copy())
joint_torque = np.stack(motor_torque.copy())

path = join(debug_dir, f'tune_urdf.xlsx')
with pd.ExcelWriter(path) as f:
    pd.DataFrame(np.hstack([joint_action]), columns=names_dofs).to_excel(f, 'joint_act', index=False)
    pd.DataFrame(np.hstack([joint_position]), columns=names_dofs).to_excel(f, 'joint_pos', index=False)
    pd.DataFrame(np.hstack([joint_velocity]), columns=names_dofs).to_excel(f, 'joint_vel', index=False)
    pd.DataFrame(np.hstack([joint_torque]), columns=names_dofs).to_excel(f, 'joint_tau', index=False)
print(f"Debug data has been saved to {path}.")

if render:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
