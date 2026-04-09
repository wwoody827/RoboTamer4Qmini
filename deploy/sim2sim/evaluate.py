"""
Policy evaluation across multiple conditions.

Runs sim2sim headlessly over a test matrix (friction × cmd_vx × cmd_yaw),
each condition repeated N times, and outputs a CSV summary.

Usage:
    python deploy/sim2sim/evaluate.py [--config deploy/sim2sim/configs/qmini_birl.yaml]
                                      [--runs 10]
                                      [--duration 10]
                                      [--out experiments/my_run/eval.csv]
"""

import os
import sys
import argparse
import itertools
import csv
from collections import deque
from math import tau

import numpy as np
import mujoco
import yaml

# reuse sim2sim helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim import (
    build_mujoco_model, PhaseModulator,
    quat_to_euler_xyz, quat_rotate_inverse, scale_transform,
)

import onnxruntime as ort


def run_episode(cfg, cmd_vx, cmd_yaw, floor_friction, duration, seed=None):
    """
    Run one episode. Returns a dict of scalar metrics.
    """
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
    act_low    = np.array(cfg['action_inc_low'],   dtype=np.float32)
    act_high   = np.array(cfg['action_inc_high'],  dtype=np.float32)
    jlim_low   = np.array(cfg['joint_limit_low'],  dtype=np.float32)
    jlim_high  = np.array(cfg['joint_limit_high'], dtype=np.float32)
    num_legs   = cfg['num_legs']
    obs_hist   = cfg['obs_history']
    obs_dim    = cfg['num_obs_per_step']
    static_thr = cfg['static_cmd_threshold']

    commands    = np.array([cmd_vx, cmd_yaw], dtype=np.float32)
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

    mujoco.mj_resetData(model, data)
    # randomize initial phase slightly for varied episodes
    data.qpos[QPOS_START:QPOS_START + NUM_JOINTS] = ref_joint
    mujoco.mj_forward(model, data)

    pm = PhaseModulator(dt=policy_dt, num_legs=num_legs)
    pm.reset()
    current_joint_act = ref_joint.copy()
    obs_history = deque(maxlen=obs_hist)
    for _ in range(obs_hist):
        obs_history.append(np.zeros(obs_dim, dtype=np.float32))

    def get_obs():
        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        if imu_body_id >= 0:
            quat         = data.xquat[imu_body_id]
            world_angvel = data.cvel[imu_body_id][0:3]
        else:
            quat         = data.qpos[3:7]
            world_angvel = data.qvel[3:6]
        euler        = quat_to_euler_xyz(quat)
        base_euler   = euler[:2]
        base_ang_vel = quat_rotate_inverse(quat, world_angvel)
        pm_phase_val = np.concatenate([np.sin(pm.phase), np.cos(pm.phase)]) * static_flag
        pm_f_val     = (pm.frequency * 0.3 - 1.0) * static_flag
        obs = np.concatenate([
            commands, base_euler, base_ang_vel * 0.5,
            q - ref_joint, dq * 0.1, current_joint_act - q,
            pm_phase_val, pm_f_val,
        ]).astype(np.float32)
        return np.clip(obs, -3.0, 3.0)

    def compute_torques(target_q, q, dq):
        error = target_q - q
        return kps * error + kds - dq + tor_offset - 3.5 * np.sign(dq) * vel_sign

    total_steps  = int(duration / sim_dt)
    fall_thresh  = 0.25   # z below this = fallen

    # accumulators
    vx_errors   = []
    vy_abs      = []
    roll_rms_acc = []
    pitch_rms_acc = []
    torque_acc  = []
    survived    = True
    survive_steps = total_steps

    for step in range(total_steps):
        if step % decimation == 0:
            obs_now = get_obs()
            obs_history.append(obs_now)
            obs_stacked = np.concatenate(list(obs_history))[np.newaxis, :]
            net_out = session.run(None, {input_name: obs_stacked})[0][0]
            scaled  = scale_transform(net_out, act_low, act_high)
            pm.compute(scaled[:num_legs])
            current_joint_act[:] += scaled[num_legs:] * policy_dt
            current_joint_act[:] = np.clip(current_joint_act, jlim_low, jlim_high)
            static_flag = float(np.linalg.norm(commands) >= static_thr)

        q  = data.qpos[QPOS_START:QPOS_START + NUM_JOINTS]
        dq = data.qvel[QVEL_START:QVEL_START + NUM_JOINTS]
        torques = compute_torques(current_joint_act, q, dq)
        data.ctrl[:NUM_JOINTS] = torques
        mujoco.mj_step(model, data)

        z = data.qpos[2]
        if z < fall_thresh:
            survived = False
            survive_steps = step
            break

        # log every policy step
        if step % decimation == 0:
            vx_errors.append(abs(data.qvel[0] - cmd_vx))
            vy_abs.append(abs(data.qvel[1]))
            quat  = data.xquat[imu_body_id] if imu_body_id >= 0 else data.qpos[3:7]
            euler = quat_to_euler_xyz(quat)
            roll_rms_acc.append(euler[0] ** 2)
            pitch_rms_acc.append(euler[1] ** 2)
            torque_acc.append(np.sum(np.abs(torques * dq)))  # power proxy

    # final position
    x_final = data.qpos[0]
    y_final = data.qpos[1]

    # CoT: sum(|τ·dq|) / (mg * |Δx|), only meaningful when moving
    total_mass = 7.0   # kg (approximate)
    g = 9.81
    dx = abs(x_final)
    cot = (np.sum(torque_acc) * policy_dt) / (total_mass * g * dx) if dx > 0.05 else float('nan')

    return {
        'survived':       int(survived),
        'survive_time':   survive_steps * sim_dt,
        'x_final':        x_final,
        'y_final':        y_final,
        'vx_error_mean':  float(np.mean(vx_errors)) if vx_errors else float('nan'),
        'vy_abs_mean':    float(np.mean(vy_abs))    if vy_abs    else float('nan'),
        'roll_rms':       float(np.sqrt(np.mean(roll_rms_acc)))  if roll_rms_acc  else float('nan'),
        'pitch_rms':      float(np.sqrt(np.mean(pitch_rms_acc))) if pitch_rms_acc else float('nan'),
        'cot':            cot,
    }


def evaluate(cfg, runs, duration, frictions, vx_list, yaw_list, out_path):
    conditions = list(itertools.product(frictions, vx_list, yaw_list))
    total = len(conditions) * runs
    print(f"Evaluating {len(conditions)} conditions × {runs} runs = {total} episodes")
    print(f"Policy: {cfg['policy_path']}\n")

    fieldnames = [
        'friction', 'cmd_vx', 'cmd_yaw', 'run',
        'survived', 'survive_time', 'x_final', 'y_final',
        'vx_error_mean', 'vy_abs_mean', 'roll_rms', 'pitch_rms', 'cot',
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        done = 0
        for friction, cmd_vx, cmd_yaw in conditions:
            results = []
            for run_i in range(runs):
                metrics = run_episode(cfg, cmd_vx, cmd_yaw, friction, duration, seed=run_i)
                row = {'friction': friction, 'cmd_vx': cmd_vx, 'cmd_yaw': cmd_yaw,
                       'run': run_i, **metrics}
                writer.writerow(row)
                f.flush()
                results.append(metrics)
                done += 1

            # print per-condition summary
            surv_rate = np.mean([r['survived'] for r in results])
            vx_err    = np.nanmean([r['vx_error_mean'] for r in results])
            vy_drift  = np.nanmean([r['vy_abs_mean'] for r in results])
            roll      = np.nanmean([r['roll_rms'] for r in results])
            print(f"friction={friction:.1f} vx={cmd_vx:+.1f} yaw={cmd_yaw:+.1f} | "
                  f"survival={surv_rate*100:.0f}%  vx_err={vx_err:.3f}  "
                  f"vy_drift={vy_drift:.3f}  roll_rms={np.degrees(roll):.1f}deg  "
                  f"[{done}/{total}]")

    print(f"\nResults saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='deploy/sim2sim/configs/qmini_birl.yaml')
    parser.add_argument('--runs',     type=int,   default=10,   help='Runs per condition')
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds per episode')
    parser.add_argument('--out',      default=None, help='Output CSV path')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.out is None:
        policy_dir = os.path.dirname(cfg['policy_path'])
        args.out = os.path.join(policy_dir, 'eval.csv')

    # Test matrix
    frictions = [0.5, 1.0, 1.5, 3.0]
    vx_list   = [-0.3, 0.0, 0.3, 0.5, 0.7]
    yaw_list  = [0.0]

    evaluate(cfg, args.runs, args.duration, frictions, vx_list, yaw_list, args.out)


if __name__ == '__main__':
    main()
