# -*- coding: utf-8 -*-
import isaacgym
import numpy as np
import os
from os.path import exists, join
import torch.nn as nn

from model import load_actor
from env.utils import get_args
from config.loader import load_config, CfgNode, config_to_dict
import onnxruntime as ort
import torch
import yaml

args = get_args()
exp_dir = join('experiments', args.name)
model_dir = join(exp_dir, 'model')
deploy_dir = join(exp_dir, 'deploy')
os.makedirs(deploy_dir, exist_ok=True)

cfg = load_config(join(model_dir, 'cfg.yaml'))
params = config_to_dict(cfg)


def convert(name: str, model: nn.Module, input: np.ndarray):
    print(f'\n******************************** {name} ********************************************\n')
    deploy_path = join(deploy_dir, f'{name}.onnx')
    torch.onnx.export(model, torch.from_numpy(input), deploy_path, verbose=False, opset_version=12, input_names=['input'], output_names=['output'])
    print('Pytorch')
    print(model(torch.from_numpy(input)).detach().cpu().numpy())
    ort_session = ort.InferenceSession(deploy_path)
    print('Onnx')
    print(ort_session.run(None, {'input': input})[0])
    gap = model(torch.from_numpy(input)).detach().cpu().numpy() - ort_session.run(None, {'input': input})[0]
    print('Gap')
    print(gap)

    # Save manifest alongside ONNX for self-describing deployment
    manifest = _build_manifest(params, name)
    manifest_path = join(deploy_dir, f'{name}_manifest.yaml')
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    print(f'Manifest saved: {manifest_path}')


def _build_manifest(params, onnx_name):
    """Build a self-describing manifest from training config."""
    policy_cfg = params.get('policy', {})
    action_cfg = params.get('action', {})
    task_cfg = params.get('task', {})
    pd_cfg = params.get('pd_gains', {})

    obs_per_step = policy_cfg.get('num_observations', 0) // 3  # 3 history steps
    is_mirl = task_cfg.get('cfg', 'BIRL').startswith('MIRL')

    manifest = {
        'format_version': 1,
        'task_type': task_cfg.get('cfg', 'BIRL'),
        'obs_per_step': obs_per_step,
        'obs_history': 3,
        'obs_total': policy_cfg.get('num_observations', 0),
        'action_dim': policy_cfg.get('num_actions', 0),
        'action_mode': 'increment' if action_cfg.get('use_increment', True) else 'absolute',
        'action_scaling': {
            'low': action_cfg.get('inc_low_ranges', action_cfg.get('low_ranges')),
            'high': action_cfg.get('inc_high_ranges', action_cfg.get('high_ranges')),
        },
        'ref_joint_pos': action_cfg.get('ref_joint_pos'),
        'pd_gains': {
            'kps': _stiffness_to_list(pd_cfg.get('stiffness', {})),
            'kds': _damping_to_list(pd_cfg.get('damping', {})),
            'decimation': pd_cfg.get('decimation', 15),
        },
        'joint_limits': {
            'low': action_cfg.get('action_limit_low'),
            'high': action_cfg.get('action_limit_up'),
        },
        'phase_modulator': {
            'enabled': not is_mirl,
            'num_legs': 2,
        },
        'use_teacher_obs': task_cfg.get('use_teacher_obs', False),
        'hidden_layers': list(policy_cfg.get('hidden_layers', [512, 256])),
        'activation': policy_cfg.get('activation', 'relu'),
    }
    return manifest


def _stiffness_to_list(stiffness):
    """Convert stiffness dict to ordered list [L hip_yaw..ankle, R hip_yaw..ankle]."""
    if isinstance(stiffness, list):
        return stiffness
    order = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle']
    vals = [stiffness.get(k, 0.) for k in order]
    return vals + vals  # L + R


def _damping_to_list(damping):
    if isinstance(damping, list):
        return damping
    order = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle']
    vals = [damping.get(k, 0.) for k in order]
    return vals + vals


policy_dict = cfg.policy.to_dict() if isinstance(cfg.policy, CfgNode) else params['policy']
policy = load_actor(policy_dict, deploy=True).eval()
if args.iter is not None:
    policy_path = join(model_dir, 'all', f'policy_{args.iter}.pt')
else:
    policy_path = join(model_dir, 'policy.pt')
assert exists(policy_path), policy_path
policy.load_state_dict(torch.load(policy_path, map_location='cpu')['actor'], strict=False)
onnx_name = f'policy_{args.iter}' if args.iter is not None else 'policy'
for i in range(3):
    input = torch.rand([1, cfg.policy.num_observations]).cpu().numpy()
    convert(onnx_name, policy, input)


