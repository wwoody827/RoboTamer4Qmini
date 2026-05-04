# -*- coding: utf-8 -*-
import isaacgym
import numpy as np
import os
from os.path import exists, join
import torch.nn as nn

from rl.policy import load_actor
from env.utils import get_args
from config.loader import load_config, CfgNode, config_to_dict
from deploy.manifest import build_manifest, save_manifest
import onnxruntime as ort
import torch

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
    manifest = build_manifest(params)
    manifest_path = join(deploy_dir, f'{name}_manifest.yaml')
    save_manifest(manifest, manifest_path)
    print(f'Manifest saved: {manifest_path}')


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


