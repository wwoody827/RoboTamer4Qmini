"""
Build self-describing deployment manifests from training config dicts.

Used by both export_pt2onnx.py (offline export) and train.py (inline sim2sim eval).
"""

import yaml


def build_manifest(params):
    """Build a manifest dict from a training config dict (flat keys).

    Contains everything needed for sim2sim / SDK deployment — no separate
    config files required.
    """
    policy_cfg = params.get('policy', {})
    action_cfg = params.get('action', {})
    task_cfg = params.get('task', {})
    pd_cfg = params.get('pd_gains', {})
    sim_cfg = params.get('sim', {})
    asset_cfg = params.get('asset', {})

    obs_per_step = policy_cfg.get('num_observations', 0) // 3  # 3 history steps
    is_mirl = task_cfg.get('cfg', 'BIRL').startswith('MIRL')

    manifest = {
        'format_version': 2,
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
            'low': _to_list(action_cfg.get('action_limit_low')),
            'high': _to_list(action_cfg.get('action_limit_up')),
        },
        'phase_modulator': {
            'enabled': not is_mirl,
            'num_legs': 2,
            'static_cmd_threshold': 0.15,
        },
        'use_teacher_obs': task_cfg.get('use_teacher_obs', False),
        'hidden_layers': list(policy_cfg.get('hidden_layers', [512, 256])),
        'activation': policy_cfg.get('activation', 'relu'),
        'simulation_dt': sim_cfg.get('dt', 0.001),
        'urdf_path': asset_cfg.get('file', 'assets/q1/urdf/q1.urdf'),
        'init_height': 0.5,
    }
    return manifest


def save_manifest(manifest, path):
    """Write manifest dict to YAML."""
    with open(path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def _to_list(val):
    """Convert numpy arrays to plain lists for YAML serialization."""
    if val is None:
        return None
    if hasattr(val, 'tolist'):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    return val


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
