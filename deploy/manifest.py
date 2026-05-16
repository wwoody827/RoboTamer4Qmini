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

    obs_cfg = params.get('observation', {})
    phase_cfg = params.get('phase', {})
    obs_history = obs_cfg.get('history', 3) if obs_cfg else 3
    obs_skip = obs_cfg.get('skip', 1) if obs_cfg else 1
    obs_per_step = policy_cfg.get('num_observations', 0) // obs_history
    phase_mode = phase_cfg.get('mode', 'output') if phase_cfg else 'output'

    action_mode = action_cfg.get('action_mode', 'increment')

    manifest = {
        'format_version': 3,
        'task_type': task_cfg.get('cfg', 'BIRL'),
        'obs_per_step': obs_per_step,
        'obs_history': obs_history,
        'obs_skip': obs_skip,
        'obs_slots': obs_cfg.get('slots') if obs_cfg else None,
        'obs_total': policy_cfg.get('num_observations', 0),
        'action_dim': policy_cfg.get('num_actions', 0),
        'action_mode': action_mode,
        'action_lowpass_alpha': float(action_cfg.get('action_lowpass_alpha', 1.0)),
        'action_scaling': {
            'inc_low': _to_list(action_cfg.get('inc_low_ranges')),
            'inc_high': _to_list(action_cfg.get('inc_high_ranges')),
            'abs_low': _to_list(action_cfg.get('abs_low_ranges')),
            'abs_high': _to_list(action_cfg.get('abs_high_ranges')),
            'residual_low': _to_list(action_cfg.get('residual_low_ranges')),
            'residual_high': _to_list(action_cfg.get('residual_high_ranges')),
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
            'enabled': phase_mode == 'output',
            'mode': phase_mode,
            'num_legs': 2,
            'static_cmd_threshold': 0.15,
            'freq_low': float(phase_cfg.get('freq_low', 2.0)) if phase_cfg else 2.0,
            'freq_high': float(phase_cfg.get('freq_high', 3.0)) if phase_cfg else 3.0,
            'freq_default': float(phase_cfg.get('freq_default', 2.5)) if phase_cfg else 2.5,
        },
        'use_teacher_obs': task_cfg.get('use_teacher_obs', False),
        'hidden_layers': list(policy_cfg.get('hidden_layers', [512, 256])),
        'activation': policy_cfg.get('activation', 'relu'),
        'simulation_dt': sim_cfg.get('dt', 0.001),
        'urdf_path': asset_cfg.get('file', 'assets/q1/urdf/q1.urdf'),
        'init_height': float(params.get('init_state', {}).get('pos', [0, 0, 0.5])[2]),
        # Obs delay ranges (mirrored by sim2sim — was missing, contributed to
        # the IG↔MuJoCo sim2real gap by feeding policy ~0-step delayed obs at
        # eval vs ~10-50 step delayed obs at training).
        'delay_joint_ranges': list(params.get('domain_rand', {}).get('delay_joint_ranges', [10, 40])),
        'delay_angle_ranges': list(params.get('domain_rand', {}).get('delay_angle_ranges', [20, 50])),
        'delay_rate_ranges':  list(params.get('domain_rand', {}).get('delay_rate_ranges',  [20, 50])),
    }

    # Recovery-task-specific section (consumed by high-level controller in deploy).
    if manifest['task_type'] == 'Recovery':
        manifest['recovery'] = {
            'target_height': float(task_cfg.get('target_height', 0.45)),
            'success_pose_err': float(task_cfg.get('success_pose_err', 0.3)),
            'success_tilt_deg': float(task_cfg.get('success_tilt_deg', 25.0)),
            'episode_length_s': float(params.get('runner', {}).get('episode_length_s', 5.0)),
            # Pool used for training — auto-eval should evaluate on the SAME
            # pool to avoid the false-negative trap (e.g. balance policy tested
            # on fallen poses → 0% success).
            'init_states_path': task_cfg.get('init_states_path', 'data/recovery_init_states.npz'),
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
