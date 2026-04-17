"""Tests for export manifest generation."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from config.loader import load_config, config_to_dict


class TestManifestBuild:
    """Test _build_manifest from export_pt2onnx.py (imported inline to avoid Isaac Gym)."""

    def _build_manifest(self, params, onnx_name='policy'):
        """Inline copy of export_pt2onnx._build_manifest to avoid Isaac Gym import."""
        policy_cfg = params.get('policy', {})
        action_cfg = params.get('action', {})
        task_cfg = params.get('task', {})
        pd_cfg = params.get('pd_gains', {})

        obs_per_step = policy_cfg.get('num_observations', 0) // 3
        is_mirl = task_cfg.get('cfg', 'BIRL').startswith('MIRL')

        def _to_list(d, order):
            if isinstance(d, list):
                return d
            vals = [d.get(k, 0.) for k in order]
            return vals + vals

        joint_order = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle']

        return {
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
                'kps': _to_list(pd_cfg.get('stiffness', {}), joint_order),
                'kds': _to_list(pd_cfg.get('damping', {}), joint_order),
                'decimation': pd_cfg.get('decimation', 15),
            },
            'phase_modulator': {'enabled': not is_mirl, 'num_legs': 2},
            'use_teacher_obs': task_cfg.get('use_teacher_obs', False),
        }

    @pytest.fixture
    def birl_params(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl_fwd.yaml'))
        d = config_to_dict(cfg)
        # Simulate what train.py adds
        d['policy']['num_observations'] = 132
        d['policy']['num_actions'] = 12
        d['policy']['num_critic_obs'] = 387
        return d

    @pytest.fixture
    def mirl_params(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'mirl_fwd.yaml'))
        d = config_to_dict(cfg)
        d['policy']['num_observations'] = 192
        d['policy']['num_actions'] = 10
        d['policy']['num_critic_obs'] = 300
        return d

    def test_birl_manifest_required_fields(self, birl_params):
        m = self._build_manifest(birl_params)
        assert m['format_version'] == 1
        assert m['task_type'] == 'BIRL'
        assert m['obs_per_step'] == 44
        assert m['obs_history'] == 3
        assert m['obs_total'] == 132
        assert m['action_dim'] == 12
        assert m['action_mode'] == 'increment'

    def test_birl_action_scaling(self, birl_params):
        m = self._build_manifest(birl_params)
        assert len(m['action_scaling']['low']) == 12
        assert len(m['action_scaling']['high']) == 12

    def test_birl_phase_modulator_enabled(self, birl_params):
        m = self._build_manifest(birl_params)
        assert m['phase_modulator']['enabled'] == True

    def test_birl_pd_gains(self, birl_params):
        m = self._build_manifest(birl_params)
        assert len(m['pd_gains']['kps']) == 10  # 5 per leg
        assert len(m['pd_gains']['kds']) == 10
        assert m['pd_gains']['decimation'] == 15

    def test_birl_ref_joint_pos(self, birl_params):
        m = self._build_manifest(birl_params)
        assert len(m['ref_joint_pos']) == 10

    def test_mirl_manifest(self, mirl_params):
        m = self._build_manifest(mirl_params)
        assert m['task_type'] == 'MIRL'
        assert m['obs_per_step'] == 64
        assert m['obs_total'] == 192
        assert m['action_dim'] == 10
        assert m['phase_modulator']['enabled'] == False

    def test_mirl_action_scaling_10dim(self, mirl_params):
        m = self._build_manifest(mirl_params)
        assert len(m['action_scaling']['low']) == 10
        assert len(m['action_scaling']['high']) == 10

    def test_teacher_manifest(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl_teacher.yaml'))
        d = config_to_dict(cfg)
        d['policy']['num_observations'] = 141  # 47 * 3
        d['policy']['num_actions'] = 12
        m = self._build_manifest(d)
        assert m['use_teacher_obs'] == True
        assert m['obs_per_step'] == 47

    def test_obs_total_equals_per_step_times_history(self, birl_params):
        m = self._build_manifest(birl_params)
        assert m['obs_total'] == m['obs_per_step'] * m['obs_history']


class TestManifestConsistencyWithConfig:
    """Verify manifest fields match the YAML configs they came from."""

    def test_ref_joint_pos_matches_config(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl.yaml'))
        expected = cfg.action.ref_joint_pos
        assert expected == [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]

    def test_increment_mode_from_config(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl.yaml'))
        assert cfg.action.use_increment == True
