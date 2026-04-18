"""Tests for export manifest generation and manifest → sim2sim conversion."""
import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from config.loader import load_config, config_to_dict
from deploy.manifest import build_manifest, save_manifest


class TestManifestBuild:
    """Test build_manifest from deploy.manifest module."""

    @pytest.fixture
    def birl_params(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl_fwd.yaml'))
        d = config_to_dict(cfg)
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
        m = build_manifest(birl_params)
        assert m['format_version'] == 3
        assert m['task_type'] == 'BIRL'
        assert m['obs_per_step'] == 44
        assert m['obs_history'] == 3
        assert m['obs_total'] == 132
        assert m['action_dim'] == 12
        assert m['action_mode'] == 'increment'

    def test_birl_action_scaling(self, birl_params):
        m = build_manifest(birl_params)
        assert len(m['action_scaling']['inc_low']) == 12
        assert len(m['action_scaling']['inc_high']) == 12

    def test_birl_phase_modulator_enabled(self, birl_params):
        m = build_manifest(birl_params)
        assert m['phase_modulator']['enabled'] == True

    def test_birl_pd_gains(self, birl_params):
        m = build_manifest(birl_params)
        assert len(m['pd_gains']['kps']) == 10  # 5 per leg
        assert len(m['pd_gains']['kds']) == 10
        assert m['pd_gains']['decimation'] == 15

    def test_birl_ref_joint_pos(self, birl_params):
        m = build_manifest(birl_params)
        assert len(m['ref_joint_pos']) == 10

    def test_mirl_manifest(self, mirl_params):
        m = build_manifest(mirl_params)
        assert m['task_type'] == 'BIRL'
        assert m['obs_per_step'] == 64
        assert m['obs_total'] == 192
        assert m['action_dim'] == 10
        assert m['phase_modulator']['enabled'] == False
        assert m['phase_modulator']['mode'] == 'none'

    def test_mirl_action_scaling_10dim(self, mirl_params):
        m = build_manifest(mirl_params)
        assert len(m['action_scaling']['inc_low']) == 10
        assert len(m['action_scaling']['inc_high']) == 10

    def test_teacher_manifest(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl_teacher.yaml'))
        d = config_to_dict(cfg)
        d['policy']['num_observations'] = 141  # 47 * 3
        d['policy']['num_actions'] = 12
        m = build_manifest(d)
        assert m['use_teacher_obs'] == True
        assert m['obs_per_step'] == 47

    def test_obs_total_equals_per_step_times_history(self, birl_params):
        m = build_manifest(birl_params)
        assert m['obs_total'] == m['obs_per_step'] * m['obs_history']

    def test_sim2sim_fields_present(self, birl_params):
        m = build_manifest(birl_params)
        assert 'simulation_dt' in m
        assert 'urdf_path' in m
        assert 'init_height' in m
        assert m['simulation_dt'] > 0
        assert m['init_height'] > 0

    def test_joint_limits_present_when_set(self):
        """Joint limits are None in raw config (filled by train.py from URDF).
        Verify they propagate when set."""
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl_fwd.yaml'))
        d = config_to_dict(cfg)
        d['policy']['num_observations'] = 132
        d['policy']['num_actions'] = 12
        d['action']['action_limit_low'] = [-0.1] * 10
        d['action']['action_limit_up'] = [0.7] * 10
        m = build_manifest(d)
        assert len(m['joint_limits']['low']) == 10
        assert len(m['joint_limits']['high']) == 10


    def test_bdx_manifest(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'bdx.yaml'))
        d = config_to_dict(cfg)
        d['policy']['num_observations'] = 126  # 42 * 3
        d['policy']['num_actions'] = 10
        m = build_manifest(d)
        assert m['task_type'] == 'BIRL'
        assert m['obs_per_step'] == 42
        assert m['action_dim'] == 10
        assert m['action_mode'] == 'absolute'
        assert m['action_lowpass_alpha'] == 0.75
        assert m['phase_modulator']['enabled'] == False
        assert m['phase_modulator']['mode'] == 'input'
        assert m['phase_modulator']['base_freq'] == 1.0
        assert m['phase_modulator']['vel_scale'] == 1.0


class TestManifestConsistencyWithConfig:
    """Verify manifest fields match the YAML configs they came from."""

    def test_ref_joint_pos_matches_config(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl.yaml'))
        expected = cfg.action.ref_joint_pos
        assert expected == [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]

    def test_increment_mode_from_config(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl.yaml'))
        assert cfg.action.action_mode == 'increment'


class TestSaveManifest:
    """Test manifest save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        import yaml
        manifest = {
            'format_version': 2,
            'task_type': 'BIRL',
            'obs_per_step': 44,
            'ref_joint_pos': [0.1, 0.2, 0.3],
        }
        path = str(tmp_path / 'test_manifest.yaml')
        save_manifest(manifest, path)

        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == manifest


class TestManifestToSim2simCfg:
    """Test manifest → sim2sim cfg conversion."""

    @pytest.fixture
    def birl_manifest(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl_fwd.yaml'))
        d = config_to_dict(cfg)
        d['policy']['num_observations'] = 132
        d['policy']['num_actions'] = 12
        return build_manifest(d)

    @pytest.fixture
    def mirl_manifest(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'mirl_fwd.yaml'))
        d = config_to_dict(cfg)
        d['policy']['num_observations'] = 192
        d['policy']['num_actions'] = 10
        return build_manifest(d)

    def _convert(self, manifest, policy_path='/tmp/fake_policy.onnx'):
        # Import here to avoid sim2sim path issues at module level
        sim2sim_dir = os.path.join(os.path.dirname(__file__), '..', 'deploy', 'sim2sim')
        if sim2sim_dir not in sys.path:
            sys.path.insert(0, sim2sim_dir)
        from sim2sim import manifest_to_sim2sim_cfg
        return manifest_to_sim2sim_cfg(manifest, policy_path)

    def test_birl_cfg_has_required_keys(self, birl_manifest):
        cfg = self._convert(birl_manifest)
        required = [
            'policy_path', 'urdf_path', 'simulation_dt', 'control_decimation',
            'init_height', 'ref_joint_pos', 'kps', 'kds', 'joint_tor_offset',
            'joint_vel_sign', 'action_inc_low', 'action_inc_high',
            'joint_limit_low', 'joint_limit_high', 'num_obs_per_step',
            'obs_history', 'num_legs', 'static_cmd_threshold',
        ]
        for key in required:
            assert key in cfg, f'Missing key: {key}'

    def test_birl_obs_dim(self, birl_manifest):
        cfg = self._convert(birl_manifest)
        assert cfg['num_obs_per_step'] == 44

    def test_mirl_obs_dim(self, mirl_manifest):
        cfg = self._convert(mirl_manifest)
        assert cfg['num_obs_per_step'] == 64

    def test_action_scaling_length_matches_manifest(self, birl_manifest):
        cfg = self._convert(birl_manifest)
        assert len(cfg['action_inc_low']) == len(birl_manifest['action_scaling']['inc_low'])
        assert len(cfg['action_inc_high']) == len(birl_manifest['action_scaling']['inc_high'])

    def test_policy_path_propagated(self, birl_manifest):
        cfg = self._convert(birl_manifest, '/some/path/policy.onnx')
        assert cfg['policy_path'] == '/some/path/policy.onnx'

    def test_decimation_matches(self, birl_manifest):
        cfg = self._convert(birl_manifest)
        assert cfg['control_decimation'] == birl_manifest['pd_gains']['decimation']

    def test_v3_fields_propagated(self, birl_manifest):
        cfg = self._convert(birl_manifest)
        assert cfg['action_mode'] == 'increment'
        assert cfg['action_lowpass_alpha'] == 1.0
        assert cfg['phase_mode'] == 'output'

    def test_bdx_sim2sim_cfg(self):
        bdx_cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'bdx.yaml'))
        d = config_to_dict(bdx_cfg)
        d['policy']['num_observations'] = 126
        d['policy']['num_actions'] = 10
        m = build_manifest(d)
        cfg = self._convert(m)
        assert cfg['action_mode'] == 'absolute'
        assert cfg['action_lowpass_alpha'] == 0.75
        assert cfg['phase_mode'] == 'input'
        assert cfg['phase_base_freq'] == 1.0
        assert cfg['phase_vel_scale'] == 1.0
