"""Tests for the modular observation builder."""
import os
import sys
import importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

# Import obs_builder directly to avoid env/__init__.py → isaacgym chain
_obs_builder_path = os.path.join(os.path.dirname(__file__), '..', 'env', 'obs_builder.py')
spec = importlib.util.spec_from_file_location('obs_builder', _obs_builder_path)
obs_builder_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(obs_builder_mod)
ObsBuilder = obs_builder_mod.ObsBuilder
_SLOT_REGISTRY = obs_builder_mod._SLOT_REGISTRY
obs_slot = obs_builder_mod.obs_slot

from config.loader import load_config, config_to_dict


class MockExtClock:
    """Minimal mock of ExternalPhaseClock for obs builder tests."""
    def __init__(self, num_envs):
        self._sin_cos = torch.zeros(num_envs, 4)

    def sin_cos(self):
        return self._sin_cos


class MockTask:
    """Minimal mock of BIRLTask/MIRLTask for obs builder tests."""

    def __init__(self, num_envs=4, device='cpu'):
        self.num_envs = num_envs
        self.device = device
        self.commands = torch.zeros(num_envs, 8, device=device)
        self.base_euler = torch.zeros(num_envs, 3, device=device)
        self.base_ang_vel = torch.zeros(num_envs, 3, device=device)
        self.base_lin_vel = torch.zeros(num_envs, 3, device=device)
        self.joint_pos = torch.zeros(num_envs, 10, device=device)
        self.joint_vel = torch.zeros(num_envs, 10, device=device)
        self.joint_pos_error = torch.zeros(num_envs, 10, device=device)
        self.ref_joint_action = torch.zeros(num_envs, 10, device=device)
        self.pm_phase = torch.zeros(num_envs, 4, device=device)
        self.pm_f = torch.ones(num_envs, 2, device=device)
        self.static_flag = torch.ones(num_envs, 1, device=device)
        # MIRL ref clip attributes
        self._has_ref = False
        self._ref_joint_pos_now = torch.zeros(num_envs, 10, device=device)
        self._ref_joint_vel_now = torch.zeros(num_envs, 10, device=device)
        self._ref_phase_progress = torch.zeros(num_envs, 1, device=device)
        # BD_X external phase clock
        self._ext_clock = MockExtClock(num_envs)


class TestSlotRegistry:
    def test_standard_slots_registered(self):
        expected = [
            'commands_3', 'commands_8', 'base_euler', 'base_ang_vel',
            'joint_pos_err', 'joint_vel', 'joint_tracking_err',
            'phase_sin_cos', 'phase_freq', 'phase_clock', 'base_lin_vel',
            'ref_joint_pos_err', 'ref_joint_vel', 'ref_phase_progress',
        ]
        for name in expected:
            assert name in _SLOT_REGISTRY, f"Slot '{name}' not registered"

    def test_slot_dims(self):
        expected_dims = {
            'commands_3': 3, 'commands_8': 8,
            'base_euler': 2, 'base_ang_vel': 3,
            'joint_pos_err': 10, 'joint_vel': 10, 'joint_tracking_err': 10,
            'phase_sin_cos': 4, 'phase_freq': 2, 'phase_clock': 4,
            'base_lin_vel': 3,
            'ref_joint_pos_err': 10, 'ref_joint_vel': 10, 'ref_phase_progress': 1,
        }
        for name, expected_dim in expected_dims.items():
            _, dim = _SLOT_REGISTRY[name]
            assert dim == expected_dim, f"Slot '{name}' dim={dim}, expected {expected_dim}"


class TestObsBuilder:
    def test_birl_standard_layout(self):
        task = MockTask()
        slots = [
            'commands_3', 'base_euler', 'base_ang_vel',
            'joint_pos_err', 'joint_vel', 'joint_tracking_err',
            'phase_sin_cos', 'phase_freq',
        ]
        builder = ObsBuilder(task, slot_names=slots)
        assert builder.obs_dim == 44

    def test_birl_teacher_layout(self):
        task = MockTask()
        slots = [
            'commands_3', 'base_euler', 'base_ang_vel',
            'joint_pos_err', 'joint_vel', 'joint_tracking_err',
            'phase_sin_cos', 'phase_freq', 'base_lin_vel',
        ]
        builder = ObsBuilder(task, slot_names=slots)
        assert builder.obs_dim == 47

    def test_mirl_layout(self):
        task = MockTask()
        slots = [
            'commands_8', 'base_euler', 'base_ang_vel',
            'joint_pos_err', 'joint_vel', 'joint_tracking_err',
            'ref_joint_pos_err', 'ref_joint_vel', 'ref_phase_progress',
        ]
        builder = ObsBuilder(task, slot_names=slots)
        assert builder.obs_dim == 64

    def test_build_output_shape(self):
        task = MockTask(num_envs=8)
        slots = ['commands_3', 'base_euler', 'base_ang_vel']
        builder = ObsBuilder(task, slot_names=slots)
        obs = builder.build()
        assert obs.shape == (8, 8)  # 3 + 2 + 3

    def test_build_clips_output(self):
        task = MockTask()
        task.commands[:, 0] = 10.0  # exceeds clip range
        builder = ObsBuilder(task, slot_names=['commands_3'])
        obs = builder.build()
        assert obs.max().item() <= 3.0

    def test_unknown_slot_raises(self):
        task = MockTask()
        with pytest.raises(ValueError, match="Unknown obs slot"):
            ObsBuilder(task, slot_names=['nonexistent_slot'])

    def test_get_layout(self):
        task = MockTask()
        builder = ObsBuilder(task, slot_names=['commands_3', 'base_euler', 'base_ang_vel'])
        layout = builder.get_layout()
        assert len(layout) == 3
        assert layout[0] == {'name': 'commands_3', 'dim': 3, 'offset': 0}
        assert layout[1] == {'name': 'base_euler', 'dim': 2, 'offset': 3}
        assert layout[2] == {'name': 'base_ang_vel', 'dim': 3, 'offset': 5}

    def test_mirl_ref_slots_zero_when_no_clip(self):
        task = MockTask()
        task._has_ref = False
        builder = ObsBuilder(task, slot_names=['ref_joint_pos_err', 'ref_joint_vel', 'ref_phase_progress'])
        obs = builder.build()
        assert (obs == 0).all()

    def test_mirl_ref_slots_nonzero_with_clip(self):
        task = MockTask()
        task._has_ref = True
        task._ref_joint_pos_now = torch.ones(4, 10)
        builder = ObsBuilder(task, slot_names=['ref_joint_pos_err'])
        obs = builder.build()
        assert obs.abs().sum() > 0

    def test_bdx_layout(self):
        task = MockTask()
        slots = [
            'commands_3', 'base_euler', 'base_ang_vel',
            'joint_pos_err', 'joint_vel', 'joint_tracking_err',
            'phase_clock',
        ]
        builder = ObsBuilder(task, slot_names=slots)
        assert builder.obs_dim == 42  # 3+2+3+10+10+10+4

    def test_phase_clock_output_shape(self):
        task = MockTask(num_envs=8)
        builder = ObsBuilder(task, slot_names=['phase_clock'])
        obs = builder.build()
        assert obs.shape == (8, 4)


class TestObsConfigInYaml:
    def test_birl_config_has_obs_slots(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl.yaml'))
        assert cfg.observation is not None
        assert cfg.observation.slots is not None
        assert len(cfg.observation.slots) == 8  # standard BIRL: 8 slots = 44 dim

    def test_birl_teacher_config_has_extra_slot(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl_teacher.yaml'))
        assert 'base_lin_vel' in cfg.observation.slots
        assert len(cfg.observation.slots) == 9

    def test_mirl_config_has_obs_slots(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'mirl.yaml'))
        assert cfg.observation is not None
        slots = cfg.observation.slots
        assert 'commands_8' in slots
        assert 'ref_joint_pos_err' in slots
        assert 'ref_joint_vel' in slots
        assert 'ref_phase_progress' in slots
        # No phase modulator slots
        assert 'phase_sin_cos' not in slots
        assert 'phase_freq' not in slots

    def test_all_config_slots_are_registered(self):
        """Every slot name in every config must exist in the registry."""
        configs_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
        for name in ['birl.yaml', 'birl_fwd.yaml', 'birl_teacher.yaml', 'mirl.yaml', 'mirl_fwd.yaml', 'bdx.yaml']:
            cfg = load_config(os.path.join(configs_dir, name))
            if cfg.observation and cfg.observation.slots:
                for slot in cfg.observation.slots:
                    assert slot in _SLOT_REGISTRY, f"Config {name} references unknown slot '{slot}'"

    def test_birl_obs_dim_matches_expected(self):
        """ObsBuilder dim from config should match expected 44."""
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl.yaml'))
        task = MockTask()
        builder = ObsBuilder(task, slot_names=cfg.observation.slots)
        assert builder.obs_dim == 44

    def test_mirl_obs_dim_matches_expected(self):
        """ObsBuilder dim from config should match expected 64."""
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'mirl.yaml'))
        task = MockTask()
        builder = ObsBuilder(task, slot_names=cfg.observation.slots)
        assert builder.obs_dim == 64

    def test_bdx_obs_dim_matches_expected(self):
        """ObsBuilder dim from bdx.yaml should match expected 42."""
        cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'bdx.yaml'))
        task = MockTask()
        builder = ObsBuilder(task, slot_names=cfg.observation.slots)
        assert builder.obs_dim == 42
