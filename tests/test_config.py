"""Tests for config loader — inheritance, merging, overrides, validation."""
import os
import tempfile
import pytest
import yaml

# Ensure project root is on sys.path so imports work
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.loader import (
    CfgNode, load_config, config_to_dict, save_config,
    _deep_merge, _apply_overrides, _load_yaml_recursive, _validate_config,
)


# ---------------------------------------------------------------------------
# CfgNode basics
# ---------------------------------------------------------------------------

class TestCfgNode:
    def test_dot_access(self):
        cfg = CfgNode({'a': 1, 'b': {'c': 2}})
        assert cfg.a == 1
        assert cfg.b.c == 2

    def test_missing_key_returns_none(self):
        cfg = CfgNode({'a': 1})
        assert cfg.nonexistent is None

    def test_nested_dict_becomes_cfgnode(self):
        cfg = CfgNode({'outer': {'inner': 42}})
        assert isinstance(cfg.outer, CfgNode)
        assert cfg.outer.inner == 42

    def test_to_dict_roundtrip(self):
        original = {'a': 1, 'b': {'c': [1, 2, 3], 'd': 'hello'}}
        cfg = CfgNode(original)
        recovered = cfg.to_dict()
        assert recovered == original

    def test_setattr(self):
        cfg = CfgNode({'a': 1})
        cfg.a = 99
        assert cfg.a == 99
        cfg.new_field = 'hello'
        assert cfg.new_field == 'hello'

    def test_contains(self):
        cfg = CfgNode({'a': 1})
        assert 'a' in cfg
        assert 'b' not in cfg

    def test_keys_and_items(self):
        cfg = CfgNode({'x': 1, 'y': 2})
        assert set(cfg.keys()) == {'x', 'y'}
        assert dict(cfg.items()) == {'x': 1, 'y': 2}

    def test_getitem(self):
        cfg = CfgNode({'a': 1, 'b': {'c': 2}})
        assert cfg['a'] == 1
        assert cfg['b']['c'] == 2

    def test_getitem_missing_returns_none(self):
        cfg = CfgNode({'a': 1})
        assert cfg['nonexistent'] is None

    def test_setitem(self):
        cfg = CfgNode({'a': 1})
        cfg['a'] = 99
        assert cfg['a'] == 99
        assert cfg.a == 99

    def test_iterate_and_bracket_access(self):
        """Mimics legged_robot.py PD gain loop: iterate keys, access by bracket."""
        stiffness = CfgNode({'hip_yaw': 55, 'hip_roll': 105, 'knee': 150})
        names = list(stiffness)
        assert set(names) == {'hip_yaw', 'hip_roll', 'knee'}
        for name in stiffness:
            assert stiffness[name] == stiffness.__dict__[name]

    def test_empty_node(self):
        cfg = CfgNode()
        assert cfg.anything is None
        assert cfg.to_dict() == {}


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_flat_override(self):
        base = {'a': 1, 'b': 2}
        override = {'b': 99}
        result = _deep_merge(base, override)
        assert result == {'a': 1, 'b': 99}

    def test_nested_merge(self):
        base = {'x': {'a': 1, 'b': 2}, 'y': 3}
        override = {'x': {'b': 99, 'c': 100}}
        result = _deep_merge(base, override)
        assert result == {'x': {'a': 1, 'b': 99, 'c': 100}, 'y': 3}

    def test_override_adds_new_keys(self):
        base = {'a': 1}
        override = {'b': 2}
        result = _deep_merge(base, override)
        assert result == {'a': 1, 'b': 2}

    def test_override_replaces_non_dict_with_dict(self):
        base = {'x': 1}
        override = {'x': {'nested': True}}
        result = _deep_merge(base, override)
        assert result == {'x': {'nested': True}}

    def test_base_not_mutated(self):
        base = {'x': {'a': 1}}
        override = {'x': {'a': 99}}
        _deep_merge(base, override)
        assert base['x']['a'] == 1

    def test_list_replaced_not_merged(self):
        base = {'items': [1, 2, 3]}
        override = {'items': [4, 5]}
        result = _deep_merge(base, override)
        assert result['items'] == [4, 5]


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_flat_override(self):
        d = {'a': 1, 'b': 2}
        result = _apply_overrides(d, {'a': 99})
        assert result['a'] == 99

    def test_dotted_key(self):
        d = {'x': {'y': 1}}
        result = _apply_overrides(d, {'x.y': 42})
        assert result['x']['y'] == 42

    def test_creates_intermediate_dicts(self):
        d = {'a': 1}
        result = _apply_overrides(d, {'b.c.d': 'deep'})
        assert result['b']['c']['d'] == 'deep'

    def test_original_not_mutated(self):
        d = {'a': 1}
        _apply_overrides(d, {'a': 99})
        assert d['a'] == 1


# ---------------------------------------------------------------------------
# YAML loading with _base inheritance
# ---------------------------------------------------------------------------

class TestYAMLInheritance:
    def _write_yaml(self, tmpdir, name, content):
        path = os.path.join(tmpdir, name)
        with open(path, 'w') as f:
            yaml.dump(content, f)
        return path

    def test_simple_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_yaml(tmpdir, 'test.yaml', {'task': {'cfg': 'BIRL'}, 'runner': {'num_envs': 100}})
            d = _load_yaml_recursive(os.path.join(tmpdir, 'test.yaml'))
            assert d['task']['cfg'] == 'BIRL'
            assert d['runner']['num_envs'] == 100

    def test_single_inheritance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_yaml(tmpdir, 'parent.yaml', {
                'task': {'cfg': 'Base'},
                'runner': {'num_envs': 4096, 'max_iterations': 5000},
            })
            self._write_yaml(tmpdir, 'child.yaml', {
                '_base': os.path.join(tmpdir, 'parent.yaml'),
                'task': {'cfg': 'BIRL'},
                'runner': {'max_iterations': 10000},
            })
            d = _load_yaml_recursive(os.path.join(tmpdir, 'child.yaml'))
            assert d['task']['cfg'] == 'BIRL'  # child overrides
            assert d['runner']['num_envs'] == 4096  # inherited from parent
            assert d['runner']['max_iterations'] == 10000  # child overrides

    def test_multi_level_inheritance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_yaml(tmpdir, 'grandparent.yaml', {
                'task': {'cfg': 'Base'},
                'command': {'num_commands': 4, 'resampling_time': 5.},
            })
            self._write_yaml(tmpdir, 'parent.yaml', {
                '_base': os.path.join(tmpdir, 'grandparent.yaml'),
                'task': {'cfg': 'BIRL'},
            })
            self._write_yaml(tmpdir, 'child.yaml', {
                '_base': os.path.join(tmpdir, 'parent.yaml'),
                'command': {'num_commands': 8},
            })
            d = _load_yaml_recursive(os.path.join(tmpdir, 'child.yaml'))
            assert d['task']['cfg'] == 'BIRL'  # from parent
            assert d['command']['num_commands'] == 8  # child overrides
            assert d['command']['resampling_time'] == 5.  # inherited from grandparent

    def test_circular_inheritance_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, 'a.yaml')
            path_b = os.path.join(tmpdir, 'b.yaml')
            self._write_yaml(tmpdir, 'a.yaml', {'_base': path_b, 'x': 1})
            self._write_yaml(tmpdir, 'b.yaml', {'_base': path_a, 'y': 2})
            with pytest.raises(ValueError, match="Circular"):
                _load_yaml_recursive(path_a)

    def test_base_removed_from_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_yaml(tmpdir, 'parent.yaml', {'task': {'cfg': 'Base'}})
            self._write_yaml(tmpdir, 'child.yaml', {
                '_base': os.path.join(tmpdir, 'parent.yaml'),
            })
            d = _load_yaml_recursive(os.path.join(tmpdir, 'child.yaml'))
            assert '_base' not in d


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_config_no_warnings(self):
        d = {'task': {'cfg': 'BIRL'}, 'runner': {'num_envs': 100}}
        warnings = _validate_config(d)
        assert len(warnings) == 0

    def test_missing_task_cfg_warns(self):
        d = {'runner': {'num_envs': 100}}
        warnings = _validate_config(d)
        assert any('task.cfg' in w for w in warnings)

    def test_unknown_section_strict_warns(self):
        d = {'task': {'cfg': 'BIRL'}, 'foobar': {'x': 1}}
        warnings = _validate_config(d, strict=True)
        assert any('foobar' in w for w in warnings)

    def test_unknown_section_non_strict_ok(self):
        d = {'task': {'cfg': 'BIRL'}, 'foobar': {'x': 1}}
        warnings = _validate_config(d, strict=False)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Real config files (integration)
# ---------------------------------------------------------------------------

class TestRealConfigs:
    """Test loading actual YAML configs from configs/ directory."""

    @pytest.fixture
    def configs_dir(self):
        return os.path.join(os.path.dirname(__file__), '..', 'configs')

    def _load(self, configs_dir, name):
        path = os.path.join(configs_dir, name)
        if not os.path.exists(path):
            pytest.skip(f"{name} not found")
        return load_config(path)

    def test_base_loads(self, configs_dir):
        cfg = self._load(configs_dir, 'base.yaml')
        assert cfg.task.cfg == 'BIRL'
        assert cfg.runner.num_envs == 4096
        assert cfg.algorithm.learning_rate == 0.001

    def test_birl_inherits_base(self, configs_dir):
        cfg = self._load(configs_dir, 'birl.yaml')
        assert cfg.task.cfg == 'BIRL'
        # Should inherit runner from base
        assert cfg.runner.num_envs == 4096
        assert cfg.runner.max_iterations == 5000
        # Action overridden
        assert cfg.action.inc_high_ranges[0] == 3.5

    def test_birl_fwd_inherits_birl(self, configs_dir):
        cfg = self._load(configs_dir, 'birl_fwd.yaml')
        assert cfg.task.cfg == 'BIRL'
        # command overridden
        assert cfg.command.lin_vel_y_range == [0., 0.]
        assert cfg.command.use_heading_reward == True
        # action inherited from birl.yaml
        assert cfg.action.inc_high_ranges[0] == 3.5
        # actuator delay enabled
        assert cfg.action.use_actuator_delay == True
        # reward override
        assert cfg.reward.heading == 0.3

    def test_birl_teacher_inherits_birl_fwd(self, configs_dir):
        cfg = self._load(configs_dir, 'birl_teacher.yaml')
        assert cfg.task.cfg == 'BIRL'
        assert cfg.task.use_teacher_obs == True
        # inherited from birl_fwd
        assert cfg.command.lin_vel_y_range == [0., 0.]
        assert cfg.action.use_actuator_delay == True

    def test_mirl_fwd(self, configs_dir):
        cfg = self._load(configs_dir, 'mirl_fwd.yaml')
        assert cfg.task.cfg == 'BIRL'
        assert cfg.phase.mode == 'none'
        assert cfg.command.num_commands == 8
        assert cfg.command.lin_vel_y_range == [0., 0.]
        # 10-dim actions
        assert len(cfg.action.inc_high_ranges) == 10

    def test_mirl_combined(self, configs_dir):
        cfg = self._load(configs_dir, 'mirl_combined.yaml')
        assert cfg.task.cfg == 'BIRL'
        assert cfg.phase.mode == 'none'
        assert cfg.command.lin_vel_x_range == [-0.3, 0.7]
        assert cfg.command.lin_vel_y_range == [-0.3, 0.3]

    def test_no_inner_class_shadowing(self, configs_dir):
        """The main bug this refactoring fixes: child configs inherit ALL parent fields."""
        cfg = self._load(configs_dir, 'birl_fwd.yaml')
        # These should come from base.yaml, NOT be lost due to shadowing
        assert cfg.runner.num_steps_per_env == 24
        assert cfg.runner.save_interval == 200
        assert cfg.runner.episode_length_s == 10
        assert cfg.runner.send_timeouts == True
        assert cfg.runner.use_mirror_augmentation == True
        # These should come from birl_fwd.yaml command override
        assert cfg.command.lin_vel_y_range == [0., 0.]
        # But these should still exist from base
        assert cfg.command.curriculum == False
        assert cfg.command.resampling_time == 5.

    def test_override_at_load_time(self, configs_dir):
        cfg = self._load(configs_dir, 'birl_fwd.yaml')
        cfg2 = load_config(
            os.path.join(configs_dir, 'birl_fwd.yaml'),
            overrides={'runner.num_envs': 512, 'reward.fwd_vel': 5.0}
        )
        assert cfg2.runner.num_envs == 512
        assert cfg2.reward.fwd_vel == 5.0
        # Original not affected
        assert cfg.runner.num_envs == 4096

    def test_bdx_config(self, configs_dir):
        cfg = self._load(configs_dir, 'bdx.yaml')
        assert cfg.task.cfg == 'BIRL'
        assert cfg.task.foot_mask_mode == 'contact'
        assert cfg.phase.mode == 'input'
        assert cfg.phase.base_freq == 1.0
        assert cfg.phase.vel_scale == 1.0
        assert cfg.action.action_mode == 'absolute'
        assert cfg.action.action_lowpass_alpha == 0.75
        # abs_*_ranges null = use URDF limits at runtime
        assert cfg.action.abs_high_ranges is None
        assert cfg.action.abs_low_ranges is None
        assert 'phase_clock' in cfg.observation.slots
        assert 'phase_sin_cos' not in cfg.observation.slots

    def test_all_configs_have_reward_section(self, configs_dir):
        """Every config should inherit the reward section from base."""
        for name in ['birl.yaml', 'birl_fwd.yaml', 'mirl.yaml', 'mirl_fwd.yaml', 'bdx.yaml']:
            cfg = self._load(configs_dir, name)
            assert cfg.reward is not None, f"{name} missing reward section"
            assert cfg.reward.fwd_vel is not None, f"{name} missing reward.fwd_vel"


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_reload(self):
        cfg = CfgNode({
            'task': {'cfg': 'BIRL'},
            'runner': {'num_envs': 100},
            'reward': {'fwd_vel': 2.3},
        })
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w') as f:
            save_config(cfg, f.name)
            loaded = load_config(f.name)
        os.unlink(f.name)
        assert loaded.task.cfg == 'BIRL'
        assert loaded.runner.num_envs == 100
        assert loaded.reward.fwd_vel == 2.3
