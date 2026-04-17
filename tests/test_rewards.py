"""Tests for configurable reward weights."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from config.loader import load_config, config_to_dict


class TestRewardWeightsInConfig:
    """Verify reward weights are present in YAML configs."""

    @pytest.fixture
    def birl_cfg(self):
        return load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl.yaml'))

    @pytest.fixture
    def birl_fwd_cfg(self):
        return load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'birl_fwd.yaml'))

    @pytest.fixture
    def mirl_cfg(self):
        return load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'mirl.yaml'))

    def test_birl_has_reward_section(self, birl_cfg):
        assert birl_cfg.reward is not None

    def test_birl_default_weights(self, birl_cfg):
        w = birl_cfg.reward.to_dict()
        # Spot-check key weights match base.yaml defaults
        assert w['fwd_vel'] == 2.3
        assert w['yaw_rat'] == 2.5
        assert w['balance'] == 1.5
        assert w['twist'] == 2.5
        assert w['act_smo'] == 1.5
        assert w['heading'] == 0.0
        assert w['yaw_smooth'] == 0.0

    def test_birl_fwd_overrides_heading(self, birl_fwd_cfg):
        assert birl_fwd_cfg.reward.heading == 0.3

    def test_mirl_disables_phase_rewards(self, mirl_cfg):
        w = mirl_cfg.reward.to_dict()
        assert w['pmf'] == 0.0
        assert w['foot_phase'] == 0.0

    def test_mirl_enables_power_and_air_time(self, mirl_cfg):
        w = mirl_cfg.reward.to_dict()
        assert w['power'] == 0.1
        assert w['air_time'] == 1.0

    def test_birl_power_disabled_by_default(self, birl_cfg):
        assert birl_cfg.reward.power == 0.0

    def test_all_birl_reward_keys_present(self, birl_cfg):
        """Every key used in birl_task.py rew_dict must exist in config."""
        expected_keys = {
            'constant', 'base_heit', 'balance', 'fwd_vel', 'yaw_rat',
            'lateral_vel', 'vertical_vel', 'ang_vel', 'twist', 'base_acc',
            'foot_clr', 'foot_supt', 'foot_heit', 'leg_width_rew',
            'act_const', 'sa_const', 'foot_phase', 'jnt_pos_err',
            'act_smo', 'net_smo', 'net_out_val', 'foot_slip', 'foot_vz',
            'foot_acc', 'foot_sft', 'jnt_vel', 'feet_py', 'feet_frc',
            'joint_tor', 'pmf', 'heading', 'yaw_smooth',
        }
        w = birl_cfg.reward.to_dict()
        missing = expected_keys - set(w.keys())
        assert not missing, f"Missing reward keys in config: {missing}"

    def test_all_mirl_reward_keys_present(self, mirl_cfg):
        """MIRL config should have all base keys plus power/air_time."""
        expected_keys = {
            'constant', 'base_heit', 'balance', 'fwd_vel', 'yaw_rat',
            'lateral_vel', 'vertical_vel', 'ang_vel', 'twist', 'base_acc',
            'foot_clr', 'foot_supt', 'foot_heit', 'leg_width_rew',
            'act_const', 'sa_const', 'foot_phase', 'jnt_pos_err',
            'act_smo', 'net_smo', 'net_out_val', 'foot_slip', 'foot_vz',
            'foot_acc', 'foot_sft', 'jnt_vel', 'feet_py', 'feet_frc',
            'joint_tor', 'pmf', 'heading', 'yaw_smooth',
            'power', 'air_time',
        }
        w = mirl_cfg.reward.to_dict()
        missing = expected_keys - set(w.keys())
        assert not missing, f"Missing reward keys in MIRL config: {missing}"

    def test_zero_weight_disables_term(self, birl_cfg):
        """A weight of 0 should result in zero contribution (verified at config level)."""
        w = birl_cfg.reward.to_dict()
        disabled = [k for k, v in w.items() if v == 0.0]
        # In base BIRL config, heading and yaw_smooth are disabled
        assert 'heading' in disabled
        assert 'yaw_smooth' in disabled

    def test_no_negative_weights(self, birl_cfg):
        """Weights should be non-negative (penalties are encoded in the raw reward term)."""
        w = birl_cfg.reward.to_dict()
        for k, v in w.items():
            assert v >= 0, f"Negative weight for {k}: {v}"

    def test_reward_weights_are_floats(self, birl_cfg):
        """All reward weights should be numeric."""
        w = birl_cfg.reward.to_dict()
        for k, v in w.items():
            assert isinstance(v, (int, float)), f"Weight {k} is {type(v)}, not numeric"
