"""Tests for BIRLMirror — L↔R symmetry transforms."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from utils.mirror import BIRLMirror


@pytest.fixture
def mirror():
    return BIRLMirror(obs_history=3, device='cpu')


class TestMirrorRoundTrip:
    def test_obs_double_mirror_is_identity(self, mirror):
        """mirror(mirror(obs)) should give back the original."""
        obs = torch.randn(4, 132)  # 44 * 3
        mirrored = mirror.mirror_obs(obs)
        back = mirror.mirror_obs(mirrored)
        torch.testing.assert_close(back, obs, atol=1e-6, rtol=1e-6)

    def test_action_double_mirror_is_identity(self, mirror):
        """mirror(mirror(action)) should give back the original."""
        act = torch.randn(4, 12)
        mirrored = mirror.mirror_actions(act)
        back = mirror.mirror_actions(mirrored)
        torch.testing.assert_close(back, act, atol=1e-6, rtol=1e-6)


class TestMirrorObsPermutation:
    """Verify specific index swaps and sign flips in obs."""

    def test_cmd_vx_unchanged(self, mirror):
        """cmd_vx (idx 0) should keep sign, not swap."""
        obs = torch.zeros(1, 132)
        obs[0, 0] = 1.0  # cmd_vx in first history step
        m = mirror.mirror_obs(obs)
        assert m[0, 0].item() == pytest.approx(1.0)

    def test_cmd_vy_negated(self, mirror):
        """cmd_vy (idx 1) should be negated."""
        obs = torch.zeros(1, 132)
        obs[0, 1] = 1.0
        m = mirror.mirror_obs(obs)
        assert m[0, 1].item() == pytest.approx(-1.0)

    def test_cmd_yaw_negated(self, mirror):
        """cmd_yaw (idx 2) should be negated."""
        obs = torch.zeros(1, 132)
        obs[0, 2] = 1.0
        m = mirror.mirror_obs(obs)
        assert m[0, 2].item() == pytest.approx(-1.0)

    def test_roll_negated(self, mirror):
        """roll (idx 3) should be negated."""
        obs = torch.zeros(1, 132)
        obs[0, 3] = 0.5
        m = mirror.mirror_obs(obs)
        assert m[0, 3].item() == pytest.approx(-0.5)

    def test_pitch_unchanged(self, mirror):
        """pitch (idx 4) should NOT be negated."""
        obs = torch.zeros(1, 132)
        obs[0, 4] = 0.5
        m = mirror.mirror_obs(obs)
        assert m[0, 4].item() == pytest.approx(0.5)

    def test_left_right_joint_swap(self, mirror):
        """L joints [8:13] swap with R joints [13:18], both negated."""
        obs = torch.zeros(1, 132)
        # Set L hip_yaw = 1.0
        obs[0, 8] = 1.0
        m = mirror.mirror_obs(obs)
        # Should end up at R hip_yaw (idx 13), negated
        assert m[0, 13].item() == pytest.approx(-1.0)
        # Original L position should now be 0 (was 0 at R before mirror)
        assert m[0, 8].item() == pytest.approx(0.0)

    def test_joint_vel_swap_and_negate(self, mirror):
        """Joint vel L[18:23] ↔ R[23:28], negated."""
        obs = torch.zeros(1, 132)
        obs[0, 20] = 2.0  # L hip_pitch vel
        m = mirror.mirror_obs(obs)
        assert m[0, 25].item() == pytest.approx(-2.0)  # R hip_pitch vel

    def test_phase_sin_swap(self, mirror):
        """sin phase [38, 39] swap (no negate)."""
        obs = torch.zeros(1, 132)
        obs[0, 38] = 0.7  # sin_L
        obs[0, 39] = 0.3  # sin_R
        m = mirror.mirror_obs(obs)
        assert m[0, 38].item() == pytest.approx(0.3)
        assert m[0, 39].item() == pytest.approx(0.7)

    def test_freq_swap(self, mirror):
        """freq [42, 43] swap (no negate)."""
        obs = torch.zeros(1, 132)
        obs[0, 42] = 1.5
        obs[0, 43] = 2.5
        m = mirror.mirror_obs(obs)
        assert m[0, 42].item() == pytest.approx(2.5)
        assert m[0, 43].item() == pytest.approx(1.5)

    def test_history_steps_mirrored_independently(self, mirror):
        """Each of the 3 history steps should be mirrored the same way."""
        obs = torch.zeros(1, 132)
        # Set cmd_vy in each history step
        for h in range(3):
            obs[0, h * 44 + 1] = float(h + 1)
        m = mirror.mirror_obs(obs)
        for h in range(3):
            assert m[0, h * 44 + 1].item() == pytest.approx(-float(h + 1))


class TestMirrorActionPermutation:
    def test_freq_swap(self, mirror):
        """freq [0, 1] swap, no negate."""
        act = torch.zeros(1, 12)
        act[0, 0] = 1.5
        act[0, 1] = 2.5
        m = mirror.mirror_actions(act)
        assert m[0, 0].item() == pytest.approx(2.5)
        assert m[0, 1].item() == pytest.approx(1.5)

    def test_joint_swap_and_negate(self, mirror):
        """L joints [2:7] ↔ R joints [7:12], negated."""
        act = torch.zeros(1, 12)
        act[0, 2] = 1.0  # L hip_yaw
        m = mirror.mirror_actions(act)
        assert m[0, 7].item() == pytest.approx(-1.0)  # R hip_yaw
        assert m[0, 2].item() == pytest.approx(0.0)

    def test_all_joints_negate(self, mirror):
        """All 10 joint outputs should be negated after swap."""
        act = torch.zeros(1, 12)
        for i in range(2, 12):
            act[0, i] = 1.0
        m = mirror.mirror_actions(act)
        for i in range(2, 12):
            assert m[0, i].item() == pytest.approx(-1.0)


class TestMirrorOutputShape:
    def test_obs_shape_preserved(self, mirror):
        obs = torch.randn(16, 132)
        m = mirror.mirror_obs(obs)
        assert m.shape == (16, 132)

    def test_action_shape_preserved(self, mirror):
        act = torch.randn(16, 12)
        m = mirror.mirror_actions(act)
        assert m.shape == (16, 12)

    def test_single_env(self, mirror):
        obs = torch.randn(1, 132)
        m = mirror.mirror_obs(obs)
        assert m.shape == (1, 132)


class TestMirrorSymmetricInput:
    """If input is already symmetric, mirror should produce same result."""

    def test_symmetric_obs_ref_pos(self, mirror):
        """An obs with ref_joint_pos already symmetric L↔R should be a fixed point
        (after accounting for sign convention)."""
        obs = torch.zeros(1, 132)
        # ref_joint_pos: L=[0.4, -0.1, -1.5, 1.0, -1.3], R=[-0.4, 0.1, 1.5, -1.0, 1.3]
        # joint_pos - ref is zero when at ref → all joint slots are 0
        # Commands symmetric: vx=0.5, vy=0, yaw=0
        obs[0, 0] = 0.5
        m = mirror.mirror_obs(obs)
        # With zero joints and symmetric commands, obs should equal mirrored obs
        torch.testing.assert_close(m, obs, atol=1e-6, rtol=1e-6)
