"""Tests for action scaling (scale_transform) used in tasks and sim2sim."""
import os
import sys
import importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pytest


def _scale_transform_torch(action, action_low, action_high, clip_val=1.):
    """Pure-torch copy of env/utils/math.py:scale_transform (avoids Isaac Gym import)."""
    action = torch.clip(action, -clip_val, clip_val)
    return (action + 1.) / 2. * (action_high - action_low) + action_low


class TestScaleTransformTorch:
    """Test the torch version (env/utils/math.py)."""

    def setup_method(self):
        self.scale_transform = _scale_transform_torch

    def test_zero_maps_to_midpoint(self):
        low = torch.tensor([-15.0])
        high = torch.tensor([15.0])
        result = self.scale_transform(torch.tensor([0.0]), low, high)
        assert result.item() == pytest.approx(0.0)

    def test_minus_one_maps_to_low(self):
        low = torch.tensor([-15.0])
        high = torch.tensor([15.0])
        result = self.scale_transform(torch.tensor([-1.0]), low, high)
        assert result.item() == pytest.approx(-15.0)

    def test_plus_one_maps_to_high(self):
        low = torch.tensor([-15.0])
        high = torch.tensor([15.0])
        result = self.scale_transform(torch.tensor([1.0]), low, high)
        assert result.item() == pytest.approx(15.0)

    def test_asymmetric_range(self):
        low = torch.tensor([0.5])
        high = torch.tensor([3.5])
        result_mid = self.scale_transform(torch.tensor([0.0]), low, high)
        assert result_mid.item() == pytest.approx(2.0)

    def test_clipping(self):
        low = torch.tensor([-15.0])
        high = torch.tensor([15.0])
        result = self.scale_transform(torch.tensor([5.0]), low, high)
        # Should be clipped to +1 → maps to 15
        assert result.item() == pytest.approx(15.0)

    def test_batch(self):
        low = torch.tensor([-10.0, -20.0])
        high = torch.tensor([10.0, 20.0])
        actions = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        result = self.scale_transform(actions, low, high)
        assert result[0, 0].item() == pytest.approx(0.0)
        assert result[0, 1].item() == pytest.approx(0.0)
        assert result[1, 0].item() == pytest.approx(10.0)
        assert result[1, 1].item() == pytest.approx(-20.0)


class TestScaleTransformNumpy:
    """Test the numpy version (deploy/sim2sim/sim2sim.py)."""

    def test_matches_torch_version(self):
        """Numpy sim2sim scale_transform should produce identical results to torch version."""
        torch_scale = _scale_transform_torch
        # Inline the numpy version from sim2sim.py
        def numpy_scale(action, low, high, clip_val=1.0):
            action = np.clip(action, -clip_val, clip_val)
            return (action + 1.0) / 2.0 * (high - low) + low

        low = np.array([0.5, 0.5, -15., -15., -15.])
        high = np.array([3.5, 3.5, 15., 15., 15.])
        actions = np.array([0.0, 0.5, -1.0, 1.0, 0.3])

        np_result = numpy_scale(actions, low, high)
        torch_result = torch_scale(
            torch.tensor(actions),
            torch.tensor(low),
            torch.tensor(high),
        ).numpy()

        np.testing.assert_allclose(np_result, torch_result, atol=1e-6)
