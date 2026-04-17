"""Tests for RolloutStorage — add, clear, GAE computation."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from rl.storage import Transition, RolloutStorage


@pytest.fixture
def storage():
    """Small storage: 4 envs, 8 steps, obs=10, act=3."""
    return RolloutStorage(
        num_envs=4,
        num_transitions_per_env=8,
        critic_obs_shape=[10],
        actor_obs_shape=[10],
        actions_shape=[3],
        device='cpu',
    )


class TestTransition:
    def test_clear(self):
        t = Transition()
        t.observations = torch.randn(4, 10)
        t.clear()
        assert t.observations is None


class TestStorageAdd:
    def test_add_transitions(self, storage):
        t = Transition()
        t.observations = torch.randn(4, 10)
        t.critic_obs = torch.randn(4, 10)
        t.actions = torch.randn(4, 3)
        t.rewards = torch.randn(4)
        t.dones = torch.zeros(4)
        t.values = torch.randn(4, 1)
        t.actions_log_prob = torch.randn(4)
        t.action_mean = torch.randn(4, 3)
        t.action_sigma = torch.ones(4, 3)

        storage.add_transitions(t)
        assert storage.step == 1

    def test_overflow_raises(self, storage):
        t = Transition()
        t.observations = torch.randn(4, 10)
        t.critic_obs = torch.randn(4, 10)
        t.actions = torch.randn(4, 3)
        t.rewards = torch.randn(4)
        t.dones = torch.zeros(4)
        t.values = torch.randn(4, 1)
        t.actions_log_prob = torch.randn(4)
        t.action_mean = torch.randn(4, 3)
        t.action_sigma = torch.ones(4, 3)

        for _ in range(8):
            storage.add_transitions(t)
        with pytest.raises(AssertionError):
            storage.add_transitions(t)

    def test_clear_resets_step(self, storage):
        t = Transition()
        t.observations = torch.randn(4, 10)
        t.critic_obs = torch.randn(4, 10)
        t.actions = torch.randn(4, 3)
        t.rewards = torch.randn(4)
        t.dones = torch.zeros(4)
        t.values = torch.randn(4, 1)
        t.actions_log_prob = torch.randn(4)
        t.action_mean = torch.randn(4, 3)
        t.action_sigma = torch.ones(4, 3)

        storage.add_transitions(t)
        storage.clear()
        assert storage.step == 0


def _fill_storage(storage, num_steps=8):
    """Helper: fill storage with random data."""
    for _ in range(num_steps):
        t = Transition()
        t.observations = torch.randn(4, 10)
        t.critic_obs = torch.randn(4, 10)
        t.actions = torch.randn(4, 3)
        t.rewards = torch.rand(4) * 2  # positive rewards
        t.dones = (torch.rand(4) > 0.9).float()
        t.values = torch.randn(4, 1)
        t.actions_log_prob = torch.randn(4)
        t.action_mean = torch.randn(4, 3)
        t.action_sigma = torch.ones(4, 3) * 0.5
        storage.add_transitions(t)


class TestGAE:
    def test_returns_shape(self, storage):
        _fill_storage(storage)
        last_values = torch.randn(4, 1)
        storage.compute_returns(last_values, gamma=0.99, gae_lambda=0.95)
        assert storage.returns.shape == (8, 4, 1)
        assert storage.advantages.shape == (8, 4, 1)

    def test_returns_finite(self, storage):
        _fill_storage(storage)
        last_values = torch.randn(4, 1)
        storage.compute_returns(last_values, gamma=0.99, gae_lambda=0.95)
        assert torch.isfinite(storage.returns).all()
        assert torch.isfinite(storage.advantages).all()

    def test_advantages_normalized(self, storage):
        _fill_storage(storage)
        last_values = torch.randn(4, 1)
        storage.compute_returns(last_values, gamma=0.99, gae_lambda=0.95)
        adv = storage.advantages.flatten()
        # Should be approximately mean=0, std=1
        assert abs(adv.mean().item()) < 0.1
        assert abs(adv.std().item() - 1.0) < 0.2

    def test_gamma_zero_returns_equal_rewards(self, storage):
        """With gamma=0, returns should equal rewards (no bootstrapping)."""
        _fill_storage(storage)
        last_values = torch.zeros(4, 1)
        storage.compute_returns(last_values, gamma=0.0, gae_lambda=0.95)
        # returns = rewards + 0 * next_values - values + values = rewards
        # This is approximate because GAE with gamma=0 gives returns = rewards
        for step in range(8):
            torch.testing.assert_close(
                storage.returns[step],
                storage.rewards[step],
                atol=1e-5, rtol=1e-5,
            )


class TestMiniBatchGenerator:
    def test_yields_correct_number_of_batches(self, storage):
        _fill_storage(storage)
        storage.compute_returns(torch.zeros(4, 1), gamma=0.99, gae_lambda=0.95)
        batches = list(storage.mini_batch_generator(num_mini_batches=4, num_epochs=2))
        assert len(batches) == 4 * 2  # 4 mini-batches × 2 epochs

    def test_batch_shapes(self, storage):
        _fill_storage(storage)
        storage.compute_returns(torch.zeros(4, 1), gamma=0.99, gae_lambda=0.95)
        batch_size = 4 * 8 // 4  # num_envs * steps / num_mini_batches = 8
        for batch in storage.mini_batch_generator(num_mini_batches=4, num_epochs=1):
            obs, cri_obs, actions, values, advantages, returns, old_logp, old_mu, old_sigma = batch
            assert obs.shape == (batch_size, 10)
            assert actions.shape == (batch_size, 3)
            assert values.shape == (batch_size, 1)
            assert old_logp.shape == (batch_size, 1)
            break  # just check first batch
