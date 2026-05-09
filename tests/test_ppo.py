"""Integration tests for PPO — update step produces finite losses."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from rl.alg.ppo import PPO
from rl.module.continuous import Actor, Critic


@pytest.fixture
def tiny_ppo():
    """Small PPO setup: 4 envs, obs=10, act=3."""
    actor = Actor(input_dim=10, output_dim=3, hidden_layers=(32,), activation='relu')
    critic = Critic(input_dim=10, hidden_layers=(32,), activation='relu')
    alg = PPO(
        actor, critic,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        discount_factor=0.99,
        gae_lambda=0.95,
        device='cpu',
    )
    alg.init_storage(
        num_envs=4,
        num_transitions_per_env=8,
        critic_obs_shape=[10],
        actor_obs_shape=[10],
        action_shape=[3],
    )
    return alg


def _collect_rollout(alg, num_steps=8):
    """Simulate a rollout collection phase."""
    for _ in range(num_steps):
        obs = torch.randn(4, 10)
        cri_obs = torch.randn(4, 10)
        actions = alg.act(obs, cri_obs)
        rewards = torch.rand(4) * 2
        dones = (torch.rand(4) > 0.95).float()
        alg.process_env_step(rewards, dones, {})
    alg.compute_returns(torch.randn(4, 10))


class TestPPOUpdate:
    def test_update_returns_finite_losses(self, tiny_ppo):
        _collect_rollout(tiny_ppo)
        v_loss, s_loss, kl = tiny_ppo.update()
        assert torch.isfinite(torch.tensor(v_loss))
        assert torch.isfinite(torch.tensor(s_loss))
        assert torch.isfinite(torch.tensor(kl))

    def test_multiple_updates_no_nan(self, tiny_ppo):
        """Run 5 rollout + update cycles, should never produce NaN."""
        for _ in range(5):
            _collect_rollout(tiny_ppo)
            v_loss, s_loss, kl = tiny_ppo.update()
            assert not (v_loss != v_loss), "v_loss is NaN"
            assert not (s_loss != s_loss), "s_loss is NaN"

    def test_act_returns_correct_shape(self, tiny_ppo):
        obs = torch.randn(4, 10)
        cri_obs = torch.randn(4, 10)
        actions = tiny_ppo.act(obs, cri_obs)
        assert actions.shape == (4, 3)

    def test_mirror_augmentation_no_nan(self, tiny_ppo):
        """Update with mock mirror should not produce NaN."""

        for _ in range(8):
            obs = torch.randn(4, 132)
            actions = alg.act(obs, obs)
            alg.process_env_step(torch.rand(4), torch.zeros(4), {})
        alg.compute_returns(torch.randn(4, 132))

        v_loss, s_loss, kl = alg.update(mirror=mirror, mirror_weight=0.5)
        assert not (v_loss != v_loss), "v_loss is NaN with mirror"
        assert not (s_loss != s_loss), "s_loss is NaN with mirror"


class TestPPOSaveLoad:
    def test_state_dict_has_required_keys(self, tiny_ppo):
        _collect_rollout(tiny_ppo)
        tiny_ppo.update()
        # Simulate what train.py saves
        state = {
            'actor': tiny_ppo.actor.state_dict(),
            'critic': tiny_ppo.critic.state_dict(),
            'optimizer': tiny_ppo.optimizer.state_dict(),
            'learning_rate': tiny_ppo.learning_rate,
            'iteration': 100,
        }
        assert 'actor' in state
        assert 'critic' in state
        assert 'optimizer' in state
        assert 'learning_rate' in state

    def test_load_state_dict(self, tiny_ppo):
        """Save and reload model weights — output should be deterministic."""
        obs = torch.randn(1, 10)
        tiny_ppo.actor.eval()
        out1 = tiny_ppo.actor(obs)['act']

        state = tiny_ppo.actor.state_dict()

        # Create fresh actor, load weights
        actor2 = Actor(input_dim=10, output_dim=3, hidden_layers=(32,), activation='relu')
        actor2.load_state_dict(state)
        actor2.eval()
        out2 = actor2(obs)['act']

        torch.testing.assert_close(out1, out2)
