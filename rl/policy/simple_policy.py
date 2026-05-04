import torch
from typing import Tuple
import rl


class Actor(rl.ContinuousActor):
    def __init__(self, num_observations: int, num_actions: int, hidden_layers: Tuple[int, ...], activation: str, device: str = 'cpu', deploy=False, **kwargs):
        super(Actor, self).__init__(num_observations, num_actions, hidden_layers, activation, deploy=deploy)
        self.device = device


class Critic(rl.ContinuousCritic):
    def __init__(self, num_critic_obs: int, hidden_layers: Tuple[int, ...], activation: str, device: str = 'cpu', **kwargs):
        super(Critic, self).__init__(num_critic_obs, hidden_layers, activation)
        self.device = device


if __name__ == '__main__':
    c = Critic(10, (128,))
    o = c(torch.zeros(0, 10))
    x = torch.ones(10, 1)
    c(o, x=o)
