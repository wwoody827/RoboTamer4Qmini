import torch
from torch import nn
from torch.distributions import Normal
from typing import Tuple

from .common import Net


class Actor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: Tuple[int, ...] = (128,),
            activation: str = 'tanh',
            init_noise_std: float = 0.8,
            deploy: bool = False
    ) -> None:
        super().__init__()
        self.preprocess = Net(input_dim, 0, hidden_layers, activation)
        self.mu = nn.Linear(hidden_layers[-1], output_dim)
        self.deploy = deploy
        self.output_dim = output_dim
        self.std = nn.Parameter(torch.ones(output_dim) * float(init_noise_std))
        # self.mu.weight.data.normal_(0, 0.01)
        Normal.set_default_validate_args = False  # disable args validation for speedup

    def forward(self, s: torch.Tensor):
        s = s.type(torch.float32)
        net_out = self.preprocess(s)
        # mu = torch.tanh(self.mu(net_out) + s[:, -self.output_dim:])
        mu = torch.tanh(self.mu(net_out))
        # mu = self.mu(net_out)
        if self.deploy:
            return mu
        logits = mu, self.std
        if self.training:
            dist = Normal(*logits)
            return {'logits': logits, 'dist': dist, 'act': dist.sample()}
        else:
            return {'logits': logits, 'act': mu}


class Critic(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_layers: Tuple[int, ...] = (128,),
            activation: str = 'tanh'
    ) -> None:
        super().__init__()
        self.preprocess = Net(input_dim, 0, hidden_layers, activation)
        self.value = nn.Linear(hidden_layers[-1], 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.value(self.preprocess(s))
