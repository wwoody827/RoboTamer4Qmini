"""Welford running mean/std for value/return normalization in PPO.

Maintains a numerically-stable running estimate of mean and variance over
a stream of batched samples. Used by PPO to normalize returns so the
critic predicts in [-1, 1]-ish range regardless of reward scale, which
makes value-loss gradients independent of return magnitude.
"""

import torch


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4, device='cpu'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(epsilon, device=device)
        self.epsilon = epsilon
        self.device = device

    @torch.no_grad()
    def update(self, x):
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def std(self, eps=1e-8):
        return self.var.sqrt() + eps

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state):
        self.mean = state['mean'].to(self.device)
        self.var = state['var'].to(self.device)
        self.count = state['count'].to(self.device) if torch.is_tensor(state['count']) else \
                     torch.tensor(float(state['count']), device=self.device)
