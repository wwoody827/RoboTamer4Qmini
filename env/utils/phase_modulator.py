from math import sin, pi, tau
import numpy as np
from isaacgym.torch_utils import to_torch, torch_rand_float
import torch


class PhaseModulator:
    def __init__(self, time_step, num_envs, num_legs, device):
        self.num_legs = num_legs
        self._phase = torch.zeros(num_envs, num_legs, dtype=torch.float, device=device, requires_grad=False)
        self._frequency = torch.ones(num_envs, num_legs, dtype=torch.float, device=device, requires_grad=False) * 0.5
        self._time_step = time_step
        self.device = device
        self.num_envs = num_envs
        self.reset(env_ids=torch.arange(num_envs))

    def reset(self, convert_phi=pi, env_ids=None, render=False):
        if render:
            init_phase = to_torch([0, 0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        else:
            init_phase = torch_rand_float(0, 2 * pi, (len(env_ids), self.num_legs), device=self.device)
        self._phase[env_ids] = init_phase % tau
        self._frequency[env_ids] = torch.ones(len(env_ids), self.num_legs, dtype=torch.float, device=self.device,
                                              requires_grad=False) * 0.5

    def compute(self, frequency):
        self._frequency = frequency
        self._phase = (self._phase + tau * frequency * self._time_step) % tau
        return self._phase

    @property
    def frequency(self):
        return self._frequency

    @property
    def phase(self):
        return self._phase


class ExternalPhaseClock:
    """Phase clock driven by velocity command magnitude (BD_X style).

    Instead of the policy outputting leg frequencies, the phase is derived
    from the commanded velocity:
        freq = base_freq + vel_scale * ||cmd_vel||

    Two legs are anti-phase (offset by pi). The policy receives sin/cos of
    each leg's phase as an input observation, not an action output.
    """

    def __init__(self, dt, num_envs, num_legs, device, base_freq=1.0, vel_scale=1.0):
        self.num_legs = num_legs
        self.num_envs = num_envs
        self.device = device
        self._dt = dt
        self.base_freq = base_freq
        self.vel_scale = vel_scale
        self._phase = torch.zeros(num_envs, num_legs, dtype=torch.float, device=device)
        self._frequency = torch.ones(num_envs, num_legs, dtype=torch.float, device=device) * base_freq
        # Anti-phase offset: leg 0 at 0, leg 1 at pi
        self._leg_offset = torch.zeros(num_envs, num_legs, dtype=torch.float, device=device)
        self._leg_offset[:, 1] = pi

    def reset(self, env_ids, render=False):
        if render:
            self._phase[env_ids] = 0.0
        else:
            self._phase[env_ids] = torch_rand_float(0, tau, (len(env_ids), 1), device=self.device).expand(-1, self.num_legs)
        self._frequency[env_ids] = self.base_freq

    def update(self, cmd_vel_norm):
        """Advance phase based on velocity command magnitude.

        Args:
            cmd_vel_norm: [num_envs, 1] — ||[vx, vy]|| or ||[vx, vy, yaw]||
        """
        freq = self.base_freq + self.vel_scale * cmd_vel_norm  # [num_envs, 1]
        self._frequency = freq.expand(-1, self.num_legs)
        self._phase = (self._phase + tau * freq * self._dt) % tau

    @property
    def phase(self):
        """Raw phase per leg: [num_envs, num_legs]."""
        return self._phase

    @property
    def phase_with_offset(self):
        """Phase with anti-phase offset applied: [num_envs, num_legs]."""
        return (self._phase + self._leg_offset) % tau

    @property
    def frequency(self):
        return self._frequency

    def sin_cos(self):
        """Return [sin(L), sin(R), cos(L), cos(R)] — 4-dim per env."""
        p = self.phase_with_offset
        return torch.cat([torch.sin(p), torch.cos(p)], dim=1)  # [num_envs, 4]
