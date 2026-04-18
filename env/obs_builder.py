"""
Modular observation builder — drives obs construction from config.

Each obs "slot" is a named method that returns a [num_envs, dim] tensor.
The config lists which slots to include and in what order.

Usage in a task class:
    self.obs_builder = ObsBuilder(self, slot_names=cfg.observation.slots)
    ...
    def pure_observation(self):
        return self.obs_builder.build()

Adding a new slot:
    1. Add a method decorated with @obs_slot('name', dim=N) to the task class
       or register it via ObsBuilder.register_slot().
    2. Add the slot name to the config YAML observation.slots list.
    3. No other code changes needed.
"""

import torch
from collections import OrderedDict


# Global registry of slot functions: name -> (fn, dim)
# fn signature: fn(task) -> Tensor[num_envs, dim]
_SLOT_REGISTRY = OrderedDict()


def obs_slot(name, dim):
    """Decorator to register an observation slot function.

    The decorated function takes a task instance and returns [num_envs, dim].
    """
    def decorator(fn):
        _SLOT_REGISTRY[name] = (fn, dim)
        return fn
    return decorator


class ObsBuilder:
    """Builds observations by concatenating named slots in config order.

    Args:
        task: The task instance (BIRLTask or MIRLTask) providing sensor state.
        slot_names: List of slot name strings from config.
        clip_range: Tuple (min, max) to clip the final observation.
    """

    def __init__(self, task, slot_names, clip_range=(-3.0, 3.0)):
        self.task = task
        self.clip_min, self.clip_max = clip_range
        self.slots = []  # list of (name, fn, dim)

        for name in slot_names:
            if name in _SLOT_REGISTRY:
                fn, dim = _SLOT_REGISTRY[name]
                self.slots.append((name, fn, dim))
            else:
                raise ValueError(
                    f"Unknown obs slot '{name}'. "
                    f"Available: {list(_SLOT_REGISTRY.keys())}"
                )

        self._obs_dim = sum(d for _, _, d in self.slots)

    @property
    def obs_dim(self):
        return self._obs_dim

    def get_layout(self):
        """Return obs layout as list of dicts for manifest export."""
        layout = []
        offset = 0
        for name, _, dim in self.slots:
            layout.append({'name': name, 'dim': dim, 'offset': offset})
            offset += dim
        return layout

    def build(self):
        """Concatenate all slots into a single obs tensor."""
        parts = [fn(self.task) for _, fn, _ in self.slots]
        obs = torch.cat(parts, dim=1)
        return obs.clip(min=self.clip_min, max=self.clip_max)


# ─── BIRL obs slots (44-dim standard layout) ────────────────────────────────

@obs_slot('commands_3', dim=3)
def _commands_3(task):
    """vx, vy, yaw commands."""
    return task.commands[:, :3]


@obs_slot('commands_8', dim=8)
def _commands_8(task):
    """8-slot commands: [vx, vy, yaw, height, 0, 0, 0, 0]."""
    return task.commands[:, :8]


@obs_slot('base_euler', dim=2)
def _base_euler(task):
    """Roll, pitch from delayed IMU."""
    return task.base_euler[:, :2]


@obs_slot('base_ang_vel', dim=3)
def _base_ang_vel(task):
    """Angular velocity × 0.5 from delayed IMU."""
    return task.base_ang_vel * 0.5


@obs_slot('joint_pos_err', dim=10)
def _joint_pos_err(task):
    """joint_pos - ref_joint_pos (delayed)."""
    return task.joint_pos - task.ref_joint_action


@obs_slot('joint_vel', dim=10)
def _joint_vel(task):
    """Joint velocities × 0.1 (delayed)."""
    return task.joint_vel * 0.1


@obs_slot('joint_tracking_err', dim=10)
def _joint_tracking_err(task):
    """current_joint_act - joint_pos (tracking error)."""
    return task.joint_pos_error


@obs_slot('phase_sin_cos', dim=4)
def _phase_sin_cos(task):
    """sin/cos of leg phases × static_flag (BIRL only)."""
    return task.pm_phase * task.static_flag


@obs_slot('phase_freq', dim=2)
def _phase_freq(task):
    """(freq × 0.3 - 1.0) × static_flag (BIRL only)."""
    return (task.pm_f * 0.3 - 1.0) * task.static_flag


@obs_slot('phase_clock', dim=4)
def _phase_clock(task):
    """sin/cos of external phase clock × static_flag (BD_X style, phase.mode=input)."""
    return task._ext_clock.sin_cos() * task.static_flag


@obs_slot('base_lin_vel', dim=3)
def _base_lin_vel(task):
    """Base linear velocity (privileged — teacher obs only)."""
    return task.base_lin_vel


# ─── MIRL-specific obs slots ────────────────────────────────────────────────

@obs_slot('ref_joint_pos_err', dim=10)
def _ref_joint_pos_err(task):
    """ref_joint_pos[t] - joint_pos (reference clip tracking, zeros if no clip)."""
    if task._has_ref:
        return (task._ref_joint_pos_now - task.joint_pos).clip(-3., 3.)
    return torch.zeros(task.num_envs, 10, dtype=torch.float, device=task.device)


@obs_slot('ref_joint_vel', dim=10)
def _ref_joint_vel(task):
    """ref_joint_vel[t] (reference clip velocity, zeros if no clip)."""
    if task._has_ref:
        return task._ref_joint_vel_now.clip(-3., 3.)
    return torch.zeros(task.num_envs, 10, dtype=torch.float, device=task.device)


@obs_slot('ref_phase_progress', dim=1)
def _ref_phase_progress(task):
    """Phase progress 0→1 in reference clip (zero if no clip)."""
    if task._has_ref:
        return task._ref_phase_progress
    return torch.zeros(task.num_envs, 1, dtype=torch.float, device=task.device)
