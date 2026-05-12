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


# Global registry of slot functions: name -> (fn, dim, mirror_spec)
# fn signature: fn(task) -> Tensor[num_envs, dim]
# mirror_spec: None (identity) | (perm, sign) tuple of length-dim lists
_SLOT_REGISTRY = OrderedDict()


# ─── Mirror spec helpers ─────────────────────────────────────────────────────
# Each returns a (perm, sign) tuple describing how a slot transforms under
# left↔right body reflection (sagittal-plane mirror).

def mirror_signs(signs):
    """Sign flip without permutation. e.g. [1, -1, -1] flips dims 1 and 2."""
    return (list(range(len(signs))), [float(s) for s in signs])


def mirror_swap_negate(dim):
    """L/R swap + negate-all. Layout: [L0..Ln-1, R0..Rn-1] with n = dim/2.

    Used for joint-pos/vel/tracking-error slots (10-dim, 5L+5R).
    """
    half = dim // 2
    perm = list(range(half, dim)) + list(range(0, half))
    sign = [-1.0] * dim
    return (perm, sign)


def mirror_pair_swap(dim):
    """Swap consecutive (L, R) pairs without sign flip. e.g. 4-dim
    [sinL, sinR, cosL, cosR] → [sinR, sinL, cosR, cosL]. Used for phase_clock.
    """
    perm = list(range(dim))
    for i in range(0, dim, 2):
        perm[i], perm[i + 1] = i + 1, i
    return (perm, [1.0] * dim)


def mirror_split_swap(dim):
    """Swap [L_block, R_block] halves without sign flip. e.g. 4-dim
    [sinL, sinR, cosL, cosR] is NOT this layout; this fits [sinL, cosL, sinR, cosR]
    with halves [sinL, cosL] and [sinR, cosR]. Pick the right helper for the slot.
    """
    half = dim // 2
    perm = list(range(half, dim)) + list(range(0, half))
    return (perm, [1.0] * dim)


def obs_slot(name, dim, mirror=None):
    """Decorator to register an observation slot function.

    Args:
        name: slot identifier used in config YAML.
        dim:  number of dims this slot contributes to obs.
        mirror: how this slot transforms under L/R body mirror.
            None (default) → identity (no permutation, no sign flip).
            (perm, sign)   → length-dim lists. Use the helpers above.
    """
    if mirror is None:
        mirror_spec = None
    else:
        perm, sign = mirror
        if len(perm) != dim or len(sign) != dim:
            raise ValueError(f"Slot '{name}' mirror spec length mismatch (dim={dim})")
        mirror_spec = (perm, sign)

    def decorator(fn):
        _SLOT_REGISTRY[name] = (fn, dim, mirror_spec)
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
                fn, dim, _mirror_spec = _SLOT_REGISTRY[name]
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

@obs_slot('commands_3', dim=3, mirror=mirror_signs([1, -1, -1]))
def _commands_3(task):
    """vx, vy, yaw commands."""
    return task.commands[:, :3]


@obs_slot('commands_8', dim=8, mirror=mirror_signs([1, -1, -1, 1, 1, 1, 1, 1]))
def _commands_8(task):
    """8-slot commands: [vx, vy, yaw, height, 0, 0, 0, 0]."""
    return task.commands[:, :8]


@obs_slot('base_euler', dim=2, mirror=mirror_signs([-1, 1]))
def _base_euler(task):
    """Roll, pitch from delayed IMU."""
    return task.base_euler[:, :2]


@obs_slot('projected_gravity', dim=3, mirror=mirror_signs([1, -1, 1]))
def _projected_gravity(task):
    """Projected gravity in body frame (3-dim, IMU-style — what real robots
    output directly).  Equivalent to roll/pitch but more robust at large
    tilts. Used by BDX-R-MjLab / Isaac Lab humanoid configs.

    Mirror: gravity_x stays (forward axis), gravity_y flips, gravity_z stays.
    """
    return task.env.projected_gravity


@obs_slot('base_ang_vel', dim=3, mirror=mirror_signs([-1, 1, -1]))
def _base_ang_vel(task):
    """Angular velocity × 0.5 from delayed IMU."""
    return task.base_ang_vel * 0.5


@obs_slot('joint_pos_err', dim=10, mirror=mirror_swap_negate(10))
def _joint_pos_err(task):
    """joint_pos - ref_joint_pos (delayed)."""
    return task.joint_pos - task.ref_joint_action


@obs_slot('joint_vel', dim=10, mirror=mirror_swap_negate(10))
def _joint_vel(task):
    """Joint velocities × 0.1 (delayed)."""
    return task.joint_vel * 0.1


@obs_slot('joint_tracking_err', dim=10, mirror=mirror_swap_negate(10))
def _joint_tracking_err(task):
    """current_joint_act - joint_pos (tracking error)."""
    return task.joint_pos_error


@obs_slot('phase_sin_cos', dim=4, mirror=mirror_pair_swap(4))
def _phase_sin_cos(task):
    """sin/cos of leg phases × static_flag (BIRL only).
    Layout [sinL, sinR, cosL, cosR] — pair-swap under L↔R reflection.
    """
    return task.pm_phase * task.static_flag


@obs_slot('phase_freq', dim=2, mirror=mirror_pair_swap(2))
def _phase_freq(task):
    """(freq × 0.3 - 1.0) × static_flag (BIRL only). [freqL, freqR] — swap."""
    return (task.pm_f * 0.3 - 1.0) * task.static_flag


@obs_slot('phase_clock', dim=4, mirror=mirror_pair_swap(4))
def _phase_clock(task):
    """sin/cos of external phase clock × static_flag (BD_X style, phase.mode=input).
    Layout [sinL, sinR, cosL, cosR] — pair-swap.
    """
    return task._ext_clock.sin_cos() * task.static_flag


@obs_slot('phase_freq_cmd', dim=1, mirror=mirror_signs([1]))
def _phase_freq_cmd(task):
    """Normalized commanded phase frequency (BD_X style, phase.mode=input).

    Maps cmd_freq ∈ [freq_low, freq_high] → [-1, 1]. Multiplied by static_flag
    so the signal is zeroed when the robot is standing (same as phase_clock).
    Scalar — invariant under L↔R.
    """
    return ((task._cmd_freq - task._freq_mid) / task._freq_scale) * task.static_flag


@obs_slot('base_lin_vel', dim=3, mirror=mirror_signs([1, -1, 1]))
def _base_lin_vel(task):
    """Base linear velocity (privileged — teacher obs only). [vx, vy, vz]."""
    return task.base_lin_vel


# ─── MIRL-specific obs slots ────────────────────────────────────────────────

@obs_slot('ref_joint_pos_err', dim=10, mirror=mirror_swap_negate(10))
def _ref_joint_pos_err(task):
    """ref_joint_pos[t] - joint_pos (reference clip tracking, zeros if no clip)."""
    if task._has_ref:
        return (task._ref_joint_pos_now - task.joint_pos).clip(-3., 3.)
    return torch.zeros(task.num_envs, 10, dtype=torch.float, device=task.device)


@obs_slot('ref_joint_vel', dim=10, mirror=mirror_swap_negate(10))
def _ref_joint_vel(task):
    """ref_joint_vel[t] (reference clip velocity, zeros if no clip)."""
    if task._has_ref:
        return task._ref_joint_vel_now.clip(-3., 3.)
    return torch.zeros(task.num_envs, 10, dtype=torch.float, device=task.device)


@obs_slot('ref_phase_progress', dim=1, mirror=mirror_signs([1]))
def _ref_phase_progress(task):
    """Phase progress 0→1 in reference clip (zero if no clip). Scalar — identity."""
    if task._has_ref:
        return task._ref_phase_progress
    return torch.zeros(task.num_envs, 1, dtype=torch.float, device=task.device)
