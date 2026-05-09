"""
Left-right mirror transform for BIRL policy observations and actions.

Used for symmetry data augmentation during PPO training: each real
mini-batch is augmented with its mirrored counterpart, which doubles
the effective data and enforces a symmetric policy without extra envs.

BIRL obs layout (44 dims/step × 3 history = 132 total):
  [0]     cmd_vx
  [1]     cmd_vy        → negate
  [2]     cmd_yaw       → negate
  [3]     roll          → negate
  [4]     pitch
  [5]     ang_vel_x×0.5 → negate (roll rate)
  [6]     ang_vel_y×0.5
  [7]     ang_vel_z×0.5 → negate (yaw rate)
  [8-12]  L joint pos − ref  (hip_yaw, hip_roll, hip_pitch, knee, ankle)
  [13-17] R joint pos − ref
  [18-22] L joint vel × 0.1
  [23-27] R joint vel × 0.1
  [28-32] L joint pos error
  [33-37] R joint pos error
  [38-39] sin phase [L, R]   → swap
  [40-41] cos phase [L, R]   → swap
  [42]    freq_L              → swap with freq_R
  [43]    freq_R

Mirror rule for joints: ALL 5 joint types negate when L↔R swap.
This is because the URDF defines opposite-sign joint axes for the
two legs (verified from ref_joint_pos:
  L=[0.4, -0.1, -1.5, 1.0, -1.3], R=[-0.4, 0.1, 1.5, -1.0, 1.3]).

BIRL action layout (12 dims):
  [0]    freq_L   → swap with freq_R (no negate)
  [1]    freq_R
  [2-6]  L joint deltas (hip_yaw, hip_roll, hip_pitch, knee, ankle)
  [7-11] R joint deltas
  Mirror: L↔R swap + negate all joint deltas.
"""

import torch
from env.obs_builder import _SLOT_REGISTRY


# ─── Generic slot-driven Mirror (Phase 4) ────────────────────────────────────

class Mirror:
    """Mirror augmentation built from obs_slots metadata.

    Works for any obs layout: each slot self-describes its (perm, sign) under
    L↔R body reflection via the `mirror=` arg on @obs_slot. Action mirror is
    determined by `action_dim` (10 for joint-only or 10-dim BDX, 12 for BIRL
    with 2 freq + 10 joint).

    Args:
        obs_slots: list of slot names (same as cfg.observation.slots).
        obs_history: number of frames the policy sees (cfg.observation.history).
        action_dim: 10 or 12.
        device: torch device for the permutation/sign tensors.
    """

    def __init__(self, obs_slots, obs_history, action_dim, device='cpu'):
        self.obs_slots = list(obs_slots)
        self.obs_history = int(obs_history)
        self.action_dim = int(action_dim)
        self.device = device

        step_perm, step_sign = self._build_step_transform()
        self.step_dim = len(step_perm)
        self._obs_perm, self._obs_sign = self._expand_to_stack(step_perm, step_sign)
        self._act_perm, self._act_sign = self._build_action_transform()
        self.to(device)

    def _build_step_transform(self):
        step_perm, step_sign = [], []
        offset = 0
        for name in self.obs_slots:
            if name not in _SLOT_REGISTRY:
                raise ValueError(f"Mirror: unknown obs slot '{name}'")
            _fn, dim, mirror_spec = _SLOT_REGISTRY[name]
            if mirror_spec is None:
                raise ValueError(
                    f"Mirror: slot '{name}' has no mirror spec — annotate the "
                    f"@obs_slot decorator with mirror=..."
                )
            local_perm, local_sign = mirror_spec
            step_perm.extend(offset + p for p in local_perm)
            step_sign.extend(local_sign)
            offset += dim
        return step_perm, step_sign

    def _expand_to_stack(self, step_perm, step_sign):
        full_perm, full_sign = [], []
        for h in range(self.obs_history):
            base = h * self.step_dim
            full_perm += [base + p for p in step_perm]
            full_sign += step_sign
        return (
            torch.tensor(full_perm, dtype=torch.long),
            torch.tensor(full_sign, dtype=torch.float32),
        )

    def _build_action_transform(self):
        D = self.action_dim
        if D == 10:
            # Joint-only L/R swap + negate all (5L + 5R).
            perm = list(range(5, 10)) + list(range(0, 5))
            sign = [-1.0] * 10
        elif D == 12:
            # BIRL: [freq_L, freq_R, L joints×5, R joints×5].
            perm = [1, 0] + list(range(7, 12)) + list(range(2, 7))
            sign = [1.0, 1.0] + [-1.0] * 10
        else:
            raise ValueError(f"Mirror: unsupported action_dim={D}")
        return (
            torch.tensor(perm, dtype=torch.long),
            torch.tensor(sign, dtype=torch.float32),
        )

    def to(self, device):
        self._obs_perm = self._obs_perm.to(device)
        self._obs_sign = self._obs_sign.to(device)
        self._act_perm = self._act_perm.to(device)
        self._act_sign = self._act_sign.to(device)
        self.device = device
        return self

    def mirror_obs(self, obs):
        return obs[..., self._obs_perm] * self._obs_sign

    def mirror_action(self, act):
        return act[..., self._act_perm] * self._act_sign

    # PPO calls .mirror_actions (plural) — keep alias for parity with BIRLMirror.
    def mirror_actions(self, act):
        return self.mirror_action(act)




