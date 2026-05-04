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




class BDXMirror:
    """Mirror for BD_X-style policy (phase.mode=input, 43-dim obs, 10-dim action).

    Obs layout (43 dims/step × 3 history = 129 total):
      [0]     cmd_vx
      [1]     cmd_vy        → negate
      [2]     cmd_yaw       → negate
      [3]     roll          → negate
      [4]     pitch
      [5]     ang_vel_x×0.5 → negate (roll rate)
      [6]     ang_vel_y×0.5
      [7]     ang_vel_z×0.5 → negate (yaw rate)
      [8-12]  L joint pos − ref  → swap with R, negate all
      [13-17] R joint pos − ref
      [18-22] L joint vel × 0.1  → swap with R, negate all
      [23-27] R joint vel × 0.1
      [28-32] L tracking error   → swap with R, negate all
      [33-37] R tracking error
      [38-39] sin phase [L, R]   → swap
      [40-41] cos phase [L, R]   → swap
      [42]    phase_freq_cmd     → identity (scalar, symmetric under L↔R)

    Action layout (10 dims):
      [0-4]  L joints (hip_yaw, hip_roll, hip_pitch, knee, ankle)
      [5-9]  R joints
      Mirror: L↔R swap + negate all.
    """
    STEP_DIM = 43

    def __init__(self, obs_history: int = 3, device='cpu'):
        self.obs_history = obs_history
        self.device = device

        step_perm, step_sign = self._build_step_transform()
        self._obs_perm, self._obs_sign = self._expand_to_stack(step_perm, step_sign)
        self._act_perm, self._act_sign = self._build_action_transform()
        self.to(device)

    def _build_step_transform(self):
        D = self.STEP_DIM
        perm = list(range(D))
        sign = [1.0] * D

        sign[1] = -1.0   # cmd_vy
        sign[2] = -1.0   # cmd_yaw
        sign[3] = -1.0   # roll
        sign[5] = -1.0   # roll rate
        sign[7] = -1.0   # yaw rate

        # Joint sections: L[8:13] ↔ R[13:18], L[18:23] ↔ R[23:28], L[28:33] ↔ R[33:38]
        for base in (8, 18, 28):
            for i in range(5):
                perm[base + i]     = base + 5 + i
                perm[base + 5 + i] = base + i
                sign[base + i]     = -1.0
                sign[base + 5 + i] = -1.0

        # Phase clock: [38=sin_L, 39=sin_R] → swap, [40=cos_L, 41=cos_R] → swap
        perm[38], perm[39] = 39, 38
        perm[40], perm[41] = 41, 40

        # [42] phase_freq_cmd: scalar, invariant under L↔R (identity).
        # perm[42] = 42 and sign[42] = 1.0 are already set by the init above.

        return perm, sign

    def _expand_to_stack(self, step_perm, step_sign):
        full_perm, full_sign = [], []
        for h in range(self.obs_history):
            offset = h * self.STEP_DIM
            full_perm += [offset + p for p in step_perm]
            full_sign += step_sign
        return (
            torch.tensor(full_perm, dtype=torch.long),
            torch.tensor(full_sign, dtype=torch.float32),
        )

    def _build_action_transform(self):
        perm = list(range(10))
        sign = [1.0] * 10
        # L[0:5] ↔ R[5:10], negate all
        for i in range(5):
            perm[i]     = 5 + i
            perm[5 + i] = i
            sign[i]     = -1.0
            sign[5 + i] = -1.0
        return (
            torch.tensor(perm, dtype=torch.long),
            torch.tensor(sign, dtype=torch.float32),
        )

    def to(self, device):
        self._obs_perm  = self._obs_perm.to(device)
        self._obs_sign  = self._obs_sign.to(device)
        self._act_perm  = self._act_perm.to(device)
        self._act_sign  = self._act_sign.to(device)
        self.device = device
        return self

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self._obs_perm] * self._obs_sign

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions[:, self._act_perm] * self._act_sign


class BIRLMirror:
    STEP_DIM = 44

    def __init__(self, obs_history: int = 3, device='cpu'):
        self.obs_history = obs_history
        self.device = device

        step_perm, step_sign = self._build_step_transform()
        self._obs_perm, self._obs_sign = self._expand_to_stack(step_perm, step_sign)
        self._act_perm, self._act_sign = self._build_action_transform()
        self.to(device)

    # ──────────────────────────────────────────────────────────────────────────
    def _build_step_transform(self):
        D = self.STEP_DIM
        perm = list(range(D))
        sign = [1.0] * D

        # Commands
        sign[1] = -1.0   # cmd_vy
        sign[2] = -1.0   # cmd_yaw

        # Base orientation
        sign[3] = -1.0   # roll

        # Angular velocity
        sign[5] = -1.0   # roll rate (ang_vel_x)
        sign[7] = -1.0   # yaw  rate (ang_vel_z)

        # Joint sections: L[8:13] ↔ R[13:18], all negated
        # Joint vel:      L[18:23] ↔ R[23:28], all negated
        # Joint err:      L[28:33] ↔ R[33:38], all negated
        for base in (8, 18, 28):
            for i in range(5):
                perm[base + i]     = base + 5 + i
                perm[base + 5 + i] = base + i
                sign[base + i]     = -1.0
                sign[base + 5 + i] = -1.0

        # sin phase: [38=sin_L, 39=sin_R] → swap
        perm[38], perm[39] = 39, 38
        # cos phase: [40=cos_L, 41=cos_R] → swap
        perm[40], perm[41] = 41, 40

        # Freqs: [42=freq_L, 43=freq_R] → swap, no negate
        perm[42], perm[43] = 43, 42

        return perm, sign

    def _expand_to_stack(self, step_perm, step_sign):
        full_perm, full_sign = [], []
        for h in range(self.obs_history):
            offset = h * self.STEP_DIM
            full_perm += [offset + p for p in step_perm]
            full_sign += step_sign
        return (
            torch.tensor(full_perm, dtype=torch.long),
            torch.tensor(full_sign, dtype=torch.float32),
        )

    def _build_action_transform(self):
        perm = list(range(12))
        sign = [1.0] * 12

        # Frequencies: swap L↔R, no negate
        perm[0], perm[1] = 1, 0

        # Joint deltas: L[2:7] ↔ R[7:12], all negated
        for i in range(5):
            perm[2 + i] = 7 + i
            perm[7 + i] = 2 + i
            sign[2 + i] = -1.0
            sign[7 + i] = -1.0

        return (
            torch.tensor(perm, dtype=torch.long),
            torch.tensor(sign, dtype=torch.float32),
        )

    # ──────────────────────────────────────────────────────────────────────────
    def to(self, device):
        self._obs_perm  = self._obs_perm.to(device)
        self._obs_sign  = self._obs_sign.to(device)
        self._act_perm  = self._act_perm.to(device)
        self._act_sign  = self._act_sign.to(device)
        self.device = device
        return self

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [N, obs_dim] → mirrored [N, obs_dim]"""
        return obs[:, self._obs_perm] * self._obs_sign

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [N, 12] → mirrored [N, 12]"""
        return actions[:, self._act_perm] * self._act_sign
