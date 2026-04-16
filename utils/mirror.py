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
