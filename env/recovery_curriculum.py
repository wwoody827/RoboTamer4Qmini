"""
Recovery curriculum manager (Phase 5 scaffold).

Selects which init-state subset each env starts from, and advances stages once
the rolling success rate clears a threshold. Old-stage replay (default 20%) is
mixed into every stage past 1 so the policy doesn't forget earlier conditions.

Stages (default):
    1. near_nominal   — only init states whose tilt is small (<= 30°)
                        and base height >= 70% of nominal. Easiest.
    2. moderate       — tilt up to 60°, base height >= 40% nominal.
    3. heavy_fall     — tilt up to 110°, includes full prone / supine.
    4. all            — entire init pool, plus optional DR amplification.

Usage:
    curriculum = RecoveryCurriculum(pool_meta, cfg.task.curriculum)
    # at reset(env_ids):
    init_ids = curriculum.sample_indices(env_ids)
    # after each episode end, with a [num_envs] success bool:
    curriculum.record_outcomes(env_ids, success_mask)
    curriculum.maybe_advance()
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch


@dataclass
class StageSpec:
    name: str
    tilt_max_deg: float
    height_min_frac: float          # of nominal_height
    pose_labels: Optional[Sequence[str]] = None  # whitelist; None = no filter
    success_threshold: float = 0.7  # advance when rolling success >= this
    min_episodes: int = 200         # before allowed to advance


def default_stages() -> List[StageSpec]:
    return [
        StageSpec(name='stage1_near_nominal', tilt_max_deg=30.0,  height_min_frac=0.70,
                  pose_labels=None, success_threshold=0.70, min_episodes=200),
        StageSpec(name='stage2_moderate',     tilt_max_deg=60.0,  height_min_frac=0.40,
                  pose_labels=None, success_threshold=0.60, min_episodes=300),
        StageSpec(name='stage3_heavy_fall',   tilt_max_deg=110.0, height_min_frac=0.0,
                  pose_labels=None, success_threshold=0.45, min_episodes=400),
        StageSpec(name='stage4_all',          tilt_max_deg=180.0, height_min_frac=0.0,
                  pose_labels=None, success_threshold=0.0,  min_episodes=10**9),
    ]


def _quat_wxyz_to_tilt_deg(q_wxyz: np.ndarray) -> np.ndarray:
    qx = q_wxyz[:, 1]
    qy = q_wxyz[:, 2]
    cos_tilt = 1.0 - 2.0 * (qx * qx + qy * qy)
    cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt))


class RecoveryCurriculum:
    """Tracks a single global stage shared by all envs, with per-env replay
    of earlier stages."""

    def __init__(self,
                 pool_quat_wxyz: np.ndarray,        # [N, 4]
                 pool_base_pos: np.ndarray,         # [N, 3]
                 pool_pose_label: Optional[np.ndarray],  # [N] str or None
                 nominal_height: float = 0.45,
                 stages: Optional[List[StageSpec]] = None,
                 replay_old_frac: float = 0.20,
                 device: torch.device = torch.device('cpu'),
                 rolling_window: int = 1024):
        self.device = device
        self.stages = stages or default_stages()
        self.replay_old_frac = float(replay_old_frac)
        self.rolling_window = rolling_window

        # Pre-compute per-pool tilt + height-ratio.
        tilts = _quat_wxyz_to_tilt_deg(pool_quat_wxyz)
        height_frac = pool_base_pos[:, 2] / max(nominal_height, 1e-6)
        labels = np.array(pool_pose_label) if pool_pose_label is not None else None

        # Build per-stage index lists once.
        self._stage_indices: List[np.ndarray] = []
        self._stage_fallback: List[bool] = []
        for s in self.stages:
            mask = (tilts <= s.tilt_max_deg) & (height_frac >= s.height_min_frac)
            if s.pose_labels is not None and labels is not None:
                lset = set(s.pose_labels)
                mask &= np.array([str(l) in lset for l in labels])
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                # fall back to entire pool (data quality guard)
                idx = np.arange(pool_quat_wxyz.shape[0])
                self._stage_fallback.append(True)
                print(f"[RecoveryCurriculum] WARNING: stage '{s.name}' has 0 "
                      f"matching states (tilt<={s.tilt_max_deg}, "
                      f"height_frac>={s.height_min_frac}); falling back to full pool")
            else:
                self._stage_fallback.append(False)
            self._stage_indices.append(idx)

        self._stage_indices_torch = [
            torch.as_tensor(idx, dtype=torch.long, device=self.device)
            for idx in self._stage_indices
        ]

        self.current_stage = 0
        self._episode_count = 0
        self._success_buf = np.zeros(self.rolling_window, dtype=np.float32)
        self._buf_idx = 0
        self._buf_filled = 0

        print(f"[RecoveryCurriculum] {len(self.stages)} stages, "
              + ", ".join(f"{s.name}:{len(idx)} states"
                          for s, idx in zip(self.stages, self._stage_indices)))

    # ──────────────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────────────

    def sample_indices(self, n: int) -> torch.Tensor:
        """Sample n init-pool indices: most from current stage, replay_old_frac
        from a uniformly-chosen earlier stage."""
        cur = self._stage_indices_torch[self.current_stage]
        if self.current_stage == 0 or self.replay_old_frac <= 0.0:
            idx_within = torch.randint(0, len(cur), (n,), device=self.device)
            return cur[idx_within]
        # Mix: (1 - replay_old_frac) from current, replay_old_frac from a random
        # earlier stage, sampled per-element.
        n_old = int(round(n * self.replay_old_frac))
        n_new = n - n_old
        out = torch.empty(n, dtype=torch.long, device=self.device)
        if n_new > 0:
            out[:n_new] = cur[torch.randint(0, len(cur), (n_new,), device=self.device)]
        if n_old > 0:
            old_stage = int(np.random.randint(0, self.current_stage))
            old = self._stage_indices_torch[old_stage]
            out[n_new:] = old[torch.randint(0, len(old), (n_old,), device=self.device)]
        # Shuffle so consecutive env ids don't all share a stage.
        perm = torch.randperm(n, device=self.device)
        return out[perm]

    # ──────────────────────────────────────────────────────────────────────
    # Outcome bookkeeping & stage progression
    # ──────────────────────────────────────────────────────────────────────

    def record_outcomes(self, success_mask):
        """success_mask: bool tensor or numpy of episode outcomes (one per
        finished episode this update). Order doesn't matter."""
        if torch.is_tensor(success_mask):
            success_mask = success_mask.detach().cpu().numpy()
        success_mask = success_mask.astype(np.float32).reshape(-1)
        for v in success_mask:
            self._success_buf[self._buf_idx] = v
            self._buf_idx = (self._buf_idx + 1) % self.rolling_window
            self._buf_filled = min(self._buf_filled + 1, self.rolling_window)
            self._episode_count += 1

    def rolling_success_rate(self) -> float:
        if self._buf_filled == 0:
            return 0.0
        return float(self._success_buf[: self._buf_filled].mean())

    def maybe_advance(self) -> bool:
        """If rolling success >= current stage threshold AND enough episodes
        have been observed since last advance, move to the next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False
        s = self.stages[self.current_stage]
        if self._episode_count < s.min_episodes:
            return False
        if self.rolling_success_rate() < s.success_threshold:
            return False
        old = self.current_stage
        self.current_stage += 1
        # reset rolling stats so next stage gets fresh measurement
        self._success_buf[:] = 0.0
        self._buf_idx = 0
        self._buf_filled = 0
        self._episode_count = 0
        print(f"[RecoveryCurriculum] advanced {self.stages[old].name} "
              f"→ {self.stages[self.current_stage].name}")
        return True

    # Reporting -----------------------------------------------------------

    def info(self) -> dict:
        return {
            'curriculum/stage': self.current_stage,
            'curriculum/stage_name': self.stages[self.current_stage].name,
            'curriculum/rolling_success': self.rolling_success_rate(),
            'curriculum/episodes_in_stage': self._episode_count,
        }
