"""
High-level controller: orchestrates between walking and recovery policies.

State machine:

    WALKING ──fall_detected──> FALLEN_WAIT ──settled──> RECOVERING ──upright_stable──> WALKING
       ▲                                                      │
       └────────────────────────recover_timeout───────────────┘ (give up; back to WALKING)

The controller owns:
  - two ONNX policies (walking, recovery) loaded from their respective manifests
  - obs builders for each policy (different obs layouts)
  - the smoothed action targets fed to the PD controller

It does NOT own MuJoCo / robot I/O — caller passes raw sensor readings each tick
and receives joint position targets back. This keeps the controller usable from
both MuJoCo (sim2sim) and the C++ SDK glue layer.

Usage sketch (MuJoCo):
    hl = HighLevelController(walking_manifest_path, recovery_manifest_path)
    hl.reset()
    while running:
        sensor_pkt = SensorPacket(base_pos, base_quat_wxyz, base_lin_vel,
                                  base_ang_vel, base_euler, joint_pos, joint_vel,
                                  cmd_vx, cmd_vy, cmd_yaw)
        joint_target = hl.step(sensor_pkt)
        # apply PD control with joint_target ...

Notes:
  - This module is policy-agnostic on the obs side: obs vectors are built by
    helper functions matching the slot layouts the policies were trained with.
    A walking policy from `birl.yaml` and a recovery policy from `recovery.yaml`
    have very different obs shapes and that's OK — the manifests describe both.
  - For the v0 cut, obs construction supports the recovery policy's slots and
    the BIRL walking policy's slots. Other policy types can be added incrementally.
"""

from __future__ import annotations

import os
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import onnxruntime as ort
import yaml


# ─── Sensor input ─────────────────────────────────────────────────────────────

@dataclass
class SensorPacket:
    """Per-tick robot state. All fields are 1D numpy arrays unless noted.

    base_quat is wxyz convention (matches MuJoCo qpos[3:7]).
    base_euler is [roll, pitch, yaw] in radians.
    """
    base_pos: np.ndarray            # [3]
    base_quat_wxyz: np.ndarray      # [4]
    base_lin_vel: np.ndarray        # [3]   body frame
    base_ang_vel: np.ndarray        # [3]   body frame
    base_euler: np.ndarray          # [3]   roll, pitch, yaw
    joint_pos: np.ndarray           # [10]
    joint_vel: np.ndarray           # [10]
    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_yaw: float = 0.0
    dt: float = 0.015


# ─── State machine ────────────────────────────────────────────────────────────

class HLMode(Enum):
    WALKING = "walking"
    FALLEN_WAIT = "fallen_wait"      # robot has fallen; let it settle before recovery
    RECOVERING = "recovering"        # recovery policy active
    RECOVERY_DONE = "recovery_done"  # upright + stable; brief hold before WALKING


@dataclass
class HLConfig:
    # Fall detection
    fall_tilt_deg: float = 35.0          # body z vs world z exceeds this → fallen
    fall_height_frac: float = 0.55       # base_z below this fraction of nominal → fallen
    fall_persist_steps: int = 7          # need this many consecutive fallen frames
    # Settle (within FALLEN_WAIT)
    settle_lin_vel: float = 0.2
    settle_ang_vel: float = 1.0
    settle_persist_steps: int = 10
    settle_max_wait_steps: int = 200      # safety: trigger recovery anyway after this
    # Recovery success
    upright_tilt_deg: float = 20.0
    upright_height_frac: float = 0.85
    upright_persist_steps: int = 30
    recovery_max_steps: int = 500         # ~7.5s @ 67Hz; force back to walking after this
    # Hand-off
    handoff_hold_steps: int = 15          # hold last targets after RECOVERY_DONE before WALKING


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_manifest(policy_path: str):
    policy_dir = os.path.dirname(os.path.abspath(policy_path))
    stem = os.path.splitext(os.path.basename(policy_path))[0]
    candidates = [
        os.path.join(policy_dir, f'{stem}_manifest.yaml'),
        os.path.join(policy_dir, 'manifest.yaml'),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No manifest found next to {policy_path}")


def _quat_wxyz_to_tilt_deg(q):
    qx, qy = q[1], q[2]
    cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
    return math.degrees(math.acos(cos_tilt))


def _projected_gravity(q_wxyz):
    """Gravity vector in body frame (gravity is world -z)."""
    qw, qx, qy, qz = q_wxyz
    # body-frame components of world +z then negated.
    bx = 2.0 * (qx * qz - qw * qy)
    by = 2.0 * (qy * qz + qw * qx)
    bz = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.array([-bx, -by, -bz], dtype=np.float32)


# ─── Policy bundle ────────────────────────────────────────────────────────────

class _Policy:
    """Wraps an ONNX policy + its manifest + per-step state (action history)."""

    def __init__(self, manifest_path_or_dict, policy_path: Optional[str] = None):
        if isinstance(manifest_path_or_dict, dict):
            self.manifest = manifest_path_or_dict
            self.policy_path = policy_path
        else:
            self.manifest = _load_manifest(manifest_path_or_dict)
            self.policy_path = policy_path or manifest_path_or_dict
        self.session = ort.InferenceSession(self.policy_path,
                                             providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.obs_history = int(self.manifest['obs_history'])
        self.obs_skip = int(self.manifest.get('obs_skip', 1))
        self.obs_per_step = int(self.manifest['obs_per_step'])
        self.obs_total = int(self.manifest['obs_total'])
        self.action_dim = int(self.manifest['action_dim'])
        self.action_mode = self.manifest['action_mode']
        self.lp_alpha = float(self.manifest.get('action_lowpass_alpha', 1.0))
        scaling = self.manifest['action_scaling']
        if self.action_mode == 'absolute':
            self.act_low = np.array(scaling['abs_low'] or self.manifest['joint_limits']['low'],
                                    dtype=np.float32)
            self.act_high = np.array(scaling['abs_high'] or self.manifest['joint_limits']['high'],
                                     dtype=np.float32)
        else:
            self.act_low = np.array(scaling['inc_low'], dtype=np.float32)
            self.act_high = np.array(scaling['inc_high'], dtype=np.float32)
        self.ref_joint_pos = np.array(self.manifest['ref_joint_pos'], dtype=np.float32)
        self._buf_len = (self.obs_history - 1) * self.obs_skip + 1
        self.obs_history_buf = deque(maxlen=self._buf_len)
        self.lp_target = self.ref_joint_pos.copy()
        self.last_target = self.ref_joint_pos.copy()
        self.episode_step = 0
        self.episode_length = max(1, int(self.manifest.get('recovery', {})
                                                    .get('episode_length_s', 5.0)
                                                    / 0.015))

    def reset(self):
        self.obs_history_buf.clear()
        self.lp_target = self.ref_joint_pos.copy()
        self.last_target = self.ref_joint_pos.copy()
        self.episode_step = 0

    def predict(self, obs_per_step):
        if len(self.obs_history_buf) == 0:
            for _ in range(self._buf_len):
                self.obs_history_buf.append(obs_per_step.copy())
        else:
            self.obs_history_buf.append(obs_per_step)
        buf = list(self.obs_history_buf)
        idx = [i * self.obs_skip for i in range(self.obs_history)]
        obs = np.concatenate([buf[i] for i in idx], axis=0).astype(np.float32)
        out = self.session.run(None, {self.input_name: obs[None, :]})[0][0]
        return out  # raw net out [action_dim]

    def to_target(self, net_out):
        """Map raw [-1,1] net_out → joint position target (deg of low-pass applied)."""
        clipped = np.clip(net_out, -1.0, 1.0)
        scaled = self.act_low + 0.5 * (clipped + 1.0) * (self.act_high - self.act_low)
        if self.action_mode == 'absolute':
            self.lp_target = self.lp_alpha * scaled + (1.0 - self.lp_alpha) * self.lp_target
            target = self.lp_target.copy()
        else:  # increment
            target = self.last_target + scaled * 0.015
        self.last_target = target.copy()
        self.episode_step += 1
        return target


# ─── Obs builders (slot-aware) ────────────────────────────────────────────────

def _build_obs_per_step(slots: Sequence[str], pkt: SensorPacket,
                        ref_joint_pos: np.ndarray, last_target: np.ndarray,
                        episode_step: int, episode_length: int):
    """Construct one timestep's observation matching the slot list.

    Supports the slots used by `configs/recovery/recovery.yaml` and the BIRL configs.
    Unknown slots raise — fail fast at deploy time rather than ship junk obs.
    """
    parts = []
    for s in slots:
        if s == 'commands_3':
            parts.append(np.array([pkt.cmd_vx, pkt.cmd_vy, pkt.cmd_yaw], dtype=np.float32))
        elif s == 'base_euler':
            parts.append(pkt.base_euler[:2].astype(np.float32))   # roll, pitch
        elif s == 'base_ang_vel':
            parts.append((pkt.base_ang_vel * 0.5).astype(np.float32))
        elif s == 'joint_pos_err':
            parts.append((pkt.joint_pos - ref_joint_pos).astype(np.float32))
        elif s == 'joint_vel':
            parts.append((pkt.joint_vel * 0.1).astype(np.float32))
        elif s == 'joint_tracking_err':
            parts.append((last_target - pkt.joint_pos).astype(np.float32))
        elif s == 'projected_gravity':
            parts.append(_projected_gravity(pkt.base_quat_wxyz))
        elif s == 'episode_progress':
            p = float(min(episode_step, episode_length)) / max(1, episode_length)
            parts.append(np.array([p], dtype=np.float32))
        elif s == 'joint_pos_abs':
            parts.append(pkt.joint_pos.astype(np.float32))
        elif s == 'phase_sin_cos':
            # not used by walking-as-trained-here / recovery; would need PM state
            raise ValueError(f"slot 'phase_sin_cos' requires phase modulator state — "
                             "not yet wired into HighLevelController")
        elif s == 'phase_freq':
            raise ValueError("slot 'phase_freq' requires phase modulator state — "
                             "not yet wired into HighLevelController")
        elif s == 'phase_clock':
            raise ValueError("slot 'phase_clock' requires external phase clock — "
                             "not yet wired into HighLevelController")
        else:
            raise ValueError(f"Unknown obs slot for high-level controller: '{s}'")
    return np.concatenate(parts, axis=0).astype(np.float32)


# ─── HighLevelController ──────────────────────────────────────────────────────

class HighLevelController:
    """Loads a walking + recovery policy, dispatches to whichever the state machine selects.

    Both `walking_policy_path` and `recovery_policy_path` are .onnx files; their
    manifests are auto-discovered via deploy/manifest.py conventions.
    """

    def __init__(self,
                 walking_policy_path: str,
                 recovery_policy_path: str,
                 cfg: Optional[HLConfig] = None,
                 nominal_height: Optional[float] = None):
        self.cfg = cfg or HLConfig()
        self.walking = _Policy(_load_manifest(walking_policy_path), walking_policy_path)
        self.recovery = _Policy(_load_manifest(recovery_policy_path), recovery_policy_path)
        # Recovery manifest carries thresholds; prefer those when present.
        rec_meta = self.recovery.manifest.get('recovery', {})
        self._target_height = float(rec_meta.get('target_height',
                                                 nominal_height or 0.45))
        self._upright_tilt = float(rec_meta.get('success_tilt_deg', self.cfg.upright_tilt_deg))
        self._upright_h = float(rec_meta.get('target_height_ratio', 0.85)) * self._target_height
        # Slots
        self._walking_slots = self.walking.manifest.get('obs_slots') or []
        self._recovery_slots = self.recovery.manifest.get('obs_slots') or []

        self.mode = HLMode.WALKING
        self._fall_count = 0
        self._settle_count = 0
        self._fallen_wait_count = 0
        self._upright_count = 0
        self._recovery_step = 0
        self._handoff_count = 0
        self._last_target = self.walking.ref_joint_pos.copy()

    def reset(self):
        self.walking.reset()
        self.recovery.reset()
        self.mode = HLMode.WALKING
        self._fall_count = 0
        self._settle_count = 0
        self._fallen_wait_count = 0
        self._upright_count = 0
        self._recovery_step = 0
        self._handoff_count = 0
        self._last_target = self.walking.ref_joint_pos.copy()

    # ─── Detection helpers ────────────────────────────────────────────────

    def _is_fallen(self, pkt: SensorPacket) -> bool:
        tilt = _quat_wxyz_to_tilt_deg(pkt.base_quat_wxyz)
        z = float(pkt.base_pos[2])
        return tilt > self.cfg.fall_tilt_deg or z < self.cfg.fall_height_frac * self._target_height

    def _is_upright(self, pkt: SensorPacket) -> bool:
        tilt = _quat_wxyz_to_tilt_deg(pkt.base_quat_wxyz)
        z = float(pkt.base_pos[2])
        return tilt < self._upright_tilt and z > self._upright_h

    def _is_settled(self, pkt: SensorPacket) -> bool:
        return (np.linalg.norm(pkt.base_lin_vel) < self.cfg.settle_lin_vel and
                np.linalg.norm(pkt.base_ang_vel) < self.cfg.settle_ang_vel)

    # ─── Main step ────────────────────────────────────────────────────────

    def step(self, pkt: SensorPacket) -> np.ndarray:
        if self.mode == HLMode.WALKING:
            target = self._step_walking(pkt)
            if self._is_fallen(pkt):
                self._fall_count += 1
            else:
                self._fall_count = 0
            if self._fall_count >= self.cfg.fall_persist_steps:
                self.mode = HLMode.FALLEN_WAIT
                self._fall_count = 0
                self._settle_count = 0
                self._fallen_wait_count = 0

        elif self.mode == HLMode.FALLEN_WAIT:
            # Hold the last target (compliant); wait for the body to stop tumbling.
            target = self._last_target
            self._fallen_wait_count += 1
            if self._is_settled(pkt):
                self._settle_count += 1
            else:
                self._settle_count = 0
            if (self._settle_count >= self.cfg.settle_persist_steps or
                self._fallen_wait_count >= self.cfg.settle_max_wait_steps):
                self.mode = HLMode.RECOVERING
                self.recovery.reset()
                self._recovery_step = 0
                self._upright_count = 0

        elif self.mode == HLMode.RECOVERING:
            target = self._step_recovery(pkt)
            self._recovery_step += 1
            if self._is_upright(pkt):
                self._upright_count += 1
            else:
                self._upright_count = 0
            if self._upright_count >= self.cfg.upright_persist_steps:
                self.mode = HLMode.RECOVERY_DONE
                self._handoff_count = 0
            elif self._recovery_step >= self.cfg.recovery_max_steps:
                # give up; let walking try to take over
                self.mode = HLMode.WALKING
                self.walking.reset()

        elif self.mode == HLMode.RECOVERY_DONE:
            # Hold the recovery's last target briefly so the policy hand-off
            # doesn't introduce a target jump.
            target = self._last_target
            self._handoff_count += 1
            if self._handoff_count >= self.cfg.handoff_hold_steps:
                self.mode = HLMode.WALKING
                self.walking.reset()

        else:
            raise RuntimeError(f"Unhandled HLMode: {self.mode}")

        self._last_target = target
        return target

    # ─── Per-mode action computation ──────────────────────────────────────

    def _step_walking(self, pkt: SensorPacket) -> np.ndarray:
        obs = _build_obs_per_step(
            self._walking_slots, pkt,
            self.walking.ref_joint_pos, self.walking.last_target,
            self.walking.episode_step, self.walking.episode_length,
        )
        net_out = self.walking.predict(obs)
        return self.walking.to_target(net_out)

    def _step_recovery(self, pkt: SensorPacket) -> np.ndarray:
        obs = _build_obs_per_step(
            self._recovery_slots, pkt,
            self.recovery.ref_joint_pos, self.recovery.last_target,
            self.recovery.episode_step, self.recovery.episode_length,
        )
        net_out = self.recovery.predict(obs)
        return self.recovery.to_target(net_out)

    # ─── Reporting ────────────────────────────────────────────────────────

    def info(self):
        return {
            'mode': self.mode.value,
            'fall_count': self._fall_count,
            'settle_count': self._settle_count,
            'fallen_wait_count': self._fallen_wait_count,
            'upright_count': self._upright_count,
            'recovery_step': self._recovery_step,
        }
