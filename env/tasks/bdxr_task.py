"""BDXRTask — faithful reproduction of BDX-R-MjLab's velocity-tracking task.

Source: https://github.com/BDX-R/BDX-R-MjLab — Disney BDX-R bipedal trained
in mjlab (MuJoCo + Isaac Lab API). This task class inherits all of BIRLTask's
machinery (PD, DR, mirror, asymmetric critic, etc.) but expects a config
that uses only BDX-R's reward set:

  • NO phase clock in observation (phase.mode = none)
  • Asymmetric actor-critic (handled by BIRLTask.critic_observation)
  • Action mode: scale + default offset (via abs_low/high_ranges = ref±0.5)
  • Rewards:
      - track_linear_velocity / track_angular_velocity (gaussian std)
      - upright (positive bell exp(-||g_xy||²/σ²))
      - pose (speed-conditional, per-joint std)
      - feet_swing_height_peak (sparse at touchdown)
      - foot_clearance, foot_slip, soft_landing
      - action_rate_l2, body_ang_vel, dof_pos_limits, air_time

The reproduction differs from BDX-R-MjLab in:
  • Policy rate 66.67 Hz (vs BDX-R 50 Hz) — kept for Qmini hardware match
  • No terrain levels / heading-command curriculum (separate enhancements)
  • Some DR knobs (encoder_bias not yet supported)

See `configs/walk_bdxr_full.yaml` for the canonical recipe.
"""

from env.tasks.birl_task import BIRLTask
from env.tasks.null_task import register


@register
class BDXRTask(BIRLTask):
    """No-clock BDX-R-MjLab style task. Identical machinery to BIRLTask;
    distinct registry name is the only change — keeps configs and run
    bookkeeping cleanly separated from walk_v34-lineage BIRL runs.

    Use via `task.cfg: BDXR` in the training YAML.
    """
    pass
