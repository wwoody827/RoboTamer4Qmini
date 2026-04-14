"""
BIRL forward/backward expert config.

Differences from BIRL:
  - vx only: vy=0, yaw=0
  - Higher yaw_rat and twist weights to penalize drifting
  - Intended use: train a clean fwd/bwd reference clip for MIRL imitation

Usage:
    python train.py --config BIRL_Fwd --name birl_ref_v1 --sim2sim_interval 500 --num_envs 4096
"""
from .Base import SetDict2Class, Base


class BIRL_Fwd(Base):
    def __init__(self):
        super(BIRL_Fwd, self).__init__

    class task(SetDict2Class):
        cfg = 'BIRL'

    class action(SetDict2Class):
        action_limit_up  = None
        action_limit_low = None

        high_ranges = [3.] * 2 + [1.] * 10
        low_ranges  = [0.5] * 2 + [-1.] * 10

        ref_joint_pos   = [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]
        use_increment   = True
        inc_high_ranges = [3.5] * 2 + [15.] * 10
        inc_low_ranges  = [0.5] * 2 + [-15.] * 10

        use_actuator_delay          = True
        use_actuator_filter         = True

    class command(SetDict2Class):
        curriculum        = False
        num_commands      = 3
        resampling_time   = 5.
        heading_command   = False
        use_heading_reward = False

        # Forward/backward only — zero lateral and yaw
        lin_vel_x_range   = [-0.3, 0.7]
        lin_vel_y_range   = [0., 0.]
        ang_vel_yaw_range = [0., 0.]

    # Note: reward weights are hardcoded in birl_task.py, not configurable here.
    # With cmd_yaw=0 and cmd_vy=0, the existing yaw_rat and lateral_vel rewards
    # naturally penalize any drift — no extra tuning needed.
