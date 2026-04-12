"""
MIRL config — Motion Imitation RL for Qmini.

Differences from BIRL:
  - task.cfg = 'MIRLTask'            (new task class)
  - action: 10-dim increments       (no leg-frequency outputs)
  - command.num_commands = 8        (forward-compatible padding)
    active:   [0] vx, [1] vy, [2] yaw, [3] height (reserved, zero for now)
    reserved: [4-7] zero
"""

import numpy as np
from math import pi

from .Base import SetDict2Class, Base


class MIRL(Base):
    def __init__(self):
        super(MIRL, self).__init__

    class task(SetDict2Class):
        cfg = 'MIRL'  # load_task_cls appends "Task" → resolves to MIRLTask

    class action(SetDict2Class):
        action_limit_up = None
        action_limit_low = None

        # Dummy prefix for train.py compat (low_ranges[2:] → 10 joint elements)
        high_ranges = [3.] * 2 + [1.] * 10
        low_ranges  = [0.5] * 2 + [-1.] * 10

        # Increment ranges — wider than BIRLTask to allow freer exploration
        use_increment = True
        inc_high_ranges = [15.] * 10
        inc_low_ranges = [-15.] * 10

        # Standing pose (same as BIRL / Base)
        ref_joint_pos = [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]

    class command(SetDict2Class):
        curriculum = False
        max_curriculum = 1.
        # 8 slots: [vx, vy, yaw, height(reserved), 0, 0, 0, 0]
        num_commands = 8
        resampling_time = 5.     # seconds between command resamples
        heading_command = False

        lin_vel_x_range = [-0.3, 0.7]
        lin_vel_y_range = [-0.3, 0.3]
        ang_vel_yaw_range = [-1.0, 1.0]
        heading_range = [0., pi]
