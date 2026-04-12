"""MIRL expert: sideways walking only (vy, no vx, no yaw)."""
from math import pi
from .Base import SetDict2Class, Base


class MIRL_Strafe(Base):
    def __init__(self):
        super(MIRL_Strafe, self).__init__

    class task(SetDict2Class):
        cfg = 'MIRL'
        ref_clip_paths = []

    class action(SetDict2Class):
        high_ranges   = [3.] * 2 + [1.] * 10
        low_ranges    = [0.5] * 2 + [-1.] * 10
        use_increment = True
        inc_high_ranges = [15.] * 10
        inc_low_ranges  = [-15.] * 10
        ref_joint_pos   = [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]

    class command(SetDict2Class):
        num_commands      = 8
        resampling_time   = 5.
        heading_command   = False
        lin_vel_x_range   = [0.,   0.]    # no forward
        lin_vel_y_range   = [-0.3, 0.3]  # sideways only
        ang_vel_yaw_range = [0.,   0.]    # no turning
        heading_range     = [0.,   pi]
