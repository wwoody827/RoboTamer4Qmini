"""
MIRL combined: full directional control with imitation loss from expert clips.

Usage (after step 1 experts are trained and clips collected):
  1. Collect clips:
       python deploy/sim2sim/sim2sim.py --config deploy/sim2sim/configs/qmini_mirl.yaml \
           --policy experiments/mirl_fwd_v1/deploy/policy_1000.onnx \
           --record data/reference_clips/walk_fwd.npz --record_skill walk --record_loop \
           --headless --duration 10

       (repeat for strafe and turn experts → walk_strafe.npz, walk_turn.npz)

  2. Fill in ref_clip_paths below, then launch:
       python train.py --config MIRL_Combined --name mirl_combined_v1

  3. For MIRL fine-tuning from a pre-trained base, resume:
       python train.py --config MIRL_Combined --name mirl_combined_v1 \
           --resume mirl_base_v1 --max_iterations 3000
"""
from math import pi
from .Base import SetDict2Class, Base


class MIRL_Combined(Base):
    def __init__(self):
        super(MIRL_Combined, self).__init__

    class task(SetDict2Class):
        cfg = 'MIRL'

        # Fill these in after collecting clips from the 3 expert policies.
        # Each .npz is recorded from sim2sim at 67 Hz (policy rate).
        ref_clip_paths = [
            # 'data/reference_clips/walk_fwd.npz',
            # 'data/reference_clips/walk_strafe.npz',
            # 'data/reference_clips/walk_turn.npz',
        ]

        # Imitation weight schedule (applied in reward()):
        #   Start: w_imit=0.8 forces policy to track reference motion closely
        #   End:   w_imit=0.2 lets task reward dominate
        # For now both are 0.5 (balanced); tune after seeing initial behaviour.
        w_imit = 0.5
        w_task = 0.5

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
        lin_vel_x_range   = [-0.3, 0.7]
        lin_vel_y_range   = [-0.3, 0.3]
        ang_vel_yaw_range = [-1.,  1.]
        heading_range     = [0.,   pi]
