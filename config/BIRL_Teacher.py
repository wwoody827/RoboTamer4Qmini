"""
BIRL Teacher config — privileged observations for better reference clip recording.

Extends BIRL_Fwd with base_lin_vel appended to the actor observation (44 → 47 dims/step).
The Teacher can see its true velocity, allowing it to self-correct lateral drift and walk
straighter than the standard policy.

Usage:
    python train.py --config BIRL_Teacher --name birl_teacher_v1 \
        --sim2sim_interval 500 --num_envs 4096

Sim2sim (uses qmini_birl_teacher.yaml which sets num_obs_per_step=47):
    python deploy/sim2sim/sim2sim.py \
        --policy experiments/birl_teacher_v1/deploy/policy_2000.onnx \
        --config deploy/sim2sim/configs/qmini_birl_teacher.yaml \
        --cmd_vx 0.5 --cmd_vy 0.0 --cmd_yaw 0.0
"""
from .BIRL_Fwd import BIRL_Fwd
from .Base import SetDict2Class


class BIRL_Teacher(BIRL_Fwd):
    def __init__(self):
        super(BIRL_Teacher, self).__init__

    class task(SetDict2Class):
        cfg = 'BIRL'
        use_teacher_obs = True   # append base_lin_vel → 47-dim obs/step × 3 = 141 total
