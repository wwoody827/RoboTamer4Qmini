import numpy as np
from math import pi


class SetDict2Class():
    def set_dict(self, dict):
        for key, value in dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Base:
    def __init__(self):
        super(Base, self).__init__

    class task(SetDict2Class):
        cfg = 'Base'

    class viewer:
        pos = [0, 0, 0.6]  # [m]
        lookat = [0., 0, 0.6]  # [m]
        fixed_robot_id = 0
        fixed_offset = [0., 2., 0.6]

    class runner(SetDict2Class):
        seed = 1
        max_iterations = 5000  # number of policy updates
        num_steps_per_env = 24  # 24  # per iteration
        save_interval = 200  # check for potential saves every this many iterations
        num_envs = 4096
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 10  # episode length in seconds

    class policy(SetDict2Class):
        name = 'simple_policy'
        num_actions = None
        num_critic_obs = None
        num_observations = None
        hidden_layers = (512, 256)
        activation = 'relu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm():
        value_loss_coef = 1.
        use_clipped_value_loss = True
        eps_clip = 0.2
        entropy_coef = 0.0005
        num_learning_epochs = 3
        num_mini_batches = 4  # mini batch size = num_envs*n_steps / n_minibatches #
        learning_rate = 1e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        discount_factor = 0.995
        gae_lambda = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class action(SetDict2Class):
        action_limit_up = None
        action_limit_low = None

        high_ranges = [3.] * 2 + [1.] * 10
        low_ranges = [0.5] * 2 + [-1.] * 10

        ref_joint_pos = [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]

        use_increment = True
        inc_high_ranges = [10.] * 10
        inc_low_ranges = [-10.] * 10

    class pd_gains(SetDict2Class):
        decimation = 15
        stiffness = {'hip_yaw': 55., 'hip_roll': 105., 'hip_pitch': 75., 'knee': 45., 'ankle': 30.}
        damping = {'hip_yaw': 0.3, 'hip_roll': 2.5, 'hip_pitch': 0.3, 'knee': 0.5, 'ankle': 0.25}

    class init_state(SetDict2Class):
        random_rot = True
        num_legs = 2
        pos = [0., 0., 0.45]  # x,y,z [m]
        rot = [0., 0., 0., 1.]  # x,y,z,w [quat]
        lin_vel = [0.] * 3  # x,y,z [m/s]
        ang_vel = [0.] * 3  # x,y,z [rad/s]
        reset_joint_pos = [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]

    class domain_rand(SetDict2Class):
        randomize_friction = True
        friction_range = [0.2, 3.0]
        rest_offset_range = [0.0, 0.0]  # no surface sinking simulation
        randomize_mass = True
        added_mass_range = [0.5, 1.5]
        added_inertia_range = [0.5, 1.5]
        randomize_damping = True
        added_friction_range = [0.8, 1.2]
        added_damping_range = [0.8, 1.2]
        randomize_torque = True
        torque_range = [0.8, 1.2]
        randomize_gains = True
        gains_range = [0.8, 1.2]
        push_robots = True
        max_push_vel_xy = 0.5
        max_push_rate_xyz = 0.5
        push_interval_s = 3.
        delay_observation = True
        delay_joint_ranges = [10, 40]  # for q, dq
        delay_rate_ranges = [20, 50]  # for angle velocity
        delay_angle_ranges = [20, 50]  # base euler and base linear velocity

    class noise_values(SetDict2Class):
        randomize_noise = True
        use_state_filter = False
        lin_vel = 0.3
        gravity = 0.15
        ang_vel = 0.3
        foot_frc = 5.
        dof_pos = 0.1
        dof_vel = 1.2
        base_acc = 3.

    class command(SetDict2Class):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 5.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        lin_vel_x_range = [-0.3, 0.7]
        lin_vel_y_range = [-0.3, 0.3]
        ang_vel_yaw_range = [-1, 1]
        heading_range = [0., pi]

    class terrain(SetDict2Class):
        mesh_type = 'trimesh'  # none, plane, trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.01  # [m]
        border_size = 5  # [m]
        static_friction = 1.
        dynamic_friction = 1.
        restitution = 0.05
        # rough terrain only:
        measured_points_x = [-0.05, 0, 0.05]
        measured_points_y = [-0.05, 0, 0.05]
        curriculum = False
        measure_heights = False
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 3  # starting curriculum state
        terrain_length = 15
        terrain_width = 15
        num_rows = 20  # number of terrain rows (levels)  15
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [1., 0., 0., 0., 0.]
        # trimesh only:
        slope_treshold = 0.0  # slopes above this threshold will be corrected to vertical surfaces

    class sim:
        dt = 0.001  # 0.001
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z
        gravity = [0., 0., -9.81]  # [m/s^2]

        class physx:
            solver_type = 1  # 0: pgs, 1: tgs
            num_threads = 10
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            use_gpu = True

    class asset:
        enable_bar = False
        file = "assets/q1/urdf/q1.urdf"

        imu_name = "imu_in_torso"
        foot_name = ['ankle_pitch']
        base_name = "base_link"
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["knee", "base", "hip"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot == on rack
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up
        use_mesh_materials = True  # color!!! False have color

        density = 0.1
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.
        thickness = 0.01
