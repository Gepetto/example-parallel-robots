import pinocchio as pin
from .loader_tools import load
from toolbox_parallel_robots import freezeJoints


def talosOnlyLeg():
    new_model, constraint_models, actuation_model, visual_model, collision_model = load(
        "talos_full_closed"
    )

    names_joints_to_lock = [
        # 'universe',
        # 'root_joint',
        "torso_1_joint",
        "torso_2_joint",
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        # 'arm_left_4_joint',
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
        "gripper_left_inner_double_joint",
        "gripper_left_fingertip_1_joint",
        "gripper_left_fingertip_2_joint",
        "gripper_left_inner_single_joint",
        "gripper_left_fingertip_3_joint",
        "gripper_left_joint",
        "gripper_left_motor_single_joint",
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        # 'arm_right_4_joint',
        "arm_right_5_joint",
        "arm_right_6_joint",
        "arm_right_7_joint",
        "gripper_right_inner_double_joint",
        "gripper_right_fingertip_1_joint",
        "gripper_right_fingertip_2_joint",
        "gripper_right_inner_single_joint",
        "gripper_right_fingertip_3_joint",
        "gripper_right_joint",
        "gripper_right_motor_single_joint",
        "head_1_joint",
        "head_2_joint",
    ]

    ids_joints_to_lock = [
        i for (i, n) in enumerate(new_model.names) if n in names_joints_to_lock
    ]
    q0 = pin.neutral(new_model)
    (
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    ) = freezeJoints(
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
        ids_joints_to_lock,
        q0,
    )
    return (
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    )
