import numpy as np
import example_robot_data as robex
import pinocchio as pin
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import hppfcl
import re
from toolbox_parallel_robots import ActuationModel, freezeJoints

import sandbox_pinocchio_parallel_robots as sppr

def TalosClosed(closed_loop=True, only_legs=True, free_flyer=True):
    """
    Create the talos robot with closed loop constraints
    """
    # In this code, the robot is loaded from example robot data and is modified to add the closed loop constraints
    # The closed loop is defined to form a parallelogram named ABCD in the code 
    # A being the motor in the calf, B being the free ankle joint
    # C being the rod attach point on the foot and D being the joint between the motor rod and the free rod
    robot = robex.load("talos")
    model = robot.model
    visual_model = robot.visual_model
    collision_model = robot.collision_model
    if not free_flyer:
        model, [visual_model, collision_model] = pin.buildReducedModel(
            model, [visual_model, collision_model], [1], pin.neutral(model)
        )
        id_A_parent_left = id_B_parent_left  = 4
        id_A_parent_right = id_B_parent_right = 10
        id_B_left = id_parent_C_left = 5
        id_B_right = id_parent_C_right = 11
    else:
        id_A_parent_left = id_B_parent_left  = 5
        id_A_parent_right = id_B_parent_right = 11
        id_B_left = id_parent_C_left = 6
        id_B_right = id_parent_C_right = 12

    I4 = pin.SE3.Identity()
    inertia = pin.Inertia(
        1e-3, np.array([0.0, 0.0, 0.0]), np.eye(3) * 1e-3**2
    )  # inertia of a 1g sphere

    # * Creation of free bar on foot B
    # Defining placement wrt to the parent joints
    BMC_right = I4.copy()
    BMC_right.translation = np.array([-0.08, -0.105, 0.02])
    BMC_left = I4.copy()
    BMC_left.translation = np.array([-0.08, 0.105, 0.02])
    # * Creation of the motor joint A
    kneeMA_right = I4.copy()
    kneeMA_right.translation = np.array([-0.015, -0.105, -0.11])
    kneeMA_left = I4.copy()
    kneeMA_left.translation = np.array([-0.015, 0.105, -0.11])

    ## Get the length of the rods
    # * Rod DC
    kneeMB_right = model.jointPlacements[id_B_right]
    kneeMB_left = model.jointPlacements[id_B_left]
    AMB_left = kneeMA_left.inverse() * kneeMB_left
    AMB_right = kneeMA_right.inverse() * kneeMB_right

    bar_length = np.linalg.norm(AMB_left.translation[[0, 2]])
    assert np.allclose(bar_length, np.linalg.norm(AMB_right.translation[[0, 2]]))

    # * Rod AD
    mot_bar_length = np.linalg.norm(BMC_left.translation[[0, 2]])
    assert np.allclose(mot_bar_length, np.linalg.norm(BMC_right.translation[[0, 2]]))
    bar_width = 1.5e-2
    thickness = 1e-2

    # * Adding corresponding visual and collision models
    alpha = 0.5
    bar_length_bottom = bar_length * alpha
    bar_length_top = bar_length * (1 - alpha)

    ## Adding new joints and links for the parallel actuation
    # Adding new joints
    id_A_right = model.addJoint(
        id_A_parent_right, pin.JointModelRY(), kneeMA_right, "mot_calf_right"
    )
    id_A_left = model.addJoint(
        id_A_parent_left, pin.JointModelRY(), kneeMA_left, "mot_calf_left"
    )

    # Adding corresponding Geometry objects using hpp-fcl
    bar_mot = hppfcl.Box(thickness, bar_width, mot_bar_length)
    AMbar = I4.copy()
    AMbar.translation = np.array([0, 0, mot_bar_length / 2])
    bar_mot_right = pin.GeometryObject("mot_calf_right", id_A_right, AMbar, bar_mot)
    color = [1, 0, 0, 1]
    bar_mot_right.meshColor = np.array(color)
    bar_mot_left = pin.GeometryObject("mot_calf_left", id_A_left, AMbar, bar_mot)
    color = [0, 1, 0, 1]
    bar_mot_left.meshColor = np.array(color)
    visual_model.addGeometryObject(bar_mot_right)
    visual_model.addGeometryObject(bar_mot_left)

    DMFrame = I4.copy() # Placement of the contact frame of the rod wrt the u joint
    DMFrame.translation = np.array([0, 0, bar_length_top])
    CMFrame = I4.copy() # Placement of the contact frame of the rod wrt the u joint
    CMFrame.translation = np.array([0, 0, bar_length_bottom])

    # * Create free joint linked to previous motor bar
    # Defining placements wrt parent joints
    AMD_left = I4.copy()
    AMD_left.translation = np.array([0, 0, mot_bar_length])
    AMD_right = I4.copy()
    AMD_right.translation = np.array([0, 0, mot_bar_length])
    # Adding joints
    id_D_right_X = model.addJoint(
        id_A_right,
        pin.JointModelRX(),
        AMD_right,
        "free_calf_right_X",
    )
    id_D_right = model.addJoint(
        id_D_right_X,
        pin.JointModelRY(),
        pin.SE3.Identity(),
        "free_calf_right_Y",
    )

    id_D_left_X = model.addJoint(
        id_A_left,
        pin.JointModelRX(),
        AMD_left,
        "free_calf_left_X",
    )
    id_D_left = model.addJoint(
        id_D_left_X,
        pin.JointModelRY(),
        pin.SE3.Identity(),
        "free_calf_left_Y",
    )

    # Adding bodies to joints with no displacement
    model.appendBodyToJoint(id_D_right, inertia, DMFrame)
    model.appendBodyToJoint(id_D_left, inertia, DMFrame)
    # * Adding the joints on C
    id_C_right = model.addJoint(
        id_parent_C_right,
        pin.JointModelSpherical(),
        BMC_right,
        "free_ankle_spherical_right",
    )
    id_C_left = model.addJoint(
        id_parent_C_left,
        pin.JointModelSpherical(),
        BMC_left,
        "free_ankle_spherical_left",
    )
    # Adding bodies to joints with no displacement
    model.appendBodyToJoint(id_C_right, inertia, CMFrame)
    model.appendBodyToJoint(id_C_left, inertia, CMFrame)
    # Adding corresponding visual and collision model
    bar_free = hppfcl.Box(thickness, bar_width, bar_length_top)
    DMrod = I4.copy()
    DMrod.translation = np.array([0, 0, bar_length_top / 2])
    half_rod_top_right = pin.GeometryObject(
        "half_rod_calf_right", id_D_right, DMrod, bar_free
    )
    color = [0, 0, 1, 1]
    half_rod_top_right.meshColor = np.array(color)
    half_rod_top_left = pin.GeometryObject(
        "half_rod_calf_left", id_D_left, DMrod, bar_free
    )
    color = [0, 1, 1, 1]
    half_rod_top_left.meshColor = np.array(color)
    visual_model.addGeometryObject(half_rod_top_right)
    visual_model.addGeometryObject(half_rod_top_left)
    #
    bar_free = hppfcl.Box(thickness, bar_width, bar_length_bottom)
    CMrod = I4.copy()
    CMrod.translation = np.array([0, 0, bar_length_bottom / 2])
    half_rod_bottom_right = pin.GeometryObject("half_rod_right", id_C_right, CMrod, bar_free)
    color = [1, 1, 0, 1]
    half_rod_bottom_right.meshColor = np.array(color)
    half_rod_bottom_left = pin.GeometryObject("half_rod_left", id_C_left, CMrod, bar_free)
    color = [0, 1, 0, 1]
    half_rod_bottom_left.meshColor = np.array(color)
    visual_model.addGeometryObject(half_rod_bottom_right)
    visual_model.addGeometryObject(half_rod_bottom_left)

    # * Create the frames corresponding to the closed loop contacts
    closure_right_A = pin.Frame(
        "closure_right_A", id_D_right, DMFrame, pin.OP_FRAME
    )
    closure_left_A = pin.Frame(
        "closure_left_A", id_D_left, DMFrame, pin.OP_FRAME
    )
    closure_right_B = pin.Frame(
        "closure_right_B", id_C_right, CMFrame, pin.OP_FRAME
    )
    closure_left_B = pin.Frame(
        "closure_left_B", id_C_left, CMFrame, pin.OP_FRAME
    )
    model.addFrame(closure_right_A)
    model.addFrame(closure_right_B)
    model.addFrame(closure_left_A)
    model.addFrame(closure_left_B)

    constraint_models = []

    # * Create the new model
    new_model = pin.Model()
    new_model.name = "talos_closed"  # Defining the model name
    # Renaming the non-actuated joints
    for jp, iner, name, i, jm in zip(
        model.jointPlacements[1:],
        model.inertias[1:],
        model.names[1:],
        model.parents[1:],
        model.joints[1:],
    ):
        match1 = re.search("leg", name)
        match2 = re.search("5", name)
        match3 = re.search("mot", name)
        match4 = re.search("free", name)
        if match1 and match2:
            name = "free_" + name
        elif not (match3) and not (match4):
            name = "mot_" + name
        jid = new_model.addJoint(i, jm, jp, name)
        new_model.appendBodyToJoint(jid, iner, pin.SE3.Identity())

    # Adding new frames
    # ? Is this really necessary or can we just frame.copy() ?
    for frame in model.frames[1:]:
        name = frame.name
        parent_joint = (
            frame.parentJoint
        )  # Parent joints for frames may be incorrect dur to the changes in the joints order
        placement = frame.placement
        frame = pin.Frame(name, parent_joint, placement, pin.BODY)
        new_model.addFrame(frame, False)
    # Define new neutral configuration
    q0 = pin.neutral(new_model)

    # Define current actuation model (will change)
    actuation_model = ActuationModel(new_model, ["mot"])
    # * Freezing required joints
    # Use only legs (ie freeze top joints)
    if only_legs:
        print("Freezing upper body")
        names_joints_to_lock = [
            # "universe",
            # "root_joint",
            "mot_arm_left_1_joint",
            "mot_arm_left_2_joint",
            "mot_arm_left_3_joint",
            # "arm_left_4_joint",
            "mot_arm_left_5_joint",
            "mot_arm_left_6_joint",
            "mot_arm_left_7_joint",
            "mot_arm_right_1_joint",
            "mot_arm_right_2_joint",
            "mot_arm_right_3_joint",
            # "arm_right_4_joint",
            "mot_arm_right_5_joint",
            "mot_arm_right_6_joint",
            "mot_arm_right_7_joint",
            "mot_gripper_left_joint",
            "mot_gripper_right_joint",
            "mot_head_1_joint",
            "mot_head_2_joint",
            "mot_torso_1_joint",
            "mot_torso_2_joint",
        ]
        ids_joints_to_lock = [
            i for (i, n) in enumerate(new_model.names) if n in names_joints_to_lock
        ]
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
        q0 = pin.neutral(new_model)

    # Freeze joints if required
    if not closed_loop:  # If we want to consider the robot in open loop
        new_model.name = "talos_open"
        print("Freezing closed loop joints")
        # Remove inertia from the rod and add it back in the foot
        new_model.inertias[new_model.getJointId("free_calf_right_Y")] = pin.Inertia.Zero()
        new_model.inertias[new_model.getJointId("free_calf_left_Y")] = pin.Inertia.Zero()
        new_model.appendBodyToJoint(
            [i for i,n in enumerate(new_model.names) if model.names[id_B_right] in n][0], 
            inertia, 
            BMC_right
        )
        new_model.appendBodyToJoint(
            [i for i,n in enumerate(new_model.names) if model.names[id_B_left] in n][0], 
            inertia, 
            BMC_left
        )
        names_joints_to_lock = [
            "mot_calf_right",
            "mot_calf_left",
            "free_ankle_spherical_right",
            "free_ankle_spherical_left",
            "free_calf_right_X",
            "free_calf_right_Y",
            "free_calf_left_X",
            "free_calf_left_Y",
        ]
        ids_joints_to_lock = [
            i for (i, n) in enumerate(new_model.names) if n in names_joints_to_lock
        ]
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
        q0 = pin.neutral(new_model)
        # Retransform free joints into actuated joints
        new_model.names[
            new_model.getJointId("free_leg_right_5_joint")
        ] = "mot_leg_right_5_joint"
        new_model.names[
            new_model.getJointId("free_leg_left_5_joint")
        ] = "mot_leg_left_5_joint"
        actuation_model = ActuationModel(new_model, ["mot"])

    else:  # if we consider closed loop
        # Create constraint models
        closure_right_A = new_model.frames[new_model.getFrameId("closure_right_A")]
        closure_right_B = new_model.frames[new_model.getFrameId("closure_right_B")]
        closure_left_A = new_model.frames[new_model.getFrameId("closure_left_A")]
        closure_left_B = new_model.frames[new_model.getFrameId("closure_left_B")]
        inverse_rot = pin.SE3.Identity()
        inverse_rot.rotation[1, 1] = -1; inverse_rot.rotation[2, 2] = -1
        constraint_right = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            new_model,
            closure_right_A.parentJoint,
            inverse_rot.act(closure_right_A.placement.inverse()),
            closure_right_B.parentJoint,
            closure_right_B.placement,
            pin.ReferenceFrame.LOCAL,
        )
        constraint_right.name = "Ankle_loop_right"
        constraint_left = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            new_model,
            closure_left_A.parentJoint,
            inverse_rot.act(closure_left_A.placement.inverse()),
            closure_left_B.parentJoint,
            closure_left_B.placement,
            pin.ReferenceFrame.LOCAL,
        )
        constraint_left.name = "Ankle_loop_left"
        constraint_models = [constraint_right, constraint_left]

    ## Define contact Ids for control problems - these are floor contacts
    contact_ids = [i for i, f in enumerate(new_model.frames) if "sole_link" in f.name]
    soot_size_X = 0.1
    soot_size_Y = 0.05

    ## Adding new frames for control problems
    for cid in contact_ids:
        f = new_model.frames[cid]
        rootName = f.name.partition("sole_link")[0]
        for side in {"left", "right"}:
            new_model.addFrame(
                pin.Frame(
                    f"{rootName}tow_{side}",
                    f.parentJoint,
                    f.parentFrame,
                    f.placement
                    * pin.SE3(
                        np.eye(3),
                        np.array(
                            [
                                soot_size_X,
                                soot_size_Y * (1 if side == "left" else -1),
                                0,
                            ]
                        ),
                    ),
                    pin.FrameType.OP_FRAME,
                )
            )
            new_model.addFrame(
                pin.Frame(
                    f"{rootName}heel_{side}",
                    f.parentJoint,
                    f.parentFrame,
                    f.placement
                    * pin.SE3(
                        np.eye(3),
                        np.array(
                            [
                                -soot_size_X,
                                soot_size_Y * (1 if side == "left" else -1),
                                0,
                            ]
                        ),
                    ),
                    pin.FrameType.OP_FRAME,
                )
            )
    new_model, geometry_models, constraint_models = sppr.reorganizeModels(
        new_model, [visual_model, collision_model], constraint_models
    )
    # Actuation models
    actuation_model = ActuationModel(
        new_model, [n for n in new_model.names if "mot" in n]
    )
    visual_model, collision_model = geometry_models[0], geometry_models[1]

    return (
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    )


if __name__ == "__main__":
    from vizutils import *
    model, cm, am, visual_model, collision_model = TalosClosed(
        closed_loop=True, only_legs=True
    )
    q0 = pin.neutral(model)
    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
    viz.display(q0)
    data = model.createData()
    cd = [c.createData() for c in cm]
    from toolbox_parallel_robots.mounting import closedLoopMountScipy
    closedLoopMountScipy(model, data, cm, cd, q0)

    from toolbox_parallel_robots.foo import createSlidersInterface
    createSlidersInterface(model, cm, visual_model, am.mot_ids_q, viz, q0=q0)
    #
    # input()
    # q0 = pin.neutral(new_model)
    # viz = MeshcatVisualizer(new_model, new_visual_model, new_visual_model)
    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.clean()
    # viz.loadViewerModel(rootNodeName="universe")
    # viz.display(q0)
