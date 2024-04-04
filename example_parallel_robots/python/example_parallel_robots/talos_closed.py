import numpy as np
import example_robot_data as robex
import pinocchio as pin
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import hppfcl
import re
from .actuation_model import ActuationModel
from .freeze_joints import freezeJoints


def reorganizeModelDepthFirst(model):
    def propagate(stack, new_model, i):
        if len(stack) == 0:
            return new_model
        if i == 500:
            raise (RecursionError("Reached max depth when reorganizing the model"))
        else:
            (jointId, parentId) = stack.pop()
            jId = new_model.addJoint(
                parentId,  #
                model.joints[jointId],  #
                model.jointPlacements[jointId],  #
                model.names[jointId],
            )  #
            new_model.appendBodyToJoint(
                jId,
                model.inertias[jointId],
                pin.SE3.Identity(),  #
            )  #
            children = model.children[jointId]
            for c in children:
                stack.append((c, jId))
        propagate(stack, new_model, i + 1)

    new_model = pin.Model()
    new_model.name = model.name
    propagate([(1, 0)], new_model, 0)
    return new_model


def reorganizeModels(old_model, old_geometry_models, constraint_models):
    # Model
    model = reorganizeModelDepthFirst(old_model)
    # Frames
    for frame in old_model.frames[1:]:
        name = frame.name
        parent_joint = model.getJointId(
            old_model.names[frame.parentJoint]
        )  # Should be a joint Id
        placement = frame.placement
        frame = pin.Frame(name, parent_joint, placement, pin.BODY)
        model.addFrame(frame, False)
    # Geometry model
    geometry_models = []
    for old_geom_model in old_geometry_models:
        geom_model = old_geom_model.copy()
        for gm in geom_model.geometryObjects:
            gm.parentJoint = model.getJointId(old_model.names[gm.parentJoint])
            gm.parentFrame = model.getFrameId(old_model.frames[gm.parentFrame].name)
        geometry_models.append(geom_model)
    # Constraint models
    new_constraint_models = []
    for cm in constraint_models:
        new_constraint_models.append(
            pin.RigidConstraintModel(
                cm.type,
                model,
                model.getJointId(old_model.names[cm.joint1_id]),
                cm.joint1_placement,
                model.getJointId(old_model.names[cm.joint2_id]),
                cm.joint2_placement,
                pin.ReferenceFrame.LOCAL,
            )
        )
    return (model, geometry_models, new_constraint_models) 


def TalosClosed(closed_loop=True, only_legs=True, free_flyer=True):
    robot = robex.load("talos")
    model = robot.model
    visual_model = robot.visual_model
    collision_model = robot.collision_model
    if not free_flyer:
        model, [visual_model, collision_model] = pin.buildReducedModel(
            model, [visual_model, collision_model], [1], pin.neutral(model)
        )
        id_knee_left = 4
        id_knee_right = 10
        id_ankle_left = 5
        id_ankle_right = 11
    else:
        id_knee_left = 5
        id_knee_right = 11
        id_ankle_left = 6
        id_ankle_right = 12

    I4 = pin.SE3.Identity()
    inertia = pin.Inertia(
        1e-3, np.array([0.0, 0.0, 0.0]), np.eye(3) * 1e-3**2
    )  # inertia of a 1g sphere

    # * Creation of free bar on foot
    # Defining placement wrt to the parent joints
    ankleMrodright = I4.copy()
    ankleMrodright.translation = np.array([-0.08, -0.105, 0.02])
    ankleMrodleft = I4.copy()
    ankleMrodleft.translation = np.array([-0.08, 0.105, 0.02])
    # Adding the joints
    rod_right_X = model.addJoint(
        id_ankle_right,
        pin.JointModelRX(),
        ankleMrodright,
        "free_rod_right_X",
    )
    rod_right_Y = model.addJoint(
        rod_right_X,
        pin.JointModelRY(),
        pin.SE3.Identity(),
        "free_rod_right_Y",
    )
    rod_left_X = model.addJoint(
        id_ankle_left,
        pin.JointModelRX(),
        ankleMrodleft,
        "free_rod_left_X",
    )
    rod_left_Y = model.addJoint(
        rod_left_X,
        pin.JointModelRY(),
        pin.SE3.Identity(),
        "free_rod_left_Y",
    )
    # Adding bodies to joints with no displacement
    model.appendBodyToJoint(rod_right_Y, inertia, I4)
    model.appendBodyToJoint(rod_left_Y, inertia, I4)
    # Adding corresponding visual and collision models
    free_bar_length = 10e-2
    free_bar_width = 1.5e-2
    thickness = 1e-2
    bar_free = hppfcl.Box(thickness, free_bar_width, free_bar_length)
    jMbar = I4.copy()
    jMbar.translation = np.array([0, 0, free_bar_length / 2])
    half_rod_right = pin.GeometryObject("half_rod_right", rod_right_Y, jMbar, bar_free)
    color = [1, 0, 0, 1]
    half_rod_right.meshColor = np.array(color)
    half_rod_left = pin.GeometryObject("half_rod_left", rod_left_Y, jMbar, bar_free)
    color = [0, 1, 0, 1]
    half_rod_left.meshColor = np.array(color)
    visual_model.addGeometryObject(half_rod_right)
    visual_model.addGeometryObject(half_rod_left)

    ## Adding new joints and links for the parallel actuation
    # * Creation of the motor bar
    # Defining SE3 placements of new joints wrt parent joints
    kneeMcalfright = I4.copy()
    kneeMcalfright.translation = np.array([-0.015, -0.105, -0.11])
    kneeMcalfleft = I4.copy()
    kneeMcalfleft.translation = np.array([-0.015, 0.105, -0.11])
    # Adding new joints
    mot_calf_right = model.addJoint(
        id_knee_right, pin.JointModelRY(), kneeMcalfright, "mot_calf_right"
    )
    mot_calf_left = model.addJoint(
        id_knee_left, pin.JointModelRY(), kneeMcalfleft, "mot_calf_left"
    )
    # Adding bodies to new joints with no displacement
    model.appendBodyToJoint(mot_calf_right, inertia, I4)
    model.appendBodyToJoint(mot_calf_left, inertia, I4)
    # Adding corresponding Geometry objects using hpp-fcl
    mot_bar_length = 8e-2
    mot_bar_width = 1.5e-2
    thickness = 1e-2
    bar_mot = hppfcl.Box(thickness, mot_bar_width, mot_bar_length)
    jMbar = I4.copy()
    jMbar.translation = np.array([0, 0, mot_bar_length / 2])
    bar_mot_right = pin.GeometryObject("mot_calf_right", mot_calf_right, jMbar, bar_mot)
    color = [1, 0, 0, 1]
    bar_mot_right.meshColor = np.array(color)
    bar_mot_left = pin.GeometryObject("mot_calf_left", mot_calf_left, jMbar, bar_mot)
    color = [0, 1, 0, 1]
    bar_mot_left.meshColor = np.array(color)
    visual_model.addGeometryObject(bar_mot_right)
    visual_model.addGeometryObject(bar_mot_left)

    # * Create free joint linked to previous motor bar
    # Defining placements wrt parent joints
    calfMfreejointright = I4.copy()
    calfMfreejointright.translation = np.array([0, 0, mot_bar_length])
    calfMfreejointleft = I4.copy()
    calfMfreejointleft.translation = np.array([0, 0, mot_bar_length])
    # Adding joints
    free_calf_right = model.addJoint(
        mot_calf_right,
        pin.JointModelSpherical(),
        calfMfreejointright,
        "free_calf_right",
    )
    free_calf_left = model.addJoint(
        mot_calf_left,
        pin.JointModelSpherical(),
        calfMfreejointleft,
        "free_calf_left",
    )
    # Adding bodies to joints with no displacement
    model.appendBodyToJoint(free_calf_right, inertia, I4)
    model.appendBodyToJoint(free_calf_left, inertia, I4)
    # Adding corresponding visual and collision model
    free_bar_length = 10e-2
    free_bar_width = 1.5e-2
    thickness = 1e-2
    bar_free = hppfcl.Box(thickness, free_bar_width, free_bar_length)
    half_rod_calf_right = pin.GeometryObject(
        "half_rod_calf_right", free_calf_right, jMbar, bar_free
    )
    half_rod_calf_left = pin.GeometryObject(
        "half_rod_calf_left", free_calf_left, jMbar, bar_free
    )
    visual_model.addGeometryObject(half_rod_calf_right)
    visual_model.addGeometryObject(half_rod_calf_left)

    # * Create the frames corresponding to the closed loop contacts
    Rx = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 0]))
    fplacement = I4.copy()
    fplacement.translation = np.array([0, 0, free_bar_length])
    closure_right_A = pin.Frame(
        "closure_right_A", free_calf_right, fplacement * Rx, pin.OP_FRAME
    )
    closure_left_A = pin.Frame(
        "closure_left_A", free_calf_left, fplacement * Rx, pin.OP_FRAME
    )
    closure_right_B = pin.Frame(
        "closure_right_B", rod_right_Y, fplacement, pin.OP_FRAME
    )
    closure_left_B = pin.Frame("closure_left_B", rod_left_Y, fplacement, pin.OP_FRAME)
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
        names_joints_to_lock = [
            "mot_calf_right",
            "mot_calf_left",
            "free_rod_right_X",
            "free_rod_right_Y",
            "free_rod_left_X",
            "free_rod_left_Y",
            "free_calf_right",
            "free_calf_left",
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
        new_model.names[new_model.getJointId("free_leg_right_5_joint")] = (
            "mot_leg_right_5_joint"
        )
        new_model.names[new_model.getJointId("free_leg_left_5_joint")] = (
            "mot_leg_left_5_joint"
        )
        actuation_model = ActuationModel(new_model, ["mot"])

    else:  # if we consider closed loop
        # Create constraint models
        closure_right_A = new_model.frames[new_model.getFrameId("closure_right_A")]
        closure_right_B = new_model.frames[new_model.getFrameId("closure_right_B")]
        closure_left_A = new_model.frames[new_model.getFrameId("closure_left_A")]
        closure_left_B = new_model.frames[new_model.getFrameId("closure_left_B")]
        constraint_right = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            new_model,
            closure_right_A.parentJoint,
            closure_right_A.placement,
            closure_right_B.parentJoint,
            closure_right_B.placement,
            pin.ReferenceFrame.LOCAL,
        )
        constraint_right.name = "Ankle_loop_right"
        constraint_left = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            new_model,
            closure_left_A.parentJoint,
            closure_left_A.placement,
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
    new_model, geometry_models, constraint_models = reorganizeModels(
        new_model, [visual_model, collision_model], constraint_models
    )
    # Actuation models
    actuation_model = ActuationModel(new_model, [n for n in new_model.names if "mot" in n])
    visual_model, collision_model = geometry_models[0], geometry_models[1]

    return (
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    )


if __name__ == "__main__":
    model, cm, am, visual_model, collision_model = TalosClosed(
        closed_loop=True, only_legs=True
    )
    # new_model, geometry_models, cm, am = reorganizeModels(model, [visual_model, collision_model], cm)
    # new_visual_model, new_collision_model = geometry_models[0], geometry_models[1]
    #
    q0 = pin.neutral(model)
    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
    viz.display(q0)
    #
    # input()
    # q0 = pin.neutral(new_model)
    # viz = MeshcatVisualizer(new_model, new_visual_model, new_visual_model)
    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.clean()
    # viz.loadViewerModel(rootNodeName="universe")
    # viz.display(q0)
