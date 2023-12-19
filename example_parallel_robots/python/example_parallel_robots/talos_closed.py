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
        if len(stack)==0:
            return(new_model)
        if i==500:
            raise(RecursionError("Reached max depth when reorganizing the model"))
        else:
            (jointId, parentId) = stack.pop()
            jId = new_model.addJoint(parentId, #
                                     model.joints[jointId], #
                                     model.jointPlacements[jointId], #
                                     model.names[jointId]) #
            new_model.appendBodyToJoint(jId, #
                                        model.inertias[jointId], #
                                        pin.SE3.Identity()) #
            children = model.children[jointId]
            for c in children:
                stack.append((c, jId))
        propagate(stack, new_model, i+1)
    new_model = pin.Model()
    new_model.name = model.name
    propagate([(1, 0)], new_model, 0)
    return(new_model)

def reorganizeModels(old_model, old_geometry_models, constraint_models):
    # Model
    model = reorganizeModelDepthFirst(old_model)
    # Frames
    for frame in old_model.frames[1:]:
        name = frame.name
        parent_joint = model.getJointId(old_model.names[frame.parentJoint]) # Should be a joint Id
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
        new_constraint_models.append(pin.RigidConstraintModel(
                cm.type,
                model,
                model.getJointId(old_model.names[cm.joint1_id]),
                cm.joint1_placement,
                model.getJointId(old_model.names[cm.joint2_id]),  # To the world
                cm.joint2_placement,
                pin.ReferenceFrame.LOCAL,
        ) )
    # Actuation models
    actuation_model = ActuationModel(model, ["mot"])

    return(model, geometry_models, new_constraint_models, actuation_model)

def TalosClosed(closed_loop=True, only_legs=True, free_flyer=True):
    robot = robex.load("talos")
    model = robot.model
    visual_model = robot.visual_model
    collision_model = robot.collision_model
    if not free_flyer:
        model, [visual_model, collision_model] = pin.buildReducedModel(model, [visual_model, collision_model], [1], pin.neutral(model))
        id_genoux_left = 4
        id_genoux_right = 10
        id_cheville_left = 5
        id_cheville_right = 11
    else:
        id_genoux_left = 5
        id_genoux_right = 11
        id_cheville_left = 6
        id_cheville_right = 12
        
    I4 = pin.SE3.Identity()
    Inertia = pin.Inertia(
        1e-3, np.array([0.0, 0.0, 0.0]), np.eye(3) * 1e-3**2
    )  # Inertia of a 1g sphere

    # * Creation of free bar on foot
    # Defining placement wrt to the parent joints
    chevilleMtendonright = I4.copy()
    chevilleMtendonright.translation = np.array([-0.08, -0.105, 0.02])
    chevilleMtendonleft = I4.copy()
    chevilleMtendonleft.translation = np.array([-0.08, 0.105, 0.02])
    # Adding the joints
    tendon_right = model.addJoint(
    id_cheville_right,
    pin.JointModelSpherical(),
    chevilleMtendonright,
    "free_tendon_right",
    )
    tendon_left = model.addJoint(
    id_cheville_left,
    pin.JointModelSpherical(),
    chevilleMtendonleft,
    "free_tendon_left",
    )
    # Adding bodies to joints with no displacement
    model.appendBodyToJoint(tendon_right, Inertia, I4)
    model.appendBodyToJoint(tendon_left, Inertia, I4)
    # Adding corresponding visual and collision models
    longueur_bar_free = 10e-2
    largeur_bar_free = 1.5e-2
    epaisseur = 1e-2
    bar_free = hppfcl.Box(epaisseur, largeur_bar_free, longueur_bar_free)
    jMbarre = I4.copy()
    jMbarre.translation = np.array([0, 0, longueur_bar_free / 2])
    demi_tendon_right = pin.GeometryObject(
        "demi_tendon_right", tendon_right, jMbarre, bar_free
    )
    color = [1, 0, 0, 1]
    demi_tendon_right.meshColor = np.array(color)
    demi_tendon_left = pin.GeometryObject(
        "demi_tendon_left", tendon_left, jMbarre, bar_free
    )
    color = [0, 1, 0, 1]
    demi_tendon_left.meshColor = np.array(color)
    visual_model.addGeometryObject(demi_tendon_right)
    visual_model.addGeometryObject(demi_tendon_left)

    ## Adding new joints and links for the parallel actuation
    # * Creation of the motor bar
    # Defining SE3 placements of new joints wrt parent joints
    genouxMmoletright = I4.copy()
    genouxMmoletright.translation = np.array([-0.015, -0.105, -0.11])
    genouxMmoletleft = I4.copy()
    genouxMmoletleft.translation = np.array([-0.015, 0.105, -0.11])
    # Adding new joints
    mot_molet_right = model.addJoint(
        id_genoux_right, pin.JointModelRY(), genouxMmoletright, "mot_molet_right"
    )
    mot_molet_left = model.addJoint(
        id_genoux_left, pin.JointModelRY(), genouxMmoletleft, "mot_molet_left"
    )
    # Adding bodies to new joints with no displacement
    model.appendBodyToJoint(mot_molet_right, Inertia, I4)
    model.appendBodyToJoint(mot_molet_left, Inertia, I4)
    # Adding corresponding Geometry objects using hpp-fcl
    longueur_bar_mot = 8e-2
    largeur_bar_mot = 1.5e-2
    epaisseur = 1e-2
    bar_mot = hppfcl.Box(epaisseur, largeur_bar_mot, longueur_bar_mot)
    jMbarre = I4.copy()
    jMbarre.translation = np.array([0, 0, longueur_bar_mot / 2])
    barre_mot_right = pin.GeometryObject(
        "mot_molet_right", mot_molet_right, jMbarre, bar_mot
    )
    color = [1, 0, 0, 1]
    barre_mot_right.meshColor = np.array(color)
    barre_mot_left = pin.GeometryObject(
        "mot_molet_left", mot_molet_left, jMbarre, bar_mot
    )
    color = [0, 1, 0, 1]
    barre_mot_left.meshColor = np.array(color)
    visual_model.addGeometryObject(barre_mot_right)
    visual_model.addGeometryObject(barre_mot_left)

    # * Create free joint linked to previous motor bar
    # Defining placements wrt parent joints
    moletMfreejointright = I4.copy()
    moletMfreejointright.translation = np.array([0, 0, longueur_bar_mot])
    moletMfreejointleft = I4.copy()
    moletMfreejointleft.translation = np.array([0, 0, longueur_bar_mot])
    # Adding joints
    free_molet_right = model.addJoint(
        mot_molet_right,
        pin.JointModelSpherical(),
        moletMfreejointright,
        "free_molet_right",
    )
    free_molet_left = model.addJoint(
        mot_molet_left,
        pin.JointModelSpherical(),
        moletMfreejointleft,
        "free_molet_left",
    )
    # Adding bodies to joints with no displacement
    model.appendBodyToJoint(free_molet_right, Inertia, I4)
    model.appendBodyToJoint(free_molet_left, Inertia, I4)
    # Adding corresponding visual and collision model
    longueur_bar_free = 10e-2
    largeur_bar_free = 1.5e-2
    epaisseur = 1e-2
    bar_free = hppfcl.Box(epaisseur, largeur_bar_free, longueur_bar_free)
    demi_tendon_molet_right = pin.GeometryObject(
        "demi_tendon_molet_right", free_molet_right, jMbarre, bar_free
    )
    demi_tendon_molet_left = pin.GeometryObject(
        "demi_tendon_molet_left", free_molet_left, jMbarre, bar_free
    )
    visual_model.addGeometryObject(demi_tendon_molet_right)
    visual_model.addGeometryObject(demi_tendon_molet_left)

    # * Create the frames corresponding to the closed loop contacts
    Rx = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 0]))
    fplacement = I4.copy()
    fplacement.translation = np.array([0, 0, longueur_bar_free])
    fermeture_right_A = pin.Frame(
        "fermeture_right_A", free_molet_right, fplacement * Rx, pin.OP_FRAME
    )
    fermeture_left_A = pin.Frame(
        "fermeture_left_A", free_molet_left, fplacement * Rx, pin.OP_FRAME
    )
    fermeture_right_B = pin.Frame(
        "fermeture_right_B", tendon_right, fplacement, pin.OP_FRAME
    )
    fermeture_left_B = pin.Frame(
        "fermeture_left_B", tendon_left, fplacement, pin.OP_FRAME
    )
    model.addFrame(fermeture_right_A)
    model.addFrame(fermeture_right_B)
    model.addFrame(fermeture_left_A)
    model.addFrame(fermeture_left_B)

    constraint_models = []

    # * Create the new model
    new_model = pin.Model()
    new_model.name = "talos_closed" # Defining the model name
    # Renaming the non-actuated joints
    for jp, iner, name, i, jm in zip(
        model.jointPlacements[1:], model.inertias[1:], model.names[1:], model.parents[1:], model.joints[1:]
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
        parent_joint = frame.parentJoint # Parent joints for frames may be incorrect dur to the changes in the joints order
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
        jointToLockNames = [
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
        jointToLockIds = [
            i for (i, n) in enumerate(new_model.names) if n in jointToLockNames
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
            jointToLockIds,
            q0,
        )
        q0 = pin.neutral(new_model)

    # Freeze joints if required
    if not closed_loop:     # If we want to consider the robot in open loop
        new_model.name = "talos_open"
        print("Freezing closed loop joints")
        jointToLock = [
            'mot_molet_right',
            'mot_molet_left',
            'free_tendon_right',
            'free_tendon_left',
            'free_molet_right',
            'free_molet_left',
        ]
        jointToLockIds = [
            i for (i, n) in enumerate(new_model.names) if n in jointToLock
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
            jointToLockIds,
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
    
    else: # if we consider closed loop
        # Create constraint models
        fermeture_right_A = new_model.frames[new_model.getFrameId("fermeture_right_A")]
        fermeture_right_B = new_model.frames[new_model.getFrameId("fermeture_right_B")]
        fermeture_left_A = new_model.frames[new_model.getFrameId("fermeture_left_A")]
        fermeture_left_B = new_model.frames[new_model.getFrameId("fermeture_left_B")]
        contactConstraint_right = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            new_model,
            fermeture_right_A.parentJoint,
            fermeture_right_A.placement,
            fermeture_right_B.parentJoint, 
            fermeture_right_B.placement,
            pin.ReferenceFrame.LOCAL,
        )
        contactConstraint_right.name = "Ankle_loop_right"
        contactConstraint_left = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            new_model,
            fermeture_left_A.parentJoint,
            fermeture_left_A.placement,
            fermeture_left_B.parentJoint, 
            fermeture_left_B.placement,
            pin.ReferenceFrame.LOCAL,
        )
        contactConstraint_left.name = "Ankle_loop_left"
        constraint_models = [contactConstraint_right, contactConstraint_left]

    ## Define contact Ids for control problems - these are floor contacts
    contactIds = [i for i, f in enumerate(new_model.frames) if "sole_link" in f.name]
    footSizeX = 0.1
    footSizeY = 0.05

    ## Adding new frames for control problems
    for cid in contactIds:
        f = new_model.frames[cid]
        rootName = f.name.partition('sole_link')[0]
        for side in {'left', 'right'}:
            new_model.addFrame(
                pin.Frame(
                    f"{rootName}tow_{side}",
                    f.parentJoint,
                    f.parentFrame,
                    f.placement * pin.SE3(np.eye(3), np.array([footSizeX, footSizeY*(1 if side=='left' else -1) , 0])),
                    pin.FrameType.OP_FRAME,
                )
            )
            new_model.addFrame(
                pin.Frame(
                    f"{rootName}heel_{side}",
                    f.parentJoint,
                    f.parentFrame,
                    f.placement * pin.SE3(np.eye(3), np.array([-footSizeX, footSizeY*(1 if side=='left' else -1), 0])),
                    pin.FrameType.OP_FRAME,
                )
            )
    for cm in constraint_models:
        print(cm.colwise_span_indexes, cm.joint1_id, cm.joint2_id)
    
    new_model, geometry_models, constraint_models, actuation_model = reorganizeModels(new_model, [visual_model, collision_model], constraint_models)
    visual_model, collision_model = geometry_models[0], geometry_models[1]

    for cm in constraint_models:
        print(cm.colwise_span_indexes, cm.joint1_id, cm.joint2_id)
    return (
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    )

if __name__ == "__main__":
    model, cm, am, visual_model, collision_model = TalosClosed(closed_loop=True, only_legs=True)
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