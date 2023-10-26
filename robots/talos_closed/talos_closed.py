import numpy as np
import example_robot_data as robex
import pinocchio as pin
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import hppfcl
import re
from actuation_model import ActuationModel
from robot_utils import freezeJoints


def TalosClosed(closed_loop=True, only_legs=True):
    robot = robex.load("talos")
    model = robot.model
    visual_model = robot.visual_model
    collision_model = robot.collision_model

    I4 = pin.SE3.Identity()
    Inertia = pin.Inertia(
        1e-3, np.array([0.0, 0.0, 0.0]), np.eye(3) * 1e-3**2
    )  # Inertia of a 1g sphere

    ## Adding new joints and links for the parallel actuation
    # * Creation of the motor bar
    id_genoux_gauche = 5
    id_genoux_droite = 11
    # Defining SE3 placements of new joints wrt parent joints
    genouxMmoletdroit = I4.copy()
    genouxMmoletdroit.translation = np.array([-0.015, -0.105, -0.11])
    genouxMmoletgauche = I4.copy()
    genouxMmoletgauche.translation = np.array([-0.015, 0.105, -0.11])
    # Adding new joints
    mot_molet_droit = model.addJoint(
        id_genoux_droite, pin.JointModelRY(), genouxMmoletdroit, "mot_molet_droit"
    )
    mot_molet_gauche = model.addJoint(
        id_genoux_gauche, pin.JointModelRY(), genouxMmoletgauche, "mot_molet_gauche"
    )
    # Adding bodies to new joints with no displacement
    model.appendBodyToJoint(mot_molet_droit, Inertia, I4)
    model.appendBodyToJoint(mot_molet_gauche, Inertia, I4)
    # Adding corresponding Geometry objects using hpp-fcl
    longueur_bar_mot = 8e-2
    largeur_bar_mot = 1.5e-2
    epaisseur = 1e-2
    bar_mot = hppfcl.Box(epaisseur, largeur_bar_mot, longueur_bar_mot)
    jMbarre = I4.copy()
    jMbarre.translation = np.array([0, 0, longueur_bar_mot / 2])
    barre_mot_droit = pin.GeometryObject(
        "mot_molet_droit", mot_molet_droit, jMbarre, bar_mot
    )
    color = [1, 0, 0, 1]
    barre_mot_droit.meshColor = np.array(color)
    barre_mot_gauche = pin.GeometryObject(
        "mot_molet_gauche", mot_molet_gauche, jMbarre, bar_mot
    )
    color = [0, 1, 0, 1]
    barre_mot_gauche.meshColor = np.array(color)
    visual_model.addGeometryObject(barre_mot_droit)
    visual_model.addGeometryObject(barre_mot_gauche)

    # * Creation of free bar on foot
    # Defining placement wrt to the parent joints
    id_cheville_gauche = 6
    id_cheville_droite = 12
    chevilleMtendondroit = I4.copy()
    chevilleMtendondroit.translation = np.array([-0.08, -0.105, 0.02])
    chevilleMtendongauche = I4.copy()
    chevilleMtendongauche.translation = np.array([-0.08, 0.105, 0.02])
    # Adding the joints
    tendon_droit = model.addJoint(
        id_cheville_droite,
        pin.JointModelSpherical(),
        chevilleMtendondroit,
        "free_tendon_droit",
    )
    tendon_gauche = model.addJoint(
        id_cheville_gauche,
        pin.JointModelSpherical(),
        chevilleMtendongauche,
        "free_tendon_gauche",
    )
    # Adding bodies to joints with no displacement
    model.appendBodyToJoint(tendon_droit, Inertia, I4)
    model.appendBodyToJoint(tendon_gauche, Inertia, I4)
    # Adding corresponding visual and collision models
    longueur_bar_free = 10e-2
    largeur_bar_free = 1.5e-2
    epaisseur = 1e-2
    bar_free = hppfcl.Box(epaisseur, largeur_bar_free, longueur_bar_free)
    jMbarre = I4.copy()
    jMbarre.translation = np.array([0, 0, longueur_bar_free / 2])
    demi_tendon_droit = pin.GeometryObject(
        "demi_tendon_droit", tendon_droit, jMbarre, bar_free
    )
    color = [1, 0, 0, 1]
    demi_tendon_droit.meshColor = np.array(color)
    demi_tendon_gauche = pin.GeometryObject(
        "demi_tendon_gauche", tendon_gauche, jMbarre, bar_free
    )
    color = [0, 1, 0, 1]
    demi_tendon_gauche.meshColor = np.array(color)
    visual_model.addGeometryObject(demi_tendon_droit)
    visual_model.addGeometryObject(demi_tendon_gauche)

    # * Create free joint linked to previous motor bar
    # Defining placements wrt parent joints
    moletMfreejointdroit = I4.copy()
    moletMfreejointdroit.translation = np.array([0, 0, longueur_bar_mot])
    moletMfreejointgauche = I4.copy()
    moletMfreejointgauche.translation = np.array([0, 0, longueur_bar_mot])
    # Adding joints
    free_molet_droit = model.addJoint(
        mot_molet_droit,
        pin.JointModelSpherical(),
        moletMfreejointdroit,
        "free_molet_droit",
    )
    free_molet_gauche = model.addJoint(
        mot_molet_gauche,
        pin.JointModelSpherical(),
        moletMfreejointgauche,
        "free_molet_gauche",
    )
    # Adding bodies to joints with no displacement
    model.appendBodyToJoint(free_molet_droit, Inertia, I4)
    model.appendBodyToJoint(free_molet_gauche, Inertia, I4)
    # Adding corresponding visual and collision model
    demi_tendon_molet_droit = pin.GeometryObject(
        "demi_tendon_molet_droit", free_molet_droit, jMbarre, bar_free
    )
    demi_tendon_molet_gauche = pin.GeometryObject(
        "demi_tendon_molet_gauche", free_molet_gauche, jMbarre, bar_free
    )
    visual_model.addGeometryObject(demi_tendon_molet_droit)
    visual_model.addGeometryObject(demi_tendon_molet_gauche)

    # * Create the frames corresponding to the closed loop contacts
    Rx = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 0]))
    fplacement = I4.copy()
    fplacement.translation = np.array([0, 0, longueur_bar_free])
    fermeture_droite_A = pin.Frame(
        "fermeture_droite_A", free_molet_droit, fplacement * Rx, pin.OP_FRAME
    )
    fermeture_gauche_A = pin.Frame(
        "fermeture_gauche_A", free_molet_gauche, fplacement * Rx, pin.OP_FRAME
    )
    fermeture_droite_B = pin.Frame(
        "fermeture_droite_B", tendon_droit, fplacement, pin.OP_FRAME
    )
    fermeture_gauche_B = pin.Frame(
        "fermeture_gauche_B", tendon_gauche, fplacement, pin.OP_FRAME
    )
    model.addFrame(fermeture_droite_A)
    model.addFrame(fermeture_droite_B)
    model.addFrame(fermeture_gauche_A)
    model.addFrame(fermeture_gauche_B)

    constraint_models = []

    # * Create the new model
    new_model = pin.Model()
    new_model.name = "talos_closed" # Defining the model name
    # Renaming the non-actuated joints 
    first = True
    i = 0
    for jp, iner, name, i, jm in zip(
        model.jointPlacements, model.inertias, model.names, model.parents, model.joints
    ):
        if first:
            first = False
        else:
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
        parent_joint = frame.parentJoint
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
            'mot_molet_droit',
            'mot_molet_gauche',
            'free_tendon_droit',
            'free_tendon_gauche',
            'free_molet_droit',
            'free_molet_gauche',
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
        fermeture_droite_A = new_model.frames[new_model.getFrameId("fermeture_droite_A")]
        fermeture_droite_B = new_model.frames[new_model.getFrameId("fermeture_droite_B")]
        fermeture_gauche_A = new_model.frames[new_model.getFrameId("fermeture_gauche_A")]
        fermeture_gauche_B = new_model.frames[new_model.getFrameId("fermeture_gauche_B")]
        contactConstraint_right = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            new_model,
            fermeture_droite_A.parentJoint,
            fermeture_droite_A.placement,
            fermeture_droite_B.parentJoint, 
            fermeture_droite_B.placement,
            pin.ReferenceFrame.LOCAL,
        )
        contactConstraint_right.name = "Ankle_loop_right"
        contactConstraint_left = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            new_model,
            fermeture_gauche_A.parentJoint,
            fermeture_gauche_A.placement,
            fermeture_gauche_B.parentJoint, 
            fermeture_gauche_B.placement,
            pin.ReferenceFrame.LOCAL,
        )
        contactConstraint_left.name = "Ankle_loop_left"
        constraint_models = [contactConstraint_right, contactConstraint_left]

    ## Define contact Ids for control problems - these are floor contacts
    contactIds = [i for i, f in enumerate(new_model.frames) if "sole_link" in f.name]
    ankleToTow = 0.1
    ankleToHeel = -0.1

    ## Adding new frames for control problems
    for cid in contactIds:
        f = new_model.frames[cid]
        rootName = f.name.partition('sole_link')[0]
        for side in {'left', 'right'}:
            new_model.addFrame(
                pin.Frame(
                    f"{rootName}_tow_{side}",
                    f.parentJoint,
                    f.parentFrame,
                    f.placement * pin.SE3(np.eye(3), np.array([ankleToTow, 0, 0])),
                    pin.FrameType.OP_FRAME,
                )
            )
            new_model.addFrame(
                pin.Frame(
                    f"{rootName}_heel_{side}",
                    f.parentJoint,
                    f.parentFrame,
                    f.placement * pin.SE3(np.eye(3), np.array([ankleToHeel, 0, 0])),
                    pin.FrameType.OP_FRAME,
                )
            )

    return (
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    )


if __name__ == "__main__":
    model, cm, am, visual_model, collision_model = TalosClosed(closed_loop=True, only_legs=True)
    q0 = np.zeros(model.nq)
    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.loadViewerModel(rootNodeName="number 1")
    viz.display(q0)
