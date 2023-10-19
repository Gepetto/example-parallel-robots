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

    ##creation of the motor bar
    id_genoux_gauche = 5
    id_genoux_droite = 11

    genouxMmoletdroit = I4.copy()
    genouxMmoletdroit.translation = np.array([-0.015, -0.105, -0.11])

    genouxMmoletgauche = I4.copy()
    genouxMmoletgauche.translation = np.array([-0.015, 0.105, -0.11])

    mot_molet_droit = model.addJoint(
        id_genoux_droite, pin.JointModelRY(), genouxMmoletdroit, "mot_molet_droit"
    )

    mot_molet_gauche = model.addJoint(
        id_genoux_gauche, pin.JointModelRY(), genouxMmoletgauche, "mot_molet_gauche"
    )

    model.appendBodyToJoint(mot_molet_droit, Inertia, I4)
    model.appendBodyToJoint(mot_molet_gauche, Inertia, I4)

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

    ## creation of free bar on foot

    id_cheville_gauche = 6
    id_cheville_droite = 12

    chevilleMtendondroit = I4.copy()
    chevilleMtendondroit.translation = np.array([-0.08, -0.105, 0.02])

    chevilleMtendongauche = I4.copy()
    chevilleMtendongauche.translation = np.array([-0.08, 0.105, 0.02])

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

    model.appendBodyToJoint(tendon_droit, Inertia, I4)
    model.appendBodyToJoint(tendon_gauche, Inertia, I4)

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

    ## add free joint on bar mot

    moletMfreejointdroit = I4.copy()
    moletMfreejointdroit.translation = np.array([0, 0, longueur_bar_mot])

    moletMfreejointgauche = I4.copy()
    moletMfreejointgauche.translation = np.array([0, 0, longueur_bar_mot])

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

    model.appendBodyToJoint(free_molet_droit, Inertia, I4)
    model.appendBodyToJoint(free_molet_gauche, Inertia, I4)

    demi_tendon_molet_droit = pin.GeometryObject(
        "demi_tendon_molet_droit", free_molet_droit, jMbarre, bar_free
    )

    demi_tendon_molet_gauche = pin.GeometryObject(
        "demi_tendon_molet_gauche", free_molet_gauche, jMbarre, bar_free
    )

    visual_model.addGeometryObject(demi_tendon_molet_droit)
    visual_model.addGeometryObject(demi_tendon_molet_gauche)

    ## creation of frame

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

    contactConstraint_right = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        model,
        fermeture_droite_A.parentJoint,
        fermeture_droite_A.placement,
        fermeture_droite_B.parentJoint,  # To the world
        fermeture_droite_B.placement,
        pin.ReferenceFrame.LOCAL,
    )
    contactConstraint_right.name = "Ankle_loop_right"
    contactConstraint_left = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        model,
        fermeture_gauche_A.parentJoint,
        fermeture_gauche_A.placement,
        fermeture_gauche_B.parentJoint,  # To the world
        fermeture_gauche_B.placement,
        pin.ReferenceFrame.LOCAL,
    )
    contactConstraint_left.name = "Ankle_loop_left"
    constraint_models = [contactConstraint_right, contactConstraint_left]

    # Create New model
    new_model = pin.Model()
    new_model.name = "talos_closed"
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

    for frame in model.frames[1:]:
        name = frame.name
        parent_joint = frame.parentJoint
        placement = frame.placement
        frame = pin.Frame(name, parent_joint, placement, pin.BODY)
        _ = new_model.addFrame(frame, False)

    q0 = pin.neutral(new_model)

    # Freeze joints if required
    if not closed_loop:
        new_model.name = "talos_open"
        print("Freezing closed loop joints")
        jointToLock = [
            mot_molet_droit,
            mot_molet_gauche,
            tendon_droit,
            tendon_gauche,
            free_molet_droit,
            free_molet_gauche,
        ]

        new_model, [collision_model, visual_model] = pin.buildReducedModel(
            new_model,
            [collision_model, visual_model],
            jointToLock,
            q0,
        )
        new_model.names[
            new_model.getJointId("free_leg_right_5_joint")
        ] = "mot_leg_right_5_joint"
        new_model.names[
            new_model.getJointId("free_leg_left_5_joint")
        ] = "mot_leg_left_5_joint"
        constraint_models = []
        q0 = pin.neutral(new_model)

    actuation_model = ActuationModel(new_model, ["mot"])
    # Use only legs (ie freeze top joints)
    if only_legs:
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

        ## Define contact Ids for control problems - these are floor contacts
        contactIds = [i for i, f in enumerate(new_model.frames) if "sole_link" in f.name]
        ankleToTow = 0.1
        ankleToHeel = -0.1

        ## Adding new frames for control problems
        for cid in contactIds:
            f = new_model.frames[cid]
            new_model.addFrame(
                pin.Frame(
                    f"{f.name}_tow",
                    f.parentJoint,
                    f.parentFrame,
                    f.placement * pin.SE3(np.eye(3), np.array([ankleToTow, 0, 0])),
                    pin.FrameType.OP_FRAME,
                )
            )
            new_model.addFrame(
                pin.Frame(
                    f"{f.name}_heel",
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
    model, data, visual_model = TalosClosed()
    q0 = np.zeros(model.nq)
    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.loadViewerModel(rootNodeName="number 1")
    viz.display(q0)
