import numpy as np
import example_robot_data as robex
import pinocchio as pin
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import hppfcl
import re
from actuation_model import ActuationModel
from robot_utils import freezeJoints
from numpy import random
from numpy.linalg import norm

def create_visual(model):
    Lframeinterest=[145, 146, 147, 148]
    Lframeinterest.append(40)
    Lframeinterest.append(54)
    # Lframeinterest.append(idcentre)
    
    ### GENERATION OF VISUAL MODEL
    import hppfcl
    gmodel=pin.GeometryModel()

    #generation of the joint :

    for j in range(len(model.joints[2:])): #exclude the root joint, exclude the two first joint when freeflyer base
        i=j+2
        geom_cylinder_robot = pin.GeometryObject(
            "cylinder_robot"+str(i), i, pin.SE3.Identity(), hppfcl.Cylinder(2e-2, 5e-2)
        )

        if i==0:
            color = [0, 0, 0, 0.01]
            geom_cylinder_robot = pin.GeometryObject(
            "cylinder_robot"+str(i), i, pin.SE3.Identity(), hppfcl.Sphere(2e-2)
        )
        else:
            color = [0, 0, 1, 1]

        if "mot" in model.names[i]:
            color = [1,0,0,1]

        if "to_rotule" in model.names[i]:
            geom_cylinder_robot = pin.GeometryObject(
            "cylinder_robot"+str(i), i, pin.SE3.Identity(), hppfcl.Sphere(2e-2)
        )
            color = [1,0,0,1]
        geom_cylinder_robot.meshColor = np.array(color)
        gmodel.addGeometryObject(geom_cylinder_robot)
        print('Generate joint')
    
    Lposparent=[]
    for f in model.frames:
        if "frame" not in f.name:
            Lposparent.append(f.placement.translation.tolist()+[f.parentJoint])
    for idparent,Lchild in enumerate(model.children):
        for i in Lchild:
            Lposparent.append(model.jointPlacements[i].translation.tolist()+[idparent])
    Lposparent=np.array(Lposparent)


    Lcolor=[]
    nj=model.njoints
    for i in range(nj):
        Lcolor.append(np.array([1/nj*i,1/nj*i,1/nj*i,1]))
    random.shuffle(Lcolor)


#generation of the bar
    for parent in range(model.njoints):
        Lpos_joint=[] # pos of parent joint
        bx=0
        by=0
        bz=0
        for L in Lposparent:
            if L[3]==parent:
                Lpos_joint.append(np.array(L[0:3]))
                bx+=L[0]
                by+=L[1]
                bz+=L[2]

        barricentre=np.array([bx/len(Lpos_joint),by/len(Lpos_joint),bz/len(Lpos_joint)])



        se3barricentre=pin.SE3.Identity()
        se3barricentre.translation=barricentre
        # se3barricentre.rotation=np.array([x.tolist(),y.tolist(),z.tolist()]).T

        for pos in Lpos_joint:
            dpos=pos-barricentre
            d=norm(dpos)
            dir=pin.SE3.Identity()
            dir.translation=dpos/2
            z=dpos/norm(dpos)
            x=np.array([z[1],-z[0],0])
            if norm(x)<1e-8:
                x=np.array([z[2],0,0])
            x=x/norm(x)
            y=np.cross(x,z)

            dir.rotation=np.array([x.tolist(),y.tolist(),z.tolist()]).T
            # dir.rotation=np.array([[0,0,1],[0,1,0],[1,0,0]])
            geom_cylinder_robot = pin.GeometryObject("cylinder_robotdv"+str(np.random.rand()), parent, se3barricentre*dir, hppfcl.Cylinder(1e-2,d))
            # color = [0, 1, 0, 1]
            # if parent == 0:
            #     color=[1,1,1,1]
            geom_cylinder_robot.meshColor = np.array(Lcolor[parent])
            gmodel.addGeometryObject(geom_cylinder_robot)
        
    for idf in Lframeinterest:
        frame=model.frames[idf]
        i=frame.parentJoint
        if "3d" in frame.name or "3D" in frame.name:
            geom_cylinder_robot = pin.GeometryObject(
                "cylinder_robotdv"+str(np.random.rand()), i, frame.placement, hppfcl.Sphere(2e-2)
            )
        else:
            geom_cylinder_robot = pin.GeometryObject(
                "cylinder_robotdv"+str(np.random.rand()), i, frame.placement, hppfcl.Box(2e-2,2e-2,2e-2)
            )
        color = [0, 0, 1, 1]
        if "pied" in frame.name :
            geom_cylinder_robot = pin.GeometryObject(
                "cylinder_robotdv"+str(np.random.rand()), i, frame.placement, hppfcl.Box(10e-2,5e-2,2e-2)
            )
            color = [0, 1, 1, 1]

        if "hanche" in frame.name :
            geom_cylinder_robot = pin.GeometryObject(
                "cylinder_robotdv"+str(np.random.rand()), i, frame.placement, hppfcl.Box(2e-2,2e-2,2e-2)
            )
            color = [0, 1, 0, 1]

        geom_cylinder_robot.meshColor = np.array(Lcolor[i])
        gmodel.addGeometryObject(geom_cylinder_robot)

    from pinocchio.visualize import MeshcatVisualizer
    import meshcat
    return(gmodel)
    # viz = MeshcatVisualizer(model, gmodel, gmodel)
    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.clean()
    # viz.loadViewerModel(rootNodeName="number 1")
    # viz.display(q0)

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
    id_corres = {}
    for part in ['root', 'arm_left', 'arm_right', 'left', 'right']:
        for jp, iner, name, i, jm, vgm, cgm in zip(
            model.jointPlacements[1:], model.inertias[1:], model.names[1:], model.parents[1:], model.joints[1:],
            visual_model.geometryObjects[1:], collision_model.geometryObjects[1:]
        ):
            side_left = re.search("left", name)
            side_right = re.search("right", name)
            side_arm_left = re.search('arm_left', name) 
            side_arm_right = re.search('arm_right', name)
            if part=='root' and (side_left or side_right or side_arm_left or side_arm_right): 
                # If root, ignore arms and legs
                continue
            if (part=='arm_right' and not side_arm_right) or (part=='arm_left' and not side_arm_left): 
                # If arm_right, ignore all but right arm part, same for arm_left
                continue
            elif (part=='left' and (not side_left or side_arm_left)) or (part=='right' and (not side_right or side_arm_right)):
                # If leg_left, ignore all but left leg parts, same for leg_right
                continue
            match1 = re.search("leg", name)
            match2 = re.search("5", name)
            match3 = re.search("mot", name)
            match4 = re.search("free", name)
            if match1 and match2:
                name = "free_" + name
            elif not (match3) and not (match4):
                name = "mot_" + name
            # To find the parent, look from their names in the new model from their names in the old model
            parent = int(np.min([
                        new_model.getJointId('mot_'+model.names[i]), 
                        new_model.getJointId('free_'+model.names[i]),
                        new_model.getJointId(model.names[i])]))    # TODO This should really be done in a better way
            id_corres[i] = parent # Create a correspondance between model ids and new_models_ids (will only containt)
            jid = new_model.addJoint(parent, jm, jp, name)
            new_model.appendBodyToJoint(jid, iner, pin.SE3.Identity())
    
    # print(id_corres, model)
    # for gm in visual_model.geometryObjects:
    #     gm.parentJoint = id_corres[gm.parentJoint]

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
            new_visual_model,
            new_collision_model,
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

    visual_model = collision_model = create_visual(new_model)

    return (
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    )

if __name__ == "__main__":
    model, cm, am, visual_model, collision_model = TalosClosed(closed_loop=False, only_legs=True)
    q0 = pin.neutral(model)
    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="number 1")
    viz.display(q0)

    print(model)