"""
-*- coding: utf-8 -*-
Virgile Batto & Ludovic De Matteis - December 2023

Tools to load and parse URDF and YAML robot files. Can also generate YANL file from an URDF with naming conventions
"""

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import re
import yaml
from yaml.loader import SafeLoader
from warnings import warn
from os.path import dirname, exists, join
import sys
import numpy as np

from .actuation_model import ActuationModel
from .robot_options import ROBOTS
from .path import EXAMPLE_PARALLEL_ROBOTS_MODEL_DIR, EXAMPLE_PARALLEL_ROBOTS_SOURCE_DIR


def getNameFrameConstraint(model, name_loop="fermeture", cstr_frames_ids=[]):
    """
    Extracts the names of constrained frames based on a kinematic loop in the robot model.

    Args:
        model (Pinocchio.RobotModel): The Pinocchio robot model.
        name_loop (str, optional): Identifier of the names of the frames to set in contact for closing the loop.
            This identifier is used to match specific frame names. By default, it is set to "fermeture".
            The frames must be named using a convention where the identifier is followed by a numeric value
            corresponding to the index of the kinematic loop. For example, if name_loop="fermeture" and
            the index is 1, the frames would be named "fermeture1_frame_A" and "fermeture1_frame_B".
        cstr_frames_ids (list, optional): List of kinematic loop indexes to select. Defaults to [] (select all).

    Returns:
        list: A list of frame name pairs that should be in contact to close the kinematic loop.
            Each pair is represented as ['name_frame1_A', 'name_frame1_B'], ['name_frame2_A', 'name_frame2_B'],
            and so on, where names_frameX_A and names_frameX_B are frames in forced contact by the kinematic loop.
    """
    warn(
        "Function getNameFrameConstraint depreceated - prefer using a YAML file as complement to the URDF. Should only be used to generate a YAML file"
    )

    warn(
        "Function getNameFrameConstraint depreceated - prefer using a YAML file as complement to the URDF. Should only be used to generate a YAML file"
    )
    if cstr_frames_ids == []:
        cstr_frames_ids = range(len(model.frames) // 2)
    cstr_frames_names = []
    for id in cstr_frames_ids:
        pair_names = []
        for f in model.frames:
            name = f.name
            match = re.search(name_loop + str(id), name)
            match2 = re.search("frame", f.name)
            if match and not (match2):
                pair_names.append(name)
        if len(pair_names) == 2:
            cstr_frames_names.append(pair_names)
    return cstr_frames_names


def generateYAML(path, name_mot="mot", name_spherical="to_rotule", file=None):
    """
    Generates or updates a YAML file to describe robot constraints and actuation.

    This function generates a YAML file containing information about constrained frames, constraints types, motor joint names, specific joint names, and types.
    If the YAML file exists and is specified, the function adds information to this file.

    The robot motor name should contain `name_mot`, and the joints that are to be converted to spherical joints should contain `name_spherical` in their name.

    Args:
        path (str): Path to the folder containing the robot.urdf file and where the YAML file will be generated.
        name_mot (str, optional): Identifier of the motors names. Defaults to "mot".
        name_spherical (str, optional): Identifier of the spherical joints names. Defaults to "to_rotule".
        file (file object, optional): Existing YAML file to append to. Defaults to None.

    Returns:
        None
    """
    rob = RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    joint_names = []
    joint_types = []
    mot_joints_names = []
    for name in rob.model.names:
        match = re.search(name_spherical, name)
        match_mot = re.search(name_mot, name)
        if match:
            joint_names.append(name)
            joint_types.append("SPHERICAL")
        if match_mot:
            mot_joints_names.append(name)

    name_frame_constraint = getNameFrameConstraint(rob.model, name_loop="fermeture")
    constraint_type = ["6d"] * len(
        name_frame_constraint
    )  # Constraint is default to 6D... that is not very general...

    if file is None:
        with open(path + "/robot.yaml", "w") as f:
            f.write("closed_loop: " + str(name_frame_constraint) + "\n")
            f.write("type: " + str(constraint_type) + "\n")
            f.write("name_mot: " + str(mot_joints_names) + "\n")
            f.write("joint_name: " + str(joint_names) + "\n")
            f.write("joint_type: " + str(joint_types) + "\n")
    else:
        file.write("closed_loop: " + str(name_frame_constraint) + "\n")
        file.write("type: " + str(constraint_type) + "\n")
        file.write("name_mot: " + str(mot_joints_names) + "\n")
        file.write("joint_name: " + str(joint_names) + "\n")
        file.write("joint_type: " + str(joint_types) + "\n")


def getYAMLcontents(path, name_yaml="robot.yaml"):
    """
    Retrieves the content of the specified YAML file.

    Args:
        path (str): Path to the folder containing the YAML file.
        name_yaml (str, optional): Name of the YAML file. Defaults to 'robot.yaml'.

    Returns:
        dict: Content of the YAML file.
    """
    with open(path + "/" + name_yaml, "r") as yaml_file:
        contents = yaml.load(yaml_file, Loader=SafeLoader)
    return contents


def completeRobotLoader(
    path, name_urdf="robot.urdf", name_yaml="robot.yaml", freeflyer=False
):
    """
    Generates a complete robot model from URDF and YAML files.

    This function builds a comprehensive robot model including constraints and actuation based on URDF and YAML files.

    Args:
        path (str): Path to the folder containing the URDF and YAML files.
        name_urdf (str, optional): Name of the URDF file. Defaults to "robot.urdf".
        name_yaml (str, optional): Name of the YAML file. Defaults to "robot.yaml".
        freeflyer (bool, optional): Specifies whether the root joint should be a free-flyer (True) or world-fixed (False). Defaults to False.

    Returns:
        model (Pinocchio.RobotModel): Pinocchio robot model.
        constraints_models (list): List of Pinocchio robot constraint model.
        actuation_model (object): Robot actuation model (custom object defined in the library).
        visual_model (Pinocchio.RobotVisualModel): Pinocchio robot visual model.
        collision_model (Pinocchio.RobotCollisionModel): Pinocchio robot collision model.
    """
    # Load the robot model using the pinocchio URDF parser
    if freeflyer:
        robot = RobotWrapper.BuildFromURDF(
            path + "/" + name_urdf, path, root_joint=pin.JointModelFreeFlyer()
        )
    else:
        robot = RobotWrapper.BuildFromURDF(path + "/" + name_urdf, path)
    model = robot.model

    yaml_content = getYAMLcontents(path, name_yaml)

    # Update model
    update_joint = yaml_content["joint_name"]
    joints_types = yaml_content["joint_type"]
    fixed_joints_names = []
    new_model = pin.Model()
    for place, iner, name, parent_old, joint in list(
        zip(
            model.jointPlacements,
            model.inertias,
            model.names,
            model.parents,
            model.joints,
        )
    )[1:]:
        parent = new_model.getJointId(model.names[parent_old])
        if name in update_joint:
            joint_type = joints_types[update_joint.index(name)]
            if joint_type == "SPHERICAL":
                jm = pin.JointModelSpherical()
            elif joint_type == "FIXED":
                jm = joint
                fixed_joints_names.append(joint.id)
            elif joint_type == "CARDAN":
                parent = new_model.addJoint(
                    parent, pin.JointModelRX(), place, name + "_X"
                )
                jm = pin.JointModelRY()
                place = pin.SE3.Identity()
                name = name + "_Y"
        else:
            jm = joint
        jid = new_model.addJoint(parent, jm, place, name)
        new_model.appendBodyToJoint(jid, iner, pin.SE3.Identity())
    # Frames
    for f in model.frames:
        n, parent_old, placement = f.name, f.parentJoint, f.placement
        if (
            model.names[parent_old] in update_joint
            and joints_types[update_joint.index(model.names[parent_old])] == "CARDAN"
        ):
            parent = new_model.getJointId(model.names[parent_old] + "_Y")
        else:
            parent = new_model.getJointId(model.names[parent_old])
            # print(parent)
        frame = pin.Frame(n, parent, placement, f.type)
        new_model.addFrame(frame, False)

    # Geometry models
    geometry_models = []
    for old_geom_model in [robot.visual_model, robot.collision_model]:
        geom_model = old_geom_model.copy()
        for gm in geom_model.geometryObjects:
            if (
                model.names[gm.parentJoint] in update_joint
                and joints_types[update_joint.index(model.names[gm.parentJoint])]
                == "CARDAN"
            ):
                gm.parentJoint = new_model.getJointId(
                    model.names[gm.parentJoint] + "_Y"
                )
            else:
                gm.parentJoint = new_model.getJointId(model.names[gm.parentJoint])
            gm.parentFrame = new_model.getFrameId(model.frames[gm.parentFrame].name)
        geometry_models.append(geom_model)
    visual_model, collision_model = geometry_models[0], geometry_models[1]

    new_model.frames.__delitem__(0)
    new_model, visual_model = pin.buildReducedModel(
        new_model, visual_model, fixed_joints_names, pin.neutral(new_model)
    )

    model = new_model
    constraints_models = []
    # check if type is associated,else 6D is used
    try:
        name_frame_constraint = yaml_content["closed_loop"]
        constraint_type = yaml_content["type"]

        # construction of constraint model

        for L, ctype in zip(name_frame_constraint, constraint_type):
            name1, name2 = L
            id1 = model.getFrameId(name1)
            id2 = model.getFrameId(name2)
            Se3joint1 = model.frames[id1].placement
            Se3joint2 = model.frames[id2].placement
            parentjoint1 = model.frames[id1].parentJoint
            parentjoint2 = model.frames[id2].parentJoint
            if ctype == "3D" or ctype == "3d":
                constraint = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_3D,
                    model,
                    parentjoint1,
                    Se3joint1,
                    parentjoint2,
                    Se3joint2,
                    pin.ReferenceFrame.LOCAL,
                )
                constraint.name = name1 + "C" + name2
            else:
                constraint = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_6D,
                    model,
                    parentjoint1,
                    Se3joint1,
                    parentjoint2,
                    Se3joint2,
                    pin.ReferenceFrame.LOCAL,
                )
                constraint.name = name1 + "C" + name2
            constraints_models.append(constraint)
    except RuntimeError:
        print("no constraint")

    actuation_model = ActuationModel(model, yaml_content["name_mot"])
    return (model, constraints_models, actuation_model, visual_model, collision_model)


def getModelPath(subpath, verbose=True):
    """Looks for robot directory subpath based on installation path"""
    source = dirname(dirname(dirname(__file__)))  # top level source directory
    paths = [
        # function called from "make release" in build/ dir
        join(dirname(dirname(dirname(source))), "robots"),
        # function called from a build/ dir inside top level source
        join(dirname(source), "robots"),
        # function called from top level source dir
        join(source, "robots"),
    ]
    print(EXAMPLE_PARALLEL_ROBOTS_MODEL_DIR, EXAMPLE_PARALLEL_ROBOTS_SOURCE_DIR)
    try:
        # function called from installed project
        paths.append(EXAMPLE_PARALLEL_ROBOTS_MODEL_DIR)
        # function called from off-tree build dir
        paths.append(EXAMPLE_PARALLEL_ROBOTS_SOURCE_DIR)
    except NameError:
        pass
    paths += [join(p, "../../../share/example-robot-data/robots") for p in sys.path]
    for path in paths:
        print(f"Checkin {join(path, subpath.strip('/'))}")
        if exists(join(path, subpath.strip("/"))):
            if verbose:
                print("using %s as modelPath" % path)
            return join(path, subpath.strip("/"))
    raise IOError("%s not found" % subpath)


def load(robot_name, free_flyer=None, only_legs=None):
    """
    Load a model of a robot and return model objects containing all information about the robot.

    Args:
        robot_name (str): Name of the robot. See models to see possible options.
        free_flyer (bool, optional): Load the robot with a free flyer base. Uses the robot's default setting if not specified.
        only_legs (bool, optional): Freeze all joints outside of the legs, only used for full body models. Uses the robot's default setting if not specified.

    Returns:
        model (Pinocchio.RobotModel): Pinocchio robot model.
        constraint_models (list): List of Pinocchio robot constraint models.
        actuation_model (object): Robot actuation model (custom object defined in the library).
        visual_model (Pinocchio.RobotVisualModel): Pinocchio robot visual model.
        collision_model (Pinocchio.RobotCollisionModel): Pinocchio robot collision model.
    """
    if robot_name not in ROBOTS.keys():
        raise ValueError(
            f"Robot {robot_name} does not exist.\n Call 'models()' to see the list of available models"
        )
    robot = ROBOTS[robot_name]
    if robot.urdf_file is not None:
        ff = robot.free_flyer if free_flyer is None else free_flyer
        models_stack = completeRobotLoader(
            getModelPath(robot.path), robot.urdf_file, robot.yaml_file, ff
        )
        return models_stack
    else:  # This concerns full body models
        ff = robot.free_flyer if free_flyer is None else free_flyer
        ol = robot.only_legs if only_legs is None else only_legs
        models_stack = robot.exec(robot.closed_loop, ol, ff)
        return models_stack


def models():
    """Displays the list of available robot names"""
    print(f"Available models are: \n {ROBOTS.keys()}\n Generate model with method load")


def simplifyModel(model, visual_model):
    """
    Checks if any revolute joints can be replaced with spherical joints.

    Args:
        model (Pinocchio.RobotModel): Pinocchio robot model.
        visual_model (Pinocchio.RobotVisualModel): Pinocchio robot visual model.

    Returns:
        Pinocchio.RobotModel: The simplified Pinocchio robot model.
        Pinocchio.RobotVisualModel: The simplified Pinocchio robot visual model.
    """
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.randomConfiguration(model))
    new_model = pin.Model()
    fixed_joints_ids = []
    for jid, place, iner, name, parent_old, jtype in list(
        zip(
            range(len(model.joints)),
            model.jointPlacements,
            model.inertias,
            model.names,
            model.parents,
            model.joints,
        )
    ):
        vectors = []
        joints_mass = []
        points = []
        parent = new_model.getJointId(model.names[parent_old])
        for jid2, jtype in zip(range(3), model.joints[jid : jid + 3]):
            joint_id = jid + jid2
            oMi = data.oMi[joint_id]
            if "RX" in jtype.shortname():
                vec = oMi.rotation[:, 0]
            elif "RY" in jtype.shortname():
                vec = oMi.rotation[:, 1]
            elif "RZ" in jtype.shortname():
                vec = oMi.rotation[:, 2]
            else:
                break
            mass = model.inertias[joint_id].mass
            joints_mass.append(mass)
            vectors.append(vec)
            points.append(oMi.translation)
        if len(vectors) == 3:
            if joints_mass[-1] < 1e-3 and joints_mass[-2] < 1e-3:
                if (
                    np.linalg.norm(np.cross(vectors[0], vectors[1])) > 1e-6
                    and np.linalg.norm(np.cross(vectors[0], vectors[2])) > 1e-6
                    and np.linalg.norm(np.cross(vectors[2], vectors[1])) > 1e-6
                ):
                    if (
                        abs(
                            np.dot(
                                np.cross(vectors[0], vectors[1]), points[0] - points[1]
                            )
                        )
                        < 1e-5
                        and abs(
                            np.dot(
                                np.cross(vectors[0], vectors[2]), points[0] - points[2]
                            )
                        )
                        < 1e-5
                        and abs(
                            np.dot(
                                np.cross(vectors[2], vectors[1]), points[2] - points[1]
                            )
                        )
                        < 1e-5
                    ):
                        print(jid)
                        a = vectors[0]
                        b = vectors[1]
                        A = points[0]
                        B = points[1]

                        numerateur = A[0] - B[0] - ((A[1] - B[1]) / b[1]) * b[0]
                        denominateur = (a[1] / b[1]) * b[0] - a[0]
                        if numerateur < 1e-5:
                            t = 0
                        elif denominateur != 0:
                            t = numerateur / denominateur  # intersection point
                        else:
                            t = 0
                            print("INVALID POSITION COMPUTED")

                            break

                        pos = A + t * a
                        newoMi = pin.SE3.Identity()
                        newoMi.translation = pos

                        place = data.oMi[max(jid - 1, 0)].inverse() * newoMi
                        jtype = pin.JointModelSpherical()
                        fixed_joints_ids += [jid + 1, jid + 2]
        if jid != 0:
            print(jid)
            test = new_model.addJoint(parent, jtype, place, name)
            new_model.appendBodyToJoint(test, iner, pin.SE3.Identity())

    new_model, new_visual_model = pin.buildReducedModel(
        new_model, visual_model, fixed_joints_ids, pin.neutral(new_model)
    )
    return (new_model, new_visual_model)


def unitest_SimplifyModel():
    import example_robot_data as robex
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat

    robot = robex.load("panda")
    model = robot.model
    visual_model = robot.visual_model

    model.inertias[2].mass = 0
    model.inertias[3].mass = 0

    new_model, new_visual_model = simplifyModel(model, visual_model)

    viz = MeshcatVisualizer(new_model, new_visual_model, new_visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="number 1" + str(np.random.rand()))
    viz.display(pin.randomConfiguration(new_model))
