"""
-*- coding: utf-8 -*-
Virgile Batto & Ludovic De Matteis - December 2023

Tools to load and parse URDF and YAML robot files. Can also generate YAML file from an URDF with naming conventions
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
from toolbox_parallel_robots import freezeJoints, ActuationModel
from example_parallel_robots.robot_options import ROBOTS
from example_parallel_robots.path import (
    EXAMPLE_PARALLEL_ROBOTS_MODEL_DIR,
    EXAMPLE_PARALLEL_ROBOTS_SOURCE_DIR,
)


def getNameFrameConstraint(model, name_loop="closedloop", cstr_frames_ids=[]):
    """
    Extracts the names of constrained frames based on a kinematic loop in the robot model.

    Args:
        model (pinocchio.Model): The Pinocchio robot model.
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


def generateYAML(path, name_mot="motor", name_spherical="spherical", file=None):
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

    name_frame_constraint = getNameFrameConstraint(rob.model, name_loop="closedloop")
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
        model (pinocchio.Model): Pinocchio robot model.
        constraints_models (list): List of Pinocchio robot constraint model.
        actuation_model (object): Robot actuation model (custom object defined in the library).
        visual_model (pinocchio.GeometryModel): Pinocchio robot visual model.
        collision_model (pinocchio.GeometryModel): Pinocchio robot collision model.
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
    if "joint_name" in yaml_content.keys():
        update_joint = yaml_content["joint_name"]
        joints_types = yaml_content["joint_type"]
    else:
        update_joint = []
        joints_types = []

    fixed_joints_names = []
    new_model = pin.Model()

    ujoints_type = {"X": pin.JointModelRX, "Y": pin.JointModelRY, "Z": pin.JointModelRZ}

    for old_place, old_iner, old_name, old_parent, old_joint in list(
        zip(
            model.jointPlacements,
            model.inertias,
            model.names,
            model.parents,
            model.joints,
        )
    )[1:]:
        parent = new_model.getJointId(model.names[old_parent])
        place = old_place
        name = old_name
        if old_name in update_joint:
            joint_type = joints_types[update_joint.index(old_name)]
            if joint_type == "SPHERICAL":
                joint_model = pin.JointModelSpherical()
            elif joint_type == "FIXED":
                joint_model = old_joint
                fixed_joints_names.append(old_joint.id)
            if "UJOINT" in joint_type:
                type1 = ujoints_type[joint_type[-2]]
                type2 = ujoints_type[joint_type[-1]]

                parent = new_model.addJoint(
                    parent, type1(), place, old_name + f"_{joint_type[-2]}"
                )
                joint_model = type2()
                place = pin.SE3.Identity()
                name = old_name + f"_{joint_type[-1]}"
        else:
            joint_model = old_joint
        jid = new_model.addJoint(parent, joint_model, place, name)
        new_model.appendBodyToJoint(jid, old_iner, pin.SE3.Identity())

    # Frames - Add ujoints frames
    for f in model.frames:
        old_name, old_parent, old_place, old_type = (
            f.name,
            f.parentJoint,
            f.placement,
            f.type,
        )
        # For ujoints, create frames for the joints
        if (
            old_name in update_joint
        ):  # If the frame is a joint frame for a joint that as been updated
            assert old_type == pin.JOINT, "Frame type should be JOINT"
            assert model.names[old_parent] == old_name, (
                "Frame parent should be the joint"
            )
            assert old_place == pin.SE3.Identity(), "Frame placement should be Identity"
            joint_type = joints_types[update_joint.index(old_name)]
            if "UJOINT" in joint_type:  # If this joint is a ujoint
                ujoint_frame = pin.Frame(
                    old_name + f"_{joint_type[-2]}",
                    new_model.getJointId(old_name + f"_{joint_type[-2]}"),
                    pin.SE3.Identity(),
                    pin.FrameType.JOINT,
                )
                new_model.addFrame(ujoint_frame, False)
                ujoint_frame = pin.Frame(
                    old_name + f"_{joint_type[-1]}",
                    new_model.getJointId(old_name + f"_{joint_type[-1]}"),
                    pin.SE3.Identity(),
                    pin.FrameType.JOINT,
                )
                new_model.addFrame(ujoint_frame, False)
            else:
                name = old_name
                parent = new_model.getJointId(model.names[old_parent])
                assert old_type == pin.JOINT, "Frame type should be JOINT"
                joint_frame = pin.Frame(name, parent, old_place, old_type)
                new_model.addFrame(joint_frame, False)

        elif (
            model.names[old_parent] in update_joint
        ):  # Frame is attached to a joint that has been modified (but is not this joint frame)
            joint_type = joints_types[update_joint.index(model.names[old_parent])]

            if "UJOINT" in joint_type:  # The parent joint is ujoint
                type2 = ujoints_type[joint_type[-1]]
                name = old_name
                parent = new_model.getJointId(
                    model.names[old_parent] + f"_{joint_type[-1]}"
                )
                place = old_place
                frame_type = old_type

            else:  # The parent joint is not ujoint
                name = old_name
                parent = new_model.getJointId(model.names[old_parent])
                place = old_place
                frame_type = old_type

            frame = pin.Frame(name, parent, place, frame_type)
            new_model.addFrame(frame, False)
        else:  # The frame corresponds to the joint
            name = old_name
            parent = new_model.getJointId(model.names[old_parent])
            place = old_place
            frame_type = old_type
            frame = pin.Frame(name, parent, place, frame_type)
            new_model.addFrame(frame, False)

    # Geometry models
    geometry_models = []
    for old_geom_model in [robot.visual_model, robot.collision_model]:
        geom_model = old_geom_model.copy()
        for gm in geom_model.geometryObjects:
            if (
                model.names[gm.parentJoint] in update_joint
                and "UJOINT"
                in joints_types[update_joint.index(model.names[gm.parentJoint])]
            ):
                if (
                    joints_types[update_joint.index(model.names[gm.parentJoint])]
                    == "UJOINT_XY"
                ):
                    gm.parentJoint = new_model.getJointId(
                        model.names[gm.parentJoint] + "_Y"
                    )
                elif (
                    joints_types[update_joint.index(model.names[gm.parentJoint])]
                    == "UJOINT_YZ"
                ):
                    gm.parentJoint = new_model.getJointId(
                        model.names[gm.parentJoint] + "_Z"
                    )
                elif (
                    joints_types[update_joint.index(model.names[gm.parentJoint])]
                    == "UJOINT_ZX"
                ):
                    gm.parentJoint = new_model.getJointId(
                        model.names[gm.parentJoint] + "_X"
                    )
                elif (
                    joints_types[update_joint.index(model.names[gm.parentJoint])]
                    == "UJOINT_ZY"
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
    new_model, [visual_model, collision_model] = pin.buildReducedModel(
        new_model,
        [visual_model, collision_model],
        fixed_joints_names,
        pin.neutral(new_model),
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
    model, constraints_models, actuation_model, visual_model, collision_model = (
        simplifyModel(
            model, constraints_models, actuation_model, visual_model, collision_model
        )
    )
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
    paths += ["robots/"]
    for path in paths:
        print(f"Checking {join(path, subpath.strip('/'))}")
        if exists(join(path, subpath.strip("/"))):
            if verbose:
                print("using %s as modelPath" % path)
            return join(path, subpath.strip("/"))
    raise IOError("%s not found" % subpath)


def load(robot_name, closed_loop=True, free_flyer=None, only_legs=None):
    """
    Load a model of a robot and return model objects containing all information about the robot.

    Args:
        robot_name (str): Name of the robot. See models to see possible options.
        free_flyer (bool, optional): Load the robot with a free flyer base. Uses the robot's default setting if not specified.
        only_legs (bool, optional): Freeze all joints outside of the legs, only used for full body models. Uses the robot's default setting if not specified.

    Returns:
        model (pinocchio.Model): Pinocchio robot model.
        constraint_models (list): List of Pinocchio robot constraint models.
        actuation_model (object): Robot actuation model (custom object defined in the library).
        visual_model (pinocchio.GeometryModel): Pinocchio robot visual model.
        collision_model (pinocchio.GeometryModel): Pinocchio robot collision model.
    """
    if robot_name not in ROBOTS.keys():
        raise ValueError(
            f"Robot {robot_name} does not exist.\n Call 'models()' to see the list of available models"
        )
    robot = ROBOTS[robot_name]

    if robot_name == "talos_only_leg":
        from .talos_only_legs import talosOnlyLeg

        models_stack = talosOnlyLeg()
    elif robot_name == "talos_2legs":
        from .talos_closed import TalosClosed

        (model, constraints_models, actuation_model, visual_model, collision_model) = (
            TalosClosed(closed_loop, only_legs, free_flyer)
        )
    elif robot_name == "talos_2legs_6d":
        from .talos_closed_6d import TalosClosed

        (model, constraints_models, actuation_model, visual_model, collision_model) = (
            TalosClosed(closed_loop, only_legs, free_flyer)
        )
    else:
        ff = robot.free_flyer if free_flyer is None else free_flyer
        models_stack = completeRobotLoader(
            getModelPath(robot.path), robot.urdf_file, robot.yaml_file, ff
        )
    return models_stack


def models():
    """Displays the list of available robot names"""
    print(f"Available models are: \n {ROBOTS.keys()}\n Generate model with method load")


def simplifyModel(
    model, constraint_models, actuation_model, visual_model, collision_model
):
    """
    Checks if any revolute joints can be replaced with spherical joints.

    Args:
        model (pinocchio.Model): Pinocchio robot model.
        visual_model (pinocchio.GeometryModel): Pinocchio robot visual model.

    Returns:
        pinocchio.Model: The simplified Pinocchio robot model.
        pinocchio.GeometryModel: The simplified Pinocchio robot visual model.
    """
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    new_model = pin.Model()
    fixed_joints_ids = []
    spherical = False
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
        for jid2, jtype2 in enumerate(model.joints[jid : jid + 3]):
            joint_id = jid + jid2
            oMi = data.oMi[joint_id]
            if "RX" in jtype2.shortname():
                vec = oMi.rotation[:, 0]
            elif "RY" in jtype2.shortname():
                vec = oMi.rotation[:, 1]
            elif "RZ" in jtype2.shortname():
                vec = oMi.rotation[:, 2]
            else:
                break
            mass = model.inertias[joint_id].mass
            joints_mass.append(mass)
            vectors.append(vec)
            points.append(oMi.translation)
        if len(vectors) == 3:
            if joints_mass[0] < 1e-5 and joints_mass[1] < 1e-5:
                # print("no mass")
                print(jtype.shortname())
                print(vectors)

                if (
                    np.linalg.norm(np.cross(vectors[0], vectors[1])) > 1e-6
                    and np.linalg.norm(np.cross(vectors[0], vectors[2])) > 1e-6
                    and np.linalg.norm(np.cross(vectors[2], vectors[1])) > 1e-6
                ):
                    print("no colinearité")
                    if (
                        np.linalg.norm(points[0] - points[1]) < 1e-4
                        and np.linalg.norm(points[2] - points[1]) < 1e-4
                    ):
                        spherical = True
                        last_joint = jid

                        fixed_joints_ids += [jid + 0, jid + 1]
        if jid != 0:
            if spherical and jid == last_joint + 2:
                jtype = pin.JointModelSpherical()
                spherical = False
            test = new_model.addJoint(parent, jtype, place, name)
            new_model.appendBodyToJoint(test, iner, pin.SE3.Identity())
        # Adding frames
    for f in model.frames:
        n, parent_old, placement = f.name, f.parentJoint, f.placement
        parent = new_model.getJointId(model.names[parent_old])
        frame = pin.Frame(n, parent, placement, f.type)
        new_model.addFrame(frame, False)  # We assume that there is no inertial frame

    q0 = pin.neutral(new_model)

    (
        new_model,
        new_constraint_models,
        new_actuation_model,
        new_visual_model,
        collision_model,
    ) = freezeJoints(
        new_model,
        constraint_models,
        actuation_model,
        visual_model,
        collision_model,
        fixed_joints_ids,
        q0,
    )
    return (
        new_model,
        new_constraint_models,
        new_actuation_model,
        new_visual_model,
        collision_model,
    )


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


if __name__ == "__main__":
    model, constraints_models, actuation_model, visual_model, collision_model = load(
        "battobot"
    )
    joints_lock_names = [
        "free_knee_left_Y",
        "free_knee_left_Z",
    ]
    jointToLockIds = [i for (i, n) in enumerate(model.names) if n in joints_lock_names]
    (
        reduced_model,
        (),
    ) = pin.buildReducedModel(model, [], jointToLockIds, pin.neutral(model))
    print("End")

    # from toolbox_parallel_robots.foo import createSlidersInterface
    # from pinocchio.visualize import MeshcatVisualizer
    # import meshcat

    # viz = MeshcatVisualizer(model, visual_model, visual_model)
    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.clean()
    # viz.loadViewerModel(rootNodeName="universe")

    # createSlidersInterface(
    #     model,
    #     constraints_models,
    #     visual_model,
    #     actuation_model.mot_ids_q,
    #     viz,
    #     q0=pin.neutral(model),
    # )
    # print(model)
