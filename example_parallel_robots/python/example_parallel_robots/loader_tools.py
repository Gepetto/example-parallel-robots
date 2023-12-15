'''
-*- coding: utf-8 -*-
Virgile Batto & Ludovic De Matteis - September 2023

Tools to load and parse URDF and YAML robot files. Can also generate YANL file from an URDF with naming conventions
'''

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import re
import yaml
from yaml.loader import SafeLoader
from warnings import warn
from os.path import dirname, exists, join
import sys

from .actuation_model import ActuationModel
from .robot_options import ROBOTS
from .path import EXAMPLE_PARALLEL_ROBOTS_MODEL_DIR, EXAMPLE_PARALLEL_ROBOTS_SOURCE_DIR

def nameFrameConstraint(model, name_loop="fermeture", Lid=[]):
    """
    nameFrameConstraint(model, name_loop="fermeture", Lid=[])

    Takes a robot model and returns a list of frame names that are constrained to be in contact: Ln=[['name_frame1_A','name_frame1_B'],['name_frame2_A','name_frame2_B'].....]
    where names_frameX_A and names_frameX_B are the frames in forced contact by the kinematic loop.
    The frames must be named: "...name_loopX..." where X is the number of the corresponding kinematic loop.
    The kinematics loop can be selectionned with Lid=[id_kinematcsloop1, id_kinematicsloop2 .....] = [1,2,...]
    if Lid = [] all the kinematics loop will be treated.

    Argument:
        model - Pinocchio robot model
        name_loop [Optionnal] - identifier of the names of the frame to set in contact for closing the loop - default: "fermeture"
        Lid [Optionnal] - List of kinematic loop indexes to select - default: [] (select all)
    Return:
        Lnames - List of frame names that should be in contact
    """
    warn("Function nameFrameConstraint depreceated - prefer using a YAML file as complement to the URDF. Should only be used to generate a YAML file")
    if Lid == []:
        Lid = range(len(model.frames) // 2)
    Lnames = []
    for id in Lid:
        pair_names = []
        for f in model.frames:
            name = f.name
            match = re.search(name_loop + str(id), name)
            match2 = re.search("frame", f.name)
            if match and not (match2):
                pair_names.append(name)
        if len(pair_names) == 2:
            Lnames.append(pair_names)
    return Lnames

def generateYAML(path, name_mot="mot", name_spherical="to_rotule", file=None):
    """
    generateYAML(path, name_mot="mot", name_spherical="to_rotule", file=None)

    Generate a YAML file to describe the robot constraints and actuation. If the YAML file exists and is specified, information is added to this file instead.
    The generated file contains 
    - contrained frames names
    - constraints types
    - Names of the motor joints
    - Names of the specific joints (such as spherical)
    - Type of the specific joints
    The robot motor name should contain name_mot and the joints that are to be converted to spherical joints should contain name_spherical in their name.

    Argument:
        path - path to the folder containing the robot.urdf file and in which the yaml file will be generated
        name_mot [Optionnal] - Identifier of the motors names - default: "mot"
        name_spherical [Optionnal] - Identifier of the spherical joints names - default: "to_rotule"
        file [Optionnal] - Existing YAML file to complete - default: None
    Return:
        None
    """
    rob = RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    Ljoint=[]
    Ltype=[]
    Lmot=[]
    for name in rob.model.names:
        match = re.search(name_spherical, name)
        match_mot= re.search(name_mot,name)
        if match :
            Ljoint.append(name)
            Ltype.append("SPHERICAL")
        if match_mot:
            Lmot.append(name)

    name_frame_constraint = nameFrameConstraint(rob.model, name_loop="fermeture")
    constraint_type=["6d"]*len(name_frame_constraint) # Constraint is default to 6D... that is not very general...

    if file is None:
        with open(path + '/robot.yaml', 'w') as f:
            f.write('closed_loop: '+ str(name_frame_constraint)+'\n')
            f.write('type: '+str(constraint_type)+'\n')
            f.write('name_mot: '+str(Lmot)+'\n')
            f.write('joint_name: '+str(Ljoint)+'\n')
            f.write('joint_type: '+str(Ltype)+'\n')
    else:
        file.write('closed_loop: '+ str(name_frame_constraint)+'\n')
        file.write('type: '+str(constraint_type)+'\n')
        file.write('name_mot: '+str(Lmot)+'\n')
        file.write('joint_name: '+str(Ljoint)+'\n')
        file.write('joint_type: '+str(Ltype)+'\n')

def getYAMLcontents(path, name_yaml='robot.yaml'):
    """
    Get the content of the given YAML file.
    Argument:
        path - Path to the folder containing the YAML file
        name_yaml [Optionnal] - name of the file
    Return:
        Content of the file
    """
    with open(path+"/"+name_yaml, 'r') as yaml_file:
        contents = yaml.load(yaml_file, Loader=SafeLoader)
    return(contents)

def completeRobotLoader(path, name_urdf="robot.urdf", name_yaml="robot.yaml", freeflyer=False):
    """
    Generate a robot complete model from the URDF and YAML files.
    Argument:
        path - Path to the folder containing the URDF and YAML files
        name_urdf [Optionnal] - Name of the URDF file - default: "robot.urdf"
        name_yaml [Optionnal] - Name of the YAML file - default: "robot.yaml"
        freeflyer [Optionnal, Boolean] - Set weither the root joint should a free-flyer (True) or world fixed (False) - default: False 
    Return:
        model - Pinocchio robot model
        constraint_models - Pinocchio robot constraint model
        actuation_model - Robot actuation model - Custom object defined in the lib
        visual_model - Pinocchio robot visual model
        collision_model - Pinocchio robot collision model
    """
    # Load the robot model using the pinocchio URDF parser
    if freeflyer:
        robot = RobotWrapper.BuildFromURDF(path + "/" + name_urdf, path, root_joint=pin.JointModelFreeFlyer())
    else:
        robot = RobotWrapper.BuildFromURDF(path + "/" + name_urdf, path)
    model = robot.model

    yaml_content = getYAMLcontents(path, name_yaml)

    # try to update model
    update_joint = yaml_content['joint_name']   
    joints_types = yaml_content['joint_type']
    LjointFixed=[]
    new_model = pin.Model() 
    visual_model = robot.visual_model
    for place, iner, name, parent, joint in list(zip(model.jointPlacements, model.inertias, model.names, model.parents,model.joints))[1:]:
        if name in update_joint:
            joint_type = joints_types[update_joint.index(name)]
            if joint_type=='SPHERICAL':
                jm = pin.JointModelSpherical()
            if joint_type=="FIXED":
                jm = joint
                LjointFixed.append(joint.id)
        else:
            jm = joint
        jid = new_model.addJoint(parent, jm, place, name)
        new_model.appendBodyToJoint(jid, iner, pin.SE3.Identity())
    
    for f in model.frames:
        n, parent, placement = f.name, f.parentJoint, f.placement
        frame = pin.Frame(n, parent, placement, f.type)
        new_model.addFrame(frame, False)

    new_model.frames.__delitem__(0)
    new_model, visual_model = pin.buildReducedModel(new_model,visual_model,LjointFixed,pin.neutral(new_model))

    model = new_model

    #check if type is associated,else 6D is used
    try :
        name_frame_constraint = yaml_content['closed_loop']
        constraint_type = yaml_content['type']
    
        #construction of constraint model
        Lconstraintmodel = []
        for L,ctype in zip(name_frame_constraint, constraint_type):
            name1, name2 = L
            id1 = model.getFrameId(name1)
            id2 = model.getFrameId(name2)
            Se3joint1 = model.frames[id1].placement
            Se3joint2 = model.frames[id2].placement
            parentjoint1 = model.frames[id1].parentJoint
            parentjoint2 = model.frames[id2].parentJoint
            if ctype=="3D" or ctype=="3d":
                constraint = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_3D,
                    model,
                    parentjoint1,
                    Se3joint1,
                    parentjoint2,
                    Se3joint2,
                    pin.ReferenceFrame.LOCAL,
                )
                constraint.name = name1+"C"+name2
            else :
                constraint = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_6D,
                    model,
                    parentjoint1,
                    Se3joint1,
                    parentjoint2,
                    Se3joint2,
                    pin.ReferenceFrame.LOCAL,
                )
                constraint.name = name1+"C"+name2
            Lconstraintmodel.append(constraint)
        
        constraint_models = Lconstraintmodel
    except:
        print("no constraint")

    actuation_model = ActuationModel(model,yaml_content['name_mot'])
    return(model, constraint_models, actuation_model, visual_model, robot.collision_model)

def getModelPath(subpath, verbose=True):
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
    '''
    Loads a model of a robot and return models objects containing all information on the robot
    Arguments :
        robot_name - Name of the robot, see [to be implemented] to see possible options
        free_flyer [Optionnal, Boolean] - Load the robot with a free flyer base - Use robot's default setting if not specified
        only_legs [Optionnal, Boolean] - Freeze all joints outside of the legs, only used for full body models - Use robot's default setting if not specified
    Returns:
        model - Pinocchio robot model
        constraint_models - List of Pinocchio robot constraint model
        actuation_model - Robot actuation model - Custom object defined in the lib
        visual_model - Pinocchio robot visual model
        collision_model - Pinocchio robot collision model
    '''
    if robot_name not in ROBOTS.keys():
        raise(f"Name {robot_name} does not exists.\n Call method 'models' to see the list of available models")
    robot = ROBOTS[robot_name]
    if robot.urdf_file is not None:
        ff = robot.free_flyer if free_flyer is None else free_flyer
        models_stack = completeRobotLoader(getModelPath(robot.path), robot.urdf_file, robot.yaml_file, ff)
        return(models_stack)
    else: # This concerns full body models
        ff = robot.free_flyer if free_flyer is None else free_flyer
        ol = robot.only_legs if only_legs is None else only_legs
        models_stack = robot.exec(robot.closed_loop, ol, ff)
        return(models_stack)

def models():
    print(f"Available models are: \n {ROBOTS.keys()}\n Generate model with method load")

########## TEST ZONE ##########################

import unittest
class TestRobotLoader(unittest.TestCase):
    def test_complete_loader(self):
        import io
        robots_paths = [['robot_simple_iso3D', 'unittest_iso3D.txt'],
                        ['robot_simple_iso6D', 'unittest_iso6D.txt']]

        for rp in robots_paths:
            path = "robots/"+rp[0]
            m ,cm, am, vm, collm = completeRobotLoader(path)
            joints_info = [(j.id, j.shortname(), j.idx_q, j.idx_v) for j in m.joints[1:]]
            frames_info = [(f.name, f.inertia, f.parentJoint, f.parentFrame, f.type) for f in m.frames]
            constraint_info = [(cmi.name, cmi.joint1_id, cmi.joint2_id, cmi.joint1_placement, cmi.joint2_placement, cmi.type) for cmi in cm]
            mot_info = [(am.idqfree, am.idqmot, am.idvfree, am.idvmot)]
            
            results = io.StringIO()
            results.write('\n'.join(f'{x[0]} {x[1]} {x[2]} {x[3]}' for x in joints_info))
            results.write('\n'.join(f'{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}' for x in frames_info))
            results.write('\n'.join(f'{x[0]} {x[1]} {x[2]} {x[3]} {x[4]} {x[5]}' for x in constraint_info))
            results.write('\n'.join(f'{x[0]} {x[1]} {x[2]} {x[3]}' for x in mot_info))
            results.seek(0)

            # Ground truth is defined from a known good result
            with open('unittest/'+rp[1], 'r') as truth:
                assert truth.read() == results.read()
    
    def test_generate_yaml(self):
        import io
        robots_paths = [['robot_simple_iso3D', 'unittest_iso3D_yaml.txt'],
                        ['robot_simple_iso6D', 'unittest_iso6D_yaml.txt'],
                        ['robot_delta', 'unittest_delta_yaml.txt']]

        for rp in robots_paths:
            path = "robots/"+rp[0]
            results = io.StringIO()
            generateYAML(path, file=results)
            results.seek(0)

            # Ground truth is defined from a known good result
            with open('unittest/'+rp[1], 'r') as truth:
                assert truth.read() == results.read()

        
if __name__ == "__main__":
    unittest.main()
