import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import re
import yaml
from yaml.loader import SafeLoader
from warnings import warn
import numpy as np

from actuation_model import ActuationModel

def nameFrameConstraint(model, nomferme="fermeture", Lid=[]):
    """
    nameFrameConstraint(model, nomferme="fermeture", Lid=[])

    Takes a robot model and returns a list of frame names that are constrained to be in contact: Ln=[['name_frame1_A','name_frame1_B'],['name_frame2_A','name_frame2_B'].....]
    where names_frameX_A and names_frameX_B are the frames in forced contact by the kinematic loop.
    The frames must be named: "...nomfermeX_..." where X is the number of the corresponding kinematic loop.
    The kinematics loop can be selectionned with Lid=[id_kinematcsloop1, id_kinematicsloop2 .....] = [1,2,...]
    if Lid = [] all the kinematics loop will be treated.

    Argument:
        model - Pinocchio robot model
        nom_ferme - nom de la fermeture  
        Lid - List of kinematic loop indexes to select
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
            match = re.search(nomferme + str(id), name)
            match2 = re.search("frame", f.name)
            if match and not (match2):
                pair_names.append(name)
        if len(pair_names) == 2:
            Lnames.append(pair_names)
    return Lnames

def generateYAML(path, name_mot="mot", name_spherical="to_rotule", file=None):
    """
    if robot.urdf inside the path, write a yaml file associate to the the robot.
    Write the name of the frame constrained, the type of the constraint, the presence of rotule articulation, 
    the name of the motor, idq and idv (with the sphrical joint).
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

    name_frame_constraint = nameFrameConstraint(rob.model, nomferme="fermeture")
    constraint_type=["6d"]*len(name_frame_constraint)

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
    with open(path+"/"+name_yaml, 'r') as yaml_file:
        contents = yaml.load(yaml_file, Loader=SafeLoader)
    return(contents)

def completeRobotLoader(path,name_urdf="robot.urdf",name_yaml="robot.yaml"):
    """
    Return  model and constraint model associated to a directory, where the name od the urdf is robot.urdf and the name of the yam is robot.yaml
    if no type assiciated, 6D type is applied
    """
    # Load the robot model using the pinocchio URDF parser
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
    return(model, constraint_models, actuation_model, visual_model)

## TEST ZONE

import unittest
class TestRobotLoader(unittest.TestCase):
    def test_complete_loader(self):
        import io
        robots_paths = [['robot_simple_iso3D', 'unittest_iso3D.txt'],
                        ['robot_simple_iso6D', 'unittest_iso6D.txt']]

        for rp in robots_paths:
            path = "robots/"+rp[0]
            m ,cm, am, vm = completeRobotLoader(path)
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
                        ['robot_simple_iso6D', 'unittest_iso6D_yaml.txt']]

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
