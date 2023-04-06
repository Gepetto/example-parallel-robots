"""
-*- coding: utf-8 -*-
Virgile BATTO, march 2022

Tools to load and parse a urdf file with closed loop

"""


import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import os
import re
import yaml
from yaml.loader import SafeLoader
from warnings import warn




def idmot(model, name_mot="mot"):
    """
    Lid=idmot(model,name_mot='mot')
    return the id of the motor axis
    """
    Lid = []
    for i, name in enumerate(model.names):
        match = re.search(name_mot, name)
        if match:
            idx=model.joints[i].idx_q
            Lid.append(idx)
    return Lid


def idvmot(model, name_mot="mot"):
    """
    Lid=idmot(model,name_mot='mot')
    return the id of the motor axis
    """
    Lid = []
    for i, name in enumerate(model.names):
        match = re.search(name_mot, name)
        if match:
            idx=model.joints[i].idx_v
            Lid.append(idx)
    return Lid


def q2freeq(model,q, name_mot="mot"):
    """
    freeq=q2freeq(model,q, name_mot="mot")
    return the non motor coordinate of q in accordance with the robot model
    """
    Lidmot=idmot(model, name_mot)
    freeq=[]
    for i in range(len(q.tolist())):
        if not(i in Lidmot):
            freeq.append(q[i])
    return(freeq)

def getRobotInfo(path):
    """
    (name__closedloop,name_mot,number_closedloop,type)=getRobotInfo(path)
    path the diredction of the file that contain the robot
    return the info stored in  robot.yaml file. If no yaml, default value are returned
    """
    try:
        yaml_file = open(path+"/robot.yaml", 'r')
        yaml_content = yaml.load(yaml_file, Loader=SafeLoader)
        name_closedloop=yaml_content["name_closedloop"]
        name_mot=yaml_content["name_mot"]
        type=yaml_content["type"]
        try :
            number_closedloop=yaml_content["closed_loop_number"]
        except :
            number_closedloop=-1
 
    except :
        warn("no robot.yaml found, default value applied")
        name_closedloop="fermeture"
        name_mot="mot"
        number_closedloop=-1
        type="6D"
    return(name_closedloop,name_mot,number_closedloop,type)


def getSimplifiedRobot(path):
    """
    robot=getSimplifiedRobot(path)
    path, the dir of the file that contain the urdf file & the stl files


    load a robot with N closed loop with a joint on each of the 2 branch that are closed, return a simplified model of the robot where one of this joint is fixed
    """

    try:
        yaml_file = open(path+"/robot.yaml", 'r')
        yaml_content = yaml.load(yaml_file)
        name_closedloop=yaml_content["name_closedloop"]
        name_mot=yaml_content["name_mot"]
    except :
        warn("no robot.yaml found, default value applied")
        name_closedloop="fermeture"
        name_mot="mot"
    

    rob = RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    Lid = []
    # to simplifie the conception, the two contact point are generate with a joint
    # supression of one of this joint :
    for (joint, id) in zip(rob.model.names, range(len(rob.model.names))):
        match = re.search(name_closedloop, joint)
        match2 = re.search("B", joint)
        if match and match2:
            Lid.append(id)

    rob.model, rob.visual_model = pin.buildReducedModel(
        rob.model, rob.visual_model, Lid, np.zeros(rob.nq)
    )
    rob.data = rob.model.createData()
    rob.q0 = np.zeros(rob.nq)
    return rob


def nameFrameConstraint(model, nomferme="fermeture", Lid=[]):
    """
     nameFrameConstraint(model,nomferme="fermeture",Lid=[])


    take a robot model  and return Ln=[['name_frame1_A','name_frame1_B'],['name_frame2_A','name_frame2_B'].....]
    where names_frameX_A and names_frameX_B, the frzme in forced contact by the kinematic loop.

    The frame must be named: "...nomfermeX_..." where X is the number of the associated kinematic loop.

    The kinematics loop can be selectionned with Lid=[id_kinematcsloop1, id_kinematicsloop2 .....]=[1,2,...]
    if Lid= [] all the kinematics loop will be treated.
    """
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

def getConstraintModelFromName(model, Lnjoint, ref=pin.ReferenceFrame.LOCAL):
    """
    getconstraintModelfromname(model,Lnjoint,ref=pin.ReferenceFrame.LOCAL):


    take robot model and Lnjoint=[['name_joint1_A','name_joint1_B'],['name_joint2_A','name_joint2_B'].....]
    return the list of the constraintmodel wehre joint1A is in copntact with joint1_B , joint2_A in contact with joint2_B etc
    """
    Lconstraintmodel = []
    for L in Lnjoint:
        name1 = L[0]
        name2 = L[1]
        id1 = model.getFrameId(name1)
        id2 = model.getFrameId(name2)
        Se3joint1 = model.frames[id1].placement
        Se3joint2 = model.frames[id2].placement
        parentjoint1 = model.frames[id1].parentJoint
        parentjoint2 = model.frames[id2].parentJoint
        constraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            model,
            parentjoint1,
            Se3joint1,
            parentjoint2,
            Se3joint2,
            ref,
        )
        Lconstraintmodel.append(constraint)
    return Lconstraintmodel

def constraints3D(model, data, q, nomb_boucle=-1, name_closedloop="fermeture"):
    """
    contrainte3D(model,data, q,nomb_boucle, nom_fermeture="fermeture")

    take a robot model and data with n closed kinematics loop (nomb_boucle) and a configuration q
    the joint in contact must be name nom_fermetureX where X is the number of the kinematics loop (start at 1)
    return list of 3D distance between the joints
    """
    # rob.updateGeometryPlacements(q)
    Lcont = []
    if nomb_boucle<=0:
        nomb_boucle=len(model.names)//2
    pin.framesForwardKinematics(model, data, q)
    for i in range(nomb_boucle):
        L = []
        for j, frame in enumerate(model.frames):
            match = re.search(name_closedloop + str(i + 1), frame.name)
            match = re.search(name_closedloop + str(i + 1), frame.name)
            match2 = re.search("frame", frame.name)
            if match and not (match2):
                L.append(data.oMf[j])
        if len(L) == 2:
            c = pin.log(L[0].inverse() * L[1]).vector
            L=[c[0],c[1],c[2]]
            Lcont.append(np.array(L))
    return Lcont


def constraintsPlanar(model, data, q, nomb_boucle=-1, nom_fermeture="fermeture"):
    """
    contraintePlanar(model,data, q,nomb_boucle=-1, nom_fermeture="fermeture")

    take a robot model and data with n closed kinematics loop (nomb_boucle) and a configuration q
    the joint in contact must be name nom_fermetureX where X is the number of the kinematics loop (start at 1)
    return list of the planar distance (plan x,y in local frame )
    """
    if nomb_boucle<=0:
        nomb_boucle=len(model.names)//2
    Lcont = []
    pin.framesForwardKinematics(model, data, q)
    for i in range(nomb_boucle):
        L = []
        for j, frame in enumerate(model.frames):
            match = re.search(nom_fermeture + str(i + 1), frame.name)
            match2 = re.search("frame", frame.name)
            if match and not (match2):
                L.append(data.oMf[j])
        if len(L) == 2:
            
            c = pin.log(L[0].inverse() * L[1]).vector
            L=[c[0],c[1],c[5]]
            Lcont.append(np.array(L))
    return Lcont

def constraints6D(model, data, q, nomb_boucle=-1, nom_fermeture="fermeture"):
    """
    contrainte(rob,nomb_boucle,q,nom_fermeture="fermeture")

    take a robot (rob) rob with n closed kinematics loop (nomb_boucle) and a configuration q
    the joint in contact must be name nom_fermetureX where X is the number of the kinematics loop (start at 1)
    return list  6D distance between joints
    """
    if nomb_boucle<=0:
        nomb_boucle=len(model.names)//2
    Lcont = []
    pin.framesForwardKinematics(model, data, q)
    for i in range(nomb_boucle):
        L = []
        for j, frame in enumerate(model.frames):
            match = re.search(nom_fermeture + str(i + 1), frame.name)
            match2 = re.search("frame", frame.name)
            if match and not (match2):
                L.append(data.oMf[j])
        if len(L) == 2:
            c = L[0].inverse() * L[1]
            Lcont.append(pin.log(c).vector)  # return plan constraint (2D problem here)
    return Lcont

def jointTypeUpdate(model,rotule_name="to_rotule"):
    """
    model=jointTypeUpdate(model,rotule_name="to_rotule")
    take a robot model, and update type to change the joint whith the name rotule_name inside to rotule joint

    """
    new_model = pin.Model() 
    first = True
    i = 0
    for jp, iner, name, i in zip(
        model.jointPlacements, model.inertias, model.names, model.parents
    ):
        if first:
            first = False
        else:
            match = re.search(rotule_name, name)

            if match:
                jm = pin.JointModelSpherical()
            else:
                jm = pin.JointModelRZ()
            jid = new_model.addJoint(i, jm, jp, name)
            new_model.appendBodyToJoint(jid, iner, pin.SE3.Identity())

    for frame in model.frames:
        name = frame.name
        parent_joint = frame.parentJoint
        placement = frame.placement
        frame = pin.Frame(name, parent_joint, placement, pin.BODY)
        _ = new_model.addFrame(frame, False)

    return(new_model)

##########TEST ZONE ##########################
import unittest

class TestRobotInfo(unittest.TestCase):
    def test_getRobotInfo(self):
        name__closedloop,name_mot,number_closedloop,type=getRobotInfo(path)
        #check the model parsing
        self.assertTrue(number_closedloop==3)
        self.assertTrue(name_mot=="mot")
        self.assertTrue(name__closedloop=="fermeture")
    def test_jointTypeUpdate(self):
        new_model=jointTypeUpdate(model,rotule_name="to_rotule")
        #check that there is new spherical joint
        self.assertTrue(new_model.joints[15].nq==4) #check that joint 15 is a spherical
    def test_idmot(self):
        Lid=idmot(new_model)
        self.assertTrue(Lid==[0, 1, 4, 5, 7, 12]) #check the idmot

    def test_constraint(self):
        q=pin.neutral(new_model)
        new_data=new_model.createData()
        Lc6D=constraints6D(new_model, new_data, q)
        Lc3D=constraints3D(new_model, new_data, q)
        Lc3D=np.concatenate(Lc3D).tolist()
        Lc6D=np.concatenate(Lc6D).tolist()
        #check the constraint
        self.assertTrue(Lc3D[0]==Lc6D[0])

    def test_nameFrameConstraint(self):
        Lnom=nameFrameConstraint(new_model)
        nomf1=Lnom[0][0]
        #check the parsing
        self.assertTrue(nomf1=='fermeture1_B')

if __name__ == "__main__":
    path=os.getcwd()+"/robot_marcheur_1"
    #load robot
    robot=RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    model=robot.model
    #change joint type
    new_model=jointTypeUpdate(model,rotule_name="to_rotule")
    #run test
    unittest.main()
