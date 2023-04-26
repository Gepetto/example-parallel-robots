"""
-*- coding: utf-8 -*-
Virgile BATTO, march 2022

tools to compute the force propagation inside a robot with closed loop
(not sure of the result yet...)

"""

import pinocchio as pin
import numpy as np
from robot_info import *
from closed_loop_kinematics import *
from closed_loop_jacobian import *
from pinocchio.robot_wrapper import RobotWrapper

pin.SE3.__repr__ = pin.SE3.__str__
def closedLoopForcePropagation(model,data,constraint_model,constraint_data,q,vq,Lidf=[],Lf=[],name_mot="mot"):
    """
    closedLoopForcePropagation(model,data,constraint_model,constraint_data,q,vq,Lidf,Lf,name_mot="mot")
    Propagate the force applied on frame idf (Lidf=[id1,..,idf]) with value fn (Lf=[f1,..,fn]) inside a model with constraints
    The result is stored in data 
    """
    Lidvmot=getMotId_v(model,name_mot)
    
    LJ=[]
    for (cm,cd) in zip(constraint_model,constraint_data):
        Jc=pin.getConstraintJacobian(model,data,cm,cd)
        LJ.append(Jc)

    dq_mot=dq_dqmot(model,LJ)
    

    #obtain the tau mot that generate the Lf
    taumot=np.zeros(len(Lidvmot))
    for idf,f in zip(Lidf,Lf):
        J=pin.computeFrameJacobian(model,data,q,idf,pin.LOCAL)
        J_closed=J@dq_mot

        tauq_inter=J_closed.transpose()@f.np  #tmotor torque gerenrate by the force
        taumot=tauq_inter+taumot

    #obtain the contact force from the torque 
    tauq=np.zeros(model.nv)
    j=0
    for i in Lidvmot:
        tauq[i]=taumot[j]
        j=j+1
    pin.initConstraintDynamics(model, data, constraint_model)
    a=pin.constraintDynamics(model,data,q,vq,tauq,constraint_model,constraint_data) #compute the interior force generate by the ext force
    
    #generate all exterior forces 
    Lfjoint=[]
    for i in range(robot.nq+1): #init of ext force
        Lfjoint.append(pin.Force.Zero())

    for force,id in zip(Lf,Lidf):
        j1Mp=model.frames[id].placement
        jforce=pin.Force(j1Mp.actionInverse.transpose()@force.np)
        Lfjoint[model.frames[15].parentJoint]=jforce

    for cd,cm in zip(constraint_data,constraint_model): #creation of exterior forces for each constraint
        force=cd.contact_force
        ida=cm.joint1_id
        idb=cm.joint2_id
        j1Mp=cm.joint1_placement
        j2Mp=cm.joint2_placement
        Lfjoint[ida]+=pin.Force(j1Mp.actionInverse.transpose()@force.np) #transport of force to the parent joint
        Lfjoint[idb]-=pin.Force(j2Mp.actionInverse.transpose()@force.np)


    for id,f in enumerate(Lfjoint):
        Lfjoint[id]=pin.Force(-f.np) #inversion to obtain the forces generate by the exterior force

    pin.rnea(model,data,q,vq,a*0,Lfjoint) #propagation of interiror forces in statix case


##########TEST ZONE ##########################
import unittest

class TestRobotInfo(unittest.TestCase):
    #only test inverse constraint kineatics because it runs all precedent code
    def test_forcepropagation(self):
        vapply=np.array([0,0,1,0,0,0])
        vq=inverseConstraintKinematicsSpeed(new_model,new_data,constraint_model,constraint_data,q0,34,vapply,name_mot="mot")[0]
        closedLoopForcePropagation(new_model,new_data,constraint_model,constraint_data,q0,vq)
        #check that the computing vq give the good speed 
        self.assertTrue(norm(new_data.f[5])>-1)


if __name__ == "__main__":
    #load robot
    path=os.getcwd()+"/robot_marcheur_1"
    robot=RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    model=robot.model
    visual_model = robot.visual_model
    new_model=jointTypeUpdate(model,rotule_name="to_rotule")
    
    #No gravity
    gravity=pin.Motion(np.zeros(6))
    new_model.gravity=gravity

    #create data
    new_data=new_model.createData()

    #create variable use by test
    Lidmot=getMotId_q(new_model)

    #init of the robot
    goal=np.zeros(len(Lidmot))
    q_prec=q2freeq(new_model,pin.neutral(new_model))
    q0, q_ini= closedLoopForwardKinematics(new_model, new_data,goal,q_prec=q_prec)
    vq=np.zeros(new_model.nv)
    #init of constraint
    name_constraint=nameFrameConstraint(new_model)
    constraint_model=getConstraintModelFromName(new_model,name_constraint)
    constraint_data = [c.createData() for c in constraint_model]
    Lidf=[36] # frame bout pied
    Lf=[pin.Force(np.array([0,0,1,0,0,0]))]
    closedLoopForcePropagation(new_model,new_data,constraint_model,constraint_data,q0,vq,Lidf,Lf)
    #test
    unittest.main()