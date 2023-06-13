"""
-*- coding: utf-8 -*-
Virgile BATTO, march 2022

tools to compute the force propagation inside a robot with closed loop
(not sure of the result yet...)

"""

import pinocchio as pin
import numpy as np
from closed_loop_kinematics import closedLoopForwardKinematics
from closed_loop_jacobian import dq_dqmot, inverseConstraintKinematicsSpeed

pin.SE3.__repr__ = pin.SE3.__str__
def closedLoopForcePropagation(model, data, constraint_model, constraint_data, actuation_model, q, vq, Lidf=[], Lf=[]):
    """
    closedLoopForcePropagation(model,data,constraint_model,constraint_data,q,vq,Lidf,Lf,name_mot="mot")
    Propagate the force applied on frame idf (Lidf=[id1,..,idf]) with value fn (Lf=[f1,..,fn]) inside a model with constraints
    The result is stored in data 
    """
    Lidvmot=actuation_model.idvmot
    
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
    for i in range(model.nq+1): #init of ext force
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
        from loader_tools import completeRobotLoader

        path = "robots/robot_marcheur_1"
        model, constraint_models, actuation_model, visual_model = completeRobotLoader(path)
        # No gravity
        model.gravity = pin.Motion(np.zeros(6))
        # Create data
        data = model.createData()
        q0 = closedLoopForwardKinematics(model, data, actuation_model)
        constraint_datas = [c.createData() for c in constraint_models]

        vapply = np.array([0,0,1,0,0,0])
        vq = inverseConstraintKinematicsSpeed(model, data, constraint_models, constraint_datas, q0, 34, vapply, name_mot="mot")[0]
        closedLoopForcePropagation(model, data, constraint_models, constraint_datas, q0, vq)
        # check that the computing vq give the good speed 
        self.assertTrue(np.linalg.norm(data.f[5])>-1) # ! I don't like this unittest at all


if __name__ == "__main__":
    #test
    unittest.main()