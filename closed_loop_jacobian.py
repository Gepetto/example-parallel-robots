"""
-*- coding: utf-8 -*-
Virgile BATTO, April 2023

tools to compute of jacobian inside closed loop

"""
import pinocchio as pin
import numpy as np
from numpy.linalg import norm
from robot_info import *
from closed_loop_kinematics import *
from pinocchio.robot_wrapper import RobotWrapper

def jacobianFinitDiffClosedLoop(model, idframe: int, idref: int, qmot: np.array,q_prec, dq=1e-6,name_mot='mot',fermeture='fermeture'):
    """
    J=Jacobian_diff_finis(robot ,idframe: int,idref :int,qo :np.array,dq: float)
    return the jacobian of the frame id idframe in the reference frame number idref, with the configuration of the robot rob qo
    """
    LJ = []  # the transpose of the Jacobian ( list of list)

    data = model.createData()
    q,b=closedLoopForwardKinematics(model, data, qmot, q_prec, name_mot, fermeture)
    pin.framesForwardKinematics(model, data, q)
    oMf1 = data.oMf[idframe].copy()  # require to avoid bad pointing
    oMrep = data.oMf[idref].copy()

    RrefXframe = (oMrep.inverse() * oMf1).action
    Lidmot=getMotId_q(model,name_mot)
    for i in range(len(Lidmot)):  # finit difference algorithm
        qmot[i] = qmot[i] + dq
        nq,b=closedLoopForwardKinematics(model, data, qmot, q_prec, name_mot, fermeture)
        pin.framesForwardKinematics(model, data, nq)
        oMf1p = data.oMf[idframe]
        V = pin.log(oMf1.inverse() * oMf1p).vector / dq

        LJ.append(V.tolist())
        qmot[i] = qmot[i] - dq
    
    J = np.transpose(np.array(LJ))
    J = RrefXframe @ J
    return J


def sepJc(model,actuation_model,Jn):
    """
    Jmot,Jfree=sepJc(model,Jn)

    take a constraint Jacobian, and separate it into Jcmot and Jcfree , the constrant jacobian associate to motor and free joint
    
    """
    Lidmot=actuation_model.idvmot
    LJT=Jn.T.tolist()
    Jmot=np.zeros((6,len(Lidmot)))
    Jfree=np.zeros((6,model.nv-len(Lidmot)))
    imot=0
    ifree=0
    for i,col in enumerate(LJT):
        if i in Lidmot:
            Jmot[:,imot]=col
            imot+=1
        else:
            Jfree[:,ifree]=col
            ifree+=1
    return(Jmot,Jfree)

def dqRowReorder(model,actuation_model,dq):
    """
    q=dqRowReorder(model,dq)
    take a voector/matrix organisate as [dqmot dqfree]
    return q reorganized in accordance with the model
    """
    nJ=dq.copy()
    Lidmot=actuation_model.idvmot
    Lidfree=[]
    for i in range(model.nq):
        if not(i in Lidmot):
            Lidfree.append(i)
    imot=0
    ifree=0
    for i,dq in enumerate(dq.tolist()):
        if i<len(Lidmot):
            nJ[Lidmot[imot]]=dq
            imot+=1
        else:
            nJ[Lidfree[ifree]]=dq
            ifree+=1
    return(nJ)


def dq_dqmot(model,actuation_model,LJ):
    """
    take J the constraint Jacobian and return dq/dqmot
    
    """
    Lidmot=actuation_model.idvmot
    Jmot=np.zeros((0,len(Lidmot)))
    Jfree=np.zeros((0,model.nv-len(Lidmot)))
    for J in LJ:
        [mot,free]=sepJc(model,actuation_model,J)
        Jmot=np.concatenate((mot,Jmot))
        Jfree=np.concatenate((free,Jfree))
    
    I=np.identity(len(Lidmot))
    pinvJfree=np.linalg.pinv(Jfree)
    dq=np.concatenate((I,-pinvJfree@Jmot))
    dq=dqRowReorder(model,actuation_model,dq)
    return(dq)


def inverseConstraintKinematicsSpeed(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff):
    """
    vq,Jf_closed=inverseConstraintKinematics(model,data,constraint_model,constraint_data,q0,ideff,veff,name_mot="mot")

    compute the joint velocity vq that generate the speed veff on frame ideff.
    return also the closed loop jacobian of this frame 


    """
    pin.computeJointJacobians(model,data,q0)
    LJ=[]
    for (cm,cd) in zip(constraint_model,constraint_data):
        Jc=pin.getConstraintJacobian(model,data,cm,cd)
        LJ.append(Jc)

    Lidmot=actuation_model.idvmot
    dq_dmot=dq_dqmot(model,actuation_model,LJ)

    Jf=pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL)
    Jf_closed=Jf@dq_dmot
    vqmot=np.linalg.pinv(Jf_closed)@veff 

    Jmot=np.zeros((0,len(Lidmot)))
    Jfree=np.zeros((0,model.nv-len(Lidmot)))
    for J in LJ:
        [mot,free]=sepJc(model,actuation_model,J)
        Jmot=np.concatenate((Jmot,mot))
        Jfree=np.concatenate((Jfree,free))
    vqfree=-np.linalg.pinv(Jfree)@Jmot@vqmot
    vqmotfree=np.concatenate((vqmot,vqfree))  # qmotfree=[qmot qfree]
    vq=dqRowReorder(model,actuation_model,vqmotfree)
    return(vq,Jf_closed)






##########TEST ZONE ##########################
import unittest

class TestRobotInfo(unittest.TestCase):
    #only test inverse constraint kineatics because it runs all precedent code
    def test_inverseConstraintKinematics(self):
        vapply=np.array([0,0,1,0,0,0])
        vq=inverseConstraintKinematicsSpeed(model,data,constraint_models,constraint_datas,actuation_model,q0,34,vapply)[0]
        pin.computeAllTerms(model,data,q0,vq)
        vcheck=data.v[13].np #frame 34 is center on joint 13
        #check that the computing vq give the good speed 
        self.assertTrue(norm(vcheck-vapply)<1e-6)


if __name__ == "__main__":
    #load robot
    path = os.getcwd()+"/robots/robot_marcheur_1"
    model,constraint_models,actuation_model,visual_model=completeRobotLoader(path)
    data=model.createData()
    constraint_datas=[cm.createData() for cm in constraint_models]
    q0=proximalSolver(model,data,constraint_models,constraint_datas)
    
    
    #test
    unittest.main()

