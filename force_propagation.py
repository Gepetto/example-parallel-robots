import pinocchio as pin
import numpy as np
from numpy.linalg import norm
from robot_info import *
from closed_loop_forward_kin import *
from closed_loop_jacobian import *
from pinocchio.robot_wrapper import RobotWrapper

def closedLoopForcePropagation(model,data,constraint_model,constraint_data,q,vq,Lidf,Lf,name_mot="mot"):
    """
    closedLoopForcePropagation(model,data,constraint_model,constraint_data,q,vq,Lidf,Lf,name_mot="mot")
    propagate the force apply on frame idf (Lidf=[id1,..,idf]) with value fn (Lf=[f1,..,fn]) inside a model with constraint
    resul stored in data 
    """
    Lidmot=idmot(model,name_mot)
    qmot=np.zeros(len(Lidmot))
    for id,i in enumerate(Lidmot):
            qmot[id]=q[i]
    
    LJ=[]
    for (cm,cd) in zip(constraint_model,constraint_data):
        Jc=pin.getConstraintJacobian(model,data,cm,cd)
        LJ.append(Jc)

    dq_mot=dq_dqmot(model,LJ)

    taumot=np.zeros(len(Lidmot))
    for idf,f in zip(Lidf,Lf):
        J=pin.computeFrameJacobian(model,data,q,idf,pin.LOCAL)
        J_closed=J@dq_mot

        tauq_inter=J_closed.transpose()@f.np  #tmotor torque gerenrate by the force
        taumot=tauq_inter+taumot

    tauq=np.zeros(model.nq)
    j=0
    for i in Lidmot:
        tauq[i]=taumot[j]
        j=j+1

    a=pin.constraintDynamics(model,data,q_ini,vq,tauq,constraint_model,constraint_data) #compute the interior force generate by the ext force



    for cd,cm in zip(constraint_data,constraint_model): #creation of exterior forces for each constraint
        force=cd.contact_force
        ida=cm.joint1_id
        idb=cm.joint2_id
        j1Mp=cm.joint1_placement
        j2Mp=cm.joint2_placement
        Lf[ida]+=pin.Force(j1Mp.actionInverse.transpose()@force.np) #transport of force to the parent joint
        Lf[idb]-=pin.Force(j2Mp.actionInverse.transpose()@force.np)

    tau=pin.rnea(model,data,q_ini,vq,a*0,Lf) #propagation of interiror forces
    return()

##########TEST ZONE ##########################
#No test yet