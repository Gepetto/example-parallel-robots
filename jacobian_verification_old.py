import pinocchio as pin
import numpy as np
import math as ma
from pinocchio.robot_wrapper import RobotWrapper
import meshcat
from scipy.optimize import fmin_slsqp, fmin_bfgs
from numpy.linalg import inv, pinv, norm, eig, svd
import example_robot_data as robex
from pinocchio.visualize import MeshcatVisualizer
import time
import hppfcl
import matplotlib.pyplot as plt
import re
import os
import copy 
import sys
pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=3, linewidth=300, suppress=True,threshold=10000)

# Import and init of the robot ( a simple 4 link parralel robot with 2 motor )
# the articulation with 1 or A on the right arm, articulation with 2 or B on the left arm 


def get_simplified_robot(path):
    '''
    robot=get_simplified_robot(path)
    path, the dir of the file that contain the urdf file & the stl files
    

    load a robot with 1 closed loop with a joint on each of the 2 branch that are closed, return a simplified model of the robot with 1 joint 
    that closed the loop
    '''
    path = path+'/robot_simple'
    robot = RobotWrapper.BuildFromURDF(path+"/robot.urdf", path)

    #to simplifie the conception, the two contact point are generate with a joint
    #supression of one of this joint :
    for (joint,id) in zip(robot.model.names,range(len(robot.model.names))):
        match=re.search("fermeture1_A",joint)
        if match :
            robot.model,robot.visual_model=pin.buildReducedModel(robot.model, robot.visual_model, [id], np.zeros(6))
            robot.data=pin.Data(robot.model) 
            break
    robot.q0=np.zeros(robot.nq)
    return(robot)

def Jf_calc_LWA(lbarre :float,lbarre_eff :float,q):
    '''
    Jf=Jf_calc_LWA(lbarre,lbarre_eff,q)
    compute the JAcobian of the effector frame (frame 14) of the robot
    '''
    Jf_calcT=np.array([[0., 0.                                            , 0.                                            , 0., 0., 0.],
                        [0., 0.                                            , 0.                                            , 0., 0., 0.], 
                        [0., lbarre*np.cos(q[2])+lbarre_eff*np.cos(q[2]+q[3]), lbarre*np.sin(q[2])+lbarre_eff*np.sin(q[2]+q[3]), 1., 0., 0.],
                        [0., lbarre_eff* np.cos(q[2]+q[3])                   , lbarre_eff*np.sin(q[2]+q[3])                    , 1., 0., 0.],
                        [0., 0.                                            , 0.                                            , 0., 0., 0.]])
                   
    Jf_calc=np.transpose(Jf_calcT)
    return(Jf_calc)




def Jb_calc_LWA(lbarre :float,q):
    '''
    Jb=Jb_calc_LWA(lbarre,q)
    compute the JAcobian of the frame 15 of the robot
    '''
    Jb_calcT=np.array([[0., 0.                                            , 0.                                            , 0., 0., 0.],
                      [0., 0.                                            , 0.                                            , 0., 0., 0.],
                      [0., lbarre*np.cos(q[2])+lbarre*np.cos(q[2]+q[3])  , lbarre*np.sin(q[2])+lbarre*np.sin(q[2]+q[3])  , 1., 0., 0.],
                      [0., lbarre* np.cos(q[2]+q[3])                     , lbarre*np.sin(q[2]+q[3])                      , 1., 0., 0.],
                      [0., 0.                                            , 0.                                            , 1., 0., 0.]])
    Jb_calc=np.transpose(Jb_calcT)
    return(Jb_calc)



def Ja_calc_LWA(lbarre,q):
    '''
    Ja=Ja_calc_LWA(lbarre,q)
    compute the Jacobian of the frame 1 of the robot
    '''
    Ja_calcT=np.array([[0., lbarre*np.cos(q[0])+lbarre*np.cos(q[0]+q[1])  , lbarre*np.sin(q[0])+lbarre*np.sin(q[0]+q[1])  , 1., 0., 0.],
                      [0., lbarre* np.cos(q[0]+q[1])                     , lbarre*np.sin(q[0]+q[1])                      , 1., 0., 0.],
                      [0., 0.                                            , 0.                                            , 0., 0., 0.],
                      [0., 0.                                            , 0.                                            , 0., 0., 0.],
                      [0., 0.                                            , 0.                                            , 0., 0., 0.]])
    Ja_calc=np.transpose(Ja_calcT)
    return(Ja_calc)



def aXb(ida :int,idb :int,qo,rob):
    '''
    aXb=aXb(ida,idb,qo,rob)
    return the transfert matrix that change the aplplication point of the 6d speed
    '''
    oMb =rob.framePlacement(qo, idb).copy() # require to avoid bad pointing 
    oMa=rob.framePlacement(qo, ida).copy()
    dep=(oMa.inverse() * oMb).translation
    aXb=np.array([  [1,  0,  0,        0,  -dep[2] ,dep[1]  ],
                          [0,  1,  0,   dep[2],     0   , -dep[0] ],
                          [0,  0,  1,  -dep[1],   dep[0],       0 ],
                          [0,  0,  0,        1,        0,       0 ],
                          [0,  0,  0,        0,        1,       0 ],
                          [0,  0,  0,        0,        0,       1 ]])    
    return(aXb)


def aRb(ida :int,idb :int,qo,rob):
    '''
    aRb=aRb(ida,idb,qo,rob)
    return the 6D Rotation matrix
    '''

    oMb =rob.framePlacement(qo, idb).copy() # require to avoid bad pointing 
    oMa=rob.framePlacement(qo, ida).copy()
    rot=(oMa.inverse() *  oMb).rotation
    R=np.concatenate((np.concatenate((rot,np.zeros([3,3])),axis=1),np.concatenate((np.zeros([3,3]),rot),axis=1))) 
    return(R)




def Jacobian_finit_diff(model,idframe: int,idref :int,qo :np.array,dq=1e-6):
    '''
    J=Jacobian_diff_finis(robot ,idframe: int,idref :int,qo :np.array,dq: float)
    return the jacobian of the frame id idframe in the reference frame number idref, with the configuration of the robot rob qo
    '''
    LJ=[] #the transpose of the Jacobian ( list of list)
    
    data=model.createData()
    pin.framesForwardKinematics(model,data,qo)
    oMf1 = data.oMf[idframe].copy() # require to avoid bad pointing 
    oMrep= data.oMf[idref].copy()
    dep=(oMrep.inverse() * oMf1).translation
    rot=(oMrep.inverse() *  oMf1).rotation

    refXframe=np.array([  [1,  0,  0,        0,  -dep[2] ,dep[1]  ],
                          [0,  1,  0,   dep[2],     0   , -dep[0] ],
                          [0,  0,  1,  -dep[1],   dep[0],       0 ],
                          [0,  0,  0,        1,        0,       0 ],
                          [0,  0,  0,        0,        1,       0 ],
                          [0,  0,  0,        0,        0,       1 ]])

    R=np.concatenate((np.concatenate((rot,np.zeros([3,3])),axis=1),np.concatenate((np.zeros([3,3]),rot),axis=1)))                                       
    RrefXframe= R @ refXframe

    for i in range(model.nq):  #finit difference algorithm
        qo[i]=qo[i]+dq
        pin.framesForwardKinematics(model,data,qo)
        oMf1p=data.oMf[idframe]
        V=pin.log(oMf1.inverse() * oMf1p).vector/dq
        
        LJ.append(V.tolist())
        qo[i]=qo[i]-dq

    J=np.transpose(np.array(LJ))
    J=RrefXframe @ J

  #the forwardkinematics of the robot distoted the data and the model of the robot, example without the copy and paste :
                    #Jacobian_diff_finis(robot,15,15,qo) (first execution) != Jacobian_diff_finis(robot,15,15,qo) (every other execution)
    return(J)




#creation of the robot
path = os.getcwd()
robot=get_simplified_robot(path)
model=robot.model
data=model.createData()

#initialisation 
q=np.ones(robot.nq)
pin.forwardKinematics(model,data,q)
pin.computeAllTerms(model,data,q,np.zeros(robot.nv))
#creation of constraint

ida=model.getFrameId('fermeture1_A')
idb=model.getFrameId('fermeture1_B')
j1Mca=model.frames[ida].placement
j2Mcb=model.frames[idb].placement
idj1=model.frames[ida].parentJoint
idj2=model.frames[idb].parentJoint
constraint_local=pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,idj1,j1Mca,idj2,j2Mcb,pin.ReferenceFrame.LOCAL)
constraint_local_world_aligned=pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,idj1,j1Mca,idj2,j2Mcb,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
data_constraint_local=pin.RigidConstraintData(constraint_local)
data_constraint_local_world_aligned=pin.RigidConstraintData(constraint_local_world_aligned)


#constraint jacobian automatique (ref frame: same as contact)
Jconst_auto_local=pin.getConstraintJacobian(model,data,constraint_local,data_constraint_local)
Jconst_auto_local_world_aligned=pin.getConstraintJacobian(model,data,constraint_local_world_aligned,data_constraint_local_world_aligned)


#creation of ext force on the effector_frame
f=pin.Force(np.array([10,0,0,0,0,0]))
idframe_effector=model.getFrameId("effecteur")
Mf = data.oMf[idframe_effector]
Rf = Mf.rotation
Jf=pin.computeFrameJacobian(model,data,q,idframe_effector,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
Jf_local=pin.computeFrameJacobian(model,data,q,idframe_effector,pin.ReferenceFrame.LOCAL)
Jf_world=pin.computeFrameJacobian(model,data,q,idframe_effector,pin.ReferenceFrame.WORLD)

#Jacobian of the frame in wich contact is imposed:
Ja      =pin.computeFrameJacobian(model,data,q,ida,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
Jb      =pin.computeFrameJacobian(model,data,q,idb,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
Ja_local=pin.computeFrameJacobian(model,data,q,ida,pin.ReferenceFrame.LOCAL)
Jb_local=pin.computeFrameJacobian(model,data,q,idb,pin.ReferenceFrame.LOCAL)
Jb_world=pin.computeFrameJacobian(model,data,q,idb,pin.ReferenceFrame.WORLD)
Ja_world=pin.computeFrameJacobian(model,data,q,ida,pin.ReferenceFrame.WORLD)

#cosntraint jacobian
Jconst_local_world_aligned=-Ja+Jb




#size of the rode of the robot 
lbarre=max(abs(model.frames[idb].placement.translation[1]),abs(model.frames[ida].placement.translation[1]))
lbarre_eff=abs(model.frames[idframe_effector].placement.translation[1])


#Jacobian calcul by hand Local world aligned
Jf_calc_local_world_aligned=Jf_calc_LWA(lbarre,lbarre_eff,q)
Jb_calc_local_world_aligned=Jb_calc_LWA(lbarre,q)
Ja_calc_local_world_aligned=Ja_calc_LWA(lbarre,q)

Jf_calc_world= aXb(0,idframe_effector,q,robot) @ Jf_calc_local_world_aligned
Jb_calc_world= aXb(0,idb,q,robot)              @ Jb_calc_local_world_aligned
Ja_calc_world= aXb(0,ida,q,robot)              @ Ja_calc_local_world_aligned

#Constraint jacobian calcu by hand
Jconst_calc_local_world_aligned=-Ja_calc_local_world_aligned+Jb_calc_local_world_aligned



#finit differences

dq=1e-3
Jcons_diff_finis_local=-Jacobian_finit_diff(model,idb,ida,q,dq)+Jacobian_finit_diff(model,ida,ida,q,dq)

Ja_diff_finis_local=Jacobian_finit_diff(model,ida,ida,q,dq)
Jb_diff_finis_local=Jacobian_finit_diff(model,idb,idb,q,dq)


Ja_diff_finis_world=Jacobian_finit_diff(model,ida,0,q,dq)
Jb_diff_finis_world=Jacobian_finit_diff(model,idb,0,q,dq)




if __name__=='__main__':

    print("precision jacobienne Local World Aligned")
    print(np.max([np.max(Jf-Jf_calc_local_world_aligned),np.max(Jb-Jb_calc_local_world_aligned),np.max(Ja-Ja_calc_local_world_aligned)]))

    print("precision jacobienne World ")
    print(np.max([np.max(Jf_world-Jf_calc_world),np.max(Jb_world-Jb_calc_world),np.max(Ja_world-Ja_calc_world)]))


    print("validation of Local Jacobain (max(Jfinit_diff-Jpinocchio)")
    print(np.max([np.max(Jb_local-Jb_diff_finis_local),np.max(Ja_local-Ja_diff_finis_local)]))


    print("validation of the rotation part of the local constraint jacobian (max(Jrot_finit_diff-Jrot_pinocchio)")
    print(np.max(Jcons_diff_finis_local[3:,:]-Jconst_auto_local[3:,:]))



