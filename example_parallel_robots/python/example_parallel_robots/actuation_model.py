'''
-*- coding: utf-8 -*-
Virgile Batto & Ludovic De Matteis - September 2023

Define an actuation model, usefull for closed kinematic loops robot or underactuated robots.
'''

import numpy as np
class ActuationModel():
    """
    Defines the actuation model of a robot
    robot_actuation_model = ActuationModel(model, names)
    Arguments:
        model - robot model
        names - list of the identifiers of motor joints names
    Attributes:
        self.idMotJoints - list of ids of the actuated joints
        self.idqmot - list of the indexes of the articular configuration values corresponding to actuated joints
        self.idvmot - list of the indexes of the articular velocities values corresponding to actuated joints
        self.idqfree - list of the indexes of the articular configuration values corresponding to non-actuated joints
        self.idvfree - list of the indexes of the articular velocities values corresponding to non-actuated joints
    Methodes:
        None outside those called during init
    
    """
    def __init__(self, model, names):
        self.idMotJoints = []
        self.getMotId_q(model, names)
        self.getFreeId_q(model)
        self.getMotId_v(model, names)
        self.getFreeId_v(model)

        
    def __str__(self):
        return(print("Id q motor: " + str(self.idqmot) + "\r" "Id v motor: " + str(self.idvmot) ))
    

    def getMotId_q(self, model, motnames):
        """
        getMotId_q(self[ActuationModel], model, motnames)
        Return a list of ids corresponding to the configuration indexes associated with motors joints

        Arguments:
            model - robot model from pinocchio
            motnames - list of the identifiers of actuated joints
        Return:
            None - Update self.idqmot
        """
        Lidq = []
        for i, name in enumerate(model.names):
            for motname in motnames:
                if motname in name:
                    self.idMotJoints.append(i)
                    idq = model.joints[i].idx_q
                    nq = model.joints[i].nq
                    for j in range(nq):
                        Lidq.append(idq+j)
        self.idqmot=np.unique(Lidq)

    def getMotId_v(self, model, motnames):
        """
        getMotId_v(self[ActuationModel], model, motnames)
        Return a list of ids corresponding to the articular velocity indexes associated with motors joints

        Arguments:
            model - robot model from pinocchio
            motnames - list of the identifiers of actuated joints
        Return:
            None - Update self.idvmot
        """
        Lidv = []
        for i, name in enumerate(model.names):
            for motname in motnames:
                if motname in name:
                    idv = model.joints[i].idx_v
                    nv = model.joints[i].nv
                    for j in range(nv):
                        Lidv.append(idv+j)
        self.idvmot=np.unique(Lidv)

    def getFreeId_q(self, model):
        """
        getFreeId_q(self[ActuationModel], model)
        Return a list of ids corresponding to the configuration indexes associated with non actuated joints

        Arguments:
            model - robot model from pinocchio
        Return:
            None - Update self.idqfree
        """
        Lidq=[]
        for i in range(model.nq):
            if not(i in self.idqmot):
                Lidq.append(i)
        self.idqfree=Lidq
        return(Lidq)
    
    def getFreeId_v(self, model):
        """
        getFreeId_v(self[ActuationModel], model)
        Return a list of ids corresponding to the articular velocity indexes associated with non actuated joints

        Arguments:
            model - robot model from pinocchio
        Return:
            None - Update self.idvfree
        """
        Lidv=[]
        for i in range(model.nv):
            if not(i in self.idvmot):
                Lidv.append(i)
        self.idvfree=Lidv


import pinocchio as pin

class ActuationData():
    def __init__(self, model, constraint_model,actuation_model):

        Lidmot=actuation_model.idvmot
        Lidfree=actuation_model.idvfree
        nv=model.nv
        nv_mot=actuation_model.nv_mot
        nv_free=actuation_model.nv_free
        nc=0
        for c in constraint_model:
            nc+=c.size()

        ## init different matrix
        self.Smot=np.zeros((nv,nv_mot))
        self.Sfree=np.zeros((nv,nv_free))
        self.Jmot=np.zeros((nc,nv_mot))
        self.Jfree=np.zeros((nc,nv_free))
        self.Mmot=np.zeros((nv_mot,nv_mot))
        self.dq=np.zeros((nv,nv_mot))

        #PRIVATE
        self.dq_no=np.zeros((nv,nv_mot))
    
        #init a list of constraint_jacobian
        self.LJ=[np.array(())]*len(constraint_model)
        constraint_data=[c.createData() for c in constraint_model]
        data=model.createData()
        for (cm,cd,i) in zip(constraint_model,constraint_data,range(len(self.LJ))):
            self.LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)


        #selection matrix for actuated parallel robot
        self.Smot[:,:]=0
        self.Smot[Lidmot,range(nv_mot)]=1
        self.Sfree[:,:]=0
        self.Sfree[Lidfree,range(nv_free)]=1


        # to delete
        self.Jf_closed=pin.computeFrameJacobian(model,model.createData(),np.zeros(model.nq),0,pin.LOCAL)@self.dq


        # init of different size of vector
        self.vq=np.zeros(nv)
        self.vqmot=np.zeros(nv_mot-6)
        self.vqfree=np.zeros(nv_free)
        self.vqmotfree=np.zeros(nv-6)

        #list of constraint type
        self.Lnc=[J.shape[0] for J in self.LJ]


        #to delete ?
        self.pinvJfree=np.linalg.pinv(self.Jfree)
########## TEST ZONE ##########################

# Unitary tests for actuation model are included in the ones of loader_tools
