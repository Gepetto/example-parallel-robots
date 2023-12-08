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

########## TEST ZONE ##########################

# Unitary tests for actuation model are included in the ones of loader_tools
