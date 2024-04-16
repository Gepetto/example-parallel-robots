"""
-*- coding: utf-8 -*-
Virgile Batto & Ludovic De Matteis - September 2023

Define an actuation model, usefull for closed kinematic loops robot or underactuated robots.
"""

import numpy as np


class ActuationModel:
    """
    Defines the actuation model of a robot
    robot_actuation_model = ActuationModel(model, names)
    Arguments:
        model - robot model
        names - list of the identifiers of motor joints names
    Attributes:
        self.mot_joints_ids - list of ids of the actuated joints
        self.mot_ids_q - list of the indexes of the articular configuration values corresponding to actuated joints
        self.mot_ids_v - list of the indexes of the articular velocities values corresponding to actuated joints
        self.free_ids_q - list of the indexes of the articular configuration values corresponding to non-actuated joints
        self.free_ids_v - list of the indexes of the articular velocities values corresponding to non-actuated joints
    Methodes:
        None outside those called during init

    """

    def __init__(self, model, names):
        self.model = model
        self.mot_joints_ids = []
        self.getMotId_q(model, names)
        self.getFreeId_q(model)
        self.getMotId_v(model, names)
        self.getFreeId_v(model)
        self.mot_joints_names = names

    def __str__(self):
        return print(
            "Id q motor: " + str(self.mot_ids_q) + "\r"
            "Id v motor: " + str(self.mot_ids_v)
        )

    def getMotId_q(self, model, motnames):
        """
        getMotId_q(self[ActuationModel], model, motnames)
        Return a list of ids corresponding to the configuration indexes associated with motors joints

        Arguments:
            model - robot model from pinocchio
            motnames - list of the identifiers of actuated joints
        Return:
            None - Update self.mot_ids_q
        """
        ids_q = []
        for i, name in enumerate(model.names):
            for motname in motnames:
                if motname == name:
                    self.mot_joints_ids.append(i)
                    idq = model.joints[i].idx_q
                    nq = model.joints[i].nq
                    for j in range(nq):
                        ids_q.append(idq + j)
        self.mot_ids_q = np.unique(ids_q)

    def getMotId_v(self, model, motnames):
        """
        getMotId_v(self[ActuationModel], model, motnames)
        Return a list of ids corresponding to the articular velocity indexes associated with motors joints

        Arguments:
            model - robot model from pinocchio
            motnames - list of the identifiers of actuated joints
        Return:
            None - Update self.mot_ids_v
        """
        ids_v = []
        for i, name in enumerate(model.names):
            for motname in motnames:
                if motname == name:
                    idv = model.joints[i].idx_v
                    nv = model.joints[i].nv
                    for j in range(nv):
                        ids_v.append(idv + j)
        self.mot_ids_v = np.unique(ids_v)

    def getFreeId_q(self, model):
        """
        getFreeId_q(self[ActuationModel], model)
        Return a list of ids corresponding to the configuration indexes associated with non actuated joints

        Arguments:
            model - robot model from pinocchio
        Return:
            None - Update self.free_ids_q
        """
        self.free_ids_q = []
        for i in range(model.nq):
            if i not in self.mot_ids_q:
                self.free_ids_q.append(i)

    def getFreeId_v(self, model):
        """
        getFreeId_v(self[ActuationModel], model)
        Return a list of ids corresponding to the articular velocity indexes associated with non actuated joints

        Arguments:
            model - robot model from pinocchio
        Return:
            None - Update self.free_ids_v
        """
        self.free_ids_v = []
        for i in range(model.nv):
            if i not in self.mot_ids_v:
                self.free_ids_v.append(i)


########## TEST ZONE ##########################

# Unitary tests for actuation model are included in the ones of loader_tools
