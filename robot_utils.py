"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS, April 2023

Tools to load and parse a urdf file with closed loop

"""

import unittest
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import os
import re
import yaml
from yaml.loader import SafeLoader
from warnings import warn
# from pinocchio import casadi as caspin

from actuation_model import robot_actuation_model

def qfree(actuation_model, q):
    """
    free_q = q2freeq(model, q, name_mot="mot")
    Return the non-motor coordinate of q, i.e. the configuration of the non-actuated joints

    Arguments:
        model - robot model from pinocchio
        q - complete configuration vector
        name_mot - string to be found in the motors joints names
    Return:
        Lid - List of motors configuration velocity ids
    """
    Lidmot = actuation_model.idqfree
    mask = np.zeros_like(q, bool)
    mask[Lidmot] = True
    return(q[mask])

def qmot(actuation_model, q):
    """
    free_q = qmot(model, q, name_mot="mot")
    Return the non-motor coordinate of q, i.e. the configuration of the non-actuated joints

    Arguments:
        model - robot model from pinocchio
        q - complete configuration vector
        name_mot - string to be found in the motors joints names
    Return:
        Lid - List of motors configuration velocity ids
    """
    Lidmot = actuation_model.idqmot
    mask = np.zeros_like(q, bool)
    mask[Lidmot] = True
    return(q[mask])

def vmot(actuation_model, v):
    """
    free_q = qmot(model, q, name_mot="mot")
    Return the non-motor coordinate of q, i.e. the configuration of the non-actuated joints

    Arguments:
        model - robot model from pinocchio
        q - complete configuration vector
        name_mot - string to be found in the motors joints names
    Return:
        Lid - List of motors configuration velocity ids
    """
    Lidmot = actuation_model.idvmot
    mask = np.zeros_like(v, bool)
    mask[Lidmot] = True
    return(v[mask])

def vfree(actuation_model, v):
    """
    free_q = qmot(model, q, name_mot="mot")
    Return the non-motor coordinate of q, i.e. the configuration of the non-actuated joints

    Arguments:
        model - robot model from pinocchio
        q - complete configuration vector
        name_mot - string to be found in the motors joints names
    Return:
        Lid - List of motors configuration velocity ids
    """
    Lidmot = actuation_model.idvfree
    mask = np.zeros_like(v, bool)
    mask[Lidmot] = True
    return(v[mask])

def mergeq(model, actuation_model, q_mot, q_free):
    """
    completeq = (qmot,qfree)
    concatenate qmot qfree in respect with motor and free id
    """
    q=np.zeros(model.nq)
    for q_i, idqmot in zip(q_mot, actuation_model.idqmot):
        q[idqmot] = q_i

    for q_i,idqfree in zip(q_free, actuation_model.idqfree):
        q[idqfree] = q_i
    return(q)

def mergev(model, actuation_model, q_mot, q_free):
    """
    completeq = (qmot,qfree)
    concatenate qmot qfree in respect with motor and free id
    """
    v = np.zeros(model.nv)
    for v_i, idvmot in zip(q_mot, actuation_model.idvmot):
        v[idvmot] = v_i

    for q_i,idvfree in zip(q_free, actuation_model.idvfree):
        v[idvfree] = v_i
    return(v)
            

##########TEST ZONE ##########################

# class TestRobotInfo(unittest.TestCase):

#     def test_jointTypeUpdate(self):
#         new_model = jointTypeUpdate(model, rotule_name="to_rotule")
#         # check that there is new spherical joint
#         # check that joint 15 is a spherical
#         self.assertTrue(new_model.joints[15].nq == 4)

#     def test_idmot(self):
#         Lid = getMotId_q(new_model)
#         self.assertTrue(Lid == [0, 1, 4, 5, 7, 12])  # check the idmot

#     def test_nameFrameConstraint(self):
#         Lnom = nameFrameConstraint(new_model)
#         nomf1 = Lnom[0][0]
#         # check the parsing
#         self.assertTrue(nomf1 == 'fermeture1_B')


# if __name__ == "__main__":
#     path = os.getcwd()+"/robots/robot_marcheur_1"
#     # # load robot
#     completeRobotLoader(path)
#     robot = RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
#     model = robot.model
#     # change joint type
#     new_model = jointTypeUpdate(model, rotule_name="to_rotule")
#     # run test
#     unittest.main()

