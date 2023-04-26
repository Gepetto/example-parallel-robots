"""
-*- coding: utf-8 -*-
Virgile BATTO, march 2022

Tools to compute the forwark and inverse kinematics of a robot with  closed loop 

"""
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from numpy.linalg import norm
import os
from scipy.optimize import fmin_slsqp
import cyipopt
import casadi
from pinocchio import casadi as caspin

from robot_info import *
from constraints import *

def closedLoopInverseKinematics(rmodel, rdata, cmodels, cdatas, target_frame, q_prec=[], name_eff="effecteur", onlytranslation=False):
    
    """
        q=closedLoopInverseKinematics(model,data,fgoal,constraint_model,constraint_data,q_prec=[],name_eff="effecteur",nom_fermeture="fermeture",type="6D"):

        take the target position of the motors axis of the robot (joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration who match the goal position of the effector

    """
    # * Get effector frame id
    ideff = model.getFrameId(name_eff)

    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    if len(q_prec) != (rmodel.nq):
        q_prec = pin.neutral(rmodel)
    q_prec = np.array(q_prec)
    nq = rmodel.nq

    # * Set initial guess
    if len(q_prec) < model.nq:
        q_prec = pin.neutral(model)

    # * Optimisation functions
    cx = casadi.SX.sym("x", nq, 1)
    cM = casadi.SX.sym('M', 3, 3)
    caspin.framesForwardKinematics(casmodel, casdata, cx)
    tip_translation = casadi.Function('tip_trans', [cx], [casdata.oMf[ideff].translation])
    tip_rotation = casadi.Function('tip_rot', [cx], [casdata.oMf[ideff].rotation])
    log3 = casadi.Function('log3', [cM], [caspin.log3(cM)])
    def cost(q):
        terr = target_frame.translation - tip_translation(q)
        c = casadi.norm_2(terr) ** 2
        if not onlytranslation :
            print(caspin.SE3(target_frame).actInv(casdata.oMf[ideff]))
            R_err = caspin.log6(caspin.SE3(target_frame).actInv(casdata.oMf[ideff])).vector

            c = casadi.norm_2(R_err) ** 2
        return(c)

    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, cmodels, cdatas, q, recompute=True, pinspace=caspin, quaternions=True)
        return Lc
    
    constraintsCost = casadi.Function('constraint', [cx], [constraints(cx)])

    # * Optimisation problem
    optim = casadi.Opti()
    q = optim.variable(nq)
    # * Constraints
    optim.subject_to(constraintsCost(q)==0)
    # * cost minimization
    total_cost = cost(q)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    optim.set_initial(q, q_prec)
    try:
        sol = optim.solve_limited()
        print("Solution found")
        qs = optim.value(q)
    except:
        print('ERROR in convergence, press enter to plot debug info.')

    return qs


def closedLoopForwardKinematics(rmodel, rdata, cmodels, cdatas, q_mot_target, q_prec=[]):
    """
        closedLoopForwardKinematics(model, data, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"):

        Takes the target position of the motors axis of the robot (joint with name_mot ("mot" if empty) in the name),
        the current configuration of all joint (set to robot.q0 if let empty) and the name of the joints that close the kinematic loop. And returns a configuration that matches the goal positions of the motors
        This function solves a minimization problem
        TODO - writes explicitly the minimization problem

        Argument:
            model - Pinocchio robot model
            data - Pinocchio robot data
            target - Target configuration of the motors joints. Should be have size of the number of motors
            q_prec - Previous configuration of the free joints
            n_mot - String contained in the motors joints names
            nom_fermeture - String contained in the frame names that should be in contact
            type - Constraint type
    """
    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    Lid = getMotId_q(rmodel)

    Id_free = np.delete(np.arange(rmodel.nq), Lid)
    if len(q_prec) != (rmodel.nq - len(goal)):
        q_prec = q2freeq(rmodel, pin.neutral(rmodel))
    q_prec = np.array(q_prec)
    nF = len(Id_free)

    # * Optimisation functions
    # difference = casadi.Function('difference', [cx],[caspin.difference(robot.casmodel, cx[:nq], casadi.SX(robot.q0))])
    def cost(qF):
        return(casadi.norm_2(qF)**2)
    
    def constraints(qF):
        q = casadi.SX.sym('q', rmodel.nq, 1)
        q[Lid] = q_mot_target
        q[Id_free] = qF
        Lc = constraintsResidual(casmodel, casdata, cmodels, cdatas, q, recompute=True, pinspace=caspin, quaternions=True)
        return Lc
    
    cx = casadi.SX.sym("x", nF, 1)
    constraintsCost = casadi.Function('constraint', [cx], [constraints(cx)])

    # * Optimisation problem
    optim = casadi.Opti()
    qF = optim.variable(nF)
    # * Constraints
    optim.subject_to(constraintsCost(qF)==0)
    # * Bounds
    optim.subject_to(qF>-1*np.pi)
    optim.subject_to(qF<1*np.pi)
    # * cost minimization
    total_cost = cost(qF)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    optim.set_initial(qF, q_prec)
    try:
        sol = optim.solve_limited()
        print("Solution found")
        qFs = optim.value(qF)
    except:
        print('ERROR in convergence, press enter to plot debug info.')

    q = np.empty(rmodel.nq)
    q[Lid] = q_mot_target
    q[Id_free] = qFs
    return(q, qFs)


##########TEST ZONE ##########################
import unittest

class TestRobotInfo(unittest.TestCase):
    def test_inversekinematics(self):
        InvKin = closedLoopInverseKinematics(new_model, new_data, cmodels, cdatas, fgoal, q_prec=q_prec, name_eff=frame_effector, onlytranslation=False)
        print(InvKin)
        self.assertTrue(InvKin[3]==0) # check that joint 15 is a spherical

    # def test_forwarkinematics(self):
    #     q_opt, qF_opt=closedLoopForwardKinematics(new_model, new_data, cmodels, cdatas, goal, q_prec=q_prec)
    #     constraint=norm(constraintsResidual(new_model, new_data, cmodels, cdatas, q_opt, recompute=True, pinspace=pin, quaternions=True))
    #     self.assertTrue(constraint<1e-6) #check the constraint


if __name__ == "__main__":
    # * Import robot
    path = os.getcwd()+"/robot_marcheur_1"
    robot = RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    model = robot.model

    # * Change the joint type
    new_model = jointTypeUpdate(model,rotule_name="to_rotule")
    new_data = new_model.createData()

    # * Create robot constraint models
    Lnames = nameFrameConstraint(robot.model)
    constraints = getConstraintModelFromName(robot.model, Lnames, const_type=pin.ContactType.CONTACT_6D)
    robot.constraint_models = cmodels = constraints
    robot.full_constraint_models = robot.constraint_models
    robot.full_constraint_datas = {cm: cm.createData()
                                for cm in robot.constraint_models}
    robot.constraint_datas = cdatas = [robot.full_constraint_datas[cm]
                            for cm in robot.constraint_models]
    # * Init variable used by Unitests
    Lidmot = getMotId_q(new_model)
    goal = np.zeros(len(Lidmot))
    q_prec = q2freeq(new_model,pin.neutral(new_model))
    fgoal = new_data.oMf[36]
    frame_effector = 'bout_pied_frame'
    
    # * Run test
    unittest.main()


    


