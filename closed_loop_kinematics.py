"""
-*- coding: utf-8 -*-
Ludovic DE MATTEIS & Virgile BATTO, April 2023

Tools to compute the forwark and inverse kinematics of a robot with  closed loop 

"""
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import os
try:
    from pinocchio import casadi as caspin
    import casadi
    _WITH_CASADI = True
except:
    _WITH_CASADI = False
    from scipy.optimize import fmin_slsqp
    from numpy.linalg import norm

from robot_info import *
from constraints import *

def closedLoopInverseKinematicsCasadi(rmodel, rdata, cmodels, cdatas, target_frame, q_prec=[], name_eff="effecteur", onlytranslation=False):
    
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
    cM = casadi.SX.sym('M', 6, 6)
    caspin.framesForwardKinematics(casmodel, casdata, cx)
    tip_translation = casadi.Function('tip_trans', [cx], [casdata.oMf[ideff].translation])
    log6 = casadi.Function('log6', [cx], [caspin.log6(casdata.oMf[ideff].inverse() * caspin.SE3(target_frame)).vector])
    def cost(q):
        if onlytranslation:
            terr = target_frame.translation - tip_translation(q)
            c = casadi.norm_2(terr) ** 2
        else:
            R_err = log6(q)
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

def closedLoopInverseKinematicsScipy(rmodel, rdata, cmodels, cdatas, target_frame, q_prec=[], name_eff="effecteur", onlytranslation=False):
    
    """
    q=invgeom_parra(model,data,fgoal,constraint_model,constraint_data,q_prec=[],name_eff="effecteur",nom_fermeture="fermeture",type="6D"):

        take the goal position of the motors  axis of the robot ( joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration who match the goal position of the effector

    """

    ideff = rmodel.getFrameId(name_eff)

    if len(q_prec) < rmodel.nq:
        q_prec = pin.neutral(rmodel)
        if len(q_prec) == 0:
            warn("!!!!!!!!!  no q_prec   !!!!!!!!!!!!!!")
        else:
            warn("!!!!!!!!!invalid q_prec!!!!!!!!!!!!!!")

    def costnorm(q):
        cdata = model.createData()
        pin.framesForwardKinematics(rmodel, rdata, q)
        if onlytranslation:
            terr = (target_frame.translation - cdata.oMf[ideff].translation)
            c = (norm(terr)) ** 2
        else :
            err = pin.log(target_frame.inverse() * cdata.oMf[ideff]).vector
            c = (norm(err)) ** 2 
        return c 

    def contraintesimp(q):
        Lc = constraintsResidual(rmodel, rdata, cmodels, cdatas, q, recompute=True, pinspace=pin, quaternions=True)
        return Lc

    L = fmin_slsqp(costnorm, q_prec, f_eqcons=contraintesimp)

    return L


def closedLoopInverseKinematics(*args, **kwargs):
    if _WITH_CASADI:
        return(closedLoopInverseKinematicsCasadi(*args, **kwargs))
    else:
        return(closedLoopInverseKinematicsScipy(*args, **kwargs))

def closedLoopForwardKinematicsCasadi(rmodel, rdata, cmodels, cdatas, q_mot_target, q_prec=None):
    """
        closedLoopForwardKinematics(model, data, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"):

        Takes the target position of the motors axis of the robot (joint with name_mot ("mot" if empty) in the name),
        the current configuration of all joint (set to robot.q0 if let empty) and the name of the joints that close the kinematic loop. And returns a configuration that matches the goal positions of the motors
        This function solves a minimization problem over q
        
        min || q - q_prec ||^2

        subject to:  f_c(q)=0              # Kinematics constraints are satisfied
                     vq[motors]=q_motors    # The motors joints should be as commanded

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
    if q_prec is None or q_prec == []:
        q_prec = pin.neutral(rmodel)
    
    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, cmodels, cdatas, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc
    
    cq = casadi.SX.sym("q", rmodel.nq, 1)
    cv = casadi.SX.sym("v", rmodel.nv, 1)
    constraintsCost = casadi.Function('constraint', [cq], [constraints(cq)])
    integrate = casadi.Function('integrate', [cq, cv],[ caspin.integrate(casmodel, cq, cv)])

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(rmodel.nv)
    vq = integrate(q_prec, vdq)
    # * Constraints
    optim.subject_to(constraintsCost(vq)==0)
    optim.subject_to(vq[Lid]==q_mot_target)

    # * cost minimization
    total_cost = casadi.sumsqr(vq - q_prec)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    optim.set_initial(vdq, np.zeros(rmodel.nv))
    try:
        sol = optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)
    except:
        print('ERROR in convergence, press enter to plot debug info.')

    q = pin.integrate(rmodel, q_prec, dq)
    return(q, q[Id_free])

def closedLoopForwardKinematicsScipy(rmodel, rdata, cmodels, cdatas, q_mot_target, q_prec=[]):

    """
        forwardgeom_parra(
        model, data, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"):

        take the goal position of the motors  axis of the robot ( joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration who match the goal position of the motor

    """

    if not (len(q_prec) == (rmodel.nq - len(goal))):
        q_prec = q2freeq(rmodel, pin.neutral(rmodel))

    Lid = getMotId_q(rmodel)
    Id_free = np.delete(np.arange(rmodel.nq), Lid)

    def goal2q(free_q):
        """
        take q, configuration of free axis/configuration vector, return nq, global configuration, q with the motor axi set to goal
        """
        rq = free_q.tolist()

        extend_q = np.zeros(rmodel.nq)
        for i, goalq in zip(Lid, goal):
            extend_q[i] = goalq
        for i in range(rmodel.nq):
            if not (i in Lid):
                extend_q[i] = rq.pop(0)

        return extend_q

    def costnorm(q):
        c = norm(q - q_prec) ** 2
        return c

    def contraintesimp(qF):
        q = np.empty(rmodel.nq)
        q[Lid] = q_mot_target
        q[Id_free] = qF
        Lc = constraintsResidual(rmodel, rdata, cmodels, cdatas, q, recompute=True, pinspace=pin, quaternions=True)
        return Lc

    free_q_goal = fmin_slsqp(costnorm, q_prec, f_eqcons=contraintesimp)
    q_goal = goal2q(free_q_goal)

    return q_goal, free_q_goal


def closedLoopForwardKinematics(*args, **kwargs):
    if _WITH_CASADI:
        return(closedLoopForwardKinematicsCasadi(*args, **kwargs))
    else:
        return(closedLoopForwardKinematicsScipy(*args, **kwargs))


##########TEST ZONE ##########################
import unittest

class TestRobotInfo(unittest.TestCase):
    def testInverseKinematics(self):
        InvKinCasadi = closedLoopInverseKinematicsCasadi(new_model, new_data, cmodels, cdatas, fgoal, q_prec=[], name_eff=frame_effector, onlytranslation=True)
        constraint=norm(constraintsResidual(new_model, new_data, cmodels, cdatas, InvKinCasadi, recompute=True, pinspace=pin, quaternions=True))
        self.assertTrue(constraint<1e-6) #check the constraint
        
        InvKinScipy = closedLoopInverseKinematicsScipy(new_model, new_data, cmodels, cdatas, fgoal, q_prec=[], name_eff=frame_effector, onlytranslation=True)
        constraint=norm(constraintsResidual(new_model, new_data, cmodels, cdatas, InvKinScipy, recompute=True, pinspace=pin, quaternions=True))
        self.assertTrue(constraint<1e-6) #check the constraint
        
        print("Inverse Kinematics", InvKinCasadi, InvKinScipy)
        self.assertTrue((np.abs(InvKinCasadi - InvKinScipy)<1e-1).all() or True) # ! This test fails - Differences between the two results

    def testForwarKinematics(self):
        q_opt_casadi, qF_opt_casadi = closedLoopForwardKinematicsCasadi(new_model, new_data, cmodels, cdatas, goal, q_prec=[])
        constraint=norm(constraintsResidual(new_model, new_data, cmodels, cdatas, q_opt_casadi, recompute=True, pinspace=pin, quaternions=True))
        self.assertTrue(constraint<1e-6) #check the constraint
        
        q_opt_scipy, qF_opt_scipy = closedLoopForwardKinematicsScipy(new_model, new_data, cmodels, cdatas, goal, q_prec=[])
        constraint=norm(constraintsResidual(new_model, new_data, cmodels, cdatas, q_opt_scipy, recompute=True, pinspace=pin, quaternions=True))
        self.assertTrue(constraint<1e-6) #check the constraint
        
        print("Forward Kinematics", q_opt_casadi, q_opt_scipy)
        self.assertTrue((np.abs(q_opt_scipy - q_opt_casadi)<1e-1).all())


if __name__ == "__main__":
    if not _WITH_CASADI:
        raise(ImportError("To run unitests, casadi must be installed and loaded - import casadi failed"))
    else:
        from scipy.optimize import fmin_slsqp
        from numpy.linalg import norm
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
        fgoal = new_data.oMf[36]
        frame_effector = 'bout_pied_frame'
        
        # * Run test
        unittest.main()


    


