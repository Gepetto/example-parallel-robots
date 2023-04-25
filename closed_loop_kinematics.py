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


def closedLoopForwardKinematics(
    model, data, target, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"
):

    """
        closedLoopForwardKinematics(model, data, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"):

        Takes the target position of the motors axis of the robot (joint with name_mot ("mot" if empty) in the name),
        the current configuration of all joint (set to robot.q0 if let empty) and the name of the joints that close the kinematic loop. And returns a configuration that matches the goal positions of the motors
        This function solves a minimization problem
        TODO - writes explicitly the minimization problem and change solver

        Argument:
            model - Pinocchio robot model
            data - Pinocchio robot data
            target - Target configuration of the motors joints. Should be have size of the number of motors
            q_prec - Previous configuration of the free joints
            n_mot - String contained in the motors joints names
            nom_fermeture - String contained in the frame names that should be in contact
            type - Constraint type
    """



    if len(q_prec) != (model.nq - len(target)):
        if len(q_prec) == 0:
            warn("!!!!!!!!!  No q_prec   !!!!!!!!!!!!!!")
        else:
            warn("!!!!!!!!! Invalid q_prec !!!!!!!!!!!!!!")
        q_prec = q2freeq(model, pin.neutral(target))

    n_loop = len(model.names) // 2
    Lid = getMotId_q(model, name_mot)

    def goal2q(free_q):
        """
        Takes q, configuration vector of the free axis and returns nq the global configuration with the motor axis configurations set to goal
        """
        rq = free_q.tolist()

        extend_q = np.zeros(model.nq)
        for i, goalq in zip(Lid, target):
            extend_q[i] = goalq
        for i in range(model.nq):
            if not (i in Lid):
                extend_q[i] = rq.pop(0)

        return extend_q

    def cost(q):
        c = norm(q - q_prec) ** 2
        return c

    def constraintsImp(q):
        q = goal2q(q)
        # Carefull, we assume here that all constraints are the sames
        if type == "3D":
            L1 = constraints3D(model, data, q, n_loop, nom_fermeture)
        elif type == "planar":
            L1 = constraintsPlanar(model, data, q, n_loop, nom_fermeture)
        else:   # We assume here that the input is correct
            L1 = constraints6D(model, data, q, n_loop, nom_fermeture)
        L = []
        for l in L1:
            L = L + l.tolist()
        L2 = constraintQuaternion(model, q)
        return np.array(np.array(L + L2))

    free_q_goal = fmin_slsqp(cost, q_prec, f_eqcons=constraintsImp)
    q_goal = goal2q(free_q_goal)

    return q_goal, free_q_goal


def closedLoopInverseKinematics(model,data,fgoal,q_prec=[],name_eff="effecteur",nom_fermeture="fermeture",type="6D",onlytranslation=False):
    
    """
        q=closedLoopInverseKinematics(model,data,fgoal,constraint_model,constraint_data,q_prec=[],name_eff="effecteur",nom_fermeture="fermeture",type="6D"):

        take the target position of the motors axis of the robot (joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration who match the goal position of the effector

    """

    ideff = model.getFrameId(name_eff)

    if len(q_prec) < model.nq:
        q_prec = pin.neutral(model)
        if len(q_prec) == 0:
            warn("!!!!!!!!!  no q_prec   !!!!!!!!!!!!!!")
        else:
            warn("!!!!!!!!!invalid q_prec!!!!!!!!!!!!!!")

    n_chaines = len(model.names) // 2

    def costnorm(q):
        cdata = model.createData()
        pin.framesForwardKinematics(model, cdata, q)
        if onlytranslation:
            terr = (fgoal.translation - cdata.oMf[ideff].translation)
            c = (norm(terr)) ** 2
        else :
            err = pin.log(fgoal.inverse() * cdata.oMf[ideff]).vector
            c = (norm(err)) ** 2 
        return c 

    def contraintesimp(q):
        if type == "3D":
            L1 = constraints3D(model, data, q, n_chaines, nom_fermeture)
        elif type == "planar":
            L1 = constraintsPlanar(model, data, q, n_chaines, nom_fermeture)
        else:
            L1 = constraints6D(model, data, q, n_chaines, nom_fermeture)
        L = []
        for l in L1:
            L = L + l.tolist()
        L2 = constraintQuaternion(model, q)
        return np.array(np.array(L + L2))

    L = fmin_slsqp(costnorm, q_prec, f_eqcons=contraintesimp, full_output=True)

    return L


class ipotSolver(object):
    """
    class for ipopt solver.
    Jacobian non computed yet
    only 6D closed loop
    
    """
    def __init__(self, model, goal, Lidmot):
        self.const = 0
        self.model = model
        self.goal = goal
        self.Lidmot = Lidmot

    def objective(self, q):
        q_prec = self.q_prec
        return 0.5 * norm(q - q_prec)**2 

    def gradient(self, q):
        q_prec = self.q_prec
        return (q - q_prec)

    def constraints(self, q):
        model = self.model
        goal = self.goal
        eq = q.tolist()
        extend_q = np.zeros(model.nq)
        nq = model.nq

        Lid = getMotId_q(model)
        for i, goalq in zip(Lid, goal):
            extend_q[i] = goalq
        for i in range(nq):
            if not (i in Lid):
                extend_q[i] = eq.pop(0)

        data = model.createData()
        Lc = constraints6D(model, data, extend_q,n_loop=self.number_closed_loop+1,nom_fermeture=self.name_closed_loop)
        self.nc=len(Lc)
        L = []
        n = 0
        for c in Lc:
            L = L + c.tolist()
            n = n + norm(c)

        L = L + constraintQuaternion(model, extend_q)
        self.const = norm(L)
        self.nc = len(L)
        return np.array(L)

    def jacobian(self, q):
        return np.zeros((self.model.nv,self.nc))

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        print("Constraints value at iteration #%d is - %g" % (iter_count, self.const))

def closedLoop6DIpoptForwardKinematics(
    model, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture",number_closed_loop=-1
):
    Lidmot=getMotId_q(model, name_mot)
    solv = ipotSolver(model, goal, Lidmot)

    if number_closed_loop<0:
        number_closed_loop=len(nameFrameConstraint(model,nom_fermeture))
    solv.number_closed_loop=number_closed_loop
    solv.name_closed_loop=nom_fermeture

    if not (len(q_prec) == (model.nq - len(goal))):
        if len(q_prec) == 0:
            warn("!!!!!!!!!  no q_prec   !!!!!!!!!!!!!!")
        else:
            warn("!!!!!!!!! Invalid q_prec !!!!!!!!!!!!!!")
        q_prec = q2freeq(model, pin.neutral(model))
    q_prec = np.array(q_prec)
    solv.q_prec = np.array(q_prec)
    
    lb_np = -1 * np.pi * np.ones(model.nq - len(Lidmot))
    ub_np = 1 * np.pi * np.ones(model.nq - len(Lidmot))
    
    cl_np = np.zeros(6 * number_closed_loop)

    nlp = cyipopt.Problem(
        n=model.nq - len(Lidmot),
        m=len(cl_np),
        problem_obj=solv,
        lb=lb_np,
        ub=ub_np,
        cl=cl_np,
        cu=cl_np,
    )

    nlp.add_option("mu_strategy", "adaptive")
    nlp.add_option("tol", 1e-8)
    nlp.add_option("max_iter", 100)
    nlp.add_option("jacobian_approximation", "finite-difference-values")
    q_free, info=nlp.solve(q_prec)

    def goal2q(free_q):
        """
        take q, configuration of free axis/configuration vector, return nq, global configuration, q with the motor axi set to goal
        """
        rq = free_q.tolist()

        extend_q = np.zeros(model.nq)
        for i, goalq in zip(Lidmot, goal):
            extend_q[i] = goalq
        for i in range(model.nq):
            if not (i in Lidmot):
                extend_q[i] = rq.pop(0)

        return extend_q
    q = goal2q(q_free)
    return(q, q_free)

def closedLoop6DCasadiForwardKinematics(rmodel, rdata, cmodels, cdatas, q_mot_target, q_prec=[]):
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
    # def test_forwardkinematics(self):
    #     q0, q_ini= closedLoopForwardKinematics(new_model, new_data,goal,q_prec=q_prec)
    #     constraint=norm(constraints6D(new_model,new_data,q0))
    #     self.assertTrue(constraint<1e-6) # check the constraint

    # def test_inversekinematics(self):
    #     InvKin=closedLoopInverseKinematics(new_model,new_data,fgoal,q_prec=q_prec,name_eff=frame_effector)
    #     self.assertTrue(InvKin[3]==0) #chexk that joint 15 is a spherical

    # def test_forwarkinematicsipopt(self):
    #     qipopt, info=closedLoop6DIpoptForwardKinematics(new_model, goal, q_prec=q_prec)
    #     constraint=norm(constraints6D(new_model,new_data,qipopt))
    #     self.assertTrue(constraint<1e-6) #check the constraint

    def test_forwarkinematicscasadi(self):
        q_opt, qF_opt=closedLoop6DCasadiForwardKinematics(new_model, new_data, cmodels, cdatas, goal, q_prec=q_prec)
        constraint=norm(constraintsResidual(new_model, new_data, cmodels, cdatas, q_opt, recompute=True, pinspace=pin, quaternions=True))
        self.assertTrue(constraint<1e-6) #check the constraint


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


    


