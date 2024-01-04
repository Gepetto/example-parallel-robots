'''
-*- coding: utf-8 -*-
Virgile Batto & Ludovic De Matteis - September 2023

Tools to mount a robot model, i.e. get a configuration that satisfies all contraints (both robot-robot constraints and robot-environment constraints)
Contains three methods to solve this problem, methode selection is done by setting global variables or through imports
'''

import pinocchio as pin
import numpy as np
try:
    from pinocchio import casadi as caspin
    import casadi
    _WITH_CASADI = True
except ImportError:
    _WITH_CASADI = False
from scipy.optimize import fmin_slsqp
from numpy.linalg import norm

from .constraints import constraintsResidual, derivativeConstraintsResidual

_FORCE_PROXIMAL = False

def closedLoopMountCasadi(rmodel, rdata, cmodels, cdatas, q_prec=None, W=None):
    """
        closedLoopMountCasadi(rmodel, rdata, cmodels, cdatas, q_prec=None):

        This function takes the current configuration of the robot and projects it to the nearest feasible configuration - i.e. satisfying the constraints
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        leading to an optimisation on Lie group.
        
        min || q - q_prec ||^2
        subject to:  f_c(q)=0              # Kinematics constraints are satisfied
                     dq[mots] = 0

        The problem is solved using CasADi + IPOpt

        Argument:
            rmodel - Pinocchio robot model
            rdata - Pinocchio robot data
            cmodels - Pinocchio constraint models list
            cdatas - Pinocchio constraint datas list
            q_prec [Optionnal] - Previous configuration of the free joints - default: None (set to neutral model pose)
            mots [Optionnal] - Ids to fixed fixed (i.e equals to q_prec), this typically corresponds to motors ids
        Return:
            q - Configuration vector satisfying constraints (if optimisation process succeded)
    """
    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    if q_prec is None:
        q_prec = pin.neutral(rmodel)

    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, cmodels, cdatas, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc

    sym_dq = casadi.SX.sym("dq", rmodel.nv, 1)
    cq = caspin.integrate(casmodel, casadi.SX(q_prec), sym_dq)
    constraintsCost = casadi.Function('constraint', [sym_dq], [constraints(cq)])

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(rmodel.nv)

    # * Constraints
    optim.subject_to(constraintsCost(vdq)==0)

    # * cost minimization
    if W is None:
        W = np.eye(rmodel.nv)
    print(W, np.shape(W))

    total_cost = casadi.dot(vdq, W@vdq)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    try:
        optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)
    except RuntimeError as e:
        print(e)
        print('ERROR in convergence, press enter to plot debug info.')
        input()
        dq = optim.debug.value(vdq)
        print(dq)
    q = pin.integrate(rmodel, q_prec, dq)
    return q # I always return a value even if convergence failed


def closedLoopMountScipy(rmodel, rdata, cmodels, cdatas, q_prec=None):
    """
        closedLoopMountScipy(rmodel, rdata, cmodels, cdatas, q_prec=None):

        This function takes the current configuration of the robot and projects it to the nearest feasible configuration - i.e. satisfying the constraints
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        leading to an optimisation on Lie group.
        
        min || q - q_prec ||^2
        subject to:  f_c(q)=0              # Kinematics constraints are satisfied

        The problem is solved using Scipy SLSQP solver

        Argument:
            rmodel - Pinocchio robot model
            rdata - Pinocchio robot data
            cmodels - Pinocchio constraint models list
            cdatas - Pinocchio constraint datas list
            q_prec [Optionnal] - Previous configuration of the free joints - default: None (set to neutral model pose)
        Return:
            q - Configuration vector satisfying constraints (if optimisation process succeded)
    """
    if q_prec is None:
        q_prec = pin.neutral(rmodel)

    def costnorm(vq):
        c = norm(vq) ** 2
        return c

    def contraintesimp(vq):
        q = pin.integrate(rmodel, q_prec, vq)
        Lc = constraintsResidual(rmodel, rdata, cmodels, cdatas, q, recompute=True, pinspace=pin, quaternions=True)
        return Lc

    vq_goal = fmin_slsqp(costnorm, np.zeros(rmodel.nv), f_eqcons=contraintesimp)
    q_goal = pin.integrate(rmodel, q_prec, vq_goal)
    return q_goal

def closedLoopMountProximal(rmodel, rdata, cmodels, cdatas, q_prec=None, max_it=100, eps=1e-12, rho=1e-10, mu=1e-4):
    """
        closedLoopMountProximal(rmodel, rdata, cmodels, cdatas, q_prec=None, max_it=100, eps=1e-12, rho=1e-10, mu=1e-4):

        This function takes the current configuration of the robot and projects it to the nearest feasible configuration - i.e. satisfying the constraints
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        leading to an optimisation on Lie group.
        
        min || q - q_prec ||^2
        subject to:  f_c(q)=0              # Kinematics constraints are satisfied

        The problem is solved using a proximal solver

        Argument:
            rmodel - Pinocchio robot model
            rdata - Pinocchio robot data
            cmodels - Pinocchio constraint models list
            cdatas - Pinocchio constraint datas list
            q_prec [Optionnal] - Previous configuration of the free joints - default: None (set to neutral model pose)
            max_it [Optionnal] - Maximal number of proximal iterations - default: 100
            eps [Optinnal] - Proximal parameter epsilon - default: 1e-12
            rho [Optionnal] - Proximal parameter rho - default: 1e-10
            mu [Optionnal] - Proximal parameter mu - default: 1e-4
        Return:
            q - Configuration vector satisfying constraints (if optimisation process succeded)

    Initially written by Justin Carpentier    
    raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    if q_prec is None:
        q_prec = pin.neutral(rmodel)
    q = q_prec
      
    constraint_dim=0
    for cm in cmodels:
        constraint_dim += cm.size() 

    y = np.ones((constraint_dim))
    rdata.M = np.eye(rmodel.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(rmodel,cmodels)

    for k in range(max_it):
        pin.computeJointJacobians(rmodel,rdata,q)
        kkt_constraint.compute(rmodel,rdata,cmodels,cdatas,mu)

        constraint_value=np.concatenate([(pin.log(cd.c1Mc2).np[:cm.size()]) for (cd,cm) in zip(cdatas,cmodels)])

        LJ=[]
        for (cm,cd) in zip(cmodels,cdatas):
            Jc=pin.getConstraintJacobian(rmodel,rdata,cm,cd)
            LJ.append(Jc)
        J=np.concatenate(LJ)

        primal_feas = np.linalg.norm(constraint_value,np.inf)
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y),np.inf)
        if primal_feas < eps and dual_feas < eps:
            print("Convergence achieved")
            break
        print("constraint_value:",np.linalg.norm(constraint_value))
        rhs = np.concatenate([-constraint_value - y*mu, np.zeros(rmodel.nv)])

        dz = kkt_constraint.solve(rhs) 
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        alpha = 1.
        q = pin.integrate(rmodel,q,-alpha*dq)
        y -= alpha*(-dy + y)
    return(q)

# def closedLoopMount(*args, **kwargs):
#     if _FORCE_PROXIMAL:
#         return(closedLoopMountProximal(*args, **kwargs))
#     else:
#         if _WITH_CASADI:
#             return(closedLoopMountCasadi(*args, **kwargs))
#         else:
#             return(closedLoopMountScipy(*args, **kwargs))

def configurationProjection(model, data, constraint_models, constraint_datas, q_ref, fixed_joints_ids):
    """
        closedLoopProjection(model, data, constraint_models, constraint_datas, q_ref, open_loop_ids):

        This function takes the reference configuration of the robot and returns a configuration that satisfies the constraints while keeping some joints at the same configuration
        This function solves a minimization problem over q where q is defined as q_ref+dq (this removes the need for quaternion constraints and gives less decision variables)
        
        min || f_c(q) ||^2
        subject to: dq[fixed_joints] = 0

        The problem is solved using CasADi + IPOpt

        Argument:
            TODO
        Return:
            TODO
    """
    # * Defining casadi models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()

    casconstraint_models = [caspin.RigidConstraintModel(cm) for cm in constraint_models]
    casconstraint_datas = [cm.createData() for cm in casconstraint_models]

    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, casconstraint_models, casconstraint_datas, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc
    # def derivativesConstraints(q):
    #     jac = derivativeConstraintsResidual(casmodel, casdata, casconstraint_models, casconstraint_datas, q, recompute=True, pinspace=caspin, quaternions=False)
    #     return jac

    # nc = np.sum([cm.size() for cm in casconstraint_models])
    sym_dq = casadi.SX.sym("dq", model.nv, 1)
    sym_cost = casadi.SX.sym("dq", 1, 1)
    cq = caspin.integrate(casmodel, casadi.SX(q_ref), sym_dq)
    constraintsRes = casadi.Function('constraint', [sym_dq], [constraints(cq)])

    jac_cost = casadi.Function('jac_cost', [sym_dq, sym_cost], 
        [2*constraintsRes(sym_dq).T@constraintsRes.jacobian()(sym_dq, constraintsRes(sym_dq))])
    cost = casadi.Function('cost', [sym_dq], [casadi.sumsqr(constraintsRes(sym_dq))],
                        {"custom_jacobian": jac_cost, "jac_penalty":0})

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(model.nv)

    # * Problem
    total_cost = cost(vdq)
    optim.minimize(total_cost)
    optim.subject_to(vdq[fixed_joints_ids]==0)

    opts = {}
    optim.solver("ipopt", opts)
    try:
        optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)
    except RuntimeError as e:
        print(e)
        print('ERROR in convergence, press enter to plot debug info.')
        input()
        dq = optim.debug.value(vdq)
        print(dq)

    assert(optim.value(total_cost)<1e-6)
    q = pin.integrate(model, q_ref, dq)
    return q # I always return a value even if convergence failed