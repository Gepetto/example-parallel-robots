"""
-*- coding: utf-8 -*-
Virgile Batto & Ludovic De Matteis - September 2023

Tools to mount a robot model, i.e. get a configuration that satisfies all contraints (both robot-robot constraints and robot-environment constraints)
Contains three methods to solve this problem, methode selection is done by setting global variables or through imports
"""

import pinocchio as pin
import numpy as np
from pinocchio import casadi as caspin
import casadi
from qpsolvers import solve_qp

from .constraints import constraintsResidual


## Configuration projections
def configurationProjectionProximal(
    rmodel,
    rdata,
    cmodels,
    cdatas,
    q_prec=None,
    max_it=100,
    eps=1e-12,
    rho=1e-10,
    mu=1e-4,
):
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

    constraint_dim = 0
    for cm in cmodels:
        constraint_dim += cm.size()

    y = np.ones((constraint_dim))
    rdata.M = np.eye(rmodel.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(rmodel, cmodels)

    for k in range(max_it):
        pin.computeJointJacobians(rmodel, rdata, q)
        kkt_constraint.compute(rmodel, rdata, cmodels, cdatas, mu)

        constraint_value = np.concatenate(
            [(pin.log(cd.c1Mc2).np[: cm.size()]) for (cd, cm) in zip(cdatas, cmodels)]
        )

        LJ = []
        for cm, cd in zip(cmodels, cdatas):
            Jc = pin.getConstraintJacobian(rmodel, rdata, cm, cd)
            LJ.append(Jc)
        J = np.concatenate(LJ)

        primal_feas = np.linalg.norm(constraint_value, np.inf)
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        if primal_feas < eps and dual_feas < eps:
            print("Convergence achieved")
            break
        print("constraint_value:", np.linalg.norm(constraint_value))
        rhs = np.concatenate([-constraint_value - y * mu, np.zeros(rmodel.nv)])

        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        alpha = 1.0
        q = pin.integrate(rmodel, q, -alpha * dq)
        y -= alpha * (-dy + y)
    return q


def configurationProjection(rmodel, rdata, cmodels, cdatas, q_prec=None, W=None):
    """
    This function performs a configuration projection for a given robot model and data using the Pinocchio library.

    Parameters:
    rmodel (object): The robot model in Pinocchio.
    rdata (object): The robot data in Pinocchio.
    cmodels (list): List of constraint models.
    cdatas (list): List of constraint data.
    q_prec (array, optional): The previous configuration. If not provided, the neutral configuration of the robot model is used.
    W (array, optional): The weight matrix for the optimization problem. If not provided, an identity matrix of size rmodel.nv is used.

    Returns:
    q (array): The optimized configuration that minimizes the distance to the previous configuration while satisfying the kinematic constraints.

    The function solves the following optimization problem using CasADi + IPOpt:

    min || q - q_prec ||_W^2
    subject to:  f_c(q)=0              # Kinematic constraints are satisfied
                 dq[mots] = 0

    The problem is solved using the Pinocchio library for robot dynamics and kinematics.
    """
    # * Getting ids of actuated and free joints
    if q_prec is None:
        q_prec = pin.neutral(rmodel)
    if W is None:
        W = np.eye(rmodel.nv)

    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(
            casmodel,
            casdata,
            cmodels,
            cdatas,
            q,
            recompute=True,
            pinspace=caspin,
            quaternions=False,
        )
        return Lc

    sym_dq = casadi.SX.sym("dq", rmodel.nv, 1)
    sym_cost = casadi.SX.sym("dq", 1, 1)
    cq = caspin.integrate(casmodel, casadi.SX(q_prec), sym_dq)

    constraintsCost = casadi.Function("constraint", [sym_dq], [constraints(cq)])
    jac_residual = casadi.Function(
        "jac_constraint_residual",
        [sym_dq, sym_cost],
        [
            2
            * constraintsCost(sym_dq).T
            @ constraintsCost.jacobian()(sym_dq, constraintsCost(sym_dq))
        ],
    )
    residual = casadi.Function(
        "constraint_residual",
        [sym_dq],
        [casadi.sumsqr(constraintsCost(sym_dq))],
        {"custom_jacobian": jac_residual, "jac_penalty": 0},
    )

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(rmodel.nv)

    # * Constraints
    optim.subject_to(residual(vdq) == 0)

    # * cost minimization
    total_cost = casadi.dot(vdq, W @ vdq)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    try:
        optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)
    except RuntimeError as e:
        print(e)
        print("ERROR in convergence, press enter to plot debug info.")
        input()
        dq = optim.debug.value(vdq)
        print(dq)
    q = pin.integrate(rmodel, q_prec, dq)
    return q


## Velocity projections
def velocityProjection(model, data, q, v_ref, constraint_models, constraint_datas):
    """
    This function computes the velocity projection for a given model and its constraints.

    Parameters:
    model (object): The model object that contains the system's dynamics.
    data (object): The data object that contains the system's state.
    q (array): The current configuration of the system.
    v_ref (array): The reference velocity for the system.
    constraint_models (list): A list of constraint models for the system.
    constraint_datas (list): A list of constraint data for the system.

    Returns:
    array: The projected velocity of the system.

    Note:
    This function uses Quadratic Programming (QP) to solve for the projected velocity.
    """
    nx = len(v_ref)
    pin.computeAllTerms(model, data, q, np.zeros(model.nv))
    Jac = pin.getConstraintsJacobian(model, data, constraint_models, constraint_datas)
    P = np.eye(nx)
    q = -v_ref
    A = Jac
    b = np.zeros(A.shape[0])
    G = np.zeros((1, nx))
    h = np.zeros(1)
    x = solve_qp(
        P, q, G, h, A, b, solver="proxqp", verbose=True, eps_abs=1e-5, backend="sparse"
    )
    return x


## Acceleration projections
def accelerationProjection(
    model, data, q, v, a_ref, constraint_models, constraint_datas
):
    """
    This function computes the acceleration projection for a given model and its constraints.

    Parameters:
    model (object): The model object that contains the system's dynamics.
    data (object): The data object that contains the system's state.
    q (array): The current configuration of the system.
    v (array): The current velocity of the system.
    a_ref (array): The reference acceleration for the system.
    constraint_models (list): A list of constraint models for the system.
    constraint_datas (list): A list of constraint data for the system.

    Returns:
    array: The projected acceleration of the system.

    Note:
    This function uses Quadratic Programming (QP) to solve for the projected acceleration.
    """
    nx = len(a_ref)
    pin.computeAllTerms(model, data, q, np.zeros(model.nv))
    Jac = pin.getConstraintsJacobian(model, data, constraint_models, constraint_datas)
    pin.initConstraintDynamics(model, data, constraint_models)
    pin.constraintDynamics(
        model,
        data,
        q,
        v,
        np.zeros(model.nv),
        constraint_models,
        constraint_datas,
        pin.ProximalSettings(),
    )
    gamma = np.concatenate(
        [
            cd.contact2_acceleration_drift.vector
            - cd.contact1_acceleration_drift.vector
            for cd in constraint_datas
        ]
    )
    P = np.eye(nx)
    q = -a_ref
    A = Jac
    b = gamma
    G = np.zeros((1, nx))
    h = np.zeros(1)

    x = solve_qp(
        P, q, G, h, A, b, solver="proxqp", verbose=True, eps_abs=1e-5, backend="sparse"
    )
    return x
