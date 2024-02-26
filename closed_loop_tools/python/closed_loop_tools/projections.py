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
    Project the current configuration of the robot to the nearest feasible configuration satisfying kinematic constraints using a proximal solver.

    Args:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data associated with the model.
        cmodels (list): List of Pinocchio constraint models.
        cdatas (list): List of Pinocchio constraint data associated with the constraint models.
        q_prec (np.array, optional): Previous configuration of the free joints. Defaults to None (set to neutral model pose).
        max_it (int, optional): Maximum number of proximal iterations. Defaults to 100.
        eps (float, optional): Proximal parameter epsilon. Defaults to 1e-12.
        rho (float, optional): Proximal parameter rho. Defaults to 1e-10.
        mu (float, optional): Proximal parameter mu. Defaults to 1e-4.

    Returns:
        np.array: Configuration vector satisfying constraints (if the optimization process succeeded).

    Notes:
        This function solves a minimization problem over q, where q is defined as q0+dq. This representation removes the need for quaternion constraints and reduces decision variables, leading to optimization on the Lie group.

        The optimization problem aims to minimize the squared norm of (q - q_prec) subject to kinematic constraints (f_c(q) = 0), ensuring that the robot's motion complies with its physical constraints.

        The problem is formulated as follows:
        - Minimize the squared norm of (q - q_prec) subject to kinematic constraints (f_c(q) = 0).
        - Kinematic constraints ensure that the robot's motion complies with its physical constraints.

        The optimization process aims to find the nearest feasible configuration to the initial configuration q_prec.

        If the optimization process fails to converge, it returns the current configuration q_prec.

    References:
        - Originally written by Justin Carpentier.

    Example:
        q = configurationProjectionProximal(rmodel, rdata, cmodels, cdatas, q_prec, max_it=100, eps=1e-12, rho=1e-10, mu=1e-4)
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
    Project the current robot configuration to the nearest feasible configuration satisfying constraints using CasADi + IPOpt.

    Args:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data associated with the model.
        cmodels (list): List of Pinocchio constraint models.
        cdatas (list): List of Pinocchio constraint data associated with the constraint models.
        q_prec (np.array, optional): Previous configuration of the free joints. Defaults to None (set to neutral model pose).
        W (np.array, optional): Weighting matrix for the cost function. Defaults to None (identity matrix).

    Returns:
        np.array: Configuration vector satisfying constraints (if the optimization process succeeded).

    Notes:
        This function solves a minimization problem over q, where q is defined as q0 + dq. This representation removes the need for quaternion constraints and reduces decision variables, leading to optimization on the Lie group.

        The optimization problem aims to minimize the weighted squared norm of (q - q_prec) subject to kinematic constraints (f_c(q) = 0) and dq[mots] = 0, where mots are the motor ids.

        The problem is formulated as follows:
        - Minimize the weighted squared norm of (q - q_prec) subject to kinematic constraints (f_c(q) = 0) and dq[mots] = 0.
        - Kinematic constraints ensure that the robot's motion complies with its physical constraints, while dq[mots] = 0 fixes the motor joints.
        - The weighting matrix W allows adjusting the importance of different joint velocities in the cost function.

        The optimization process aims to find the nearest feasible configuration to the initial configuration q_prec.

        If the optimization process fails to converge, it returns the current configuration q_prec.

    Example:
        q = configurationProjection(rmodel, rdata, cmodels, cdatas, q_prec, W)
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
    Project the desired velocity `v_ref` onto the feasible velocity space subject to constraints using Quadratic Programming.

    Args:
        model (pinocchio.Model): Pinocchio robot model.
        data (pinocchio.Data): Pinocchio robot data associated with the model.
        q (np.array): Current configuration of the robot.
        v_ref (np.array): Desired velocity to be projected.
        constraint_models (list): List of Pinocchio constraint models.
        constraint_datas (list): List of Pinocchio constraint data associated with the constraint models.

    Returns:
        np.array: Projected velocity satisfying constraints.

    Notes:
        This function projects the desired velocity `v_ref` onto the feasible velocity space subject to constraints. It uses Quadratic Programming (QP) to find the optimal solution.

        The QP problem is formulated as follows:
        - Minimize the quadratic cost function: (1/2) * ||x - v_ref||^2
        - Subject to: A*x <= b, where A is the constraint Jacobian and b is a vector of constraint values.

        The function uses the 'proxqp' solver for solving the QP problem.

        If the solver fails to converge, it prints a warning message and returns the unprojected desired velocity `v_ref`.

    Example:
        v_projected = velocityProjection(model, data, q, v_ref, constraint_models, constraint_datas)
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
    Project the desired acceleration `a_ref` onto the feasible acceleration space subject to constraints using Quadratic Programming.

    Args:
        model (pinocchio.Model): Pinocchio robot model.
        data (pinocchio.Data): Pinocchio robot data associated with the model.
        q (np.array): Current configuration of the robot.
        v (np.array): Current velocity of the robot.
        a_ref (np.array): Desired acceleration to be projected.
        constraint_models (list): List of Pinocchio constraint models.
        constraint_datas (list): List of Pinocchio constraint data associated with the constraint models.

    Returns:
        np.array: Projected acceleration satisfying constraints.

    Notes:
        This function projects the desired acceleration `a_ref` onto the feasible acceleration space subject to constraints.
        It uses Quadratic Programming (QP) to find the optimal solution.

        The QP problem is formulated as follows:
        - Minimize the quadratic cost function: (1/2) * ||x - a_ref||^2
        - Subject to: A*x <= b, where A is the constraint Jacobian, b is a vector of constraint values, and gamma is a vector of contact acceleration drifts.

        The function uses the 'proxqp' solver for solving the QP problem.

        If the solver fails to converge, it prints a warning message and returns the unprojected desired acceleration `a_ref`.
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
