"""
-*- coding: utf-8 -*-
Virgile Batto & Ludovic De Matteis - September 2023

Tools to mount a robot model, i.e. get a configuration that satisfies all contraints (both robot-robot constraints and robot-environment constraints)
Contains three methods to solve this problem, methode selection is done by setting global variables or through imports
"""

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

from .constraints import constraintsResidual

_FORCE_PROXIMAL = False


def closedLoopMountCasadi(rmodel, rdata, cmodels, cdatas, q_prec=None):
    """
    Perform closed-loop mounting using CasADi and IPOpt to find the nearest feasible configuration satisfying the kinematics constraints.

    Args:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data associated with the model.
        cmodels (list): List of Pinocchio constraint models.
        cdatas (list): List of Pinocchio constraint data associated with the constraint models.
        q_prec (np.array, optional): Previous configuration of the free joints. Defaults to None (set to neutral model pose).

    Returns:
        np.array: Configuration vector satisfying constraints (if optimization process succeeded).

    Raises:
        AttributeError: If the optimization process encounters an attribute error.

    Notes:
        This function solves a minimization problem over q, where q is defined as q0+dq, removing the need for quaternion constraints and reducing decision variables.
        The optimization problem is set up using CasADi and IPOpt, subject to the kinematics constraints being satisfied.

    The problem is formulated as follows:

    - Minimize the squared norm of (q - q_prec) subject to kinematics constraints (f_c(q) = 0).
    - Kinematics constraints ensure that the robot's motion complies with its physical constraints.

    The optimization process aims to find the nearest feasible configuration to the initial configuration q_prec.

    If the optimization process fails to converge, it provides debug information for troubleshooting.

    Example:
        q = closedLoopMountCasadi(rmodel, rdata, cmodels, cdatas, q_prec)
    """
    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    if q_prec is None:
        q_prec = pin.neutral(rmodel)

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

    cq = casadi.SX.sym("q", rmodel.nq, 1)
    cv = casadi.SX.sym("v", rmodel.nv, 1)
    constraintsCost = casadi.Function("constraint", [cq], [constraints(cq)])
    integrate = casadi.Function(
        "integrate", [cq, cv], [caspin.integrate(casmodel, cq, cv)]
    )

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(rmodel.nv)
    vq = integrate(q_prec, vdq)

    # * Constraints
    optim.subject_to(constraintsCost(vq) == 0)
    optim.subject_to(
        optim.bounded(rmodel.lowerPositionLimit, vq, rmodel.upperPositionLimit)
    )

    # * cost minimization
    total_cost = casadi.sumsqr(vdq)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    try:
        optim.solve_limited()
        print("Solution found")
        q = optim.value(vq)
    except AttributeError as e:
        print(e)
        print("ERROR in convergence, press enter to plot debug info.")
        input()
        q = optim.debug.value(vq)
        print(q)

    return q  # I always return a value even if convergence failed


def closedLoopMountScipy(rmodel, rdata, cmodels, cdatas, q_prec=None):
    """
    Perform closed-loop mounting using the Scipy SLSQP solver to find the nearest feasible configuration satisfying kinematic constraints.

    Args:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data associated with the model.
        cmodels (list): List of Pinocchio constraint models.
        cdatas (list): List of Pinocchio constraint data associated with the constraint models.
        q_prec (np.array, optional): Previous configuration of the free joints. Defaults to None (set to neutral model pose).

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

    Example:
        q = closedLoopMountScipy(rmodel, rdata, cmodels, cdatas, q_prec)
    """
    if q_prec is None:
        q_prec = pin.neutral(rmodel)

    def costnorm(vq):
        c = norm(vq) ** 2
        return c

    def contraintesimp(vq):
        q = pin.integrate(rmodel, q_prec, vq)
        Lc = constraintsResidual(
            rmodel,
            rdata,
            cmodels,
            cdatas,
            q,
            recompute=True,
            pinspace=pin,
            quaternions=True,
        )
        return Lc

    vq_goal = fmin_slsqp(costnorm, np.zeros(rmodel.nv), f_eqcons=contraintesimp)
    q_goal = pin.integrate(rmodel, q_prec, vq_goal)
    return q_goal


def closedLoopMountProximal(
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
    Compute the nearest feasible configuration of the robot satisfying kinematic constraints using a proximal solver.

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
        q = closedLoopMountProximal(rmodel, rdata, cmodels, cdatas, q_prec, max_it=100, eps=1e-12, rho=1e-10, mu=1e-4)
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


def closedLoopMount(*args, **kwargs):
    if _FORCE_PROXIMAL:
        return closedLoopMountProximal(*args, **kwargs)
    else:
        if _WITH_CASADI:
            return closedLoopMountCasadi(*args, **kwargs)
        else:
            return closedLoopMountScipy(*args, **kwargs)
