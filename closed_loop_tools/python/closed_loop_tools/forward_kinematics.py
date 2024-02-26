"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS - February 2024

This module provides tools to perform the forward kinematics of a closed-loop system.
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
from .actuation import mergeq, mergev


def closedLoopForwardKinematicsCasadi(
    rmodel, rdata, cmodels, cdatas, actuation_model, q_mot_target=None, q_prec=None
):
    """
    Solves the closed-loop forward kinematics problem using Casadi + IpOpt.

    Args:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data.
        cmodels (list): Pinocchio constraint models list.
        cdatas (list): Pinocchio constraint datas list.
        actuation_model: Actuation model.
        q_mot_target (Optional): Target position of the motors axis of the robot (following the actuation model). Defaults to None.
        q_prec (Optional): Previous configuration of the free joints. Defaults to None (set to neutral model pose).

    Returns:
        q (numpy.ndarray): Configuration vector satisfying constraints (if optimization process succeeded).

    Notes:
        This function solves a minimization problem over q, where q is defined as q0 + dq. This removes the need for quaternion constraints and reduces the number of decision variables.

        The problem is formulated as follows:

        min || q - q_prec ||^2

        subject to:
            f_c(q) = 0              # Kinematics constraints are satisfied
            vq[motors] = q_motors   # The motors joints should be as commanded

        - f_c(q) represents the kinematics constraints.
        - vq[motors] represents the desired positions of the motors joints.
    """

    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    mot_ids_q = actuation_model.mot_ids_q
    if q_prec is None or q_prec == []:
        q_prec = pin.neutral(rmodel)
    if q_mot_target is None:
        q_mot_target = np.zeros(len(mot_ids_q))

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
    constraints_cost = casadi.Function("constraint", [cq], [constraints(cq)])
    integrate = casadi.Function(
        "integrate", [cq, cv], [caspin.integrate(casmodel, cq, cv)]
    )

    # * Optimisation problem
    optim = casadi.Opti()
    vdqf = optim.variable(len(actuation_model.free_ids_v))
    vdq = mergev(casmodel, actuation_model, q_mot_target, vdqf, True)
    vq = integrate(q_prec, vdq)

    # * Constraints
    optim.subject_to(constraints_cost(vq) == 0)
    optim.subject_to(
        optim.bounded(rmodel.lowerPositionLimit, vq, rmodel.upperPositionLimit)
    )

    # * cost minimization
    total_cost = casadi.sumsqr(vdqf)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    optim.set_initial(vdqf, np.zeros(len(actuation_model.free_ids_v)))
    try:
        optim.solve_limited()
        print("Solution found")
        q = optim.value(vq)
    except AttributeError:
        # ? Should we raise an error here? Or just return the previous configuration?
        print("ERROR in convergence, press enter to plot debug info.")
        input()
        q = optim.debug.value(vq)
        print(q)

    return q  # I always return a value even if convergence failed


def closedLoopForwardKinematicsScipy(
    rmodel, rdata, cmodels, cdatas, actuation_model, q_mot_target=None, q_prec=[]
):
    """
    Solves the closed-loop forward kinematics problem using Scipy SLSQP.

    Args:
        rmodel: Pinocchio robot model.
        rdata: Pinocchio robot data.
        cmodels: Pinocchio constraint models list.
        cdatas: Pinocchio constraint datas list.
        actuation_model: Actuation model.
        q_mot_target: Target position of the motors axis of the robot (following the actuation model). Default is None.
        q_prec: Previous configuration of the free joints. Default is None (set to neutral model pose).
        name_eff: Name of the effector frame. Default is "effecteur".
        onlytranslation: Set to True to choose only translation (3D) and False to have 6D position. Default is False (6D).

    Returns:
        q: Configuration vector satisfying constraints (if optimization process succeeded).

    Notes:
        This function solves a minimization problem over q, where q is defined as q0 + dq.
        The goal is to minimize the norm of the difference between q and q_prec.

        Mathematically, the problem can be formulated as follows:

        min || q - q_prec ||^2

        subject to:
            f_c(q) = 0              # Kinematics constraints are satisfied
            vq[motors] = q_motors   # The motors joints should be as commanded

        The problem is solved using Scipy SLSQP.
    """
    mot_ids_q = actuation_model.mot_ids_q
    if q_prec is None or q_prec == []:
        q_prec = pin.neutral(rmodel)
    if q_mot_target is None:
        q_mot_target = np.zeros(len(mot_ids_q))
    q_free_prec = q_prec[actuation_model.free_ids_q]

    def costnorm(qF):
        c = norm(qF - q_free_prec) ** 2
        return c

    def contraintesimp(qF):
        q = mergeq(rmodel, actuation_model, q_mot_target, qF)
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

    free_q_goal = fmin_slsqp(costnorm, q_free_prec, f_eqcons=contraintesimp)
    q_goal = mergeq(rmodel, actuation_model, q_mot_target, free_q_goal)
    return q_goal


def closedLoopForwardKinematics(*args, **kwargs):
    """
    Calculates the forward kinematics for a closed-loop system.
    Calls either closedLoopForwardKinematicsCasadi or closedLoopForwardKinematicsScipy.

    Parameters:
        *args: positional arguments
        **kwargs: keyword arguments

    Returns:
        The forward kinematics of the closed-loop system.
    """
    if _WITH_CASADI:
        return closedLoopForwardKinematicsCasadi(*args, **kwargs)
    else:
        return closedLoopForwardKinematicsScipy(*args, **kwargs)
