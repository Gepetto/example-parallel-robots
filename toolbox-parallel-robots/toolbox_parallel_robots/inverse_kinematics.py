"""
-*- coding: utf-8 -*-
Ludovic DE MATTEIS & Virgile BATTO, February 2024

Tools to compute the inverse kinematics of a closed loop system.
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


### Inverse Kinematics
def closedLoopInverseKinematicsCasadi(
    rmodel,
    rdata,
    cmodels,
    cdatas,
    target_pos,
    q_prec=None,
    name_eff="effecteur",
    onlytranslation=False,
):
    """
    Computes the inverse kinematics of a closed loop system.

    Args:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data.
        cmodels (list): Pinocchio constraint models list.
        cdatas (list): Pinocchio constraint datas list.
        target_pos (numpy.array): Target position.
        q_prec (numpy.array, optional): Previous configuration of the free joints. Defaults to None (set to neutral model pose).
        name_eff (str, optional): Name of the effector frame. Defaults to "effecteur".
        onlytranslation (bool, optional): Set to True to choose only translation (3D) and False to have 6D position. Defaults to False (6D).

    Returns:
        numpy.array: Configuration vector satisfying constraints (if optimization process succeeded).

    Notes:
        This function takes a target and an effector frame and finds a configuration of the robot such that the effector is as close as possible to the target
        and the robot constraints are satisfied. It is actually a geometry problem.

        This function solves the following minimization problem over q:
        min || d(eff(q), target) ||^2
        subject to:  f_c(q) = 0              # Kinematics constraints are satisfied

        Where d(eff(q), target) is a distance measure between the effector and the target.

        The problem is solved using Casadi + IpOpt.
    """
    # * Get effector frame id
    id_eff = rmodel.getFrameId(name_eff)

    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    if q_prec is None:
        q_prec = pin.neutral(rmodel)
    q_prec = np.array(q_prec)
    nq = rmodel.nq

    # * Set initial guess
    if len(q_prec) < rmodel.nq:
        q_prec = pin.neutral(rmodel)

    # * Optimisation functions
    cq = casadi.SX.sym("q", nq, 1)
    caspin.framesForwardKinematics(casmodel, casdata, cq)
    tip_translation = casadi.Function(
        "tip_trans", [cq], [casdata.oMf[id_eff].translation]
    )
    log6 = casadi.Function(
        "log6",
        [cq],
        [caspin.log6(casdata.oMf[id_eff].inverse() * caspin.SE3(target_pos)).vector],
    )

    def cost(q):
        if onlytranslation:
            terr = target_pos.translation - tip_translation(q)
            c = casadi.norm_2(terr) ** 2
        else:
            R_err = log6(q)
            c = casadi.norm_2(R_err) ** 2
        return c

    def constraints(q):
        Lc = constraintsResidual(
            casmodel,
            casdata,
            cmodels,
            cdatas,
            q,
            recompute=True,
            pinspace=caspin,
            quaternions=True,
        )
        return Lc

    constraint_cost = casadi.Function("constraint", [cq], [constraints(cq)])

    # * Optimisation problem
    optim = casadi.Opti()
    q = optim.variable(nq)
    # * Constraints
    optim.subject_to(constraint_cost(q) == 0)
    # * cost minimization
    total_cost = cost(q)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    optim.set_initial(q, q_prec)
    try:
        optim.solve_limited()
        print("Solution found")
        qs = optim.value(q)
    except AttributeError:
        print("ERROR in convergence, press enter to plot debug info.")
    return qs


def closedLoopInverseKinematicsScipy(
    rmodel,
    rdata,
    cmodels,
    cdatas,
    target_pos,
    q_prec=None,
    name_eff="effecteur",
    onlytranslation=False,
):
    """
    Computes the inverse kinematics of a closed loop system using Scipy optimization.

    Args:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data.
        cmodels (list): Pinocchio constraint models list.
        cdatas (list): Pinocchio constraint datas list.
        target_pos (pinocchio.SE3): Target position.
        q_prec (numpy.ndarray, optional): Previous configuration of the free joints. Defaults to None (set to neutral model pose).
        name_eff (str, optional): Name of the effector frame. Defaults to "effecteur".
        onlytranslation (bool, optional): Set to True to consider only translation (3D), False for 6D position. Defaults to False.

    Returns:
        numpy.ndarray: Configuration vector satisfying constraints (if optimization process succeeded).

    Notes:
        This function solves a minimization problem over q, where q is defined as q0 + dq, removing the need for quaternion constraints and reducing the number of decision variables.
        The objective is to minimize the distance between the effector and the target position, subject to the kinematics constraints being satisfied.
        The problem is formulated as follows:

        minimize || d(eff(q), target) ||^2
        subject to:  f_c(q) = 0              # Kinematics constraints are satisfied

        where d(eff(q), target) is a distance measure between the effector and the target position.

        The problem is solved using Scipy optimization.
    """

    id_eff = rmodel.getFrameId(name_eff)

    if q_prec is None:
        q_prec = pin.neutral(rmodel)
    q_prec = np.array(q_prec)

    def costnorm(vq):
        q = pin.integrate(rmodel, q_prec, vq)
        pin.framesForwardKinematics(rmodel, rdata, q)
        if onlytranslation:
            terr = target_pos.translation - rdata.oMf[id_eff].translation
            c = (norm(terr)) ** 2
        else:
            err = pin.log(target_pos.inverse() * rdata.oMf[id_eff]).vector
            c = (norm(err)) ** 2
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

    vq_sol = fmin_slsqp(costnorm, np.zeros(rmodel.nv), f_eqcons=contraintesimp)
    q_sol = pin.integrate(rmodel, q_prec, vq_sol)
    return q_sol


def closedLoopInverseKinematicsProximal(
    rmodel,
    rdata,
    cmodels,
    cdatas,
    target_pos,
    name_eff="effecteur",
    onlytranslation=False,
    max_it=100,
    eps=1e-12,
    rho=1e-5,
    mu=1e-4,
):
    """
    Solves the inverse kinematics problem for a closed-loop system using the proximal method.

    Args:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data.
        cmodels (list): List of Pinocchio constraint models.
        cdatas (list): List of Pinocchio constraint datas.
        target_pos (numpy.ndarray): Target position.
        name_eff (str, optional): Name of the effector frame. Defaults to "effecteur".
        onlytranslation (bool, optional): Set to True to consider only translation (3D), False for 6D position. Defaults to False.
        max_it (int, optional): Maximal number of proximal iterations. Defaults to 100.
        eps (float, optional): Proximal parameter epsilon. Defaults to 1e-12.
        rho (float, optional): Proximal parameter rho. Defaults to 1e-10.
        mu (float, optional): Proximal parameter mu. Defaults to 1e-4.

    Returns:
        numpy.ndarray: Configuration vector satisfying constraints (if optimization process succeeded).

    Notes:
        This function solves the inverse kinematics problem for a closed-loop system. It finds a configuration of the robot such that the effector is as close as possible to the target and the robot constraints are satisfied. The problem is formulated as a minimization problem over q, where q is defined as q0 + dq, removing the need for quaternion constraints and reducing the number of decision variables. The objective is to minimize the distance between the effector and the target, subject to the kinematics constraints being satisfied.

        Mathematically, the problem can be formulated as follows:

        minimize || d(eff(q), target) ||^2
        subject to:  f_c(q) = 0

        where:
        - eff(q) is the effector position given the configuration q.
        - target is the desired target position.
        - d(eff(q), target) is a distance measure between the effector and the target.
        - f_c(q) represents the kinematics constraints being satisfied.

        The problem is solved using the proximal method.

        Initially written by Justin Carpentier.
        Raw code available here (L84-126): https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    model = rmodel.copy()
    constraint_model = cmodels.copy()
    # add a contact constraint
    id_eff = rmodel.getFrameId(name_eff)
    frame_constraint = model.frames[id_eff]
    parent_joint = frame_constraint.parentJoint
    placement = frame_constraint.placement
    if onlytranslation:
        final_constraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D, model, parent_joint, placement, 0, target_pos
        )
    else:
        final_constraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D, model, parent_joint, placement, 0, target_pos
        )
    constraint_model.append(final_constraint)

    data = model.createData()
    constraint_data = [cm.createData() for cm in constraint_model]

    # proximal solver
    q = pin.neutral(model)
    constraint_dim = 0
    for cm in constraint_model:
        constraint_dim += cm.size()

    y = np.ones((constraint_dim))
    data.M = np.eye(model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_model)

    for k in range(max_it):
        pin.computeJointJacobians(model, data, q)
        kkt_constraint.compute(model, data, constraint_model, constraint_data, mu)

        constraint_value = np.concatenate(
            [
                (pin.log(cd.c1Mc2).np[: cm.size()])
                for (cd, cm) in zip(constraint_data, constraint_model)
            ]
        )

        LJ = []
        for cm, cd in zip(constraint_model, constraint_data):
            Jc = pin.getConstraintJacobian(model, data, cm, cd)
            LJ.append(Jc)
        J = np.concatenate(LJ)

        primal_feas = np.linalg.norm(constraint_value, np.inf)
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        if primal_feas < eps and dual_feas < eps:
            print("Convergence achieved in " + str(k) + " iterations")
            break
        print("constraint_value:", np.linalg.norm(constraint_value))
        rhs = np.concatenate([-constraint_value - y * mu, np.zeros(model.nv)])

        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        alpha = 1.0
        q = pin.integrate(model, q, -alpha * dq)
        y -= alpha * (-dy + y)

    return q


def closedLoopInverseKinematics(*args, **kwargs):
    """
    Perform closed-loop inverse kinematics based on available methods.

    This function dynamically selects the appropriate closed-loop inverse kinematics method based on predefined settings.

    Args:
        *args: Positional arguments passed to the selected inverse kinematics function.
        **kwargs: Keyword arguments passed to the selected inverse kinematics function.

    Returns:
        numpy.ndarray: Configuration vector satisfying constraints (if optimization process succeeded).

    Notes:
        - If _FORCE_PROXIMAL is set to True, the function uses the proximal method for inverse kinematics.
        - If _WITH_CASADI is set to True and _FORCE_PROXIMAL is False, the function uses Casadi for inverse kinematics.
        - If _WITH_CASADI is set to False and _FORCE_PROXIMAL is False, the function uses Scipy for inverse kinematics.
    """
    if _FORCE_PROXIMAL:
        return closedLoopInverseKinematicsProximal(*args, **kwargs)
    else:
        if _WITH_CASADI:
            return closedLoopInverseKinematicsCasadi(*args, **kwargs)
        else:
            return closedLoopInverseKinematicsScipy(*args, **kwargs)
