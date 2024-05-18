"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS - February 2024

This module provides tools to perform loop closure on a robot model.
It is used to get configurations that minimizes the pinocchio constraints residuals under joints configurations constraints.
"""

import pinocchio as pin
import numpy as np
from pinocchio import casadi as caspin
import casadi
from warnings import warn

from .constraints import constraintsResidual


def partialLoopClosure(
    model,
    data,
    constraint_models,
    constraint_datas,
    fixed_joints_ids,
    fixed_rotations=[],
    q_ref=None,
    q_ws=None,
):
    """
    This function minimizes the constraints residuals of a robot model under joints configurations constraints.
    It uses the Pinocchio library for the kinematics and dynamics computations and the CasADi/IpOPT library for the optimization.

    Args:
        model (pinocchio.Model): The robot model in Pinocchio.
        data (pinocchio.Data): The robot data in Pinocchio.
        constraint_models (list): List of constraint models.
        constraint_datas (list): List of constraint data.
        fixed_joints_ids (list): List of IDs for joints that are fixed.
        fixed_rotations (list, optional): List of rotations that are fixed. Defaults to an empty list.
        q_ref (array, optional): The reference configuration. If not provided, the neutral configuration of the robot model is used.
        q_ws (array, optional): The warmstart configuration. If not provided, a zero array of size model.nv is used.

    Returns:
        array: The configuration q that satisfies the constraints.

    Raises:
        None

    Notes:
        The function solves the following optimization problem using CasADi + IPOpt:

        min || f(q) ||^2
        subject to:  q[i] = q_ref[i] for i in fixed_joints_ids
                     R(q[i:i+4])[0,1] = 0 for i in fixed_rotations

        where f(q) is the residual of the constraints, q is the configuration, R(q) is the rotation matrix of the configuration, and q_ref is the reference configuration.
    """
    if q_ref is None:
        q_ref = pin.neutral(model)
    if q_ws is not None:
        dq_ws = pin.difference(model, q_ref, q_ws)
    else:
        dq_ws = np.zeros(model.nv)
    # * Defining casadi models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()

    casconstraint_models = [caspin.RigidConstraintModel(cm) for cm in constraint_models]
    casconstraint_datas = [cm.createData() for cm in casconstraint_models]

    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(
            casmodel,
            casdata,
            casconstraint_models,
            casconstraint_datas,
            q,
            recompute=True,
            pinspace=caspin,
            quaternions=False,
        )
        return Lc

    sym_dq = casadi.SX.sym("dq", model.nv, 1)
    sym_cost = casadi.SX.sym("dq", 1, 1)
    cq = caspin.integrate(casmodel, casadi.SX(q_ref), sym_dq)

    integrate = casadi.Function("integrate", [sym_dq], [cq])

    constraints_res = casadi.Function("constraint", [sym_dq], [constraints(cq)])
    jac_cost = casadi.Function(
        "jac_cost",
        [sym_dq, sym_cost],
        [
            2
            * constraints_res(sym_dq).T
            @ constraints_res.jacobian()(sym_dq, constraints_res(sym_dq))
        ],
    )
    cost = casadi.Function(
        "cost",
        [sym_dq],
        [casadi.sumsqr(constraints_res(sym_dq))],
        {"custom_jacobian": jac_cost, "jac_penalty": 0},
    )

    rotations = {
        ids: casadi.Function(
            "rots_" + str(ids), [sym_dq], [caspin.exp3(sym_dq[ids : ids + 3])[1, 0]]
        )
        for ids in fixed_rotations
    }  # ! Only fixes the rotation around Z axis

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(model.nv)

    # * Problem
    total_cost = cost(vdq)
    optim.minimize(total_cost)
    for ids in fixed_rotations:
        optim.subject_to(rotations[ids](vdq) == 0)
    for ids in fixed_joints_ids:
        optim.subject_to(vdq[ids] == 0)
    optim.subject_to(
        optim.bounded(
            model.lowerPositionLimit, integrate(vdq), model.upperPositionLimit
        )
    )  # Bounding the controls to acceptable levels

    optim.set_initial(vdq, dq_ws)
    opts = {}
    s_opts = {"print_level": 0, "tol": 1e-6}
    optim.solver("ipopt", opts, s_opts)
    try:
        optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)
    except RuntimeError as e:
        # ? Should we raise an error here ? Remove the input and print the debug info ?
        print(e)
        print("ERROR in convergence, press enter to plot debug info.")
        input()
        dq = optim.debug.value(vdq)
        print(dq)
    print(optim.value(total_cost))
    if not optim.value(total_cost) < 1e-4:
        warn("Constraint not satisfied, make sure the problem is feasible")
    q = pin.integrate(model, q_ref, dq)
    return q


## Frames
def partialLoopClosureFrames(
    model,
    data,
    constraint_models,
    constraint_datas,
    framesIds=[],
    fixed_rotations=[],
    q_ref=None,
    q_ws=None,
):
    """
    This function minimizes the constraints residuals of a robot model under frames placements constraints.
    It uses the Pinocchio library for the kinematics and dynamics computations and the CasADi/IpOPT library for the optimization.

    Args:
        model (pinocchio.Model): The model on which the operation is to be performed.
        data (pinocchio.Data): The data associated with the model.
        constraint_models (list): A list of constraint models to be applied during the operation.
        constraint_datas (list): A list of data associated with each constraint model.
        framesIds (list, optional): A list of frame IDs to be considered in the operation. Defaults to an empty list.
        fixed_rotations (list, optional): A list of rotations that are to be kept fixed during the operation. Defaults to an empty list.
        q_ref (array, optional): The reference configuration. If not provided, the neutral configuration of the model is used.
        q_ws (array, optional): The working set configuration. If not provided, a zero array of size model.nv is used.

    Returns:
        q (array): The final configuration after the partial loop closure operation.

    Notes:
        The function solves the following optimization problem using CasADi + IPOpt:

        min || f(q) ||^2
        subject to:
            oMi(q)^-1 oMi(q_ref) = SE3_Id for i in framesIds
            R(q[i:i+4])[0,1] = 0 for i in fixed_rotations

        where f(q) is the residual of the constraints, q is the configuration, R(q) is the rotation matrix of the configuration, and q_ref is the reference configuration.
    """
    if q_ref is None:
        q_ref = pin.neutral(model)
    if q_ws is not None:
        dq_ws = pin.difference(model, q_ref, q_ws)
    else:
        dq_ws = np.zeros(model.nv)
    # Define frames SE3 references
    # TODO add option to only consider translation or only rotation or complete SE3
    pin.framesForwardKinematics(model, data, q_ref)
    frames_SE3 = {fId: caspin.SE3(data.oMf[fId]) for fId in framesIds}
    # Models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()
    cmodels = [caspin.RigidConstraintModel(cm) for cm in constraint_models]
    cdatas = [cm.createData() for cm in cmodels]

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

    sym_dq = casadi.SX.sym("dq", casmodel.nv, 1)
    sym_cost = casadi.SX.sym("dq", 1, 1)
    cq = caspin.integrate(casmodel, casadi.SX(q_ref), sym_dq)

    integrate = casadi.Function("integrate", [sym_dq], [cq])

    constraints_res = casadi.Function("constraint", [sym_dq], [constraints(cq)])
    jac_cost = casadi.Function(
        "jac_cost",
        [sym_dq, sym_cost],
        [
            2
            * constraints_res(sym_dq).T
            @ constraints_res.jacobian()(sym_dq, constraints_res(sym_dq))
        ],
    )
    cost = casadi.Function(
        "cost",
        [sym_dq],
        [casadi.sumsqr(constraints_res(sym_dq))],
        {"custom_jacobian": jac_cost, "jac_penalty": 0},
    )
    caspin.framesForwardKinematics(casmodel, casdata, cq)
    frames_compare = {
        fId: casadi.Function(
            "SE3_" + str(fId),
            [sym_dq],
            [caspin.log6(frames_SE3[fId].actInv(casdata.oMf[fId])).vector],
        )
        for fId in framesIds
    }

    rotations = {
        ids: casadi.Function(
            "rots_" + str(ids), [sym_dq], [caspin.exp3(sym_dq[ids : ids + 3])[1, 0]]
        )
        for ids in fixed_rotations
    }

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(model.nv)

    # * Problem
    total_cost = cost(vdq)
    optim.minimize(total_cost)
    for fId in framesIds:
        optim.subject_to(frames_compare[fId](vdq) == 0)
    for ids in fixed_rotations:
        optim.subject_to(rotations[ids](vdq) == 0)
    optim.subject_to(
        optim.bounded(
            model.lowerPositionLimit, integrate(vdq), model.upperPositionLimit
        )
    )  # Bounding the controls to acceptable levels

    optim.set_initial(vdq, dq_ws)
    opts = {}
    s_opts = {"print_level": 0, "tol": 1e-6}
    optim.solver("ipopt", opts, s_opts)
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
    print(optim.value(total_cost))
    if not optim.value(total_cost) < 1e-4:
        warn("Constraint not satisfied, make sure the problem is feasible")
    q = pin.integrate(model, q_ref, dq)
    return q  # I always return a value even if convergence failed
