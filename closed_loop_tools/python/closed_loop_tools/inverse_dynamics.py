"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS - February 2024

This module provides tools to perform the inverse dynamics of a closed-loop system.
"""

import pinocchio as pin
import numpy as np
from qpsolvers import solve_qp


### Inverse Kinematics
def closedLoopInverseDynamicsCasadi(
    rmodel, cmodels, q, vq, aq, act_matrix, u0, tol=1e-6
):
    """
    Computes the controls and contact forces that solve the following problem:

    Args:
        rmodel (pinocchio.Model): The robot model.
        cmodels (List[pinocchio.Model]): The contact models.
        q (np.ndarray): The joint positions.
        vq (np.ndarray): The joint velocities.
        aq (np.ndarray): The joint accelerations.
        act_matrix (np.ndarray): The actuation matrix.
        u0 (np.ndarray): The vector of reference controls.
        tol (float, optional): The tolerance for the QP solver. Defaults to 1e-6.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The optimized controls u[k] and the contact forces f.

    Raises:
        Exception: If there is an error in QP solving and the problem may be infeasible.

    Minimize (1/2) || u[k] - u_0||^2
    Subject to:  A u + J^T f = tau

    where:
    - A represents the actuation matrix,
    - u_0 is a vector of reference controls,
    - J is the contact Jacobian,
    - f is the contact forces,
    - tau is such that fwd(model, q, vq, tau) = aq,
    - fwd represents the integrated forward dynamics on a time dt.

    To solve the problem, the function first uses rnea to compute the controls tau such that fwd(model, q, vq, tau) = aq.
    Then, it formulates the minimization as a Quadratic Programming (QP) problem and solves it.
    """
    data = rmodel.createData()
    cdatas = [cm.createData() for cm in cmodels]
    nu = act_matrix.shape[1]
    nv = rmodel.nv
    assert nv == act_matrix.shape[0]
    nc = np.sum([cm.size() for cm in cmodels])
    nx = nu + nc

    pin.computeAllTerms(rmodel, data, q, vq)
    Jac = pin.getConstraintsJacobian(rmodel, data, cmodels, cdatas)

    P = np.diag(np.concatenate((np.ones(nu), np.zeros(nc))))
    p = np.hstack((u0, np.zeros(nc)))
    A = np.concatenate((act_matrix, Jac.transpose()), axis=1)
    b = pin.rnea(rmodel, data, q, vq, aq)
    G = np.zeros((nx, nx))
    h = np.zeros(nx)

    x = solve_qp(
        P, p, G, h, A, b, solver="proxqp", verbose=True, eps_abs=tol, max_iter=1_000_000
    )
    if x is None:
        raise ("Error in QP solving, problem may be infeasible")

    print(A @ x - b)

    return x[:nu], x[nu:]
