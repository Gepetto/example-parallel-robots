import pinocchio as pin
import numpy as np
from qpsolvers import solve_qp

### Inverse Kinematics
def closedLoopInverseDynamicsCasadi(rmodel, cmodels, q, vq, aq, act_matrix, u0, tol=1e-6):
    """
    Computes the controls and contact forces that solve the following problem.

    min (1/2) || u[k] - u_0||^2
    subject to:  A u + J^T f = tau

    where A represents the actuation matrix, 
    u_0 is a vector of reference controls, 
    J is the contact Jacobian, and tau is such that fwd(model, q, vq, tau) = aq
    where fwd represents the integrated forward dynamics on a time dt.

    To do so, we use rnea to get the controls tau such that fwd(model, q, vq, tau) = aq
    And then compute the minimization as a QP.

    Args:
        rmodel (Pinocchio.RobotModel): Pinocchio robot model.
        cmodels (list): List of Pinocchio robot constraint models.
        q (array-like): Configuration vector.
        vq (array-like): Joint velocity vector.
        aq (array-like): Joint acceleration vector.
        act_matrix (array-like): Actuation matrix.
        u0 (array-like): Vector of reference controls.
        tol (float, optional): Tolerance for the solver. Defaults to 1e-6.

    Returns:
        tuple: A tuple containing the computed controls (u) and contact forces (f).
    """
    data = rmodel.createData()
    cdatas = [cm.createData() for cm in cmodels]
    nu = act_matrix.shape[1]
    nv = rmodel.nv
    assert(nv == act_matrix.shape[0])
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

    x = solve_qp(P, p, G, h, A, b, solver="proxqp", verbose=True, eps_abs=tol, max_iter=1_000_000)
    if x is None:
        raise("Error in QP solving, problem may be infeasible")
    
    print(A@x - b)

    return x[:nu], x[nu:]
