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

def closedLoopForwardKinematicsCasadi(rmodel, rdata, cmodels, cdatas, actuation_model, q_mot_target=None, q_prec=None):
    """
    Solve a minimization problem over the configuration vector 'q' to match the goal positions of the motors and satisfy constraints.

    Arguments:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data.
        cmodels (list): List of Pinocchio constraint models.
        cdatas (list): List of Pinocchio data associated with the constraint models.
        actuation_model: Robot actuation model.
        q_mot_target (numpy.ndarray, optional): Target position of the motor joints. Defaults to None.
        q_prec (numpy.ndarray, optional): Previous configuration of the free joints. Defaults to None.

    Returns:
        numpy.ndarray: Configuration vector satisfying constraints (if optimization process succeeded).
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
        Lc = constraintsResidual(casmodel, casdata, cmodels, cdatas, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc
    
    cq = casadi.SX.sym("q", rmodel.nq, 1)
    cv = casadi.SX.sym("v", rmodel.nv, 1)
    constraints_cost = casadi.Function('constraint', [cq], [constraints(cq)])
    integrate = casadi.Function('integrate', [cq, cv],[ caspin.integrate(casmodel, cq, cv)])

    # * Optimisation problem
    optim = casadi.Opti()
    vdqf = optim.variable(len(actuation_model.free_ids_v))
    vdq = mergev(casmodel, actuation_model, q_mot_target, vdqf, True)
    vq = integrate(q_prec, vdq)

    # * Constraints
    optim.subject_to(constraints_cost(vq)==0)
    optim.subject_to(optim.bounded(rmodel.lowerPositionLimit, vq, rmodel.upperPositionLimit))

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
        print('ERROR in convergence, press enter to plot debug info.')
        input()
        q = optim.debug.value(vq)
        print(q)

    return q # I always return a value even if convergence failed


def closedLoopForwardKinematicsScipy(rmodel, rdata, cmodels, cdatas, actuation_model, q_mot_target=None, q_prec=[]):
    """
    Solve a minimization problem over the configuration vector 'q' to match the goal positions of the motors and satisfy constraints.
    
    Arguments:
        rmodel (pinocchio.Model): Pinocchio robot model.
        rdata (pinocchio.Data): Pinocchio robot data.
        cmodels (list): List of Pinocchio constraint models.
        cdatas (list): List of Pinocchio data associated with the constraint models.
        actuation_model: Robot actuation model.
        q_mot_target (numpy.ndarray, optional): Target position of the motor joints. Defaults to None.
        q_prec (numpy.ndarray, optional): Previous configuration of the free joints. Defaults to [].

    Returns:
        numpy.ndarray: Configuration vector satisfying constraints (if optimization process succeeded).
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
        Lc = constraintsResidual(rmodel, rdata, cmodels, cdatas, q, recompute=True, pinspace=pin, quaternions=True)
        return Lc

    free_q_goal = fmin_slsqp(costnorm, q_free_prec, f_eqcons=contraintesimp)
    q_goal = mergeq(rmodel, actuation_model, q_mot_target, free_q_goal)
    return q_goal

def closedLoopForwardKinematics(*args, **kwargs):
    """
    Wrapper function for closed-loop forward kinematics.

    If Casadi library is available, it calls closedLoopForwardKinematicsCasadi.
    Otherwise, it calls closedLoopForwardKinematicsScipy.

    Returns:
        numpy.ndarray: Configuration vector satisfying constraints (if optimization process succeeded).
    """

    if _WITH_CASADI:
        return(closedLoopForwardKinematicsCasadi(*args, **kwargs))
    else:
        return(closedLoopForwardKinematicsScipy(*args, **kwargs))