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
        closedLoopForwardKinematicsCasadi(rmodel, rdata, cmodels, cdatas, actuation_model, q_mot_target=None, q_prec=None)

        Takes the target position of the motors axis of the robot (following the actuation model), the current configuration of all joint (set to robot.q0 if let empty). 
        And returns a configuration that matches the goal positions of the motors and satisfies constraints
        This function solves a minimization problem over q but q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        
        min || q - q_prec ||^2

        subject to:  f_c(q)=0              # Kinematics constraints are satisfied
                     vq[motors]=q_motors    # The motors joints should be as commanded

        The problem is solved using Casadi + IpOpt

        Argument:
            rmodel - Pinocchio robot model
            rdata - Pinocchio robot data
            cmodels - Pinocchio constraint models list
            cdatas - Pinocchio constraint datas list
            target_pos - Target position
            q_prec [Optionnal] - Previous configuration of the free joints - default: None (set to neutral model pose)
            name_eff [Optionnal] - Name of the effector frame - default: "effecteur"
            onlytranslation [Optionnal] - Set to true to choose only translation (3D) and to false to have 6D position - default: False (6D)
        Return:
            q - Configuration vector satisfying constraints (if optimisation process succeded)
    """
    
    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    Lid = actuation_model.idqmot
    if q_prec is None or q_prec == []:
        q_prec = pin.neutral(rmodel)
    if q_mot_target is None:
        q_mot_target = np.zeros(len(Lid))

    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, cmodels, cdatas, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc
    
    cq = casadi.SX.sym("q", rmodel.nq, 1)
    cv = casadi.SX.sym("v", rmodel.nv, 1)
    constraintsCost = casadi.Function('constraint', [cq], [constraints(cq)])
    integrate = casadi.Function('integrate', [cq, cv],[ caspin.integrate(casmodel, cq, cv)])

    # * Optimisation problem
    optim = casadi.Opti()
    vdqf = optim.variable(len(actuation_model.idvfree))
    vdq = mergev(casmodel, actuation_model, q_mot_target, vdqf, True)
    vq = integrate(q_prec, vdq)

    # * Constraints
    optim.subject_to(constraintsCost(vq)==0)
    optim.subject_to(optim.bounded(rmodel.lowerPositionLimit, vq, rmodel.upperPositionLimit))

    # * cost minimization
    total_cost = casadi.sumsqr(vdqf)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    optim.set_initial(vdqf, np.zeros(len(actuation_model.idvfree)))
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
    closedLoopForwardKinematicsScipy(rmodel, rdata, cmodels, cdatas, actuation_model, q_mot_target=None, q_prec=None)

    Takes the target position of the motors axis of the robot (following the actuation model), the current configuration of all joint (set to robot.q0 if let empty). 
    And returns a configuration that matches the goal positions of the motors and satisfies constraints
    This function solves a minimization problem over q but q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
    
    min || q - q_prec ||^2

    subject to:  f_c(q)=0              # Kinematics constraints are satisfied
                    vq[motors]=q_motors    # The motors joints should be as commanded

    The problem is solved using Scipy SLSQP

    Argument:
        rmodel - Pinocchio robot model
        rdata - Pinocchio robot data
        cmodels - Pinocchio constraint models list
        cdatas - Pinocchio constraint datas list
        target_pos - Target position
        q_prec [Optionnal] - Previous configuration of the free joints - default: None (set to neutral model pose)
        name_eff [Optionnal] - Name of the effector frame - default: "effecteur"
        onlytranslation [Optionnal] - Set to true to choose only translation (3D) and to false to have 6D position - default: False (6D)
    Return:
        q - Configuration vector satisfying constraints (if optimisation process succeded)
    """
    Lid = actuation_model.idqmot
    if q_prec is None or q_prec == []:
        q_prec = pin.neutral(rmodel)
    if q_mot_target is None:
        q_mot_target = np.zeros(len(Lid))
    q_free_prec = q_prec[actuation_model.idqfree]

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
    if _WITH_CASADI:
        return(closedLoopForwardKinematicsCasadi(*args, **kwargs))
    else:
        return(closedLoopForwardKinematicsScipy(*args, **kwargs))