import pinocchio as pin
import numpy as np
try:
    from pinocchio import casadi as caspin
    import casadi
    _WITH_CASADI = True
except:
    _WITH_CASADI = False
from scipy.optimize import fmin_slsqp
from numpy.linalg import norm

from constraints import constraintsResidual

_FORCE_PROXIMAL = False

def closedLoopMountCasadi(rmodel, rdata, cmodels, cdatas, q_prec=None):
    """
        closedLoopMountCasadi(rmodel, rdata, cmodels, cdatas, q_prec=None):

        This function takes the current configuration of the robot and projects it to the nearest feasible configuration - i.e. satisfying the constraints
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        
        min || q - q_prec ||^2
        subject to:  f_c(q)=0              # Kinematics constraints are satisfied

        Argument:
            rmodel - Pinocchio robot model
            rdata - Pinocchio robot data
            cmodels - Pinocchio constraint models
            cdatas - Pinocchio constraint datas
            q_prec - Previous configuration of the free joints
    """
    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    if q_prec is None:
        q_prec = pin.neutral(rmodel)

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
    vdq = optim.variable(rmodel.nv)
    vq = integrate(q_prec, vdq)

    # * Constraints
    optim.subject_to(constraintsCost(vq)==0)
    optim.subject_to(optim.bounded(rmodel.lowerPositionLimit, vq, rmodel.upperPositionLimit))

    # * cost minimization
    total_cost = casadi.sumsqr(vdq)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    try:
        sol = optim.solve_limited()
        print("Solution found")
        q = optim.value(vq)
    except:
        print('ERROR in convergence, press enter to plot debug info.')
        input()
        q = optim.debug.value(vq)
        print(q)

    return q # I always return a value even if convergence failed


def closedLoopMountScipy(rmodel, rdata, cmodels, cdatas, q_prec=None):
    """
        closedLoopMountCasadi(rmodel, rdata, cmodels, cdatas, q_prec=None):

        This function takes the current configuration of the robot and projects it to the nearest feasible configuration - i.e. satisfying the constraints
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        
        min || q - q_prec ||^2
        subject to:  f_c(q)=0              # Kinematics constraints are satisfied

        Argument:
            rmodel - Pinocchio robot model
            rdata - Pinocchio robot data
            cmodels - Pinocchio constraint models
            cdatas - Pinocchio constraint datas
            q_prec - Previous configuration of the free joints
    """
    if q_prec is None:
        q_prec = pin.neutral(rmodel)

    def costnorm(q):
        c = norm(q) ** 2
        return c

    def contraintesimp(q):
        Lc = constraintsResidual(rmodel, rdata, cmodels, cdatas, q, recompute=True, pinspace=pin, quaternions=True)
        return Lc

    q_goal = fmin_slsqp(costnorm, q_prec, f_eqcons=contraintesimp)
    return q_goal

def closedLoopMountProximal(model, data, constraint_model, constraint_data, q_prec=[], max_it=100, eps=1e-12, rho=1e-10, mu=1e-4):
    """
    q=proximalSolver(model,data,constraint_model,constraint_data,max_it=100,eps=1e-12,rho=1e-10,mu=1e-4)

    Build the robot in respect to the constraints using a proximal solver.

    Args:
        model (pinocchio.Model): Pinocchio model.
        data (pinocchio.Data): Pinocchio data.
        constraint_model (list): List of constraint models.
        constraint_data (list): List of constraint data.
        q_prec (list or np.array, optional): Initial guess for joint positions. Defaults to [].
        max_it (int, optional): Maximum number of iterations. Defaults to 100.
        eps (float, optional): Convergence threshold for primal and dual feasibility. Defaults to 1e-12.
        rho (float, optional): Scaling factor for the identity matrix. Defaults to 1e-10.
        mu (float, optional): Penalty parameter. Defaults to 1e-4.

    Returns:
        np.array: Joint positions of the robot respecting the constraints.
    
    raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    if q_prec is None or q_prec == []:
        q_prec = pin.neutral(model)
    q = q_prec
      
    constraint_dim=0
    for cm in constraint_model:
        constraint_dim += cm.size() 

    y = np.ones((constraint_dim))
    data.M = np.eye(model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(model,constraint_model)

    for k in range(max_it):
        pin.computeJointJacobians(model,data,q)
        kkt_constraint.compute(model,data,constraint_model,constraint_data,mu)

        constraint_value=np.concatenate([(pin.log(cd.c1Mc2).np[:cm.size()]) for (cd,cm) in zip(constraint_data,constraint_model)])

        LJ=[]
        for (cm,cd) in zip(constraint_model,constraint_data):
            Jc=pin.getConstraintJacobian(model,data,cm,cd)
            LJ.append(Jc)
        J=np.concatenate(LJ)

        primal_feas = np.linalg.norm(constraint_value,np.inf)
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y),np.inf)
        if primal_feas < eps and dual_feas < eps:
            print("Convergence achieved")
            break
        print("constraint_value:",np.linalg.norm(constraint_value))
        rhs = np.concatenate([-constraint_value - y*mu, np.zeros(model.nv)])

        dz = kkt_constraint.solve(rhs) 
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        alpha = 1.
        q = pin.integrate(model,q,-alpha*dq)
        y -= alpha*(-dy + y)
    return(q)

def closedLoopMount(*args, **kwargs):
    if _FORCE_PROXIMAL:
        return(closedLoopMountProximal(*args, **kwargs))
    else:
        if _WITH_CASADI:
            return(closedLoopMountCasadi(*args, **kwargs))
        else:
            return(closedLoopMountScipy(*args, **kwargs))