"""
-*- coding: utf-8 -*-
Ludovic DE MATTEIS & Virgile BATTO, September 2023

Tools to compute the forwark and inverse kinematics of a robot with  closed loop 

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
def closedLoopInverseKinematicsCasadi(rmodel, rdata, cmodels, cdatas, target_pos, q_prec=None, name_eff="effecteur", onlytranslation=False):
    """
        closedLoopInverseKinematicsCasadi(rmodel, rdata, cmodels, cdatas, target_pos, q_prec=None, name_eff="effecteur", onlytranslation=False)

        This function takes a target and an effector frame and finds a configuration of the robot such that the effector is as close as possible to the target 
        and the robot constraints are satisfied. (It is actually a geometry problem)
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        leading to an optimisation on Lie group. We denoted d(eff(q), target) a distance measure between the effector and the target
        
        min || d(eff(q), target) ||^2
        subject to:  f_c(q)=0              # Kinematics constraints are satisfied

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
    # * Get effector frame id
    ideff = rmodel.getFrameId(name_eff)

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
    tip_translation = casadi.Function('tip_trans', [cq], [casdata.oMf[ideff].translation])
    log6 = casadi.Function('log6', [cq], [caspin.log6(casdata.oMf[ideff].inverse() * caspin.SE3(target_pos)).vector])

    def cost(q):
        if onlytranslation:
            terr = target_pos.translation - tip_translation(q)
            c = casadi.norm_2(terr) ** 2
        else:
            R_err = log6(q)
            c = casadi.norm_2(R_err) ** 2
        return(c)

    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, cmodels, cdatas, q, recompute=True, pinspace=caspin, quaternions=True)
        return Lc
    
    constraintsCost = casadi.Function('constraint', [cq], [constraints(cq)])

    # * Optimisation problem
    optim = casadi.Opti()
    q = optim.variable(nq)
    # * Constraints
    optim.subject_to(constraintsCost(q)==0)
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
        print('ERROR in convergence, press enter to plot debug info.')
    return qs

def closedLoopInverseKinematicsScipy(rmodel, rdata, cmodels, cdatas, target_pos, q_prec=None, name_eff="effecteur", onlytranslation=False):
    """
        closedLoopInverseKinematicsScipy(rmodel, rdata, cmodels, cdatas, target_pos, q_prec=None, name_eff="effecteur", onlytranslation=False)

        This function takes a target and an effector frame and finds a configuration of the robot such that the effector is as close as possible to the target 
        and the robot constraints are satisfied. (It is actually a geometry problem)
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        leading to an optimisation on Lie group. We denoted d(eff(q), target) a distance measure between the effector and the target
        
        min || d(eff(q), target) ||^2
        subject to:  f_c(q)=0              # Kinematics constraints are satisfied

        The problem is solved using Scipy

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

    ideff = rmodel.getFrameId(name_eff)

    if q_prec is None:
        q_prec = pin.neutral(rmodel)
    q_prec = np.array(q_prec)

    def costnorm(vq):
        q = pin.integrate(rmodel, q_prec, vq)
        pin.framesForwardKinematics(rmodel, rdata, q)
        if onlytranslation:
            terr = (target_pos.translation - rdata.oMf[ideff].translation)
            c = (norm(terr)) ** 2
        else:
            err = pin.log(target_pos.inverse() * rdata.oMf[ideff]).vector
            c = (norm(err)) ** 2 
        return c 

    def contraintesimp(vq):
        q = pin.integrate(rmodel, q_prec, vq)
        Lc = constraintsResidual(rmodel, rdata, cmodels, cdatas, q, recompute=True, pinspace=pin, quaternions=True)
        return Lc

    vq_sol = fmin_slsqp(costnorm, np.zeros(rmodel.nv), f_eqcons=contraintesimp)
    q_sol = pin.integrate(rmodel, q_prec, vq_sol)
    return q_sol

def closedLoopInverseKinematicsProximal(rmodel, rdata, cmodels, cdatas, target_pos, name_eff="effecteur", onlytranslation=False, max_it=100, eps=1e-12, rho=1e-5, mu=1e-4):
    """
        closedLoopInverseKinematicsProximal(rmodel, rdata, cmodels, cdatas, target_pos, q_prec=None, name_eff="effecteur", onlytranslation=False)

        This function takes a target and an effector frame and finds a configuration of the robot such that the effector is as close as possible to the target 
        and the robot constraints are satisfied. (It is actually a geometry problem)
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        leading to an optimisation on Lie group. We denoted d(eff(q), target) a distance measure between the effector and the target
        
        min || d(eff(q), target) ||^2
        subject to:  f_c(q)=0              # Kinematics constraints are satisfied

        The problem is solved using proximal method

        Argument:
            rmodel - Pinocchio robot model
            rdata - Pinocchio robot data
            cmodels - Pinocchio constraint models list
            cdatas - Pinocchio constraint datas list
            target_pos - Target position
            name_eff [Optionnal] - Name of the effector frame - default: "effecteur"
            onlytranslation [Optionnal] - Set to true to choose only translation (3D) and to false to have 6D position - default: False (6D)
            max_it [Optionnal] - Maximal number of proximal iterations - default: 100
            eps [Optinnal] - Proximal parameter epsilon - default: 1e-12
            rho [Optionnal] - Proximal parameter rho - default: 1e-10
            mu [Optionnal] - Proximal parameter mu - default: 1e-4
        Return:
            q - Configuration vector satisfying constraints (if optimisation process succeded)

        Initially written by Justin Carpentier    
        raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    model=rmodel.copy()
    constraint_model=cmodels.copy()
    #add a contact constraint
    ideff = rmodel.getFrameId(name_eff)
    frame_constraint=model.frames[ideff]
    parent_joint=frame_constraint.parentJoint
    placement=frame_constraint.placement
    if onlytranslation:
        final_constraint=pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,model,parent_joint,placement,0,target_pos)
    else :
        final_constraint=pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,parent_joint,placement,0,target_pos)
    constraint_model.append(final_constraint)

    data=model.createData()
    constraint_data=[cm.createData() for cm in constraint_model]

    #proximal solver (black magic)
    q=pin.neutral(model)
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
            print("Convergence achieved in " + str(k) + " iterations")
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

def closedLoopInverseKinematics(*args, **kwargs):
    if _FORCE_PROXIMAL:
        return(closedLoopInverseKinematicsProximal(*args, **kwargs))
    else:
        if _WITH_CASADI:
            return(closedLoopInverseKinematicsCasadi(*args, **kwargs))
        else:
            return(closedLoopInverseKinematicsScipy(*args, **kwargs))
