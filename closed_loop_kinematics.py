"""
-*- coding: utf-8 -*-
Ludovic DE MATTEIS & Virgile BATTO, April 2023

Tools to compute the forwark and inverse kinematics of a robot with  closed loop 

"""
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
from robot_utils import mergeq, mergev

_FORCE_PROXIMAL = False

def closedLoopInverseKinematicsCasadi(rmodel, rdata, cmodels, cdatas, target_pos, q_prec=[], name_eff="effecteur", onlytranslation=False):
    """
        q=closedLoopInverseKinematics(model,data,fgoal,constraint_model,constraint_data,q_prec=[],name_eff="effecteur",nom_fermeture="fermeture",type="6D"):

        take the target position of the motors axis of the robot (joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration that match the goal position of the effector
    """
    # * Get effector frame id
    ideff = rmodel.getFrameId(name_eff)

    # * Defining casadi models
    casmodel = caspin.Model(rmodel)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    if len(q_prec) != (rmodel.nq):
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
        sol = optim.solve_limited()
        print("Solution found")
        qs = optim.value(q)
    except:
        print('ERROR in convergence, press enter to plot debug info.')
    return qs

def closedLoopInverseKinematicsScipy(rmodel, rdata, cmodels, cdatas, target_pos, q_prec=[], name_eff="effecteur", onlytranslation=False):
    """
    q=invgeom_parra(model,data,fgoal,constraint_model,constraint_data,q_prec=[],name_eff="effecteur",nom_fermeture="fermeture",type="6D"):

        take the goal position of the motors  axis of the robot ( joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration who match the goal position of the effector
    """

    ideff = rmodel.getFrameId(name_eff)

    if len(q_prec) < rmodel.nq:
        q_prec = pin.neutral(rmodel)
    q_prec = np.array(q_prec)

    def costnorm(q):
        cdata = rmodel.createData()
        pin.forwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        if onlytranslation:
            terr = (target_pos.translation - cdata.oMf[ideff].translation)
            c = (norm(terr)) ** 2
        else:
            err = pin.log(target_pos.inverse() * cdata.oMf[ideff]).vector
            c = (norm(err)) ** 2 
        return c 

    def contraintesimp(q):
        Lc = constraintsResidual(rmodel, rdata, cmodels, cdatas, q, recompute=True, pinspace=pin, quaternions=True)
        return Lc

    L = fmin_slsqp(costnorm, q_prec, f_eqcons=contraintesimp)

    return L

def closedLoopInverseKinematicsProximal(rmodel, rdata, rconstraint_model, rconstraint_data, target_pos, name_eff="effecteur", onlytranslation=False, max_it=100, eps=1e-12, rho=1e-10, mu=1e-4):
    """
    q=inverseGeomProximalSolver(rmodel,rdata,rconstraint_model,rconstraint_data,idframe,pos,only_translation=False,max_it=100,eps=1e-12,rho=1e-10,mu=1e-4)

    make the inverse kinematics with a constraint on the frame idframe that must be placed on pos (on world coordinate)
    raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    model=rmodel.copy()
    constraint_model=rconstraint_model.copy()
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

        constraint_value = np.concatenate([pin.log(cd.c1Mc2) for cd in constraint_data])

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

def closedLoopForwardKinematicsCasadi(rmodel, rdata, cmodels, cdatas, actuation_model, q_mot_target=None, q_prec=None):
    """
        closedLoopForwardKinematics(model, data, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"):

        Takes the target position of the motors axis of the robot (joint with name_mot ("mot" if empty) in the name),
        the current configuration of all joint (set to robot.q0 if let empty) and the name of the joints that close the kinematic loop. And returns a configuration that matches the goal positions of the motors
        This function solves a minimization problem over q. q is actually defined as q0+dq (this removes the need for quaternion constraints and gives less decision variables)
        
        min || q - q_prec ||^2

        subject to:  f_c(q)=0              # Kinematics constraints are satisfied
                     vq[motors]=q_motors    # The motors joints should be as commanded

        Argument:
            model - Pinocchio robot model
            data - Pinocchio robot data
            target - Target configuration of the motors joints. Should be have size of the number of motors
            q_prec - Previous configuration of the free joints
            n_mot - String contained in the motors joints names
            nom_fermeture - String contained in the frame names that should be in contact
            type - Constraint type
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
    
    cqf = casadi.SX.sym("qf", len(actuation_model.idqfree), 1)
    cvf = casadi.SX.sym("vf", len(actuation_model.idvfree), 1)
    cq = casadi.SX.sym("q", rmodel.nq, 1)
    cv = casadi.SX.sym("v", rmodel.nv, 1)
    constraintsCost = casadi.Function('constraint', [cq], [constraints(cq)])
    integrate = casadi.Function('integrate', [cq, cv],[ caspin.integrate(casmodel, cq, cv)])

    # * Optimisation problem
    optim = casadi.Opti()
    vdqf = optim.variable(len(actuation_model.idvfree))
    vdq = mergev(casmodel, actuation_model, casadi.MX.zeros(len(actuation_model.idvmot)), vdqf, True)
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
        sol = optim.solve_limited()
        print("Solution found")
        q = optim.value(vq)
    except:
        print('ERROR in convergence, press enter to plot debug info.')
        input()
        q = optim.debug.value(vq)
        print(q)

    return q # I always return a value even if convergence failed


def closedLoopForwardKinematicsScipy(rmodel, rdata, cmodels, cdatas, actuation_model, q_mot_target=None, q_prec=[]):

    """
        forwardgeom_parra(
        model, data, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"):

        take the goal position of the motors  axis of the robot ( joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration who match the goal position of the motor
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

def closedLoopForwardKinematicsProximal(model, data, constraint_model, constraint_data, actuation_model, q_mot_target=None, q_prec=[], max_it=100, eps=1e-12, rho=1e-10, mu=1e-4):
    """
    q=proximalSolver(model,data,constraint_model,constraint_data,max_it=100,eps=1e-12,rho=1e-10,mu=1e-4)
    build the robot in respect of the constraint with a proximal solver
    raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    Lid = actuation_model.idqmot
    if q_prec is None or q_prec == []:
        q_prec = pin.neutral(model)
    if q_mot_target is None:
        q_mot_target = np.zeros(len(Lid))
    q = q_prec
    q[Lid] = q_mot_target
      
    constraint_dim=0
    for cm in constraint_model:
        constraint_dim += cm.size() 

    y = np.ones((constraint_dim))
    data.M = np.eye(model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(model,constraint_model)

    for k in range(max_it):
        pin.computeJointJacobians(model,data,q)
        kkt_constraint.compute(model,data,constraint_model,constraint_data,mu)

        constraint_value = np.concatenate([pin.log(cd.c1Mc2) for cd in constraint_data])

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

def closedLoopForwardKinematics(*args, **kwargs):
    if _FORCE_PROXIMAL:
        return(closedLoopForwardKinematicsProximal(*args, **kwargs))
    else:
        if _WITH_CASADI:
            return(closedLoopForwardKinematicsCasadi(*args, **kwargs))
        else:
            return(closedLoopForwardKinematicsScipy(*args, **kwargs))

##########TEST ZONE ##########################
import unittest

class TestRobotInfo(unittest.TestCase):
    # def testInverseKinematicsScipy(self):
    #     # * Import robot
    #     path = "robots/robot_marcheur_1"
    #     model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(path)
    #     data = model.createData()
    #     constraint_datas = [cm.createData() for cm in constraint_models]

    #     # * Init variable used by Unitests
    #     fgoal = data.oMf[36]
    #     frame_effector = 'bout_pied_frame'

    #     InvKinCasadi = closedLoopInverseKinematicsCasadi(model, data, constraint_models, constraint_datas, fgoal, q_prec=[], name_eff=frame_effector, onlytranslation=False)
    #     InvKinScipy = closedLoopInverseKinematicsScipy(model, data, constraint_models, constraint_datas, fgoal, q_prec=[], name_eff=frame_effector, onlytranslation=False)
    #     InvKinProx = closedLoopInverseKinematicsProximal(model, data, constraint_models, constraint_datas, fgoal, name_eff=frame_effector, only_translation=False)        
        
    #     print(InvKinCasadi, InvKinScipy, InvKinProx)

    def testForwardKinematics(self):
        # * Import robot
        path = "robots/robot_marcheur_1"
        model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(path)
        data = model.createData()
        constraint_datas = [cm.createData() for cm in constraint_models]

        ForwKinCasadi = closedLoopForwardKinematicsCasadi(model, data, constraint_models, constraint_datas, actuation_model)
        ForwKinScipy = closedLoopForwardKinematicsScipy(model, data, constraint_models, constraint_datas, actuation_model)
        ForwKinProx = closedLoopForwardKinematicsProximal(model, data, constraint_models, constraint_datas, actuation_model)        
        
        truth = [ 0.00000000e+00,  0.00000000e+00,  1.11436057e-01, -6.65014110e-02,
                1.11436057e-01,  1.69335893e-01, -8.16182755e-01, -3.34177420e-01,
                -3.23783260e-03,  6.79048212e-04, -3.15738635e-02,  9.99495946e-01,
                3.40103913e-01, -2.24210819e-03, -7.53600303e-04,  8.12200125e-02,
                9.96693390e-01, -2.44903747e-01, -1.68944475e-02, -9.14960840e-04,
                -6.31408794e-03,  3.45985204e-02,  9.99380927e-01,  4.95244035e-04,
                -6.26732401e-03,  1.05141563e-01,  9.94437392e-01, -4.64840023e-02,
                1.86376632e-02, -3.71682025e-01,  9.27008278e-01]
    
        np.testing.assert_allclose(truth, ForwKinProx, rtol=1e-6, atol=1e-3)
        np.testing.assert_allclose(truth, ForwKinScipy, rtol=5, atol=0.5)
        np.testing.assert_allclose(truth, ForwKinCasadi, rtol=5, atol=0.5)
        
if __name__ == "__main__":
    if not _WITH_CASADI:
        raise(ImportError("To run unitests, casadi must be installed and loaded - import casadi failed"))
    else:
        from scipy.optimize import fmin_slsqp
        from numpy.linalg import norm
        from loader_tools import completeRobotLoader
        # * Run test
        unittest.main()






    


