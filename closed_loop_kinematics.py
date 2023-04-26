"""
-*- coding: utf-8 -*-
Virgile BATTO, march 2022

Tools to compute the forwark and inverse kinematics of a robot with  closed loop 

"""
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from numpy.linalg import norm
import os
from scipy.optimize import fmin_slsqp
import ipopt

from robot_info import *


def constraintQuaternion(model, q):
    """
    L=constraintQuaternion(model, q)
    return the list of 1 - the squared norm of each quaternion inside the configuration vector q (work for free flyer and spherical joint)
    """
    L = []
    for j in model.joints:
        idx_q = j.idx_q
        nq = j.nq
        nv = j.nv
        if nq != nv:
            quat = q[idx_q : idx_q + 4]
            L.append(norm(quat) ** 2 - 1)
    return L


def closedLoopForwardKinematics(
    model, data, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"
):

    """
        forwardgeom_parra(
        model, data, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture", type="6D"):

        take the goal position of the motors  axis of the robot ( joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration who match the goal position of the motor

    """

    if not (len(q_prec) == (model.nq - len(goal))):
        
        if len(q_prec) == 0:
            warn("!!!!!!!!!  no q_prec   !!!!!!!!!!!!!!")
        else:
            warn("!!!!!!!!!invalid q_prec!!!!!!!!!!!!!!")
        q_prec = q2freeq(model,pin.neutral(model))

    nombre_chaine = len(model.names) // 2
    Lid = idmot(model, name_mot)

    def goal2q(free_q):
        """
        take q, configuration of free axis/configuration vector, return nq, global configuration, q with the motor axi set to goal
        """
        rq = free_q.tolist()

        extend_q = np.zeros(model.nq)
        for i, goalq in zip(Lid, goal):
            extend_q[i] = goalq
        for i in range(model.nq):
            if not (i in Lid):
                extend_q[i] = rq.pop(0)

        return extend_q

    def costnorm(q):
        c = norm(q - q_prec) ** 2
        return c

    def contraintesimp(q):
        q = goal2q(q)
        if type == "3D":
            L1 = constraints3D(model, data, q, nombre_chaine, nom_fermeture)
        elif type == "planar":
            L1 = constraintsPlanar(model, data, q, nombre_chaine, nom_fermeture)
        else:
            L1 = constraints6D(model, data, q, nombre_chaine, nom_fermeture)
        L = []
        for l in L1:
            L = L + l.tolist()
        L2 = constraintQuaternion(model, q)
        return np.array(np.array(L + L2))

    free_q_goal = fmin_slsqp(costnorm, q_prec, f_eqcons=contraintesimp)
    q_goal = goal2q(free_q_goal)

    return q_goal, free_q_goal



def proximalSolver(model,data,constraint_model,constraint_data,max_it=100,eps=1e-12,rho=1e-10,mu=1e-4):
    """
    tentative 
    raw here (l84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """
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

        # J = pin.getFrameJacobian(model,data,constraint_model.joint1_id,constraint_model.joint1_placement,constraint_model.reference_frame)[:3,:]

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






def closedLoopInverseKinematics(model,data,fgoal,q_prec=[],name_eff="effecteur",nom_fermeture="fermeture",type="6D",onlytranslation=False):
    
    """
    q=invgeom_parra(model,data,fgoal,constraint_model,constraint_data,q_prec=[],name_eff="effecteur",nom_fermeture="fermeture",type="6D"):

        take the goal position of the motors  axis of the robot ( joint with name_mot, ("mot" if empty) in the name), the robot model and data,
        the current configuration of all joint ( set to robot.q0 if let empty)
        the name of the joint who close the kinematic loop nom_fermeture

        return a configuration who match the goal position of the effector

    """

    ideff = model.getFrameId(name_eff)

    if len(q_prec) < model.nq:
        q_prec = pin.neutral(model)
        if len(q_prec) == 0:
            warn("!!!!!!!!!  no q_prec   !!!!!!!!!!!!!!")
        else:
            warn("!!!!!!!!!invalid q_prec!!!!!!!!!!!!!!")

    nombre_chaine = len(model.names) // 2

    def costnorm(q):
        cdata = model.createData()
        pin.framesForwardKinematics(model, cdata, q)
        if onlytranslation:
            terr = (fgoal.translation - cdata.oMf[ideff].translation)
            c = (norm(terr)) ** 2
        else :
            err = pin.log(fgoal.inverse() * cdata.oMf[ideff]).vector
            c = (norm(err)) ** 2 
        return c 

    def contraintesimp(q):
        if type == "3D":
            L1 = constraints3D(model, data, q, nombre_chaine, nom_fermeture)
        elif type == "planar":
            L1 = constraintsPlanar(model, data, q, nombre_chaine, nom_fermeture)
        else:
            L1 = constraints6D(model, data, q, nombre_chaine, nom_fermeture)
        L = []
        for l in L1:
            L = L + l.tolist()
        L2 = constraintQuaternion(model, q)
        return np.array(np.array(L + L2))

    L = fmin_slsqp(costnorm, q_prec, f_eqcons=contraintesimp, full_output=True)

    return L


class ipotSolver(object):
    """
    class for ipopt solver.
    Jacobian non computed yet
    only 6D closed loop
    
    """
    def __init__(self):
        self.const = 0
        pass

    def objective(self, q):
        q_prec = self.q_prec
        return 0.5 * norm(q - q_prec)**2 

    def gradient(self, q):
        q_prec = self.q_prec
        return (q - q_prec)

    def constraints(self, q):
        goal = self.goal
        model = self.model
        eq = q.tolist()
        extend_q = np.zeros(model.nq)
        nq = model.nq
        Lid = idmot(model)
        for i, goalq in zip(Lid, goal):
            extend_q[i] = goalq
        for i in range(nq):
            if not (i in Lid):
                extend_q[i] = eq.pop(0)

        data = model.createData()
        n_boucle = len(model.names) // 2
        Lc = constraints6D(model, data, extend_q,nomb_boucle=self.number_closed_loop+1,nom_fermeture=self.name_closed_loop)
        self.nc=len(Lc)
        L = []
        n = 0
        for c in Lc:
            L = L + c.tolist()
            n = n + norm(c)
        
        
        L = L + constraintQuaternion(model, extend_q)
        self.const = norm(L)
        self.nc=len(L)
        return np.array(np.array(L))

    def jacobian(self, q):
        T=np.zeros((self.model.nv,self.nc))

        return np.array(T)

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Constraints value at iteration #%d is - %g" % (iter_count, self.const))

def closedLoop6DIpoptForwardKinematics(
    model, goal, q_prec=[], name_mot="mot", nom_fermeture="fermeture",number_closed_loop=-1
):
    Lidmot=idmot(model,name_mot)
    solv = ipotSolver()
    solv.model = model
    solv.goal = goal
    solv.Lidmot=Lidmot
    if number_closed_loop<0:
        number_closed_loop=len(nameFrameConstraint(model,nom_fermeture))
    solv.number_closed_loop=number_closed_loop
    solv.name_closed_loop=nom_fermeture

    if not (len(q_prec) == (model.nq - len(goal))):
        
        if len(q_prec) == 0:
            warn("!!!!!!!!!  no q_prec   !!!!!!!!!!!!!!")
        else:
            warn("!!!!!!!!!invalid q_prec!!!!!!!!!!!!!!")
        q_prec = q2freeq(model,pin.neutral(model))
    q_prec=np.array(q_prec)
    solv.q_prec = np.array(q_prec)
    

    lb_np = -1 * np.pi * np.ones(model.nq - len(Lidmot))
    ub_np = 1 * np.pi * np.ones(model.nq - len(Lidmot))
    
    cl_np = np.zeros(6 * number_closed_loop)




    nlp = ipopt.problem(
        n=model.nq - len(Lidmot),
        m=len(cl_np),
        problem_obj=solv,
        lb=lb_np,
        ub=ub_np,
        cl=cl_np,
        cu=cl_np,
    )


    nlp.addOption("mu_strategy", "adaptive")
    nlp.addOption("tol", 1e-8)
    nlp.addOption("max_iter", 100)
    nlp.addOption("jacobian_approximation", "finite-difference-values")
    q_free,info=nlp.solve(q_prec)

    def goal2q(free_q):
        """
        take q, configuration of free axis/configuration vector, return nq, global configuration, q with the motor axi set to goal
        """
        rq = free_q.tolist()

        extend_q = np.zeros(model.nq)
        for i, goalq in zip(Lidmot, goal):
            extend_q[i] = goalq
        for i in range(model.nq):
            if not (i in Lidmot):
                extend_q[i] = rq.pop(0)

        return extend_q
    q = goal2q(q_free)
    return(q,q_free)




##########TEST ZONE ##########################
import unittest

class TestRobotInfo(unittest.TestCase):
    def test_forwardkinematics(self):
        q0, q_ini= closedLoopForwardKinematics(new_model, new_data,goal,q_prec=q_prec)
        constraint=norm(constraints6D(new_model,new_data,q0))
        self.assertTrue(constraint<1e-6) #check the constraint
    def test_inversekinematics(self):
        InvKin=closedLoopInverseKinematics(new_model,new_data,fgoal,q_prec=q_prec,name_eff=frame_effector)
        self.assertTrue(InvKin[3]==0) #chexk that joint 15 is a spherical
    def test_forwarkinematicsipopt(self):
        qipopt,info=closedLoop6DIpoptForwardKinematics(new_model, goal, q_prec=q_prec)
        constraint=norm(constraints6D(new_model,new_data,qipopt))
        self.assertTrue(constraint<1e-6) #check the constraint


if __name__ == "__main__":
    # import robot
    path=os.getcwd()+"/robots/robot_marcheur_4"
    robot=RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    model=robot.model
    visual_model = robot.visual_model
    #change the joint type
    new_model=jointTypeUpdate(model,rotule_name="to_rotule")
    new_data=new_model.createData()
    
    #init variable use by test
    Lidmot=idmot(new_model)
    goal=np.zeros(len(Lidmot))
    q_prec=q2freeq(new_model,pin.neutral(new_model))
    fgoal=new_data.oMf[36]
    frame_effector='bout_pied_frame'
    
    #run test
    robot=RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    model,constraint_model=completeModelFromDirectory(path)
    data=model.createData()
    constraint_data=[cm.createData() for cm in constraint_model]
    unittest.main()


    


