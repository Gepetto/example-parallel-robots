import pinocchio as pin
from pinocchio import casadi as caspin
import casadi
import os
from loader_tools import completeRobotLoader
from closed_loop_kinematics import closedLoopForwardKinematicsScipy, closedLoopForwardKinematicsCasadi, closedLoopForwardKinematicsProximal, closedLoopInverseKinematicsCasadi, closedLoopInverseKinematicsScipy, closedLoopInverseKinematicsProximal
import numpy as np
from constraints import constraintsResidual
from itertools import combinations

if __name__ == "__main__":
    # * Load robot
    path = os.getcwd()+"/robots/robot_marcheur_4"
    model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(path)
    data = model.createData()
    constraint_datas = [cm.createData() for cm in constraint_models]

    # * Create vizualizer
    Viewer = pin.visualize.MeshcatVisualizer
    viz = Viewer(model, collision_model, visual_model)
    viz.initViewer(loadModel=True, open=True)

    # * Create variables 
    Lidmot = actuation_model.idqmot
    goal = np.zeros(len(Lidmot))   
    q0 = pin.neutral(model)

    # * Initialize visualizer
    pin.framesForwardKinematics(model, data, q0)
    viz.display(q0)

    # * Get initial feasible configuration

    ## * ForwardDynamics
    # vdq_init = np.zeros(model.nv)
    vdq_init = np.array([ 8.88717274e-03, -1.18611893e-04,  8.88646914e-03,  8.89722803e-03,
        2.11933390e-03, -3.33526016e-02,  1.61191975e-01, -2.88319667e-01,
        3.97998778e-03,  3.79653087e-02,  1.82083208e-01, -7.49722142e-03,
       -6.46804581e-02,  1.06943793e-01, -3.67527502e-01,  2.56377178e-02,
       -1.63610913e-01,  1.00772025e-01, -1.13037196e-01,  3.72041458e-02,
       -5.45255147e-02,  3.74060475e-02,  4.81669858e-02,  8.97203982e-02,
       -1.86873384e-02,  3.34147548e-02,  5.16785740e-01,  3.26563321e-02,
       -3.38244516e-01, -2.21297576e-02, -4.76783081e-01,  9.11184289e-02,
        2.62265850e-02, -4.79984183e-01,  3.25771612e-02, -3.94527411e-01,
        4.02109890e-02, -3.38584490e-02,  6.13122570e-02,  1.12271337e+00,
        9.81157238e-02, -3.57735409e-01, -3.98528995e-01,  1.89157301e-02,
       -1.24860115e-01, -8.82566215e-02,  3.78357181e-01, -3.46242227e-01,
        3.17384814e-01,  1.26941497e-01,  4.24033329e-01, -1.43467250e-01,
        1.21732898e-01,  3.97152719e-01, -6.85746024e-02,  2.35214784e-02,
        1.32739337e-02, -4.06860527e-01,  3.30787516e-02,  4.02491765e-01,
        9.37853816e-02, -6.85624658e-02, -3.29434826e-02,  8.84710353e-02,
       -4.29241975e-01,  9.16066185e-02,  2.10125350e-01,  3.18565342e-01,
       -3.46359727e-01, -1.16504703e-02, -6.20834217e-02, -3.08942397e-01,
        9.42718484e-03, -6.90946320e-02, -5.08840689e-01,  1.72509590e-02,
        2.55625989e-01,  3.38560524e-02, -2.41802184e-02, -1.46831406e-04,
       -3.30028689e-02,  8.89722803e-03])
    assert (len(vdq_init)==model.nv)
    # * Defining casadi models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    Lid = actuation_model.idqmot
    q_prec = pin.neutral(model)

    false = []
    true = []

    ids = [0,1,2,3,4,5,6,7,8,9,10]
    constraint_datas_red = [constraint_datas[i] for i in ids]
    constraint_models_red = [constraint_models[i] for i in ids]
    
    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, constraint_models_red, constraint_datas_red, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc
    
    cq = casadi.SX.sym("q", model.nq, 1)
    cv = casadi.SX.sym("v", model.nv, 1)
    constraintsCost = casadi.Function('constraint', [cq], [constraints(cq)])
    integrate = casadi.Function('integrate', [cq, cv],[ caspin.integrate(casmodel, cq, cv)])

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(model.nv)
    vq = integrate(q_prec, vdq)

    # * Constraints
    if len(constraint_models_red) != 0:
        optim.subject_to(constraintsCost(vq)==0)
    q_mot_target = None
    if q_mot_target is not None:
        optim.subject_to(vq[Lid]==q_mot_target)

    # ! Computing the derivatives to look for a NaN
    print("\n\nComputing constraints derivatives\n")
    print(constraintsCost(vq))
    print("\nComputing Jacobian x:")
    print(casadi.jacobian(constraintsCost(vq), vdq))

    print('Trying to compute the derivatives of the Lagrangian')

    # * cost minimization
    total_cost = casadi.sumsqr(vdq)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    optim.set_initial(vdq, vdq_init)
    try:
        sol = optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)   
        q0 = pin.integrate(model, q_prec, dq)
        true.append(ids)
    except:
        print('ERROR in convergence, press enter to plot debug info.')
        # input()
        dq = optim.debug.value(vdq)
        q = pin.integrate(model, q_prec, dq)
        # print(dq)
        # print(q)
        q0 = q_prec
        false.append(ids)

    # print("Solution found, press enter to visualize")
    # input()

    # # * Display
    # viz.display(q0)