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
    viz.initViewer(loadModel=True, open=False)

    # * Create variables 
    Lidmot = actuation_model.idqmot
    goal = np.zeros(len(Lidmot))   
    q0 = pin.neutral(model)

    # * Initialize visualizer
    pin.framesForwardKinematics(model, data, q0)
    viz.display(q0)

    # * Get initial feasible configuration

    ## * ForwardDynamics
    vdq_init = np.zeros(model.nv)
    vdq_init = [-1.88060804e-04,  1.13203647e-15, -1.88060804e-04, -2.27788368e-16,
                -2.13165658e-02,  4.77113581e-02,  1.93108276e-02,  5.95222974e-02,
                -2.20158505e-02, -1.10566578e-02,  5.93979437e-02, -2.22440983e-02,
                2.71539071e-03, -4.02082611e-02, -2.97768048e-01,  5.76121738e-03,
                -1.33743753e-01,  2.96254411e-02, -6.85481453e-02, -4.12560712e-02,
                -4.22292456e-03, -7.23467151e-02, -6.68259655e-03, -1.25549016e-04,
                -3.11490340e-04,  7.21015424e-01, -2.96556627e-01,  1.77018135e-02,
                -1.86899928e-01, -7.73351918e-04,  5.95660773e-02, -1.79894854e-01,
                2.38591501e-04,  2.55601264e-01,  1.72236015e-02, -1.84593394e-01,
                -1.69637740e-03, -9.20432396e-02, -1.63089037e-01,  5.08683516e-03,
                -2.02109129e-01,  3.18396798e-02,  1.73646999e-01, -6.68246733e-03,
                -1.71132652e-01, -9.37050864e-02,  5.07306439e-03,  2.32068886e-01,
                7.99003966e-03,  1.70348229e-01, -4.12560835e-02, -5.83129475e-02,
                -1.92257078e-01,  3.34248220e-04, -3.13297827e-02,  3.28068354e-01,
                7.73003935e-04, -1.92696400e-02, -3.12179994e-02,  3.28515652e-01,
                1.69631115e-03, -1.96339063e-02, -2.71923190e-01,  1.34663576e-01,
                7.21014765e-01, -9.01758434e-03, -1.93480002e-02,  1.91711768e-02,
                9.46779819e-02,  1.10002211e-02,  2.07557473e-02,  9.61900744e-02,
                -6.94112276e-16,  4.75110057e-02, -1.89406826e-16]
    assert (len(vdq_init)==model.nv)
    # * Defining casadi models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    Lid = actuation_model.idqmot
    q_prec = pin.neutral(model)

    false = []
    true = []

    for r in range(len(constraint_models)):
        for ids in combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], r):
            print(ids)
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
            print("\nComputing Jacobian :")
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
    print(ids)