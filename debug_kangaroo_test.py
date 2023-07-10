import pinocchio as pin
from pinocchio import casadi as caspin
import casadi
import os
from loader_tools import completeRobotLoader
import numpy as np
from constraints import constraintsResidual
from robot_utils import mergev

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
    # * Defining casadi models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    Lid = actuation_model.idqmot
    q_prec = pin.neutral(model)
    
    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, constraint_models, constraint_datas, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc
    
    cq = casadi.SX.sym("q", model.nq, 1)
    cv = casadi.SX.sym("v", model.nv, 1)
    constraintsCost = casadi.Function('constraint', [cq], [constraints(cq)])
    integrate = casadi.Function('integrate', [cq, cv],[ caspin.integrate(casmodel, cq, cv)])

    # * Optimisation problem
    optim = casadi.Opti()
    vdqf = optim.variable(len(actuation_model.idvfree))
    vdq = mergev(casmodel, actuation_model, casadi.MX.zeros(len(actuation_model.idvmot)), vdqf, True)
    vq = integrate(q_prec, vdq)

    # * Constraints
    optim.subject_to(constraintsCost(vq)==0)

    # * cost minimization
    total_cost = casadi.sumsqr(vdq)
    optim.minimize(total_cost)

    opts = {}
    optim.solver("ipopt", opts)
    try:
        sol = optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)   
        q0 = pin.integrate(model, q_prec, dq)
    except:
        print('ERROR in convergence, press enter to plot debug info.')
        dq = optim.debug.value(vdq)
        q = pin.integrate(model, q_prec, dq)
        q0 = q_prec

    print("Solution found, press enter to visualize")
    input()

    # * Display
    viz.display(q0)