import pinocchio as pin
import os
from loader_tools import completeRobotLoader
from closed_loop_mount import closedLoopMountCasadi, closedLoopMountScipy, closedLoopMountProximal
import numpy as np

if __name__ == "__main__":
    # * Load robot
    path = os.getcwd()+"/robots/robot_marcheur_4"
    model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(path)
    # model.lowerPositionLimit[32] = -1.5
    data = model.createData()
    constraint_datas = [cm.createData() for cm in constraint_models]

    # * Create vizualizer
    Viewer = pin.visualize.MeshcatVisualizer
    viz = Viewer(model, collision_model, visual_model)
    viz.initViewer(loadModel=True, open=True)

    # * Create variables
    q0 = pin.neutral(model)

    # * Initialize visualizer
    pin.framesForwardKinematics(model, data, q0)
    viz.display(q0)

    # * Get initial feasible configuration
    # q0 = closedLoopForwardKinematicsScipy(model, data, constraint_models, constraint_datas, actuation_model, goal, q_prec=q0)
    q0 = closedLoopMountProximal(model, data, constraint_models, constraint_datas, q_prec=q0)
    # q0 = closedLoopForwardKinematicsProximal(model, data, constraint_models, constraint_datas, actuation_model, goal, q_prec=q0)
    print("Solution found, press enter to visualize")
    input()

    # * Display
    viz.display(q0)
    