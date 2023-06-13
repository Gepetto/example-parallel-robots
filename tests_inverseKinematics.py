import pinocchio as pin
import os
from loader_tools import completeRobotLoader
from closed_loop_kinematics import closedLoopForwardKinematicsScipy, closedLoopForwardKinematicsCasadi, closedLoopForwardKinematicsProximal, closedLoopInverseKinematicsCasadi, closedLoopInverseKinematicsScipy, closedLoopInverseKinematicsProximal
import numpy as np

if __name__ == "__main__":
    # * Load robot
    path = os.getcwd()+"/robots/robot_marcheur_1"
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
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    viz.display(q0)

    # * Get initial feasible configuration
    q0 = closedLoopForwardKinematicsProximal(model, data, constraint_models, constraint_datas, actuation_model, goal, q_prec=q0)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    # * Display
    viz.display(q0)
    
    # * Inverse Kinematics
    fgoal = data.oMf[36].copy()
    frame_effector = 'bout_pied_frame'

    InvKinProx = closedLoopInverseKinematicsCasadi(model, data, constraint_models, constraint_datas, fgoal, name_eff=frame_effector, onlytranslation=False)
    # InvKinProx = closedLoopInverseKinematicsScipy(model, data, constraint_models, constraint_datas, fgoal, name_eff=frame_effector, onlytranslation=False)

    print("Got a solution")
    input()

    viz.display(InvKinProx)