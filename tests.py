import pinocchio as pin
import os
YAML = True
if YAML:
    from robot_info import completeRobotLoader, getMotId_q
else:
    from robot_info import jointTypeUpdate, getMotId_q, nameFrameConstraint, getConstraintModelFromName
from closed_loop_kinematics import closedLoopForwardKinematics, closedLoopForwardKinematicsCasadi, proximalSolver, closedLoopForwardKinematicsScipy
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np

if __name__ == "__main__":
    # * Load robot
    path = os.getcwd()+"/robots/robot_marcheur_4"
    if YAML :
        robot = completeRobotLoader(path)
        rmodel = robot.model
        rdata = robot.data
        constraint_model = robot.constraint_models
        constraint_data = robot.constraint_datas
    else :
        robot = RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)

        rmodel = robot.model = jointTypeUpdate(robot.model,rotule_name="to_rotule")
        rdata = robot.data = robot.model.createData()
         
        name_constraint = nameFrameConstraint(rmodel)
        constraint_model = getConstraintModelFromName(rmodel,name_constraint)
        constraint_data = [c.createData() for c in constraint_model]

    # * Create variables 
    q0 = robot.q0 = pin.neutral(rmodel)
    Lidmot = getMotId_q(rmodel)
    goal = np.zeros(len(Lidmot))   

    # robot.viewer.gui.deleteNode('world', True)    

    # * Initialize visualizer
    viewer_type = 'Gepetto'
    if viewer_type == 'Gepetto':
        robot.initViewer(loadModel=True)
    elif viewer_type == 'Meshcat': # ! Not tested
        mv = pin.visualize.MeshcatVisualizer()
        robot.initViewer(mv, loadModel=True)

    robot.framesForwardKinematics(q0)
    robot.display(q0)

    # * Get initial feasible configuration
    idx = [2,3]
    constraint_model = [ cm for i,cm in enumerate(constraint_model) if i in idx ]
    constraint_data = [ cm for i,cm in enumerate(constraint_data) if i in idx ]

    # q0_Scipy = closedLoopForwardKinematicsScipy(rmodel, rdata, constraint_model, constraint_data, goal, q_prec=q0)
    q0 = closedLoopForwardKinematicsCasadi(rmodel, rdata, constraint_model, constraint_data, goal, q_prec=q0)
    # q0 = proximalSolver(rmodel, rdata, constraint_model, constraint_data)
    print("Solution found, press enter to visualize")
    input()

    # * Display
    robot.display(q0)
    