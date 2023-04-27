import pinocchio as pin
import os
from robot_info import *
from closed_loop_kinematics import *
from pinocchio.robot_wrapper import RobotWrapper

if __name__ == "__main__":
    #load robot
    path = os.getcwd()+"/robot_marcheur_1"
    robot = RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)

    rmodel = robot.model = jointTypeUpdate(robot.model,rotule_name="to_rotule")
    rdata = robot.data = robot.model.createData()

    q0 = robot.q0 = pin.neutral(rmodel)
    
    # * Create variables 
    Lidmot = getMotId_q(rmodel)
    goal = np.zeros(len(Lidmot))
    q_prec = q2freeq(rmodel, pin.neutral(rmodel)) # Initial guess
    
    name_constraint = nameFrameConstraint(rmodel)
    constraint_model = getConstraintModelFromName(rmodel,name_constraint)
    constraint_data = [c.createData() for c in constraint_model]

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

    q0, q_ini = closedLoopForwardKinematics(rmodel, rdata, constraint_model, constraint_data, goal, q_prec=q_prec)

    # * Display
    robot.display(q0)
    