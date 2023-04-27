import pinocchio as pin
import os
from robot_info import *
from closed_loop_kinematics import *
from pinocchio.robot_wrapper import RobotWrapper

if __name__ == "__main__":
    #load robot
    path=os.getcwd()+"/robots/robot_marcheur_1"
    robot=RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    rmodel = robot.model
    visual_model = robot.visual_model
    rmodel = jointTypeUpdate(rmodel,rotule_name="to_rotule")
    rdata = rmodel.createData()

    # * Initialize visualizer
    viewer_type = 'Gepetto'
    if viewer_type == 'Gepetto':
      robot.initViewer(loadModel=True)
    elif viewer_type == 'Meshcat': # ! Not tested
        mv = pin.visualize.MeshcatVisualizer()
        robot.initViewer(mv, loadModel=True)

    robot.display(robot.q0)

    #create variable use by test
    Lidmot = getMotId_q(rmodel)
    goal = np.zeros(len(Lidmot))
    q_prec = q2freeq(rmodel, pin.neutral(rmodel)) # Initial guess
    q0, q_ini = closedLoopForwardKinematics(rmodel, rdata, goal, q_prec=q_prec)
    
    name_constraint = nameFrameConstraint(rmodel)
    constraint_model = getConstraintModelFromName(rmodel,name_constraint)
    constraint_data = [c.createData() for c in constraint_model]

    robot.display(q0)
    