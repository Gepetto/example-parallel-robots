import pinocchio as pin
import os
from loader_tools import completeRobotLoader
from closed_loop_kinematics import closedLoopInverseKinematicsProximal
import numpy as np
import meshcat
import sys

sys.path.append("../tk-cdyn/Utils")
import vizutils

if __name__ == "__main__":
    # * Load robot
    path = os.getcwd() + "/robots/robot_marcheur_1"
    model, constraint_models, actuation_model, visual_model, collision_model = (
        completeRobotLoader(path)
    )
    data = model.createData()
    constraint_datas = [cm.createData() for cm in constraint_models]

    # * Create vizualizer
    viz = pin.visualize.MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")

    # * Create variables
    Lidmot = actuation_model.mot_ids_q
    goal = np.zeros(len(Lidmot))
    q0 = pin.neutral(model)

    # * Initialize visualizer
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    # q0 = closedLoopMount(model, data, constraint_models, constraint_datas)
    viz.display(q0)

    Mgoal = pin.SE3(pin.utils.rotate("x", 0), np.array([-0.5, 0, -0.6]))
    contactNameRight = "world/floorRight"
    vizutils.addViewerSphere(viz, contactNameRight, 0.03, [1.0, 0.2, 0.2, 0.5])
    vizutils.applyViewerConfiguration(viz, contactNameRight, pin.SE3ToXYZQUAT(Mgoal))

    # * Inverse Kinematics
    frame_effector = "bout_pied_frame"

    InvKin = closedLoopInverseKinematicsProximal(
        model,
        data,
        constraint_models,
        constraint_datas,
        Mgoal,
        name_eff=frame_effector,
        onlytranslation=False,
        max_it=100,
        eps=1e-12,
        rho=1e-5,
        mu=1e-4,
    )

    print("Got a solution")
    input()

    viz.display(InvKin)
