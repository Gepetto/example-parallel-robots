import pinocchio as pin
import meshcat

from loader_tools import completeRobotLoader
from closed_loop_mount import (
    closedLoopMountScipy,
    closedLoopMountCasadi,
    closedLoopMountProximal,
)

## Get model and usefull variables
# * Load model
path = "../closed_loop_utils/robots/robot_marcheur_1"
model, robot_constraint_models, actuation_model, visual_model, collision_model = (
    completeRobotLoader(path, freeflyer=False)
)

# * Initialize data
data = model.createData()
robot_constraint_datas = [cm.createData() for cm in robot_constraint_models]
q0 = pin.neutral(model)

## Initialize Viewer
viz = pin.visualize.MeshcatVisualizer(model, visual_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")

viz.display(q0)

input()
q_mounted = closedLoopMountCasadi(
    model, data, robot_constraint_models, robot_constraint_datas, q_prec=q0
)
viz.display(q_mounted)
print(q_mounted)
input()
q_mounted = closedLoopMountScipy(
    model, data, robot_constraint_models, robot_constraint_datas, q_prec=q0
)
viz.display(q_mounted)
print(q_mounted)
input()
q_mounted = closedLoopMountProximal(
    model, data, robot_constraint_models, robot_constraint_datas, q_prec=q0
)
viz.display(q_mounted)
print(q_mounted)
