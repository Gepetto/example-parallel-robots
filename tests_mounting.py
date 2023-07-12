import pinocchio as pin
import os
from loader_tools import completeRobotLoader
from closed_loop_mount import closedLoopMountCasadi, closedLoopMountScipy, closedLoopMountProximal
import numpy as np

# if __name__ == "__main__":
#     # * Load robot
#     path = os.getcwd()+"/robots/robot_marcheur_two_leg"
#     model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(path)
#     # model.lowerPositionLimit[32] = -1.5
#     data = model.createData()
#     constraint_datas = [cm.createData() for cm in constraint_models]

#     # * Create vizualizer
#     Viewer = pin.visualize.MeshcatVisualizer
#     viz = Viewer(model, collision_model, visual_model)
#     viz.initViewer(loadModel=True, open=True)

#     # * Create variables
#     q0 = pin.neutral(model)

#     # * Initialize visualizer
#     pin.framesForwardKinematics(model, data, q0)
#     viz.display(q0)

#     # * Get initial feasible configuration
#     # q0 = closedLoopMountScipy(model, data, constraint_models, constraint_datas, q_prec=q0)
#     # q0 = closedLoopMountProximal(model, data, constraint_models, constraint_datas, q_prec=q0)
#     q0 = closedLoopMountProximal(model, data, constraint_models, constraint_datas, q_prec=q0)
#     print("Solution found, press enter to visualize")
#     input()

#     # * Display
#     viz.display(q0)

import pinocchio as pin
from pinocchio import casadi as caspin
import meshcat

import numpy as np
import sys, os

sys.path.append('../tk-cdyn/Utils')
import vizutils as vizutils

from loader_tools import completeRobotLoader
from closed_loop_mount import closedLoopMount

# Define a path to save figures if needed
SAVE_FIGS_DIR = None #"/home/ldematteis/Images/tk_cdyn/"+time.strftime("%Y%m%d-%H%M")
if SAVE_FIGS_DIR is not None and not os.path.isdir(SAVE_FIGS_DIR):
    os.makedirs(SAVE_FIGS_DIR)
DEBUG = True

# * Load model
path = "robots/robot_marcheur_two_leg"
model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(path, freeflyer=True)
data = model.createData()
constraint_datas = [cm.createData() for cm in constraint_models]
rightFootFrameId = model.getFrameId('bout_pied_frame')         # Name of the frame to reach the target
leftFootFrameId = model.getFrameId('bout_pied_gauche_frame')
q0 = pin.neutral(model)
model.gravity = pin.Motion(np.zeros(6))  # ! Disable gravity
# Dimensions
nv, nq = model.nv, model.nq

# * Initialize Viewer
viz = pin.visualize.MeshcatVisualizer(model, visual_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")

# * Adding new constraints
# Right
floorContactPositionRight = np.array([0, -0.07, -0.7])
MContactPlacementRight = pin.SE3(pin.utils.rotate('x', 0), floorContactPositionRight)        # SE3 position of the contact
rightFootFloorConstraint = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_6D,
    model,
    model.frames[rightFootFrameId].parentJoint,
    model.frames[rightFootFrameId].placement,
    0, # To the world
    MContactPlacementRight,
    pin.ReferenceFrame.LOCAL,
)
# Left
floorContactPositionLeft = np.array([0, 0.07, -0.7])
MContactPlacementLeft = pin.SE3(pin.utils.rotate('x', 0), floorContactPositionLeft)        # SE3 position of the contact
leftFootFloorConstraint = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_6D,
    model,
    model.frames[leftFootFrameId].parentJoint,
    model.frames[leftFootFrameId].placement,
    0, # To the world
    MContactPlacementLeft,
    pin.ReferenceFrame.LOCAL,
)
# Adding to constraint lists
constraint_models.append(rightFootFloorConstraint)
constraint_datas.append(rightFootFloorConstraint.createData())
constraint_models.append(leftFootFloorConstraint)
constraint_datas.append(leftFootFloorConstraint.createData())
# Adding visuals
contactNameRight = "world/floorRight"
vizutils.addViewerBox(viz, contactNameRight, 0.3, 0.1, 0.01, [1., .2, .2, .5])
vizutils.applyViewerConfiguration(viz, contactNameRight, pin.SE3ToXYZQUAT(MContactPlacementRight))
contactNameLeft = "world/floorLeft"
vizutils.addViewerBox(viz, contactNameLeft, 0.3, 0.1, 0.01, [.2, .2, 1., .5])
vizutils.applyViewerConfiguration(viz, contactNameLeft, pin.SE3ToXYZQUAT(MContactPlacementLeft))

# * Set initial configuration
# Get initial feasible configuration
q0_feasible = closedLoopMount(model, data, constraint_models, constraint_datas, q_prec=q0)
x0 = np.concatenate((q0_feasible, np.zeros(model.nv)))
viz.display(q0_feasible) 