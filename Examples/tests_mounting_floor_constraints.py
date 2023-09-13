# import pinocchio as pin
# import os
# from loader_tools import completeRobotLoader
# from closed_loop_mount import closedLoopMountCasadi, closedLoopMountScipy, closedLoopMountProximal
# import numpy as np

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
from constraints import constraintsResidual
import casadi

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
constraint_models = list(np.array(constraint_models)[[0, 1, 2, 3, 4, 5]])
data = model.createData()
constraint_datas = [cm.createData() for cm in constraint_models]
rightFootFrameId = model.getFrameId('bout_pied_frame')         # Name of the frame to reach the target
leftFootFrameId = model.getFrameId('bout_pied_gauche_frame')
q0 = pin.neutral(model)
# q0 = np.array([ 8.56510306e-02, -1.06976882e-02,  1.34201444e-01, -8.31316228e-03,
#        -2.83978320e-01, -3.58867925e-03,  9.58787946e-01, -7.74047847e-03,
#         3.37448573e-03, -1.66196067e-03,  5.42259480e-01,  1.54755488e-01,
#         5.69476811e-02,  2.57494797e-02, -2.89787214e-02, -3.99347559e-02,
#         6.64000427e-04,  9.98781762e-01, -2.87402756e-02,  1.51948467e-02,
#        -1.16025551e-03,  9.99470743e-01,  6.86519373e-02, -2.70844612e-02,
#         3.37984874e-02,  6.64062351e-04,  9.99061386e-01, -8.20199158e-02,
#        -2.80483035e-02, -4.06883729e-02, -1.15998723e-03,  9.98777454e-01,
#         4.38277920e-02,  1.35201713e-01,  1.07457228e-04,  9.89848276e-01,
#         4.38278039e-02, -1.40053451e-01,  1.07437751e-04,  9.89173464e-01,
#         3.74917403e-02,  1.23787517e-03, -9.17580246e-02, -6.82283684e-01,
#         1.28201150e-02,  2.07034218e-01,  1.93528824e-02,  1.13090327e-02,
#        -4.56996903e-02,  4.11945912e-01,  9.09991324e-01,  1.61102237e-02,
#         3.01957456e-02, -4.96352974e-01,  8.67445908e-01,  1.64866793e-01,
#        -3.94529538e-02, -2.82514214e-02, -4.96243655e-01,  8.66826140e-01,
#        -1.45908951e-01,  2.51210940e-02, -3.59973209e-02,  4.11967323e-01,
#         9.10140675e-01, -1.35082458e-01,  4.19110631e-02,  3.09595877e-01,
#         9.40290692e-01,  8.53926776e-02,  1.19092433e-01,  3.09516467e-01,
#         9.39534268e-01])

# Dimensions
nv, nq = model.nv, model.nq

# * Initialize Viewer
viz = pin.visualize.MeshcatVisualizer(model, visual_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")

# * Adding new constraints
# Right
floorContactPositionRight = np.array([0, -0.07, 0])
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
floorContactPositionLeft = np.array([0, 0.07, 0.1])
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
# q0_feasible = closedLoopMount(model, data, constraint_models, constraint_datas, q_prec=q0)
casmodel = caspin.Model(model)
casdata = casmodel.createData()

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
vdq = optim.variable(model.nv)
vq = integrate(q0, vdq)

# * Constraints
optim.subject_to(constraintsCost(vq)==0)
# optim.subject_to(optim.bounded(model.lowerPositionLimit, vq, model.upperPositionLimit))

# * cost minimization
total_cost = casadi.sumsqr(vdq)
optim.minimize(total_cost)

opts = {}
optim.solver("ipopt", opts)
try:
    sol = optim.solve_limited()
    print("Solution found")
    q = optim.value(vq)
except:
    print('ERROR in convergence, press enter to plot debug info.')
    input()
    q = optim.debug.value(vq)
    print(q)
##
x0 = np.concatenate((q, np.zeros(model.nv)))
viz.display(q) 