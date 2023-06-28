
import tkinter as tk
from Obsolete.tk_configuration import OptimizerFrame
import pinocchio as pin
import example_robot_data as robex
import numpy as np
import os
import sys
from util_load_robots import load_from_path, addXYZAxisToJoints, replaceGeomByXYZAxis, addXYZAxisToConstraints, addXYZAxisToFrames
from functools import partial

ABA = False
if ABA:
    from cdyn_trajectory import TrajOptimizer
else:
    from QVAF.qvaf_trajectory_simplerobots import TrajOptimizer

sys.path.append('..')
from closed_loop_utils.robot_info import completeRobotLoader

# * Load model
path = "../closed_loop_utils/robots/robot_simple_iso3D"
robot = completeRobotLoader(path)

# * Set a scale factor to handle too small and too big robots
scale = 1

# * Refresh the initial configuration
robot.q0 = pin.neutral(robot.model)

# # * Adding constraints frames
# addXYZAxisToConstraints(robot.model, robot.visual_model, robot.constraint_models, scale=scale)

# ? I don't have to fix joints here I think

# * Add frames to all joint
robot.model.gravity = pin.Motion(np.zeros(6)) # ! Disable gravity
addXYZAxisToFrames(robot.model, robot.visual_model, scale=scale, world=True)
addXYZAxisToJoints(robot.model, robot.visual_model, scale=scale)
robot.rebuildData()
robot.framesForwardKinematics(robot.q0)


# * Initialize visualizer
viewer_type = 'Gepetto'
if viewer_type == 'Gepetto':
    robot.initViewer(loadModel=True)
elif viewer_type == 'Meshcat': # ! Not tested
    mv = pin.visualize.MeshcatVisualizer()
    robot.initViewer(mv, loadModel=True)

replaceGeomByXYZAxis(robot.visual_model, robot.viz, scale=scale, visible="ON")
robot.framesForwardKinematics(robot.q0)
robot.display(robot.q0)

# * Defining Baumgart corrector gains: to be general this should be done in the definition of the robot
for cm in robot.constraint_models:
    cm.corrector.Kp = 10
    cm.corrector.Kd = 2. * np.sqrt(cm.corrector.Kp)

class OptimizerManager(OptimizerFrame):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot
        self.setOptimiser()
        self.optim.setInitialConf(self.robot.q0)
        self.createTarget()

    # * Optimizer manager
    def setOptimiser(self):
        self.optim = TrajOptimizer(robot, targetFrameName="fermeture3D_1A")
    def optimize(self, display_button):
        self.optim.optimize()
        display_button["state"]="normal"
    def display(self):
        self.optim.display()

    # * Target manager
    def createTarget(self):
        self.robot.viz.viewer.gui.addSphere("world/target", 0.02*scale, [1, 0, 0, 0.8])
        self.updateTargetPosition()
    def moveTarget(self):
        self.robot.viz.viewer.gui.applyConfiguration("world/target", self.target_pos)
        self.robot.viz.viewer.gui.refresh()
    def updateTargetPosition(self, target_pos=None):
        self.target_pos = target_pos if target_pos is not None else self.target_pos
        self.optim.setTargetPos(self.target_pos) # Update the position in the optimisation process
        self.moveTarget() # Update the visual target
    
    def resetAndDisp(self, display_button):
        # This function should reset the robot to its initial configuration
        display_button["state"]="disable"
        self.robot.framesForwardKinematics(robot.q0)
        self.robot.display(self.robot.q0)
        self.setOptimiser() # Todo may be better to find another way to reset, better than creating a new optimizer
        # TODO Verify that the only copy method is working properly, we should not need to recreate the optimiser anymore
        self.optim.setInitialConf(self.robot.q0)
        self.optim.setTargetPos(self.target_pos)

root = tk.Tk()
root.bind('<Escape>', lambda ev: root.destroy())
root.title("Simple Robot Sliders")
optimManager = OptimizerManager(robot)
optimManager.createTargetSliders(root, scale=scale)

optimFrame = tk.Frame(root)
optimFrame.pack(side=tk.BOTTOM)
display_button = tk.Button(optimFrame, text="Display",
                         command=optimManager.display, state="disabled")
display_button.pack(side=tk.LEFT, padx=10, pady=10)
reset_button = tk.Button(optimFrame, text="Reset",
                         command=partial(optimManager.resetAndDisp, display_button))
reset_button.pack(side=tk.LEFT, padx=10, pady=10)
optim_button = tk.Button(optimFrame, text="Optim",
                         command=partial(optimManager.optimize, display_button))
optim_button.pack(side=tk.LEFT, padx=10, pady=10)

root.mainloop()
