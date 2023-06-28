"""
-*- coding: utf-8 -*-
Ludovic DE MATTEIS - May 2023

Create a Tkinter interface to move some joints in the robots while satisfying the desired closed loop constraints

"""

import tkinter as tk
from Obsolete.tk_configuration import RobotFrame
import pinocchio as pin
import example_robot_data as robex
from util_load_robots import addXYZAxisToJoints, replaceGeomByXYZAxis, addXYZAxisToConstraints
from casadi_projection import ProjectConfig

import sys
sys.path.append('..')
from closed_loop_utils.robot_info import completeRobotLoader

# * Load model
urdf_path = "../closed_loop_utils/robots/robot_simple_iso6D"
robot = completeRobotLoader(urdf_path)

# * Set a scale factor to handle too small and too big robots
scale = 1

# * Refresh the initial configuration
robot.q0 = pin.neutral(robot.model)

# * Adding constraints
addXYZAxisToConstraints(robot.model, robot.visual_model, robot.constraint_models, scale=scale)

# * Add frames to all joints
addXYZAxisToJoints(robot.model, robot.visual_model, scale=scale)
robot.rebuildData()

# * Initialize visualizer
viewer_type = 'Gepetto'
if viewer_type == 'Gepetto':
    robot.initViewer(loadModel=True)
elif viewer_type == 'Meshcat': # ! Not working yet
    mv = pin.visualize.meshCatVisualizer()
    robot.initViewer(mv, loadModel=True)

# * Add axis to the frames
replaceGeomByXYZAxis(robot.visual_model, robot.viz, scale=scale)

# * Display initial configuration (This one does not satisfies the constraints)
robot.display(robot.q0)

class ConstraintsManager:
    def __init__(self, robotConstraintFrame):
        self.robotConstraintFrame = robotConstraintFrame # Tkinter projection manager
        self.project = self.robotConstraintFrame.project # Projector

    def computeConstrainedConfig(self):
        qref = self.robotConstraintFrame.getConfiguration(False)    # Get the sliders configuration
        q = self.project(qref)  # Project to get the nearest feasible configuration
        self.robotConstraintFrame.resetConfiguration(q)
        self.robotConstraintFrame.display()

    def resetAndDisp(self): # Self-explanatory
        self.robotConstraintFrame.resetConfiguration(robot.q0)
        self.robotConstraintFrame.display()

class RobotConstraintFrame(RobotFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setProjector()

    def setProjector(self):
        self.project = ProjectConfig(robot)

    def slider_display(self, i, v): # Overwrites the parent function
        qref = robotFrame.getConfiguration(self.boolVar.get())
        q = self.project(qref, i)
        robotFrame.resetConfiguration(q)
        robotFrame.display()

    def setVerboseVariable(self, boolVar):
        self.boolVar = boolVar
        self.verboseProjector()

    def verboseProjector(self):
        self.project.verbose = self.boolVar.get()

# * Creating the interface
root = tk.Tk()
root.bind('<Escape>', lambda ev: root.destroy())
root.title("Simple Robot Sliders")
robotFrame = RobotConstraintFrame(robot.model, robot.q0, robot,
                                  motors=[n for n in robot.model.names if 'mot' in n])
robotFrame.createSlider(root)       # Creating sliders, main projection functions are called when the sliders are moved
robotFrame.createRefreshButons(root)

constraintsManager = ConstraintsManager(robotFrame)
optimFrame = tk.Frame(root)
optimFrame.pack(side=tk.BOTTOM)
reset_button = tk.Button(optimFrame, text="Reset",
                         command=constraintsManager.resetAndDisp)
reset_button.pack(side=tk.LEFT, padx=10, pady=10)
optim_button = tk.Button(optimFrame, text="Optim",
                         command=constraintsManager.computeConstrainedConfig)
optim_button.pack(side=tk.LEFT, padx=10, pady=10)
verbose_var = tk.BooleanVar(False)
robotFrame.setVerboseVariable(verbose_var)
verbose_checkbox = tk.Checkbutton(optimFrame, variable=verbose_var, text='Verbose',
                                  command=robotFrame.verboseProjector)
verbose_checkbox.pack(side=tk.LEFT, padx=10, pady=10)

constraintWindow = tk.Toplevel()
constraintWindow.bind('<Escape>', lambda ev: root.destroy())
## 

# ? The two following classes or tkinter interfaces, they may go in `tk_configuration.py`
# * Interface to activate or deactivate constraints on the robot
class CheckboxConstraintCmd:
    def __init__(self, boolVar, cm, project):
        self.boolVar = boolVar
        self.cm = cm
        self.project = project

    def __call__(self):
        if self.boolVar.get():
            print(f'Activate {self.cm.name}')
            assert (self.cm not in robot.constraint_models)
            robot.constraint_models.append(self.cm)
            robot.constraint_datas = [
                robot.full_constraint_datas[cm] for cm in robot.constraint_models]
        else:
            print(f'Deactivate {self.cm.name}')
            assert (self.cm in robot.constraint_models)
            robot.constraint_models.remove(self.cm)
        robot.constraint_datas = [robot.full_constraint_datas[cm]
                                  for cm in robot.constraint_models]
        self.project.recomputeConstraints()

# * Interface to display or hide constraints on the robot
class CheckboxDisplayConstraintCmd:
    def __init__(self, boolVar, cm, vm, viz):
        self.boolVar = boolVar
        self.cm = cm
        self.viz = viz
        # Get viewer object names with pinocchio convention
        idxs = [vm.getGeometryId(f'XYZ_cst_{cm.name}_1'),
                vm.getGeometryId(f'XYZ_cst_{cm.name}_2')]
        self.gname = [viz.getViewerNodeName(vm.geometryObjects[idx], pin.VISUAL)
                      for idx in idxs]

    def __call__(self):
        print(f'Set display {self.cm.name} to {self.boolVar.get()}')
        for n in self.gname:
            self.viz.viewer.gui.setVisibility(
                n, 'ON' if self.boolVar.get() else 'OFF')


# * Setting the positions of elements in the active/display constraints window
constraintFrame = tk.Frame(constraintWindow)
constraintFrame.pack(side=tk.BOTTOM)
actLabel = tk.Label(constraintFrame, text='active')
actLabel.grid(row=0, column=1)
dispLabel = tk.Label(constraintFrame, text='display')
dispLabel.grid(row=0, column=2)

for i, cm in enumerate(robot.constraint_models):
    cstLabel = tk.Label(constraintFrame, text=cm.name)
    cstLabel.grid(row=i+1, column=0)

    active_constraint_var = tk.BooleanVar(value=cm in robot.constraint_models)
    active_constraint_cmd = CheckboxConstraintCmd(
        active_constraint_var, cm, robotFrame.project)
    constraint_checkbox = tk.Checkbutton(constraintFrame, variable=active_constraint_var,
                                         command=active_constraint_cmd)
    constraint_checkbox.grid(row=i+1, column=1)

    display_constraint_var = tk.BooleanVar(value=cm in robot.constraint_models)
    display_constraint_cmd = CheckboxDisplayConstraintCmd(
        display_constraint_var, cm, robot.visual_model, robot.viz)
    display_constraint_cmd()
    display_constraint_checkbox = tk.Checkbutton(constraintFrame, variable=display_constraint_var,
                                                 command=display_constraint_cmd)
    display_constraint_checkbox.grid(row=i+1, column=2)

root.mainloop()
