"""
-*- coding: utf-8 -*-
Ludovic DE MATTEIS - May 2023

Create a Tkinter interface to move some joints in the robots while satisfying the desired closed loop constraints

"""
import meshcat
import tkinter as tk
from sliders.tk_configuration import RobotFrame
import pinocchio as pin
from sliders.util_frames import addXYZAxisToJoints, replaceGeomByXYZAxis, addXYZAxisToConstraints
from sliders.casadi_projection import ProjectConfig

from loader_tools import completeRobotLoader

# * Load model
robot_path = "robots/5bar_linkage_iso3d"
model, full_constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(robot_path)
full_constraint_datas = [cm.createData() for cm in full_constraint_models]

constraint_models, constraint_datas = full_constraint_models.copy(), full_constraint_datas.copy()
# * Set a scale factor to handle too small and too big robots
scale = 1

# * Refresh the initial configuration
q0 = pin.neutral(model)

# * Initialize the viewer
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(model, collision_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")
viz.display(q0)
print("Display q0")

# # * Adding constraints
# addXYZAxisToConstraints(model, visual_model, constraint_models, scale=scale)

# # * Add frames to all joints
# addXYZAxisToJoints(model, visual_model, scale=scale)
data = model.createData()

# * Add axis to the frames
replaceGeomByXYZAxis(visual_model, viz, scale=scale)

# * Display initial configuration (This one does not satisfies the constraints)
print("Done with replace")
viz.display(q0)
print("Display q0")

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
        self.robotConstraintFrame.resetConfiguration(q0)
        self.robotConstraintFrame.display()

class RobotConstraintFrame(RobotFrame):
    def __init__(self, model, constraint_models, actuation_model, q0, viz):
        super().__init__(model, constraint_models, actuation_model, q0, viz)
        self.model = model
        self.constraint_models = constraint_models
        self.setProjector()

    def setProjector(self):
        self.project = ProjectConfig(self.model, self.constraint_models)

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
robotFrame = RobotConstraintFrame(model, constraint_models, actuation_model, q0, viz)
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
            assert (self.cm not in constraint_models)
            constraint_models.append(self.cm)
        else:
            print(f'Deactivate {self.cm.name}')
            assert (self.cm in constraint_models)
            constraint_models.remove(self.cm)
        self.project.recomputeConstraints(constraint_models)

# * Interface to display or hide constraints on the robot
# class CheckboxDisplayConstraintCmd:
#     def __init__(self, boolVar, cm, vm, viz):
#         self.boolVar = boolVar
#         self.cm = cm
#         self.viz = viz
#         # Get viewer object names with pinocchio convention
#         idxs = [vm.getGeometryId(f'XYZ_cst_{cm.name}_1'),
#                 vm.getGeometryId(f'XYZ_cst_{cm.name}_2')]
#         self.gname = [viz.getViewerNodeName(vm.geometryObjects[idx], pin.VISUAL)
#                       for idx in idxs]

#     def __call__(self):
#         print(f'Set display {self.cm.name} to {self.boolVar.get()}')
#         for n in self.gname:
#             self.viz.viewer.gui.setVisibility(
#                 n, 'ON' if self.boolVar.get() else 'OFF')


# * Setting the positions of elements in the active/display constraints window
constraintFrame = tk.Frame(constraintWindow)
constraintFrame.pack(side=tk.BOTTOM)
actLabel = tk.Label(constraintFrame, text='active')
actLabel.grid(row=0, column=1)
dispLabel = tk.Label(constraintFrame, text='display')
dispLabel.grid(row=0, column=2)

for i, cm in enumerate(full_constraint_models):
    cstLabel = tk.Label(constraintFrame, text=cm.name)
    cstLabel.grid(row=i+1, column=0)

    active_constraint_var = tk.BooleanVar(value=cm in full_constraint_models)
    active_constraint_cmd = CheckboxConstraintCmd(
        active_constraint_var, cm, robotFrame.project)
    constraint_checkbox = tk.Checkbutton(constraintFrame, variable=active_constraint_var,
                                         command=active_constraint_cmd)
    constraint_checkbox.grid(row=i+1, column=1)

    # display_constraint_var = tk.BooleanVar(value=cm in full_constraint_models)
    # display_constraint_cmd = CheckboxDisplayConstraintCmd(
    #     display_constraint_var, cm, visual_model, viz)
    # display_constraint_cmd()
    # display_constraint_checkbox = tk.Checkbutton(constraintFrame, variable=display_constraint_var,
    #                                              command=display_constraint_cmd)
    # display_constraint_checkbox.grid(row=i+1, column=2)

root.mainloop()
