import tkinter as tk
from Obsolete.tk_configuration import RobotFrame
import pinocchio as pin
import hppfcl
import example_robot_data as robex
import numpy as np
from util_load_cassie import loadCassieAndFixIt
from constraints import constraintsResidual
from casadi_projection import ProjectConfig

cassie = loadCassieAndFixIt(initViewer=True)
cassie.full_constraint_models = cassie.constraint_models
cassie.full_constraint_datas = {cm: cm.createData()
                                for cm in cassie.constraint_models}
cassie.constraint_datas = [cassie.full_constraint_datas[cm]
                           for cm in cassie.constraint_models]

# ##############################################################################
# ### WINDOWS ##################################################################
# ##############################################################################


class ConstraintsManager:
    def __init__(self, robotConstraintFrame):
        self.robotConstraintFrame = robotConstraintFrame
        self.project = self.robotConstraintFrame.project

    def computeConstrainedConfig(self):
        qref = self.robotConstraintFrame.getConfiguration(False)
        q = self.project(qref)
        self.robotConstraintFrame.resetConfiguration(q)
        self.robotConstraintFrame.display()

    def resetAndDisp(self):
        self.robotConstraintFrame.resetConfiguration(cassie.q0)
        self.robotConstraintFrame.display()


class RobotConstraintFrame(RobotFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setProjector()

    def setProjector(self):
        # print('Set projector')
        self.project = ProjectConfig(cassie)

    def slider_display(self, i, v):
        # print('NEW SLIDE ... ',i,v)
        qref = cassieFrame.getConfiguration(False)
        q = self.project(qref, i)
        cassieFrame.resetConfiguration(q)
        cassieFrame.display()

    def setVerboseVariable(self, boolVar):
        self.boolVar = boolVar
        self.verboseProjector()

    def verboseProjector(self):
        self.project.verbose = self.boolVar.get()


root = tk.Tk()
root.bind('<Escape>', lambda ev: root.destroy())
root.title("Cassie")
cassieFrame = RobotConstraintFrame(cassie.model, cassie.q0, cassie,
                                   motors=[n for n in cassie.model.names if 'op' in n])
cassieFrame.createSlider(root)
cassieFrame.createRefreshButons(root)

constraintsManager = ConstraintsManager(cassieFrame)
optimFrame = tk.Frame(root)
optimFrame.pack(side=tk.BOTTOM)
reset_button = tk.Button(optimFrame, text="Reset",
                         command=constraintsManager.resetAndDisp)
reset_button.pack(side=tk.LEFT, padx=10, pady=10)
optim_button = tk.Button(optimFrame, text="Optim",
                         command=constraintsManager.computeConstrainedConfig)
optim_button.pack(side=tk.LEFT, padx=10, pady=10)
verbose_var = tk.BooleanVar(False)
cassieFrame.setVerboseVariable(verbose_var)
verbose_checkbox = tk.Checkbutton(optimFrame, variable=verbose_var, text='Verbose',
                                  command=cassieFrame.verboseProjector)
verbose_checkbox.pack(side=tk.LEFT, padx=10, pady=10)

constraintWindow = tk.Toplevel()
constraintWindow.bind('<Escape>', lambda ev: root.destroy())


class CheckboxConstraintCmd:
    def __init__(self, boolVar, cm, project):
        self.boolVar = boolVar
        self.cm = cm
        self.project = project

    def __call__(self):
        if self.boolVar.get():
            print(f'Activate {self.cm.name}')
            assert (self.cm not in cassie.constraint_models)
            cassie.constraint_models.append(self.cm)
            cassie.constraint_datas = [
                cassie.full_constraint_datas[cm] for cm in cassie.constraint_models]
        else:
            print(f'Deactivate {self.cm.name}')
            assert (self.cm in cassie.constraint_models)
            cassie.constraint_models.remove(self.cm)
        cassie.constraint_datas = [
            cassie.full_constraint_datas[cm] for cm in cassie.constraint_models]
        self.project.recomputeConstraints()


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


constraintFrame = tk.Frame(constraintWindow)
constraintFrame.pack(side=tk.BOTTOM)
actLabel = tk.Label(constraintFrame, text='active')
actLabel.grid(row=0, column=1)
dispLabel = tk.Label(constraintFrame, text='display')
dispLabel.grid(row=0, column=2)

for i, cm in enumerate(cassie.full_constraint_models):
    cstLabel = tk.Label(constraintFrame, text=cm.name)
    cstLabel.grid(row=i+1, column=0)

    active_constraint_var = tk.BooleanVar(value=cm in cassie.constraint_models)
    active_constraint_cmd = CheckboxConstraintCmd(
        active_constraint_var, cm, cassieFrame.project)
    constraint_checkbox = tk.Checkbutton(constraintFrame, variable=active_constraint_var,
                                         command=active_constraint_cmd)
    constraint_checkbox.grid(row=i+1, column=1)

    display_constraint_var = tk.BooleanVar(
        value=cm in cassie.constraint_models)
    display_constraint_cmd = CheckboxDisplayConstraintCmd(
        display_constraint_var, cm, cassie.visual_model, cassie.viz)
    display_constraint_cmd()
    display_constraint_checkbox = tk.Checkbutton(constraintFrame, variable=display_constraint_var,
                                                 command=display_constraint_cmd)
    display_constraint_checkbox.grid(row=i+1, column=2)


root.mainloop()

# ### DEBUG

model = cassie.model
data = cassie.data
cm = cassie.constraint_models[0]
cd = pin.RigidConstraintData(cm)
q = cassie.q0

pin.forwardKinematics(model, data, q)
pin.computeAllTerms(model, data, q, np.zeros(model.nv))
J = pin.getConstraintJacobian(model, data, cm, cd)
pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True, threshold=1e6)
