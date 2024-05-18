"""
-*- coding: utf-8 -*-
Ludovic DE MATTEIS - May 2023

Create a Tkinter interface to move some joints in the robots while satisfying the desired closed loop constraints

"""

import meshcat
import tkinter as tk
import pinocchio as pin
from sliders.tk_robot_sliders import SlidersFrame
from sliders.tk_sliders_manager import SlidersManager


# * Interface to activate or deactivate constraints on the robot
class CheckboxConstraintCmd:
    def __init__(self, boolVar, cm, project):
        self.boolVar = boolVar
        self.cm = cm
        self.project = project

    def __call__(self):
        if self.boolVar.get():
            print(f"Activate {self.cm.name}")
            assert self.cm not in constraint_models
            constraint_models.append(self.cm)
        else:
            print(f"Deactivate {self.cm.name}")
            assert self.cm in constraint_models
            constraint_models.remove(self.cm)
        self.project.recomputeConstraints(constraint_models)


def createSlidersInterface(
    model, constraint_models, visual_model, mot_ids_q, viz, q0=None
):
    """
    Create a Tkinter interface to move some joints in the robots while satisfying the desired closed loop constraints
    """
    if q0 is None:
        q0 = pin.neutral(model)

    # # * Adding constraints
    # addXYZAxisToConstraints(model, visual_model, constraint_models, scale=scale)

    # # * Add frames to all joints
    # addXYZAxisToJoints(model, visual_model, scale=scale)

    # * Create data
    data = model.createData()
    constraint_datas = [cm.createData() for cm in constraint_models]
    # * Set a scale factor to handle too small and too big robots
    scale = 1

    # replaceGeomByXYZAxis(visual_model, viz, scale=scale)
    # viz.display(q0)

    # * Create the interface
    root = tk.Tk()
    root.bind("<Escape>", lambda ev: root.destroy())
    root.title("Simple Robot Sliders")
    sliders_frame = SlidersFrame(model, mot_ids_q, q0, viz)
    # Creating sliders, main projection functions are called when the sliders are moved
    sliders_frame.createSlider(root)

    managerWindow = tk.Toplevel()
    managerWindow.bind("<Escape>", lambda ev: root.destroy())

    sliders_manager = SlidersManager(sliders_frame, constraint_models)
    sliders_manager.createButtons(managerWindow)

    root.mainloop()


if __name__ == "__main__":
    import example_parallel_robots as epr

    model, constraint_models, actuation_model, visual_model, collision_model = epr.load(
        "digit_2legs"
    )
    mot_ids_q = actuation_model.mot_ids_q
    # import example_robot_data as erd
    # robot = erd.load("solo12")
    # model, visual_model, collision_model = robot.model, robot.visual_model, robot.collision_model
    # constraint_models = []
    # mot_ids_q = [model.getJointId(joint_name) for joint_name in ["FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE", "HL_HAA", "HL_HFE", "HL_KFE", "HR_HAA", "HR_HFE", "HR_KFE"]]
    # * Create the visualizer
    import pinocchio as pin
    import meshcat

    viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
    q0 = pin.neutral(model)
    viz.display(q0)
    createSlidersInterface(model, constraint_models, visual_model, mot_ids_q, viz)
