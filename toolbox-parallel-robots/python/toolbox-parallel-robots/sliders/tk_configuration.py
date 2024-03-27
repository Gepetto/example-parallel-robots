"""
-*- coding: utf-8 -*-
Nicolas MANSARD & Ludovic DE MATTEIS - May 2023

Classes for Tkinter integration 

"""

import tkinter as tk
import numpy as np
import pinocchio as pin
from functools import partial


# * Class used to create sliders corresponding to each joints
class RobotFrame:
    """
    Create a tk.Frame and add sliders corresponding to robot joints.
    Return the tk.Frame, that must be added to the container.
    """

    NROW = 14  # Number of sliders per row

    def __init__(self, model, constraint_model, mot_ids_q, q0, viz):
        """
        motors is a list of joint names that must be highlighted in blue
        """
        self.rmodel = model
        self.constraint_model = constraint_model
        self.mot_ids_q = mot_ids_q
        self.viz = viz
        self.auto_refresh = True
        self.q0 = q0.copy()
        self.slider_vars = []

    def resetConfiguration(self, qref=None):
        if qref is None:
            qref = self.q0
        dq_ref = pin.difference(self.rmodel, self.q0, qref)
        for i, s in enumerate(self.slider_vars):
            s.set(0 if qref is None else dq_ref[i])

    def getConfiguration(self, verbose=False):
        values = [var.get() for var in self.slider_vars]
        if verbose:
            print(values)
        dq = np.array(values)
        q = pin.integrate(self.rmodel, self.q0, dq)
        return q

    # Fonction pour afficher les valeurs des sliders
    def slider_display(self, i, v):  # Overwritten by child
        if self.auto_refresh:
            self.display()

    def display(self):
        q = self.getConfiguration()
        self.viz.display(q)

    def createSlider(self, tkParent, pack=True):
        # Frame pour les sliders
        frame = self.slidersFrame = tk.Frame(tkParent)

        # Création des sliders verticaux
        iq = 0
        for j, name in enumerate(self.rmodel.names):
            if j == 0:
                continue
            for iv in range(self.rmodel.joints[j].nv):
                var = tk.DoubleVar(value=0)
                self.slider_vars.append(var)
                slider_frame = tk.Frame(
                    self.slidersFrame,
                    highlightbackground="blue",
                    highlightthickness=(
                        2
                        if self.rmodel.joints[j].idx_q in self.mot_ids_q
                        else 0
                    ),
                )
                row = iq // self.NROW
                slider_frame.grid(
                    row=row * 2, column=iq - self.NROW * row, padx=2, pady=2
                )
                name_i = name if self.rmodel.joints[j].nv == 1 else name + f"{iv}"
                slider_label = tk.Label(slider_frame, text=name_i)
                slider_label.pack(side=tk.BOTTOM)
                slider = tk.Scale(
                    slider_frame,
                    variable=var,
                    orient=tk.VERTICAL,
                    from_=-3.0,
                    to=3.0,
                    resolution=0.01,
                    command=partial(self.slider_display, iq),
                )  # When sliders are moved, call this function
                slider.pack(side=tk.BOTTOM)
                iq += 1

            class VisibilityChanger:
                def __init__(self, viz, name, var):
                    # self.gv = viz.viewer.gui
                    self.name = name
                    self.var = var
                    self()

                def __call__(self):
                    gname = f"world/pinocchio/visuals/XYZ_{self.name}"
                    # self.gv.setVisibility(gname,'ON'  if self.var.get() else 'OFF')
                    print(
                        gname,
                        "ON" if self.var.get() else "OFF",
                        self.var,
                        self.var.get(),
                    )

            XYZon = tk.BooleanVar(value=False)
            tk.Checkbutton(
                slider_frame,
                text="",
                variable=XYZon,
                command=VisibilityChanger(self.viz, name, XYZon),
            ).pack(side=tk.RIGHT)

        if pack:
            frame.pack(side=tk.TOP)
        return frame

    def setRefresh(self, v=None):
        # print(' set ' ,v)
        if v is None:
            self.auto_refresh = not self.auto_refresh
        else:
            self.auto_refresh = v

    def createRefreshButons(self, tkParent, pack=True):
        # Frame pour le bouton d'affichage et la checkbox
        self.buttonsFrame = tk.Frame(tkParent)
        if pack:
            self.buttonsFrame.pack(side=tk.BOTTOM)

        # Bouton pour afficher les valeurs des sliders manuellement
        manual_button = tk.Button(
            self.buttonsFrame, text="Display", command=self.display
        )
        manual_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Checkbox pour activer/désactiver l'auto-refresh
        self.auto_refresh_var = tk.BooleanVar(value=self.auto_refresh)
        auto_refresh_checkbox = tk.Checkbutton(
            self.buttonsFrame,
            text="Auto" + self.rmodel.name,
            variable=self.auto_refresh_var,
            command=self.checkboxCmd,
        )
        auto_refresh_checkbox.pack(side=tk.LEFT, padx=10, pady=10)

    def checkboxCmd(self):
        # print(' autt' )
        self.setRefresh(self.auto_refresh_var.get())


# * Class used for the OCP target control only
class OptimizerFrame:
    """
    Create a tk.Frame and add checkboxes to display the frames.
    Return the tk.Frame, that must be added to the container.
    This class is build to only manage the optimization problem parameters from Tkinter (ie creating and setting the sliders but not creating or moving the target, running the optimisation...)
    """

    def __init__(self, target_pos=[0, 0, 0, 1, 0, 0, 0]):
        self.sliders_vars = []
        self.target_pos = target_pos

    def createTargetSliders(self, tkParent, pack=True, scale=1):
        """
        Creates 3 sliders for directions x, y and z to move the target
        """
        frame = self.slidersFrame = tk.Frame(tkParent)
        for i, direction in enumerate(["x", "y", "z"]):
            var = tk.DoubleVar(value=0)
            self.sliders_vars.append(var)
            slider_frame = tk.Frame(self.slidersFrame)

            slider_frame.grid(row=0, column=i, padx=5, pady=5)
            slider_label = tk.Label(slider_frame, text=direction)
            slider_label.pack(side=tk.BOTTOM)
            slider = tk.Scale(
                slider_frame,
                variable=var,
                orient=tk.VERTICAL,
                from_=-1.0,
                to=1.0,
                resolution=0.02 * scale,
                command=self.slider_update,
            )
            slider.pack(side=tk.BOTTOM)
        if pack:
            frame.pack(side=tk.TOP)
        return frame

    def slider_update(self, i):
        self.target_pos = [var.get() for var in self.sliders_vars] + [1, 0, 0, 0]
        self.updateTargetPosition()  # ! child class must implement a function updateTargetPosition

    def reset(self):
        self.sliders_vars = [0 for s in self.sliders_vars]
