import tkinter as tk
import numpy as np
import pinocchio as pin
from functools import partial


class SlidersFrame:
    """
    Create a tk.Frame and add sliders corresponding to robot joints.
    Return the tk.Frame, that must be added to the container.
    """

    NROW = 5  # Number of sliders per row

    def __init__(self, model, mot_ids_q, q0, viz):
        """
        motors is a list of joint names that must be highlighted in blue
        """
        self.rmodel = model
        self.mot_ids_q = mot_ids_q
        self.viz = viz
        self.q0 = q0.copy()
        self.slider_vars = []

        self.auto_refresh = True
        self.active_constraint_models = []

    def reset(self):
        self.setConfiguration(self.q0)
        self.display()

    def setConfiguration(self, qref):
        dq_ref = pin.difference(self.rmodel, self.q0, qref)
        for i, s in enumerate(self.slider_vars):
            s.set(dq_ref[i])

    def getConfiguration(self):
        values = [var.get() for var in self.slider_vars]
        dq = np.array(values)
        q = pin.integrate(self.rmodel, self.q0, dq)
        return q

    def display(self):
        q = self.getConfiguration()
        self.viz.display(q)

    def on_slider_move(self, iq, e):
        if self.auto_refresh:
            self.display()

    def createSlider(self, tkParent):
        # Frame pour les sliders
        frame = tk.Frame(tkParent)

        # Création des sliders verticaux
        iq = 0
        for j, name in enumerate(self.rmodel.names):
            if j == 0:
                continue
            for iv in range(self.rmodel.joints[j].nv):
                var = tk.DoubleVar(value=0)
                self.slider_vars.append(var)
                slider_frame = tk.Frame(
                    frame,
                    highlightbackground=(
                        "blue"
                        if self.rmodel.joints[j].idx_q in self.mot_ids_q
                        else "black"
                    ),
                    highlightthickness=1,
                )
                row = iq // self.NROW
                slider_frame.grid(
                    row=row * 1, column=iq - self.NROW * row, padx=0, pady=0
                )
                name_i = name if self.rmodel.joints[j].nv == 1 else name + f"{iv}"
                slider_label = tk.Label(slider_frame, text=name_i)
                slider_label.pack(side=tk.BOTTOM)
                slider = tk.Scale(
                    slider_frame,
                    variable=var,
                    orient=tk.HORIZONTAL,
                    from_=-3.0,
                    to=3.0,
                    resolution=0.01,
                    command=partial(self.on_slider_move, iq),
                )  # When sliders are moved, call this function
                slider.pack(side=tk.BOTTOM)
                iq += 1
            frame.pack(side=tk.TOP)
        return frame
