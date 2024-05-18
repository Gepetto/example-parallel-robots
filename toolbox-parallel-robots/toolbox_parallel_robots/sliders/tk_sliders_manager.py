import tkinter as tk


class SlidersManager:
    def __init__(self, slidersFrame, constraint_models):
        self.slidersFrames = slidersFrame  # Tkinter sliders
        self.constraint_models = constraint_models

        # Vars
        self.refresh_var = tk.BooleanVar()

    def computeConstrainedConfig(self):
        qref = self.robotConstraintFrame.getConfiguration(
            False
        )  # Get the sliders configuration
        q = self.project(qref)  # Project to get the nearest feasible configuration
        self.robotConstraintFrame.resetConfiguration(q)
        self.robotConstraintFrame.display()

    def set_auto_refresh(self):
        self.slidersFrames.auto_refresh = self.refresh_var.get()

    def reset(self):  # Self-explanatory
        self.slidersFrames.reset()

    def refresh(self):
        self.slidersFrames.display()

    def optimize(self):
        pass

    def createButtons(self, tkParent):
        button_frame = tk.Frame(tkParent)
        button_frame.pack()

        button1 = tk.Button(button_frame, text="Reset", command=self.reset)
        button1.pack(side=tk.LEFT)

        button2 = tk.Button(button_frame, text="Refresh", command=self.refresh)
        button2.pack(side=tk.LEFT)

        check_auto_refresh = tk.Checkbutton(
            button_frame,
            text="Auto Refresh",
            command=self.set_auto_refresh,
            var=self.refresh_var,
        )
        check_auto_refresh.select()
        check_auto_refresh.pack(side=tk.LEFT)

        button_frame2 = tk.Frame(tkParent)
        button_frame2.pack()

        button4 = tk.Button(button_frame2, text="Optimize", command=self.optimize)
        button4.pack(side=tk.LEFT)

        checkbox = tk.Checkbutton(button_frame2, text="Auto Optimize")
        checkbox.pack(side=tk.LEFT)
