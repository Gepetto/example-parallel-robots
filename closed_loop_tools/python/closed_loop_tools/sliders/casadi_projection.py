"""
-*- coding: utf-8 -*-
Nicolas MANSARD & Ludovic DE MATTEIS - May 2023

Projector class to get the nearest feasible configuration to a given one.
Optimisation problem solved by casadi

"""

import pinocchio as pin
import numpy as np
import sys

sys.path.append("..")
sys.path.append("../closed_loop_utils")
from constraints import constraintsResidual
from pinocchio import casadi as caspin
import casadi


class ProjectConfig:
    """
    Define a projection function using NLP
    Given a reference configuration qref and an (optional) joint index iv (of Qdot=T_qQ),
    solve the following problem:

    search q
    which minimizes || q - qref ||**2
    subject to:
      c(q) = 0        # Kinematic constraint satisfied
      q_iv = qref_iv  # The commanded joint iv should move exactly
    """

    def __init__(self, model, constraint_models):
        self.casmodel = caspin.Model(model)
        self.casdata = self.casmodel.createData()

        cq = self.cq = casadi.SX.sym("q", self.casmodel.nq, 1)
        cv = self.cv = casadi.SX.sym("v", self.casmodel.nv, 1)

        caspin.forwardKinematics(self.casmodel, self.casdata, cq)

        self.integrate = casadi.Function(
            "integrate", [cq, cv], [caspin.integrate(self.casmodel, cq, cv)]
        )
        self.recomputeConstraints(constraint_models)
        self.verbose = True

    def recomputeConstraints(self, constraint_models):
        """
        Call this function when the constraint activation changes.
        This will force the recomputation of the computation graph of the constraint function.
        """
        self.cas_constraint_models = [
            caspin.RigidConstraintModel(cm) for cm in constraint_models
        ]
        self.cas_constraint_datas = [m.createData() for m in self.cas_constraint_models]

        constraint = constraintsResidual(
            self.casmodel,
            self.casdata,
            self.cas_constraint_models,
            self.cas_constraint_datas,
            self.cq,
            False,
            caspin,
        )
        self.constraint = casadi.Function("constraint", [self.cq], [constraint])

    def __call__(self, qref, iv=None):
        """
        Project an input configuration <qref> to the nearby feasible configuration
        If <iv> is not null, then the DOF #iv is set as hard constraints, while the other are moved.
        """
        self.opti = opti = casadi.Opti()

        # Decision variables
        self.vdq = vdq = opti.variable(self.casmodel.nv)
        self.vq = vq = self.integrate(qref, vdq)

        # Cost and constraints
        totalcost = casadi.sumsqr(vdq)
        if self.cas_constraint_models != []:
            opti.subject_to(self.constraint(vq) == 0)

        if iv is not None:
            opti.subject_to(vdq[iv] == 0)

        # Solve
        if self.verbose:
            opts = {}
        else:
            opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        opti.solver("ipopt", opts)  # set numerical backend
        opti.minimize(totalcost)
        try:
            sol = opti.solve_limited()
            self.qopt = qopt = opti.value(vq)
            self.dqopt = opti.value(vdq)
        except:
            print("ERROR in convergence, plotting debug info.")
            self.qopt = qopt = opti.debug.value(vq)
            self.dqopt = opti.debug.value(vdq)
        return qopt
