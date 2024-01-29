import pinocchio as pin
import numpy as np
from pinocchio import casadi as caspin
import casadi
from warnings import warn
from qpsolvers import solve_qp

from .constraints import constraintsResidual

## Configuration
def partialLoopClosure(model, constraint_models, q_ref, fixed_joints_ids):
    """
        # TODO Add docstring
    """
    # * Defining casadi models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()

    casconstraint_models = [caspin.RigidConstraintModel(cm) for cm in constraint_models]
    casconstraint_datas = [cm.createData() for cm in casconstraint_models]

    # * Optimisation functions
    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, casconstraint_models, casconstraint_datas, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc

    sym_dq = casadi.SX.sym("dq", model.nv, 1)
    sym_cost = casadi.SX.sym("dq", 1, 1)

    cq = caspin.integrate(casmodel, casadi.SX(q_ref), sym_dq)
    integrate = casadi.Function("integrate", [sym_dq], [cq])
    constraintsRes = casadi.Function('constraint', [sym_dq], [constraints(cq)])

    jac_cost = casadi.Function('jac_cost', [sym_dq, sym_cost], 
        [2*constraintsRes(sym_dq).T@constraintsRes.jacobian()(sym_dq, constraintsRes(sym_dq))])
    cost = casadi.Function('cost', [sym_dq], [casadi.sumsqr(constraintsRes(sym_dq))],
                        {"custom_jacobian": jac_cost, "jac_penalty":0})

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(model.nv)

    # * Problem
    total_cost = cost(vdq)
    optim.minimize(total_cost)
    optim.subject_to(vdq[fixed_joints_ids]==0)
    optim.subject_to(optim.bounded(model.lowerPositionLimit, integrate(vdq), model.upperPositionLimit)) # Bounding the controls to acceptable levels
    opts = {}
    s_opts = {"print_level": 0, 
			"tol": 1e-6
            }
    optim.solver("ipopt", opts, s_opts)
    try:
        optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)
    except RuntimeError as e:
        print(e)
        print('ERROR in convergence, press enter to plot debug info.')
        input()
        dq = optim.debug.value(vdq)
        print(dq)
    print(optim.value(total_cost))
    if not optim.value(total_cost)<1e-4: 
        warn("Constraint not satisfied, make sure the problem is feasible")
    q = pin.integrate(model, q_ref, dq)
    return q

## Frames
def partialLoopClosureFrames(model, data, constraint_models, constraint_datas, framesIds=[], q_ref=None):
    if q_ref is None:
        q_ref = pin.neutral(model)
    # Define frames SE3 references
    # TODO add option to only consider translation or only rotation or complete SE3
    pin.framesForwardKinematics(model, data, q_ref)
    frames_SE3 = {fId: caspin.SE3(data.oMf[fId]) for fId in framesIds}
    # Models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()
    cmodels = [caspin.RigidConstraintModel(cm) for cm in constraint_models]
    cdatas = [cm.createData() for cm in cmodels]

    def constraints(q):
        Lc = constraintsResidual(casmodel, casdata, cmodels, cdatas, q, recompute=True, pinspace=caspin, quaternions=False)
        return Lc
    sym_dq = casadi.SX.sym("dq", casmodel.nv, 1)
    cq = caspin.integrate(casmodel, casadi.SX(q_ref), sym_dq)
    sym_cost = casadi.SX.sym("dq", 1, 1)

    constraintsRes = casadi.Function('constraint', [sym_dq], [constraints(cq)])

    jac_cost = casadi.Function('jac_cost', [sym_dq, sym_cost], 
        [2*constraintsRes(sym_dq).T@constraintsRes.jacobian()(sym_dq, constraintsRes(sym_dq))])
    cost = casadi.Function('cost', [sym_dq], [casadi.sumsqr(constraintsRes(sym_dq))],
                        {"custom_jacobian": jac_cost, "jac_penalty":0})
    caspin.framesForwardKinematics(casmodel, casdata, cq)
    frames_compare = {fId: casadi.Function('SE3_'+str(fId), [sym_dq], [caspin.log6(frames_SE3[fId].actInv(casdata.oMf[fId])).vector]) 
                      for fId in framesIds}
    
    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(model.nv)

    # * Problem
    total_cost = cost(vdq)
    optim.minimize(total_cost)
    for fId in framesIds:
        optim.subject_to(frames_compare[fId](vdq)==0)

    opts = {}
    s_opts = {"print_level": 0, 
				"tol": 1e-6
            }
    optim.solver("ipopt", opts, s_opts)
    try:
        optim.solve_limited()
        print("Solution found")
        dq = optim.value(vdq)
    except RuntimeError as e:
        print(e)
        print('ERROR in convergence, press enter to plot debug info.')
        input()
        dq = optim.debug.value(vdq)
        print(dq)
    print(optim.value(total_cost))
    if not optim.value(total_cost)<1e-4: 
        warn("Constraint not satisfied, make sure the problem is feasible")
    q = pin.integrate(model, q_ref, dq)
    return q # I always return a value even if convergence failed