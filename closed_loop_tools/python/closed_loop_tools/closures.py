import pinocchio as pin
import numpy as np
from pinocchio import casadi as caspin
import casadi
from warnings import warn
from qpsolvers import solve_qp

from .constraints import constraintsResidual

## Configuration
def partialLoopClosure(model, data, constraint_models, constraint_datas, fixed_joints_ids, fixed_rotations=[], q_ref=None, q_ws=None):
    """
        # TODO Add docstring
    """
    if q_ref is None:
        q_ref = pin.neutral(model)
    if q_ws is not None:
        dq_ws = pin.difference(model, q_ref, q_ws)
    else:
        dq_ws = np.zeros(model.nv)
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

    constraints_res = casadi.Function('constraint', [sym_dq], [constraints(cq)])
    jac_cost = casadi.Function('jac_cost', [sym_dq, sym_cost], 
        [2*constraints_res(sym_dq).T@constraints_res.jacobian()(sym_dq, constraints_res(sym_dq))])
    cost = casadi.Function('cost', [sym_dq], [casadi.sumsqr(constraints_res(sym_dq))],
                        {"custom_jacobian": jac_cost, "jac_penalty":0})
    
    rotations = {ids: casadi.Function('rots_'+str(ids), [sym_dq], [caspin.exp3(sym_dq[ids:ids+3])[1,0]]) 
                      for ids in fixed_rotations} # ! Only fixes the rotation around Z axis

    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(model.nv)

    # * Problem
    total_cost = cost(vdq)
    optim.minimize(total_cost)
    for ids in fixed_rotations:
        optim.subject_to(rotations[ids](vdq)==0)
    for ids in fixed_joints_ids:
        optim.subject_to(vdq[ids]==0)
    optim.subject_to(optim.bounded(model.lowerPositionLimit, integrate(vdq), model.upperPositionLimit)) # Bounding the controls to acceptable levels
    
    optim.set_initial(vdq, dq_ws)
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
def partialLoopClosureFrames(model, data, constraint_models, constraint_datas, framesIds=[], fixed_rotations=[], q_ref=None, q_ws=None):
    if q_ref is None:
        q_ref = pin.neutral(model)
    if q_ws is not None:
        dq_ws = pin.difference(model, q_ref, q_ws)
    else:
        dq_ws = np.zeros(model.nv)
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
    sym_cost = casadi.SX.sym("dq", 1, 1)
    cq = caspin.integrate(casmodel, casadi.SX(q_ref), sym_dq)

    integrate = casadi.Function("integrate", [sym_dq], [cq])

    constraints_res = casadi.Function('constraint', [sym_dq], [constraints(cq)])
    jac_cost = casadi.Function('jac_cost', [sym_dq, sym_cost], 
        [2*constraints_res(sym_dq).T@constraints_res.jacobian()(sym_dq, constraints_res(sym_dq))])
    cost = casadi.Function('cost', [sym_dq], [casadi.sumsqr(constraints_res(sym_dq))],
                        {"custom_jacobian": jac_cost, "jac_penalty":0})
    caspin.framesForwardKinematics(casmodel, casdata, cq)
    frames_compare = {fId: casadi.Function('SE3_'+str(fId), [sym_dq], [caspin.log6(frames_SE3[fId].actInv(casdata.oMf[fId])).vector]) 
                      for fId in framesIds}
    
    rotations = {ids: casadi.Function('rots_'+str(ids), [sym_dq], [caspin.exp3(sym_dq[ids:ids+3])[1,0]]) 
                      for ids in fixed_rotations}
    
    # * Optimisation problem
    optim = casadi.Opti()
    vdq = optim.variable(model.nv)

    # * Problem
    total_cost = cost(vdq)
    optim.minimize(total_cost)
    for fId in framesIds:
        optim.subject_to(frames_compare[fId](vdq)==0)
    for ids in fixed_rotations:
        optim.subject_to(rotations[ids](vdq)==0)
    optim.subject_to(optim.bounded(model.lowerPositionLimit, integrate(vdq), model.upperPositionLimit)) # Bounding the controls to acceptable levels

    optim.set_initial(vdq, dq_ws)
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