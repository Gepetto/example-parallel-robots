import pinocchio as pin
from pinocchio import casadi as caspin
import casadi
import os
from loader_tools import completeRobotLoader
from closed_loop_kinematics import closedLoopForwardKinematicsScipy, closedLoopForwardKinematicsCasadi, closedLoopForwardKinematicsProximal, closedLoopInverseKinematicsCasadi, closedLoopInverseKinematicsScipy, closedLoopInverseKinematicsProximal
import numpy as np
from constraints import constraintsResidual
from itertools import combinations

if __name__ == "__main__":
    # * Load robot
    path = os.getcwd()+"/robots/robot_marcheur_4"
    model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(path)
    data = model.createData()
    constraint_datas = [cm.createData() for cm in constraint_models]
    q0 = pin.neutral(model)

    ## * ForwardDynamics
    vq_init = np.array([ 8.88717274e-03, -5.93055557e-05,  4.44320529e-03,  4.44858470e-03,
        9.99980232e-01,  2.11933390e-03, -3.33526016e-02,  1.61191975e-01,
       -2.88319667e-01,  3.97998778e-03,  3.79653087e-02,  1.82083208e-01,
       -7.49722142e-03, -6.46804581e-02,  1.06943793e-01, -3.67527502e-01,
        1.27987957e-02, -8.16774201e-02,  5.03071517e-02,  9.95306074e-01,
       -5.64782557e-02,  1.85887950e-02, -2.72432975e-02,  9.97858941e-01,
        3.74060475e-02,  2.40727380e-02,  4.48401661e-02, -9.33949663e-03,
        9.98660421e-01,  3.34147548e-02,  5.16785740e-01,  3.26563321e-02,
       -3.38244516e-01, -1.09563422e-02, -2.36053133e-01,  4.51123194e-02,
        9.70630597e-01,  1.29868250e-02, -2.37677555e-01,  1.61314899e-02,
        9.71123317e-01, -1.95964216e-01,  1.99730480e-02, -1.68177020e-02,
        9.80263367e-01,  2.90546779e-02,  5.32031880e-01,  4.64951201e-02,
        8.44947400e-01, -3.57735409e-01, -1.97816577e-01,  9.38914113e-03,
       -6.19764204e-02,  9.78232779e-01, -4.36320296e-02,  1.87051027e-01,
       -1.71174138e-01,  9.66337195e-01,  3.17384814e-01,  6.28997336e-02,
        2.10109256e-01, -7.10882731e-02,  9.73059187e-01,  6.04179157e-02,
        1.97113023e-01, -3.40346333e-02,  9.77925240e-01,  2.35214784e-02,
        6.59093439e-03, -2.02019318e-01,  1.64246625e-02,  9.79221622e-01,
        1.99777529e-01,  4.65505470e-02, -3.40311062e-02,  9.78143072e-01,
       -3.29434826e-02,  4.38669513e-02, -2.12832785e-01,  4.54216802e-02,
        9.75046033e-01,  1.03903828e-01,  1.57525775e-01, -1.71269681e-01,
        9.66983103e-01, -5.80113033e-03, -3.09132603e-02, -1.53831997e-01,
        9.87596291e-01,  4.66195631e-03, -3.41688597e-02, -2.51633240e-01,
        9.67208079e-01,  8.60149609e-03,  1.27457607e-01,  1.68809573e-02,
        9.91663051e-01, -2.41802184e-02, -7.34121290e-05, -1.65006311e-02,
        4.44839745e-03,  9.99853957e-01])
    
    # * Defining casadi models
    casmodel = caspin.Model(model)
    casdata = casmodel.createData()

    # * Getting ids of actuated and free joints
    Lid = actuation_model.idqmot
    q_prec = pin.neutral(model)

    cd = constraint_datas[3]
    cm = constraint_models[3]

    # * Optimisation functions
    def constraints(q):
        caspin.forwardKinematics(casmodel, casdata, q)
        oMc1 = casdata.oMi[cm.joint1_id]*caspin.SE3(cm.joint1_placement)
        oMc2 = casdata.oMi[cm.joint2_id]*caspin.SE3(cm.joint2_placement)
        return caspin.log6(oMc1.inverse()*oMc2).vector

    def approx_6d_constraint(cq):
        oMc1 = casdata.oMi[cm.joint1_id]*caspin.SE3(cm.joint1_placement)
        oMc2 = casdata.oMi[cm.joint2_id]*caspin.SE3(cm.joint2_placement)
        tran_error = oMc1.translation - oMc2.translation
        rot_error = casadi.diag(oMc1.rotation.T @ oMc2.rotation) - casadi.SX.ones(3)
        return(casadi.vertcat(tran_error, rot_error))
    
    cq = casadi.SX.sym("q", model.nq, 1)
    constraintsCost = casadi.Function('constraint', [cq], [constraints(cq)])

    print("Computing the Hessians")
    Gradient = [casadi.Function('gradient'+str(i), [cq], [casadi.gradient(constraints(cq)[i], cq)]) for i in range(6)]
    Jacobian = casadi.Function('Jacobian', [cq], [casadi.jacobian(constraints(cq), cq)])
    Hessian = [casadi.Function('hessian'+str(i), [cq], [casadi.hessian(constraints(cq)[i], cq)[0]]) for i in range(6)]

    print(Gradient[0](vq_init))
    print("Done")