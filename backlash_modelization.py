import pinocchio as pin
import numpy as np
from numpy.linalg import norm
from warnings import warn
from pinocchio.robot_wrapper import RobotWrapper
import os

def linearBacklash(model, q, vq, aq, r, Lf=[]):
    """
    Ldep=linearBacklash(model,q,vq,aq,r,Lf=[])

    take  robot model, data, position, velocity acceleration,, and exterior forces (set to 0 if empty) that apply on each joint (Local frame) and return the
    deplacement that would produce a linear backlash of r (Ldep a list of SE3 deplacement)
    """
    data = model.createData()
    f0 = pin.Force(np.array([0, 0, 0, 0, 0, 0]))
    if Lf == []:
        Lf = [f0 for i in range(model.nq + 1)]
    if len(Lf) < model.nq + 1:
        warn("invalid forces, set to 0")
        Lf = [f0 for i in range(model.nq + 1)]

    pin.forwardKinematics(model, data, q, vq)
    pin.rnea(model, data, q, vq, aq, Lf)
    Ldep = []
    for i in range(model.njoints - 1):
        j = i + 1  # Joint associated (exlusion of the universe joint)
        f = data.f[j]
        dl = -f.linear
        dl = dl / norm(dl) * r  # deplacement induce by the force f
        pj = model.parents[j]  # parent of the joint j
        oMj = data.oMi[j]  # current pos of the joint
        oMpj = data.oMi[pj]  # current pos of the parent joint
        jRpj = (oMpj.inverse() * oMj).rotation
        dlpj = jRpj @ dl
        jMpj = pin.SE3.Identity()
        jMpj.translation = dlpj
        Ldep.append(jMpj)
    return Ldep


def rotationalBacklash(model, data, q, vq, aq, tetamax, Lf=[]):
    """
    Lrot=rotationalBacklash(model,data,q,vq,aq,tetamax,Lf=[])

    take  robot model, data, position, velocity acceleration, and exterior forces (set to 0 if empty) that apply on each joint and return the
    deplacement that would produce a rotational backlash of max angle tetamax (Ldep a list of SE3 deplacement)

    """
    f0 = pin.Force(np.array([0, 0, 0, 0, 0, 0]))
    if Lf == []:
        Lf = [f0 for i in range(model.nq + 1)]

    if len(Lf) < model.nq + 1:
        warn("invalid forces, set to 0")
        Lf = [f0 for i in range(model.nq + 1)]
    pin.rnea(model, data, q, vq, aq, Lf)
    Lrot = []
    for i in range(model.njoints - 1):
        j = i + 1
        jointmodel = model.joints[j]
        f = data.f[j]
        dtau = -f.angular
        if jointmodel == pin.JointModelRX():
            dtau[0] = 0
        if jointmodel == pin.JointModelRY():
            dtau[1] = 0
        if jointmodel == pin.JointModelRZ():
            dtau[2] = 0
        if jointmodel == pin.JointModelSpherical():
            dtau = dtau*0
        ndetau = norm(dtau)
        dtau = dtau / ndetau
        teta = tetamax * np.arctan(1000 * ndetau) * 2 / np.pi
        ny = np.cross(np.array([0, 0, 1]), dtau)
        nz = np.array([0, 0, 1]) * np.cos(teta) + ny * np.sin(teta)
        R = pin.Quaternion(np.array([0, 0, 1]), nz).matrix()
        jMpj = pin.SE3.Identity()
        jMpj.rotation = R
        Lrot.append(jMpj)

    return Lrot


def linearFlexBacklash(model, q, vq, aq, I, Lf=[]):
    """
    Ldep=linearFlexBacklash(model,q,vq,aq,I,Lf=[])

    take  robot model, data, position, velocity acceleration,, and exterior forces (set to 0 if empty) that apply on each joint (Local frame) and return the
    deplacement that would produce a linear flexibility ( I as the deplacement is norm(f)/I ) (Ldep a list of SE3 deplacement)
    """
    data = model.createData()
    f0 = pin.Force(np.array([0, 0, 0, 0, 0, 0]))
    if Lf == []:
        Lf = [f0 for i in range(model.nq + 1)]
    if len(Lf) < model.nq + 1:
        warn("invalid forces, set to 0")
        Lf = [f0 for i in range(model.nq + 1)]

    pin.forwardKinematics(model, data, q, vq)
    pin.rnea(model, data, q, vq, aq, Lf)
    Ldep = []
    for i in range(model.njoints - 1):
        j = i + 1  # Joint associated (exlusion of the universe joint)

        f = data.f[j]
        dl = -f.linear
        r = norm(f) / I
        dl = dl / norm(dl) * r  # deplacement induce by the force f
        pj = model.parents[j]  # parent of the joint j
        oMj = data.oMi[j]  # current pos of the joint
        oMpj = data.oMi[pj]  # current pos of the parent joint
        jRpj = (oMpj.inverse() * oMj).rotation
        dlpj = jRpj @ dl
        jMpj = pin.SE3.Identity()
        jMpj.translation = dlpj
        Ldep.append(jMpj)
    return Ldep


def rotationalFlexBacklash(model, data, q, vq, aq, I, Lf=[]):
    """
    Lrot=rotationalFlexBacklash(model,data,q,vq,aq,I,Lf=[])

    take  robot model, data, position, velocity acceleration, and exterior forces (set to 0 if empty) that apply on each joint and return the
    deplacement that would produce a rotational backlash of max angle norm(torque)/I (Ldep a list of SE3 deplacement)

    """
    f0 = pin.Force(np.array([0, 0, 0, 0, 0, 0]))
    if Lf == []:
        Lf = [f0 for i in range(model.nq + 1)]

    if len(Lf) < model.nq + 1:
        warn("invalid forces, set to 0")
        Lf = [f0 for i in range(model.nq + 1)]
    pin.rnea(model, data, q, vq, aq, Lf)
    Lrot = []
    for i in range(model.njoints - 1):
        j = i + 1
        jointmodel = model.joints[j]
        f = data.f[j]
        dtau = -f.angular
        ndetau = norm(dtau)
        dtau = dtau / ndetau
        tetamax = ndetau / I
        teta = tetamax * np.arctan(1000 * ndetau) * 2 / np.pi
        ny = np.cross(np.array([0, 0, 1]), dtau)
        nz = np.array([0, 0, 1]) * np.cos(teta) + ny * np.sin(teta)
        R = pin.Quaternion(np.array([0, 0, 1]), nz).matrix()
        jMpj = pin.SE3.Identity()
        jMpj.rotation = R
        Lrot.append(jMpj)

    return Lrot

##########TEST ZONE ##########################
#No test yet

import unittest

# class TestRobotInfo(unittest.TestCase):
    #only test inverse constraint kineatics because it runs all precedent code
    # def test_backlash(self):


if __name__ == "__main__":
    path=os.getcwd()+"/robot_marcheur_1"
    robot=RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    model=robot.model

    

    unittest.main()