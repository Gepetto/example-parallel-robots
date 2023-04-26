"""
-*- coding: utf-8 -*-
Nicolas MANSARD & Ludovic DE MATTEIS & Virgile BATTO, April 2023

Tool functions to compute the constraints residuals from a robot constraint model. Also includes quaternion normalization constraint

"""

import pinocchio as pin
import numpy as np
from pinocchio import casadi as caspin
import casadi


def constraintResidual6d(model, data, cmodel, cdata, q=None, recompute=True, pinspace=pin):
    assert (cmodel.type == pin.ContactType.CONTACT_6D)
    if recompute:
        pinspace.forwardKinematics(model, data, q)
    oMc1 = data.oMi[cmodel.joint1_id]*pinspace.SE3(cmodel.joint1_placement)
    oMc2 = data.oMi[cmodel.joint2_id]*pinspace.SE3(cmodel.joint2_placement)
    return pinspace.log6(oMc1.inverse()*oMc2).vector


def constraintResidual3d(model, data, cmodel, cdata, q=None, recompute=True, pinspace=pin):
    assert (cmodel.type == pin.ContactType.CONTACT_3D)
    if recompute:
        pinspace.forwardKinematics(model, data, q)
    oMc1 = data.oMi[cmodel.joint1_id]*pinspace.SE3(cmodel.joint1_placement)
    oMc2 = data.oMi[cmodel.joint2_id]*pinspace.SE3(cmodel.joint2_placement)
    return oMc1.translation-oMc2.translation


def constraintResidual(model, data, cmodel, cdata, q=None, recompute=True, pinspace=pin):
    if cmodel.type == pin.ContactType.CONTACT_6D:
        return constraintResidual6d(model, data, cmodel, cdata, q, recompute, pinspace)
    elif cmodel.type == pin.ContactType.CONTACT_3D:
        return constraintResidual3d(model, data, cmodel, cdata, q, recompute, pinspace)
    else:
        raise(NotImplementedError("Only 6D and 3D constraints are implemented for now"))

# TODO clean quaternion constraints
def constraintQuaternion(model, q, pinspace=pin):
    """
    L=constraintQuaternion(model, q)
    Returns the list of the squared norm of each quaternion inside the configuration vector q (work for free flyer and spherical joint)

    Arguments:
        model - Pinocchio robot model
        q - Joints configuration vector
    Return:
        L - List of quaternions norms
    """
    L = []
    for j in model.joints:
        idx_q = j.idx_q
        nq = j.nq
        nv = j.nv
        if nq != nv:
            quat = q[idx_q : idx_q + 4]
            if pinspace is caspin:
                L.append(casadi.norm_2(quat) ** 2 - 1)
            else:
                L.append(np.linalg.norm(quat)**2 - 1)
    return L

# TODO add constraintsPlanar

def constraintsResidual(model, data, cmodels, cdatas, q=None, recompute=True, pinspace=pin, quaternions=False):
    res = []
    for cm, cd in zip(cmodels, cdatas):
        res.append(constraintResidual(
            model, data, cm, cd, q, recompute, pinspace))
    if pinspace is pin:
        if quaternions:
            res.append(constraintQuaternion(model, q, pinspace))
        return np.concatenate(res)
    elif pinspace is caspin:
        if quaternions:
            res += constraintQuaternion(model, q, pinspace)
        return casadi.vertcat(*res)
    else:
        assert (False and "Should never happen")
