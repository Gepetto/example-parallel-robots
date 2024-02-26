"""
-*- coding: utf-8 -*-
Nicolas MANSARD - Virgile BATTO & Ludovic DE MATTEIS - February 2024

Tool functions to compute the constraints residuals from a robot constraint model.
"""

import pinocchio as pin
import numpy as np
import casadi


## Constraints residuals
def constraintResidual6d(model, data, cmodel, cdata, q, recompute=True, pinspace=pin):
    """
    Compute the residual of a 6D contact constraint.

    Args:
        model (pin.Model): The robot model.
        data (pin.Data): The robot data.
        cmodel (pin.ContactModel6D): The 6D contact model.
        cdata (pin.ContactData6D): The 6D contact data.
        q (numpy.ndarray): The robot configuration.
        recompute (bool, optional): Whether to recompute the forward kinematics. Defaults to True.
        pinspace (pin): The pinocchio module. Defaults to pin.

    Returns:
        numpy.ndarray: The residual of the 6D contact constraint.
    """
    assert cmodel.type == pin.ContactType.CONTACT_6D
    if recompute:
        pinspace.forwardKinematics(model, data, q)
    oMc1 = data.oMi[cmodel.joint1_id] * pinspace.SE3(cmodel.joint1_placement)
    oMc2 = data.oMi[cmodel.joint2_id] * pinspace.SE3(cmodel.joint2_placement)
    return pinspace.log6(oMc1.inverse() * oMc2).vector


def constraintResidual3d(
    model, data, cmodel, cdata, q=None, recompute=True, pinspace=pin
):
    """
    Compute the residual of a 3D constraint.

    Args:
        model: The robot model.
        data: The robot data.
        cmodel: The constraint model.
        cdata: The constraint data.
        q: The robot configuration (optional).
        recompute: Whether to recompute the forward kinematics (default: True).
        pinspace: The pinocchio module (default: pin).

    Returns:
        The residual of the 3D constraint.
    """
    assert cmodel.type == pin.ContactType.CONTACT_3D
    if recompute:
        pinspace.forwardKinematics(model, data, q)
    oMc1 = data.oMi[cmodel.joint1_id] * pinspace.SE3(cmodel.joint1_placement)
    oMc2 = data.oMi[cmodel.joint2_id] * pinspace.SE3(cmodel.joint2_placement)
    return oMc1.translation - oMc2.translation


def constraintResidual(
    model, data, cmodel, cdata, q=None, recompute=True, pinspace=pin
):
    """
    Compute the residual of a constraint. Calls either constraintResidual6d or constraintResidual3d.

    Args:
        model: The robot model.
        data: The robot data.
        cmodel: The constraint model.
        cdata: The constraint data.
        q: The configuration vector (optional).
        recompute: Whether to recompute the constraint data (optional).
        pinspace: The pinocchio namespace (optional).

    Returns:
        The residual of the constraint.

    Raises:
        NotImplementedError: If the constraint type is not implemented.
    """
    if cmodel.type == pin.ContactType.CONTACT_6D:
        return constraintResidual6d(model, data, cmodel, cdata, q, recompute, pinspace)
    elif cmodel.type == pin.ContactType.CONTACT_3D:
        return constraintResidual3d(model, data, cmodel, cdata, q, recompute, pinspace)
    else:
        raise NotImplementedError("Only 6D and 3D constraints are implemented for now")


def constraintsResidual(
    model,
    data,
    cmodels,
    cdatas,
    q=None,
    recompute=True,
    pinspace=pin,
    quaternions=False,
):
    """
    Compute the residual of the constraints for a given model and data.

    Args:
        model: The model object.
        data: The data object.
        cmodels: List of constraint models.
        cdatas: List of constraint datas.
        q: The joint positions (optional).
        recompute: Flag indicating whether to recompute the constraint models and datas (default: True).
        pinspace: The pin space (default: pin).
        quaternions: Flag indicating whether to use quaternions (default: False).

    Returns:
        The concatenated residual of the constraints.
    """
    res = []
    for cm, cd in zip(cmodels, cdatas):
        res.append(constraintResidual(model, data, cm, cd, q, recompute, pinspace))
    if pinspace is pin:
        return np.concatenate(res)
    else:
        return casadi.vertcat(*res)
