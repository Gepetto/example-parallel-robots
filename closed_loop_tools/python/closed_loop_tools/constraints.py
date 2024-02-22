"""
-*- coding: utf-8 -*-
Nicolas MANSARD & Ludovic DE MATTEIS & Virgile BATTO, April 2023

Tool functions to compute the constraints residuals from a robot constraint model. Also includes quaternion normalization constraint

"""

import pinocchio as pin
import numpy as np
import casadi

## Constraints residuals
def constraintResidual6d(model, data, cmodel, cdata, q, recompute=True, pinspace=pin):
    """
    Compute the residual of a 6D constraint between two joints in the robot model.

    This function calculates the residual of a 6D constraint between two joints in the robot model.
    The residual represents the difference between the desired and actual relative transformation
    between the specified joints.

    Arguments:
        model : pinocchio.Model
            The Pinocchio robot model.
        data : pinocchio.Data
            The Pinocchio data associated with the model.
        cmodel : pinocchio.ConstraintModel
            The Pinocchio constraint model representing the 6D constraint.
        cdata : pinocchio.ConstraintData
            The Pinocchio data associated with the constraint model.
        q : numpy.ndarray
            The configuration vector (joint positions) of the robot.
        recompute : bool, optional
            Flag indicating whether to recompute the kinematics. Defaults to True.
        pinspace : module, optional
            The namespace to use for Pinocchio functions. Defaults to pin.

    Returns:
        numpy.ndarray
            The 6D residual vector representing the difference between the desired and actual
            relative transformation between the specified joints.
    """

    assert (cmodel.type == pin.ContactType.CONTACT_6D)
    if recompute:
        pinspace.forwardKinematics(model, data, q)
    oMc1 = data.oMi[cmodel.joint1_id]*pinspace.SE3(cmodel.joint1_placement)
    oMc2 = data.oMi[cmodel.joint2_id]*pinspace.SE3(cmodel.joint2_placement)
    return(pinspace.log6(oMc1.inverse() * oMc2).vector)


def constraintResidual3d(model, data, cmodel, cdata, q=None, recompute=True, pinspace=pin):

    """
    Computes the residual of a 3D contact constraint between two points in space.

    Args:
        model (pinocchio.Model): Pinocchio robot model.
        data (pinocchio.Data): Pinocchio data associated with the model.
        cmodel: Model of the 3D contact constraint.
        cdata: Data associated with the 3D contact constraint.
        q (numpy.ndarray, optional): Configuration vector. Defaults to None.
        recompute (bool, optional): Flag to recompute forward kinematics. Defaults to True.
        pinspace (module, optional): Pinocchio or CasADi module. Defaults to pin.

    Returns:
        numpy.ndarray: Residual of the 3D contact constraint between two points in space.

    Description:
        This function computes the residual of a 3D contact constraint between two points in space.
        It considers the difference between the translation components of the two specified joints,
        transformed by their respective placements within the robot model.
    """

    assert (cmodel.type == pin.ContactType.CONTACT_3D)
    if recompute:
        pinspace.forwardKinematics(model, data, q)
    oMc1 = data.oMi[cmodel.joint1_id]*pinspace.SE3(cmodel.joint1_placement)
    oMc2 = data.oMi[cmodel.joint2_id]*pinspace.SE3(cmodel.joint2_placement)
    return oMc1.translation-oMc2.translation


def constraintResidual(model, data, cmodel, cdata, q=None, recompute=True, pinspace=pin):
    """
    Computes the residual of a given constraint model.

    Args:
        model (pinocchio.Model): Pinocchio robot model.
        data (pinocchio.Data): Pinocchio data associated with the model.
        cmodel: Model of the constraint.
        cdata: Data associated with the constraint.
        q (numpy.ndarray, optional): Configuration vector. Defaults to None.
        recompute (bool, optional): Flag to recompute forward kinematics. Defaults to True.
        pinspace (module, optional): Pinocchio or CasADi module. Defaults to pin.

    Returns:
        numpy.ndarray: Residual of the constraint.

    Raises:
        NotImplementedError: If the constraint type is not implemented.

    Description:
        This function computes the residual of a given constraint model based on its type.
        It checks the type of the constraint model and delegates the computation to specialized
        functions for 6D or 3D constraints accordingly. If the constraint type is not implemented,
        it raises a NotImplementedError.
    """    
    if cmodel.type == pin.ContactType.CONTACT_6D:
        return constraintResidual6d(model, data, cmodel, cdata, q, recompute, pinspace)
    elif cmodel.type == pin.ContactType.CONTACT_3D:
        return constraintResidual3d(model, data, cmodel, cdata, q, recompute, pinspace)
    else:
        raise(NotImplementedError("Only 6D and 3D constraints are implemented for now"))

def constraintsResidual(model, data, cmodels, cdatas, q=None, recompute=True, pinspace=pin, quaternions=False):
    """
    Compute the residuals of multiple constraint models.

    Args:
        model (pinocchio.Model): Pinocchio robot model.
        data (pinocchio.Data): Pinocchio data associated with the model.
        cmodels (list): List of constraint models.
        cdatas (list): List of data associated with the constraint models.
        q (numpy.ndarray, optional): Configuration vector. Defaults to None.
        recompute (bool, optional): Flag to recompute forward kinematics. Defaults to True.
        pinspace (module, optional): Pinocchio or CasADi module. Defaults to pin.
        quaternions (bool, optional): Flag indicating whether quaternions are used. Defaults to False.

    Returns:
        numpy.ndarray or casadi.MX: Concatenated residuals of the constraint models.

    Description:
        This function computes the residuals of multiple constraint models by iterating over
        each constraint model and its associated data. It calls the constraintResidual function
        for each constraint model and appends the result to a list. If pinspace is pin, it returns
        the concatenated array of residuals; otherwise, it returns a vertical concatenation using
        CasADi's vertcat function.
    """

    res = []
    for cm, cd in zip(cmodels, cdatas):
        res.append(constraintResidual(model, data, cm, cd, q, recompute, pinspace))
    if pinspace is pin:
        return np.concatenate(res)
    else:
        return casadi.vertcat(*res)
