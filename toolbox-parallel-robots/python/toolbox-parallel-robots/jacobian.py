"""
-*- coding: utf-8 -*-
Virgile BATTO, April 2023

Tools to compute of jacobian inside closed loop

"""

import pinocchio as pin
import numpy as np


def separateConstraintJacobian(actuation_data, Jn):
    """
    Separate a constraint Jacobian `Jn` into Jcmot and Jcfree, the constraint Jacobians associated with the motor joints and free joints.

    Args:
        actuation_data (ActuationModelFreeFlyer): Actuation data containing information about motor and free joints.
        Jn (numpy.ndarray): Constraint Jacobian to be separated.

    Returns:
        tuple: A tuple containing:
            - Jmot (np.array): Constraint Jacobian associated with the motor joints.
            - Jfree (np.array): Constraint Jacobian associated with the free joints.

    Notes:
        - Jmot is obtained by multiplying Jn with Smot, which represents the selection matrix for the motor joints.
        - Jfree is obtained by multiplying Jn with Sfree, which represents the selection matrix for the free joints.
    """

    Smot = actuation_data.Smot
    Sfree = actuation_data.Sfree

    Jmot = Jn @ Smot
    Jfree = Jn @ Sfree
    return (Jmot, Jfree)


def computeDerivative_dq_dqmot(actuation_model, actuation_data, LJ):
    """
    Compute the derivative `dq/dqmot` of the joint to the motor joint.

    Args:
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        actuation_data (ActuationModelFreeFlyer): Actuation data.
        LJ (list): List of constraint Jacobians.

    Returns:
        np.array: Derivative `dq/dqmot`.

    Notes:
        - The derivative `dq/dqmot` represents the sensitivity of the joint velocities to the motor joint velocities.

    """

    mot_ids_v = actuation_model.mot_ids_v
    free_ids_v = actuation_model.free_ids_v
    constraints_sizes = actuation_data.constraints_sizes
    nprec = 0
    nv_mot = len(mot_ids_v)

    for J, n in zip(LJ, constraints_sizes):
        actuation_data.Jmot[nprec : nprec + n, :] = J @ actuation_data.Smot
        actuation_data.Jfree[nprec : nprec + n, :] = J @ actuation_data.Sfree
        nprec = nprec + n

    actuation_data.pinvJfree[:, :] = np.linalg.pinv(actuation_data.Jfree)
    actuation_data.dq_no[:nv_mot, :] = np.identity(nv_mot)
    actuation_data.dq_no[nv_mot:, :] = -actuation_data.pinvJfree @ actuation_data.Jmot
    actuation_data.dq[mot_ids_v] = actuation_model.dq_no[:nv_mot, :]
    actuation_data.dq[free_ids_v] = actuation_model.dq_no[nv_mot:, :]
    return actuation_data.dq


def computeClosedLoopFrameJacobian(
    model,
    data,
    constraint_model,
    constraint_data,
    actuation_model,
    actuation_data,
    q0,
    idframe,
):
    """
    Compute `Jf_closed`, the closed loop Jacobian on the frame `idframe`.

    Args:
        model (pinocchio.Model): Pinocchio model.
        data (pinocchio.Data): Pinocchio data associated with the model.
        constraint_model (list): List of constraint models.
        constraint_data (list): List of constraint data associated with the constraint models.
        actuation_model (ActuationModel): Actuation model.
        actuation_data (ActuationData): Actuation data.
        q0 (np.array): Initial configuration.
        idframe (int): Frame index for which the joint velocity is computed.

    Returns:
        np.array: Closed-loop Jacobian on frame `idframe`.
    """

    pin.computeJointJacobians(model, data, q0)
    LJ = actuation_data.LJ
    for cm, cd, i in zip(constraint_model, constraint_data, range(len(LJ))):
        LJ[i][:, :] = pin.getConstraintJacobian(model, data, cm, cd)
    _ = computeDerivative_dq_dqmot(actuation_model, actuation_data, LJ)
    actuation_data.Jf_closed[:, :] = (
        pin.computeFrameJacobian(model, data, q0, idframe, pin.LOCAL)
        @ actuation_data.dq
    )
    return actuation_data.Jf_closed[:, :]


def inverseConstraintKinematicsSpeed(
    model,
    data,
    constraint_model,
    constraint_data,
    actuation_model,
    actuation_data,
    q0,
    ideff,
    veff,
):
    """
    vq, Jf_closed = inverseConstraintKinematicsSpeedOptimized(model, data, constraint_model, constraint_data, actuation_model, q0, ideff, veff)

    Compute the joint velocity `vq` that generates the speed `veff` on frame `ideff`.

    Args:
        model (pinocchio.Model): Pinocchio model.
        data (pinocchio.Data): Pinocchio data associated with the model.
        constraint_model (list): List of constraint models.
        constraint_data (list): List of constraint data associated with the constraint models.
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        q0 (np.array): Initial configuration.
        ideff (int): Frame index for which the joint velocity is computed.
        veff (np.array): Desired speed on frame `ideff`.

    Returns:
        np.array: Joint velocity `vq` that generates the desired speed on frame `ideff`.
    """
    # update of the jacobian an constraint model
    pin.computeJointJacobians(model, data, q0)
    LJ = actuation_data.LJ
    for cm, cd, i in zip(constraint_model, constraint_data, range(len(LJ))):
        LJ[i][:, :] = pin.getConstraintJacobian(model, data, cm, cd)

    # init of constant
    mot_ids_v = actuation_model.mot_ids_v
    free_ids_v = actuation_model.free_ids_v
    nv_mot = len(mot_ids_v)
    constraints_sizes = actuation_data.constraints_sizes

    nprec = 0
    for J, n in zip(LJ, constraints_sizes):
        actuation_data.Jmot[nprec : nprec + n, :] = J @ actuation_data.Smot
        actuation_data.Jfree[nprec : nprec + n, :] = J @ actuation_data.Sfree

        nprec = nprec + n

    actuation_data.pinvJfree[:, :] = np.linalg.pinv(actuation_data.Jfree)
    actuation_data.dq_no[:nv_mot, :] = np.identity(nv_mot)
    actuation_data.dq_no[nv_mot:, :] = -actuation_data.pinvJfree @ actuation_data.Jmot
    actuation_model.dq[mot_ids_v] = actuation_model.dq_no[:nv_mot, :]
    actuation_model.dq[free_ids_v] = actuation_model.dq_no[nv_mot:, :]
    # computation of the closed-loop jacobian
    actuation_data.Jf_closed[:, :] = (
        pin.computeFrameJacobian(model, data, q0, ideff, pin.LOCAL) @ actuation_data.dq
    )
    actuation_data.vqmot[:] = np.linalg.pinv(actuation_data.Jf_closed) @ veff
    actuation_data.vqfree[:] = (
        -actuation_data.pinvJfree @ actuation_data.Jmot @ actuation_data.vqmot
    )
    # reorder of vq
    actuation_data.vqmotfree[:nv_mot] = actuation_data.vqmot
    actuation_data.vqmotfree[nv_mot:] = actuation_data.vqfree
    actuation_data.vq[mot_ids_v] = actuation_data.vqmotfree[:nv_mot]
    actuation_data.vq[free_ids_v] = actuation_data.vqmotfree[nv_mot:]

    return actuation_data.vq
