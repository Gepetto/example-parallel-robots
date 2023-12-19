"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS - April 2023

Tools to merge and split configuration into actuated and non-actuated parts. Also contains tools to freeze joints from a model
"""

import numpy as np
import pinocchio as pin

def qfree(actuation_model, q):
    """
    qfree(actuation_model, q)
    Return the non-actuated coordinates of q

    Arguments:
        actuation_model - robot actuation model
        q - complete configuration vector
    Return:
        q_free - non-actuated part of q, ordered as it was in q
    """
    mask = np.zeros_like(q, bool)
    mask[actuation_model.idqfree] = True
    return(q[mask])

def qmot(actuation_model, q):
    """
    qmot(actuation_model, q)
    Return the actuated coordinates of q

    Arguments:
        actuation_model - robot actuation model
        q - complete configuration vector
    Return:
        q_mot - actuated part of q, ordered as it was in q
    """
    mask = np.zeros_like(q, bool)
    mask[actuation_model.idqmot] = True
    return(q[mask])

def vmot(actuation_model, v):
    """
    vmot(actuation_model, v)
    Return the actuated coordinates of the articular velocity vector v

    Arguments:
        actuation_model - robot actuation model
        v - complete articular velocity vector
    Return:
        v_mot - actuated part of v, ordered as it was in v
    """
    mask = np.zeros_like(v, bool)
    mask[actuation_model.idvmot] = True
    return(v[mask])

def vfree(actuation_model, v):
    """
    vfree(actuation_model, v)
    Return the non-actuated coordinates of the articular velocity vector v

    Arguments:
        actuation_model - robot actuation model
        v - complete articular velocity vector
    Return:
        v_free - non-actuated part of v, ordered as it was in v
    """
    mask = np.zeros_like(v, bool)
    mask[actuation_model.idvfree] = True
    return(v[mask])

def mergeq(model, actuation_model, q_mot, q_free):
    """
    mergeq(model, actuation_model, q_mot, q_free, casadiVals=False)
    Concatenate qmot and qfree to make a configuration vector q that corresponds to the robot structure. This function includes Casadi support

    Arguments:
        model - Pinocchio robot model
        actuation_model - robot actuation model
        q_mot - the actuated part of q
        q_free - the non-actuated part of q
    Return:
        q - the merged articular configuration vector
    """
    casadiVals = (type(model) == pin.casadi.Model)
    if not casadiVals:
        q=np.zeros(model.nq)
        for q_i, idqmot in zip(q_mot, actuation_model.idqmot):
            q[idqmot] = q_i

        for q_i,idqfree in zip(q_free, actuation_model.idqfree):
            q[idqfree] = q_i
    else:
        import casadi
        q = casadi.MX.zeros(model.nq)
        for q_i, idqmot in enumerate(actuation_model.idqmot):
            q[idqmot] = q_mot[q_i]

        for q_i, idqfree in enumerate(actuation_model.idqfree):
            q[idqfree] = q_free[q_i]
    return(q)

def mergev(model, actuation_model, v_mot, v_free):
    """
    mergev(model, actuation_model, v_mot, v_free, casadiVals=False)
    Concatenate qmot and qfree to make a configuration vector q that corresponds to the robot structure. This function includes Casadi support

    Arguments:
        model - Pinocchio robot model
        actuation_model - robot actuation model
        v_mot - the actuated part of v
        v_free - the non-actuated part of v
        casadivals [Optionnal] - Set to use Casadi implementation or not - default: False
    Return:
        v - the merged articular velocity vector
    """
    casadiVals = (type(model) == pin.casadi.Model)
    if not casadiVals:
        v = np.zeros(model.nv)
        for v_i, idvmot in zip(v_mot, actuation_model.idvmot):
            v[idvmot] = v_i

        for v_i, idvfree in zip(v_free, actuation_model.idvfree):
            v[idvfree] = v_i
    else:
        import casadi
        v = casadi.MX.zeros(model.nv)
        for v_i, idvmot in enumerate(actuation_model.idvmot):
            v[idvmot] = v_mot[v_i]

        for v_i, idvfree in enumerate(actuation_model.idvfree):
            v[idvfree] = v_free[v_i]
    return(v)