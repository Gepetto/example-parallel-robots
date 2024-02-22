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
    mask[actuation_model.free_ids_q] = True
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
    mask[actuation_model.mot_ids_q] = True
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
    mask[actuation_model.mot_ids_v] = True
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
    mask[actuation_model.free_ids_v] = True
    return(v[mask])

def mergeq(model, actuation_model, q_mot, q_free):
    """
    mergeq(model, actuation_model, q_mot, q_free, casadi_vals=False)
    Concatenate qmot and qfree to make a configuration vector q that corresponds to the robot structure. This function includes Casadi support

    Arguments:
        model - Pinocchio robot model
        actuation_model - robot actuation model
        q_mot - the actuated part of q
        q_free - the non-actuated part of q
    Return:
        q - the merged articular configuration vector
    """
    casadi_vals = (type(model) == pin.casadi.Model)
    if not casadi_vals:
        q=np.zeros(model.nq)
        for q_i, mot_ids_q in zip(q_mot, actuation_model.mot_ids_q):
            q[mot_ids_q] = q_i

        for q_i,free_ids_q in zip(q_free, actuation_model.free_ids_q):
            q[free_ids_q] = q_i
    else:
        import casadi
        q = casadi.MX.zeros(model.nq)
        for q_i, mot_ids_q in enumerate(actuation_model.mot_ids_q):
            q[mot_ids_q] = q_mot[q_i]

        for q_i, free_ids_q in enumerate(actuation_model.free_ids_q):
            q[free_ids_q] = q_free[q_i]
    return(q)

def mergev(model, actuation_model, v_mot, v_free):
    """
    mergev(model, actuation_model, v_mot, v_free, casadi_vals=False)
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
    casadi_vals = (type(model) == pin.casadi.Model)
    if not casadi_vals:
        v = np.zeros(model.nv)
        for v_i, mot_ids_v in zip(v_mot, actuation_model.mot_ids_v):
            v[mot_ids_v] = v_i

        for v_i, free_ids_v in zip(v_free, actuation_model.free_ids_v):
            v[free_ids_v] = v_i
    else:
        import casadi
        v = casadi.MX.zeros(model.nv)
        for v_i, mot_ids_v in enumerate(actuation_model.mot_ids_v):
            v[mot_ids_v] = v_mot[v_i]

        for v_i, free_ids_v in enumerate(actuation_model.free_ids_v):
            v[free_ids_v] = v_free[v_i]
    return(v)