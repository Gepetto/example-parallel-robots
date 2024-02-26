"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS - February 2024

This module provides tools to merge and split configuration into actuated and non-actuated parts.
It also contains tools to freeze joints from a model.
"""

import numpy as np
import pinocchio as pin


def qfree(actuation_model, q):
    """
    qfree(actuation_model, q)
    Returns the non-actuated coordinates of q.

    Args:
        actuation_model (object): Robot actuation model.
        q (array): Complete configuration vector.

    Returns:
        array: Non-actuated part of q, ordered as it was in q.
    """
    mask = np.zeros_like(q, bool)
    mask[actuation_model.free_ids_q] = True
    return q[mask]


def qmot(actuation_model, q):
    """
    Return the actuated coordinates of q.

    Args:
        actuation_model (object): Robot actuation model.
        q (array): Complete configuration vector.

    Returns:
        array: Actuated part of q, ordered as it was in q.
    """
    mask = np.zeros_like(q, bool)
    mask[actuation_model.mot_ids_q] = True
    return q[mask]


def vmot(actuation_model, v):
    """
    Return the actuated coordinates of the articular velocity vector v.

    Args:
        actuation_model (object): Robot actuation model.
        v (array): Complete articular velocity vector.

    Returns:
        array: Actuated part of v, ordered as it was in v.
    """
    mask = np.zeros_like(v, bool)
    mask[actuation_model.mot_ids_v] = True
    return v[mask]


def vfree(actuation_model, v):
    """
    Return the non-actuated coordinates of the articular velocity vector v.

    Args:
        actuation_model (object): Robot actuation model.
        v (array): Complete articular velocity vector.

    Returns:
        array: Non-actuated part of v, ordered as it was in v.
    """
    mask = np.zeros_like(v, bool)
    mask[actuation_model.free_ids_v] = True
    return v[mask]


def mergeq(model, actuation_model, q_mot, q_free):
    """ "
    Concatenate qmot and qfree to make a configuration vector q that corresponds to the robot structure. This function includes Casadi support.

    Args:
        model (object): Pinocchio robot model.
        actuation_model (object): Robot actuation model.
        q_mot (array): The actuated part of q.
        q_free (array): The non-actuated part of q.

    Returns:
        array: The merged articular configuration vector.
    """
    casadi_vals = type(model) == pin.casadi.Model
    if not casadi_vals:
        q = np.zeros(model.nq)
        for q_i, mot_ids_q in zip(q_mot, actuation_model.mot_ids_q):
            q[mot_ids_q] = q_i

        for q_i, free_ids_q in zip(q_free, actuation_model.free_ids_q):
            q[free_ids_q] = q_i
    else:
        import casadi

        q = casadi.MX.zeros(model.nq)
        for q_i, mot_ids_q in enumerate(actuation_model.mot_ids_q):
            q[mot_ids_q] = q_mot[q_i]

        for q_i, free_ids_q in enumerate(actuation_model.free_ids_q):
            q[free_ids_q] = q_free[q_i]
    return q


def mergev(model, actuation_model, v_mot, v_free):
    """
    Concatenate qmot and qfree to make a configuration vector q that corresponds to the robot structure. This function includes Casadi support.

    Args:
        model (object): Pinocchio robot model.
        actuation_model (object): Robot actuation model.
        q_mot (array): The actuated part of q.
        q_free (array): The non-actuated part of q.

    Returns:
        array: The merged articular configuration vector.
    """
    casadi_vals = type(model) == pin.casadi.Model
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
    return v
