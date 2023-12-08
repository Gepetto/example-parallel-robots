"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS - April 2023

Tools to merge and split configuration into actuated and non-actuated parts. Also contains tools to freeze joints from a model
"""

import unittest
import numpy as np
import pinocchio as pin
from actuation_model import ActuationModel

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

def mergeq(model, actuation_model, q_mot, q_free, casadiVals=False):
    """
    mergeq(model, actuation_model, q_mot, q_free, casadiVals=False)
    Concatenate qmot and qfree to make a configuration vector q that corresponds to the robot structure. This function includes Casadi support

    Arguments:
        model - Pinocchio robot model
        actuation_model - robot actuation model
        q_mot - the actuated part of q
        q_free - the non-actuated part of q
        casadivals [Optionnal] - Set to use Casadi implementation or not - default: False
    Return:
        q - the merged articular configuration vector
    """
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

def mergev(model, actuation_model, v_mot, v_free, casadiVals=False):
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

def freezeJoints(model, constraint_models, actuation_model, visual_model, collision_model, indexToLock, reference=None):
    '''
    Reduce the model by freezing all joint needed.
    Argument:
        model - Pinocchio robot model
        constraint_models - Pinocchio robot constraint models list
        actuation_model - robot actuation model
        visual_model - Pinocchio robot visual model
        collision_model - Pinocchio robot collision model
        indexToLock: indexes of the joints to lock
        reference - reference configuration to reduce the model from, fixed joints will get their reference configuration fixed
    Return:
        reduced_model - Reduced Pinocchio robot model
        reduced_constraint_models - Reduced Pinocchio robot constraint models list
        reduced_actuation_model - Reduced robot actuation model
        reduced_visual_model - Reduced Pinocchio robot visual model
        reduced_collision_model - Reduced Pinocchio robot collision model
    '''
    if reference is None:
        reference = pin.neutral(model)
    print('Reducing the model')
    reduced_model, (reduced_visual_model, reduced_collision_model) = \
        pin.buildReducedModel(
            model, [visual_model, collision_model], indexToLock, reference)

    if constraint_models is not None:
        print('Reducing the constraint models')
        toremove = []
        for cm in constraint_models:
            print(cm.name)

            # In this function we make the huge assumption that if the parent joint of a constrained frame is fixed is grand-parent is not
            n1 = model.names[cm.joint1_id]
            n2 = model.names[cm.joint2_id]
            idf1 = reduced_model.getFrameId(n1)
            idf2 = reduced_model.getFrameId(n2)
            print(idf1, idf2)

            # Looking for corresponding new frame (ie is the joint fixed ?)
            if idf1 < reduced_model.nframes:
                # The frame has been found
                f1 = reduced_model.frames[idf1]
                cm.joint1_id = f1.parentJoint
                cm.joint1_placement = f1.placement*cm.joint1_placement # Update placement
            else:
                # The joint has not been freezed
                idj1 = reduced_model.getJointId(n1)
                cm.joint1_id = idj1
            # Same for j2
            if idf2 < reduced_model.nframes:
                # The frame has been found
                f2 = reduced_model.frames[idf2]
                cm.joint2_id = f2.parentJoint
                cm.joint2_placement = f2.placement*cm.joint2_placement # Update placement
            else:
                # The joint has not been freezed
                idj2 = reduced_model.getJointId(n2)
                cm.joint2_id = idj2

            if cm.joint1_id == cm.joint2_id:
                toremove.append(cm)
                print(f'Remove constraint {n1}//{n2} (during freeze)')
        reduced_constraint_models = [
            pin.RigidConstraintModel(
                cm.type,
                model,
                cm.joint1_id,
                cm.joint1_placement,
                cm.joint2_id,  # To the world
                cm.joint2_placement,
                pin.ReferenceFrame.LOCAL,
            )
            for cm in constraint_models if cm not in toremove]

    if actuation_model is not None:
        print('Reducing the actuation model')
        list_names = [model.names[idMot] for idMot in actuation_model.idMotJoints]
        reduced_actuation_model = ActuationModel(reduced_model,list_names)

    return(reduced_model, reduced_constraint_models, reduced_actuation_model, reduced_visual_model, reduced_collision_model)
            

########## TEST ZONE ##########################

class TestRobotInfo(unittest.TestCase):
    
    def test_q_splitting(self):
        robots_paths = [['5bar_linkage_iso3d'],
                        ['5bar_linkage_iso6d'],
                        ['robot_delta']]
        
        results = [[[1, 2],[0, 3, 4]],
                   [[1, 2],[0, 3, 4, 5, 6, 7, 8]],
                   [[5, 10],[0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13]]]

        for i, rp in enumerate(robots_paths):
            path = "robots/"+rp[0]
            m ,cm, am, vm, collm = completeRobotLoader(path)
            q = np.linspace(0, m.nq-1, m.nq)
            assert (qmot(am, q)==results[i][0]).all()
            assert (qfree(am, q)==results[i][1]).all()
            assert (mergeq(m, am, qmot(am, q), qfree(am, q)) == q).all()

    def test_v_splitting(self):
        robots_paths = [['5bar_linkage_iso3d'],
                        ['5bar_linkage_iso6d'],
                        ['robot_delta']]
        
        results = [[[1, 2],[0, 3, 4]],
                   [[1, 2],[0, 3, 4, 5, 6, 7]],
                   [[5, 10],[0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13]]]

        for i, rp in enumerate(robots_paths):
            path = "robots/"+rp[0]
            m ,cm, am, vm, collm = completeRobotLoader(path)
            v = np.linspace(0, m.nv-1, m.nv)

            assert (vmot(am, v)==results[i][0]).all()
            assert (vfree(am, v)==results[i][1]).all()
            assert (mergev(m, am, vmot(am, v), vfree(am, v)) == v).all()
        
    def test_freeze_joints(self):
        robot_path = "robots/5bar_linkage_iso3d"
        m ,cm, am, vm, collm = completeRobotLoader(robot_path)
        print("Trying to fix some joints")
        id_tofix = [2]
        rm ,rcm, ram, rvm, rcollm = freezeJoints(m, cm, am, vm, collm, id_tofix, None)
        assert (len(ram.idqmot)==1 and len(ram.idqfree)==3)
        assert (len(rcm)==1)



if __name__ == "__main__":
    from loader_tools import completeRobotLoader
    unittest.main()

