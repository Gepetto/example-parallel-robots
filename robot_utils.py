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
    free_q = q2freeq(model, q, name_mot="mot")
    Return the non-motor coordinate of q, i.e. the configuration of the non-actuated joints

    Arguments:
        model - robot model from pinocchio
        q - complete configuration vector
        name_mot - string to be found in the motors joints names
    Return:
        Lid - List of motors configuration velocity ids
    """
    Lidmot = actuation_model.idqfree
    mask = np.zeros_like(q, bool)
    mask[Lidmot] = True
    return(q[mask])

def qmot(actuation_model, q):
    """
    free_q = qmot(model, q, name_mot="mot")
    Return the non-motor coordinate of q, i.e. the configuration of the non-actuated joints

    Arguments:
        model - robot model from pinocchio
        q - complete configuration vector
        name_mot - string to be found in the motors joints names
    Return:
        Lid - List of motors configuration velocity ids
    """
    Lidmot = actuation_model.idqmot
    mask = np.zeros_like(q, bool)
    mask[Lidmot] = True
    return(q[mask])

def vmot(actuation_model, v):
    """
    free_q = qmot(model, q, name_mot="mot")
    Return the non-motor coordinate of q, i.e. the configuration of the non-actuated joints

    Arguments:
        model - robot model from pinocchio
        q - complete configuration vector
        name_mot - string to be found in the motors joints names
    Return:
        Lid - List of motors configuration velocity ids
    """
    Lidmot = actuation_model.idvmot
    mask = np.zeros_like(v, bool)
    mask[Lidmot] = True
    return(v[mask])

def vfree(actuation_model, v):
    """
    free_q = qmot(model, q, name_mot="mot")
    Return the non-motor coordinate of q, i.e. the configuration of the non-actuated joints

    Arguments:
        model - robot model from pinocchio
        q - complete configuration vector
        name_mot - string to be found in the motors joints names
    Return:
        Lid - List of motors configuration velocity ids
    """
    Lidmot = actuation_model.idvfree
    mask = np.zeros_like(v, bool)
    mask[Lidmot] = True
    return(v[mask])

def mergeq(model, actuation_model, q_mot, q_free, casadiVals=False):
    """
    completeq = (qmot,qfree)
    concatenate qmot qfree in respect with motor and free id
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
    completeq = (qmot,qfree)
    concatenate qmot qfree in respect with motor and free id
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
    Reduce the model by freezing all joint whose name contain the key string.
    robot: a robot wrapper where the result is stored (destructive mode)
    indexToLock: indexes of the joints to lock
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
            n1 = model.names[cm.joint1_id]
            n2 = model.names[cm.joint2_id]

            # The reference joints might have been frozen
            # Then seek for the corresponding frame, that might be either a joint frame
            # or a op frame.
            idf1 = reduced_model.getFrameId(n1)
            f1 = reduced_model.frames[idf1]
            idf2 = reduced_model.getFrameId(n2)
            f2 = reduced_model.frames[idf2]

            # Make the new reference joints the parent of the frame.
            cm.joint1_id = f1.parentJoint
            cm.joint2_id = f2.parentJoint
            # In the best case, the joint still exist, then it corresponds to a joint frame.
            if f1.type != pin.JOINT:
                assert (f1.type == pin.FIXED_JOINT)
                # If the joint has be freezed, the contact now should be referenced with respect
                # to the new joint, which was a parent of the previous.
                cm.joint1_placement = f1.placement*cm.joint1_placement
                # ! We assume here that the parent of the fixed joint is not also fixed
            # Same for the second joint
            if f2.type != pin.JOINT:
                assert (f2.type == pin.FIXED_JOINT)
                cm.joint2_placement = f2.placement*cm.joint2_placement

            if cm.joint1_id == cm.joint2_id:
                toremove.append(cm)
                print(f'Remove constraint {n1}//{n2} (during freeze)')

        reduced_constraint_models = [
            cm for cm in constraint_models if cm not in toremove]
    
    if actuation_model is not None:
        print('Reducing the actuation model')
        list_names = [model.names[idMot] for idMot in actuation_model.idMotJoints]
        reduced_actuation_model = ActuationModel(reduced_model,list_names)

    return(reduced_model, reduced_constraint_models, reduced_actuation_model, reduced_visual_model, reduced_collision_model)
            

            
            

########## TEST ZONE ##########################

class TestRobotInfo(unittest.TestCase):
    
    def test_q_splitting(self):
        robots_paths = [['robot_simple_iso3D'],
                        ['robot_simple_iso6D'],
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
        robots_paths = [['robot_simple_iso3D'],
                        ['robot_simple_iso6D'],
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
        robot_path = "robots/robot_simple_iso3D"
        m ,cm, am, vm, collm = completeRobotLoader(robot_path)
        print("Trying to fix some joints")
        id_tofix = [2]
        rm ,rcm, ram, rvm, rcollm = freezeJoints(m, cm, am, vm, collm, id_tofix, None)
        assert (len(ram.idqmot)==1 and len(ram.idqfree)==3)
        assert (len(rcm)==1)



if __name__ == "__main__":
    from loader_tools import completeRobotLoader
    unittest.main()

