"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS, April 2023

Tools to load and parse a urdf file with closed loop

"""
import unittest
import numpy as np
# from pinocchio import casadi as caspin


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

def mergeq(model, actuation_model, q_mot, q_free):
    """
    completeq = (qmot,qfree)
    concatenate qmot qfree in respect with motor and free id
    """
    q=np.zeros(model.nq)
    for q_i, idqmot in zip(q_mot, actuation_model.idqmot):
        q[idqmot] = q_i

    for q_i,idqfree in zip(q_free, actuation_model.idqfree):
        q[idqfree] = q_i
    return(q)

def mergev(model, actuation_model, v_mot, v_free):
    """
    completeq = (qmot,qfree)
    concatenate qmot qfree in respect with motor and free id
    """
    v = np.zeros(model.nv)
    for v_i, idvmot in zip(v_mot, actuation_model.idvmot):
        v[idvmot] = v_i

    for v_i, idvfree in zip(v_free, actuation_model.idvfree):
        v[idvfree] = v_i
    return(v)
            

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
            m ,cm, am, vm = completeRobotLoader(path)
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
            m ,cm, am, vm = completeRobotLoader(path)
            v = np.linspace(0, m.nv-1, m.nv)

            assert (vmot(am, v)==results[i][0]).all()
            assert (vfree(am, v)==results[i][1]).all()
            assert (mergev(m, am, vmot(am, v), vfree(am, v)) == v).all()



if __name__ == "__main__":
    from loader_tools import completeRobotLoader
    unittest.main()

