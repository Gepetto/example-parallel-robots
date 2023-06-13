"""
-*- coding: utf-8 -*-
Nicolas MANSARD & Ludovic DE MATTEIS & Virgile BATTO, April 2023

Tool functions to compute the constraints residuals from a robot constraint model. Also includes quaternion normalization constraint

"""

import pinocchio as pin
import numpy as np
from pinocchio import casadi as caspin
import casadi


def constraintResidual6d(model, data, cmodel, cdata, q, recompute=True, pinspace=pin):
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

def constraintsResidual(model, data, cmodels, cdatas, q=None, recompute=True, pinspace=pin, quaternions=False):
    res = []
    for cm, cd in zip(cmodels, cdatas):
        res.append(constraintResidual(model, data, cm, cd, q, recompute, pinspace))

    return np.concatenate(res)


import unittest
class TestRobotLoader(unittest.TestCase):
    def test_constraints3d(self):
        import io
        from loader_tools import completeRobotLoader

        path = "robots/robot_simple_iso3D"
        m ,cm, am, vm = completeRobotLoader(path)
        cd = [cmi.createData() for cmi in cm]
        d = m.createData()
        q = pin.neutral(m)

        # Test with pin
        result = [4.26300686e-03, 1.11022302e-16, 5.24389133e-04]
        np.testing.assert_allclose(constraintResidual3d(m, d, cm[0], cd[0], q, True, pin), result, atol=1e-8, rtol=1e-12)

        # Test with caspin
        m = caspin.Model(m)
        d = m.createData()
        cm = [caspin.RigidConstraintModel(cmi) for cmi in cm]
        cd = [cmi.createData() for cmi in cm]
        q = casadi.SX(q)
        assert (constraintResidual3d(m, d, cm[0], cd[0], q, True, caspin)-casadi.SX(result)<1e-8).is_one()
    
    def test_constraints6d(self):
        import io
        from loader_tools import completeRobotLoader

        path = "robots/robot_simple_iso6D"
        m ,cm, am, vm = completeRobotLoader(path)
        cd = [cmi.createData() for cmi in cm]
        d = m.createData()
        q = pin.neutral(m)

        # Test with pin
        result = [1.53554248e-05, 3.90264510e-03, 2.48445358e-03, 1.94165706e-02, -6.03364668e-16, 3.14153265e+00]
        np.testing.assert_allclose(constraintResidual6d(m, d, cm[0], cd[0], q, True, pin), result, atol=1e-8, rtol=1e-12)

        # Test with caspin
        m = caspin.Model(m)
        d = m.createData()
        cm = [caspin.RigidConstraintModel(cmi) for cmi in cm]
        cd = [cmi.createData() for cmi in cm]
        q = casadi.SX(q)
        assert (constraintResidual6d(m, d, cm[0], cd[0], q, True, caspin)-casadi.SX(result)<1e-8).is_one()
            
if __name__ == "__main__":
    unittest.main()
