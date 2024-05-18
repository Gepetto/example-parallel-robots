import unittest


class TestRobotInfo(unittest.TestCase):
    # only test inverse constraint kineatics because it runs all precedent code
    def test_inverseConstraintKinematics(self):
        vapply = np.array([0, 0, 1, 0, 0, 0])
        vq = inverseConstraintKinematicsSpeed(
            model,
            data,
            constraint_models,
            constraint_datas,
            actuation_model,
            q0,
            34,
            vapply,
        )[0]
        pin.computeAllTerms(model, data, q0, vq)
        vcheck = data.v[13].np  # frame 34 is center on joint 13
        # check that the computing vq give the good speed
        self.assertTrue(norm(vcheck - vapply) < 1e-6)


if __name__ == "__main__":
    # load robot
    path = os.getcwd() + "/robots/robot_marcheur_1"
    model, constraint_models, actuation_model, visual_model, collision_model = (
        completeRobotLoader(path)
    )
    data = model.createData()
    constraint_datas = [cm.createData() for cm in constraint_models]
    q0 = proximalSolver(model, data, constraint_models, constraint_datas)

    # test
    unittest.main()
