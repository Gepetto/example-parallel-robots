import pinocchio as pin
import numpy as np


class ActuationData:
    """
    Defines the actuation data of a robot.

    Example:
        robot_actuation_data = ActuationData(model, constraints_models, actuation_model)

    Args:
        model (pinocchio.Model): Robot model.
        constraints_models (list): List of the constraint models associated with the robot.
        actuation_model (ActuationModel): Robot actuation model.

    Attributes:
        Smot (np.array): Selection matrix for the motor joints.
        Sfree (np.array): Selection matrix for the free joints.
        Jmot (np.array): Constraint Jacobian associated with the motor joints.
        Jfree (np.array): Constraint Jacobian associated with the free joints.
        Mmot (np.array): Placeholder for the motor joint mass matrix.
        dq (np.array): Placeholder for the joint velocities.
        dq_no (np.array): Placeholder for joint velocities without considering constraints.
        LJ (list): List of constraint Jacobians associated with each constraint model.
        Jf_closed (np.array): Closed-loop Jacobian on the frame.
        vq (np.array): Joint velocities.
        vqmot (np.array): Joint velocities associated with motor joints.
        vqfree (np.array): Joint velocities associated with free joints.
        vqmotfree (np.array): Joint velocities associated with both motor and free joints.
        constraints_sizes (list): List of constraint sizes.
        pinvJfree (np.array): Pseudo-inverse of the free joint Jacobian.

    Methods:
        None
    """

    def __init__(self, model, constraints_models, actuation_model):

        Lidmot = actuation_model.mot_ids_v
        free_ids_v = actuation_model.free_ids_v
        nv = model.nv
        nv_mot = actuation_model.nv_mot
        nv_free = actuation_model.nv_free
        nc = 0

        # Count total size of constraints
        for c in constraints_models:
            nc += c.size()

        # Initialize matrices and arrays
        self.Smot = np.zeros((nv, nv_mot))
        self.Sfree = np.zeros((nv, nv_free))
        self.Jmot = np.zeros((nc, nv_mot))
        self.Jfree = np.zeros((nc, nv_free))
        self.Mmot = np.zeros((nv_mot, nv_mot))
        self.dq = np.zeros((nv, nv_mot))
        self.dq_no = np.zeros((nv, nv_mot))

        # init a list of constraint_jacobian
        self.LJ = [np.array(())] * len(constraints_models)
        constraint_data = [c.createData() for c in constraints_models]
        data = model.createData()
        for cm, cd, i in zip(constraints_models, constraint_data, range(len(self.LJ))):
            self.LJ[i] = pin.getConstraintJacobian(model, data, cm, cd)

        # selection matrix for actuated parallel robot
        self.Smot[:, :] = 0
        self.Smot[Lidmot, range(nv_mot)] = 1
        self.Sfree[:, :] = 0
        self.Sfree[free_ids_v, range(nv_free)] = 1

        self.Jf_closed = (
            pin.computeFrameJacobian(
                model, model.createData(), np.zeros(model.nq), 0, pin.LOCAL
            )
            @ self.dq
        )

        # init of different size of vector
        self.vq = np.zeros(nv)
        self.vqmot = np.zeros(nv_mot - 6)
        self.vqfree = np.zeros(nv_free)
        self.vqmotfree = np.zeros(nv - 6)

        # list of constraint type
        self.constraints_sizes = [J.shape[0] for J in self.LJ]
        self.pinvJfree = np.linalg.pinv(self.Jfree)
