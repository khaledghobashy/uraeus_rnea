import unittest

import numpy as np
import jax.numpy as jnp
import jax

from uraeus.rnea.quaternion.bodies import RigidBodyData
from uraeus.rnea.quaternion.joints import (
    JointConfigInputs,
    FreeJoint,
    TranslationalJoint,
    RevoluteJoint,
)
from uraeus.rnea.quaternion.topologies import MultiBodyTree, Model
from uraeus.rnea.quaternion.tree_traversals import extract_mobilizer_forces


def emulate_free_joint(
    model: MultiBodyTree, predecessor: str, successor: str, succ_data: RigidBodyData
) -> MultiBodyTree:
    dummy_body_data = RigidBodyData()

    x_axis_config = JointConfigInputs(
        pos=np.zeros((3,)), z_axis=np.array([1, 0, 0]), x_axis=None
    )
    y_axis_config = JointConfigInputs(
        pos=np.zeros((3,)), z_axis=np.array([0, 1, 0]), x_axis=None
    )
    z_axis_config = JointConfigInputs(
        pos=np.zeros((3,)), z_axis=np.array([0, 0, 1]), x_axis=None
    )

    model.add_joint(
        joint_name="x_rotation",
        predecessor=predecessor,
        successor="d1",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=x_axis_config,
    )

    model.add_joint(
        joint_name="y_rotation",
        predecessor="d1",
        successor="d2",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=y_axis_config,
    )

    model.add_joint(
        joint_name="z_rotation",
        predecessor="d2",
        successor="d3",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=z_axis_config,
    )

    model.add_joint(
        joint_name="x_trans",
        predecessor="d3",
        successor="d4",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=x_axis_config,
    )

    model.add_joint(
        joint_name="y_trans",
        predecessor="d4",
        successor="d5",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=y_axis_config,
    )

    model.add_joint(
        joint_name="z_trans",
        predecessor="d5",
        successor=successor,
        succ_data=succ_data,
        joint_type=TranslationalJoint,
        joint_data=z_axis_config,
    )

    return model


class AnalyticalMassSpringDamper(object):

    def __init__(self, mass, stiffness, damping):

        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

    def ssode(self, t, y):
        """ODE for mass-spring-damper system

        qdt2 + 2*eta*wn*qdt1 + wn^2*qdt0 = u
        where:
            wn = sqrt(k/m)
            eta = c/(2 * m * wn)
            u = F_ex/m

        Parameters
        ----------
        t : time
        """
        qdt0, qdt1 = np.split(y, 2)

        F_ex = self.mass * (-9.81)
        u = F_ex / self.mass
        wn = np.sqrt(self.stiffness / self.mass)

        eta = self.damping / (2 * self.mass * wn)
        qdt2 = u - (2 * eta * wn * qdt1) - (wn**2 * qdt0)

        return np.array([*qdt1, *qdt2])


class MassSpringDamperTest(unittest.TestCase):

    multibody_system: Model

    def setUp(self):
        self.mass = 10
        self.stiffness = 10
        self.damping = 10

        self.analytical_model = AnalyticalMassSpringDamper(
            self.mass, self.stiffness, self.damping
        )
        self.multibody_system = self._build_multibody_system(
            self.mass, self.stiffness, self.damping
        )

        self.actuator_dt0 = lambda t: 0.5 * jnp.sin(2 * t)
        self.actuator_dt1 = jax.jacfwd(self.actuator_dt0)

    def test_forward_dynamics(self):
        time_array = np.linspace(0, 2 * np.pi, 100)

        true_sol, test_sol = zip(
            *[self._evaluate_forward_dynamics(t) for t in time_array]
        )
        np.testing.assert_almost_equal(true_sol, test_sol)

    def _evaluate_forward_dynamics(self, t):

        qdt0 = np.array([self.actuator_dt0(t)])
        qdt1 = np.array([self.actuator_dt1(t)])
        ydt0 = np.array([*qdt0, *qdt1])

        ydt0_multibody = np.array(
            [0, 0, self.actuator_dt0(t), 0, 0, 0, 0, 0, self.actuator_dt1(t), 0, 0, 0]
        )

        qdt2_true = self._evaluate_analytical_forward_dynamics(t, ydt0)
        qdt2_test = self._evaluate_multibody_forward_dynamics(t, ydt0_multibody)

        print("true_sol = ", qdt2_true)
        print("test_sol = ", qdt2_test)
        print("")
        return qdt2_true, np.array([qdt2_test[2], qdt2_test[2 + 6]])

    def _evaluate_analytical_forward_dynamics(self, t, ydt0):
        return self.analytical_model.ssode(t, ydt0)

    def _evaluate_multibody_forward_dynamics(self, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        self.multibody_system.forces_map["m1"]["spring"] = self.multibody_system.fk(
            qdt0[2]
        )
        self.multibody_system.forces_map["m1"]["damper"] = self.multibody_system.fc(
            qdt1[2]
        )

        qdt2 = self.multibody_system.forward_dynamics_pass(
            qdt0, qdt1, np.zeros_like(qdt1)
        )

        return np.array([*qdt1, *qdt2])

    def _build_multibody_system(self, m1, k, c) -> Model:
        tree = MultiBodyTree("mass_spring")

        m1_data = RigidBodyData(
            np.array([0, 0, 1]), np.array([1, 0, 0, 0]), m1, np.eye(3)
        )
        j1_data = JointConfigInputs(np.array([0, 0, 1]), np.array([0, 0, 1]), None)

        tree.add_joint("j1", "ground", "m1", m1_data, FreeJoint, j1_data)
        # tree = emulate_free_joint(tree, "ground", "m1", m1_data)

        model = Model(tree)

        model.fk = lambda x: np.array([0, 0, -x * k, 0, 0, 0])
        model.fc = lambda v: np.array([0, 0, -v * c, 0, 0, 0])

        self.multibody_system = model
        return model


class DoubleMassSpringDamperTest(unittest.TestCase):

    test_system: Model
    true_system: Model

    def setUp(self):
        self.mass = 10
        self.stiffness = 10
        self.damping = 10

        self.true_system = self._build_true_system(
            self.mass, self.stiffness, self.damping
        )
        self.test_system = self._build_test_system(
            self.mass, self.stiffness, self.damping
        )

        self.actuator1_dt0 = lambda t: 0.5 * jnp.sin(2 * t)
        self.actuator1_dt1 = jax.jacfwd(self.actuator1_dt0)

        self.actuator2_dt0 = lambda t: 0.5 * jnp.sin(2 * t)
        self.actuator2_dt1 = jax.jacfwd(self.actuator2_dt0)

    def test_forward_dynamics(self):
        time_array = np.linspace(0, 2 * np.pi, 100)

        true_sol, test_sol = zip(
            *[self._evaluate_forward_dynamics(t) for t in time_array]
        )
        np.testing.assert_almost_equal(true_sol, test_sol)

    def _evaluate_forward_dynamics(self, t):

        qdt0 = np.array([self.actuator1_dt0(t), self.actuator2_dt0(t)])
        qdt1 = np.array([self.actuator1_dt1(t), self.actuator2_dt1(t)])
        ydt0 = np.array([*qdt0, *qdt1])
        ydt0_multibody = np.array(
            [
                0,
                0,
                self.actuator1_dt0(t),
                0,
                0,
                0,
                0,
                0,
                self.actuator2_dt0(t),
                0,
                0,
                0,
                0,
                0,
                self.actuator1_dt1(t),
                0,
                0,
                0,
                0,
                0,
                self.actuator2_dt1(t),
                0,
                0,
                0,
            ]
        )

        ydt2_true = self._evaluate_true_forward_dynamics(t, ydt0)
        ydt2_test = self._evaluate_test_forward_dynamics(t, ydt0_multibody)

        print("true_sol = ", ydt2_true)
        print("test_sol = ", ydt2_test)
        print("")
        return ydt2_true, np.array(
            [ydt2_test[2], ydt2_test[2 + 6], ydt2_test[2 + 12], ydt2_test[2 + 6 + 12]]
        )

    def _evaluate_test_forward_dynamics(self, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        self.test_system.forces_map["m1"]["spring"] = self.multibody_system.fk(qdt0[2])
        self.test_system.forces_map["m1"]["damper"] = self.multibody_system.fc(qdt1[2])

        self.test_system.forces_map["m2"]["spring"] = self.multibody_system.fk(
            qdt0[2 + 6]
        )
        self.test_system.forces_map["m2"]["damper"] = self.multibody_system.fc(
            qdt1[2 + 6]
        )

        qdt2 = self.test_system.forward_dynamics_pass(qdt0, qdt1, np.zeros_like(qdt1))

        return np.array([*qdt1, *qdt2])

    def _evaluate_true_forward_dynamics(self, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        self.true_system.forces_map["m1"]["spring"] = self.multibody_system.fk(qdt0[0])
        self.true_system.forces_map["m1"]["damper"] = self.multibody_system.fc(qdt1[0])

        self.true_system.forces_map["m2"]["spring"] = self.multibody_system.fk(qdt0[1])
        self.true_system.forces_map["m2"]["damper"] = self.multibody_system.fc(qdt1[1])

        qdt2 = self.true_system.forward_dynamics_pass(qdt0, qdt1, np.zeros_like(qdt1))

        return np.array([*qdt1, *qdt2])

    def _build_true_system(self, m1, k, c) -> Model:
        tree = MultiBodyTree("mass_spring")

        m1_data = RigidBodyData(
            np.array([0, 0, 1]), np.array([1, 0, 0, 0]), m1, np.eye(3)
        )
        m2_data = RigidBodyData(
            np.array([0, 0, 2]), np.array([1, 0, 0, 0]), m1, np.eye(3)
        )
        j1_data = JointConfigInputs(np.array([0, 0, 0]), np.array([0, 0, 1]), None)
        j2_data = JointConfigInputs(np.array([0, 0, 1]), np.array([0, 0, 1]), None)

        tree.add_joint("j1", "ground", "m1", m1_data, TranslationalJoint, j1_data)
        tree.add_joint("j2", "m1", "m2", m2_data, TranslationalJoint, j2_data)
        # tree = emulate_free_joint(tree, "ground", "m1", m1_data)

        model = Model(tree)

        model.fk = lambda x: np.array([0, 0, -x * k, 0, 0, 0])
        model.fc = lambda v: np.array([0, 0, -v * c, 0, 0, 0])

        self.multibody_system = model
        return model

    def _build_test_system(self, m1, k, c) -> Model:
        tree = MultiBodyTree("mass_spring")

        m1_data = RigidBodyData(
            np.array([0, 0, 1]), np.array([1, 0, 0, 0]), m1, np.eye(3)
        )
        m2_data = RigidBodyData(
            np.array([0, 0, 2]), np.array([1, 0, 0, 0]), m1, np.eye(3)
        )
        j1_data = JointConfigInputs(np.array([0, 0, 0]), np.array([0, 0, 1]), None)
        j2_data = JointConfigInputs(np.array([0, 0, 1]), np.array([0, 0, 1]), None)

        tree.add_joint("j1", "ground", "m1", m1_data, FreeJoint, j1_data)
        tree.add_joint("j2", "m1", "m2", m2_data, FreeJoint, j2_data)

        model = Model(tree)

        model.fk = lambda x: np.array([0, 0, -x * k, 0, 0, 0])
        model.fc = lambda v: np.array([0, 0, -v * c, 0, 0, 0])

        self.multibody_system = model
        return model


if __name__ == "__main__":

    unittest.main()
    # import scipy.integrate as integrate
    # import matplotlib.pyplot as plt

    # def simulate(ssode, ydt0, t_end):

    #     time_history = []
    #     qdt0_history = []
    #     qdt1_history = []
    #     qdt2_history = []

    #     stepper = integrate.BDF(ssode, 0, ydt0, 10)
    #     while stepper.status == "running":
    #         y = stepper.y
    #         ydt1 = ssode(stepper.t, y)
    #         qdt0, qdt1 = y.reshape(2, -1)
    #         _, qdt2 = ydt1.reshape(2, -1)

    #         print(qdt0)

    #         time_history.append(stepper.t)
    #         qdt0_history.append(qdt0)
    #         qdt1_history.append(qdt1)
    #         qdt2_history.append(qdt2)
    #         stepper.step()

    #     return time_history, qdt0_history, qdt1_history, qdt2_history

    # # analytical_model = AnalyticalMassSpringDamper(10, 10, 10)
    # # true_t, true_qdt0, true_qdt1, true_qdt2 = simulate(
    # #     analytical_model.ssode, np.array([0, 0]), 10
    # # )

    # # mutlibody_model = MassSpringDamperTest()
    # # mutlibody_model._build_multibody_system(10, 10, 10)
    # # test_t, test_qdt0, test_qdt1, test_qdt2 = simulate(
    # #     mutlibody_model._evaluate_multibody_forward_dynamics, np.zeros((12,)), 10
    # # )

    # test = DoubleMassSpringDamperTest()
    # test.setUp()
    # # true_system = test._build_true_system(10, 10, 10)
    # # test_system = test._build_test_system(10, 10, 10)

    # test_t, test_qdt0, test_qdt1, test_qdt2 = simulate(
    #     test._evaluate_test_forward_dynamics, np.zeros((24,)), 10
    # )

    # true_t, true_qdt0, true_qdt1, true_qdt2 = simulate(
    #     test._evaluate_true_forward_dynamics, np.zeros((4,)), 10
    # )

    # plt.figure()
    # plt.plot(true_t, np.array(true_qdt0)[:, 0])
    # plt.plot(test_t, np.array(test_qdt0)[:, 2])
    # plt.grid()

    # plt.figure()
    # plt.plot(true_t, np.array(true_qdt0)[:, 1])
    # plt.plot(test_t, np.array(test_qdt0)[:, 2 + 6])
    # plt.grid()

    # plt.show()
