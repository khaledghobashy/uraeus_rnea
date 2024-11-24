import unittest
from typing import NamedTuple

from scipy import integrate
import numpy as np
import jax.numpy as jnp
import jax

from uraeus.rnea.bodies import RigidBodyData
from uraeus.rnea.joints import (
    JointConfigInputs,
    FreeJoint,
    TranslationalJoint,
)
from uraeus.rnea.topologies import MultiBodyTree, Model, emulate_free_joint_v2
from uraeus.rnea.spatial_algebra import (
    quaternion_to_dcm,
    quaternion_from_axis_angle,
    quaternion_from_euler_angles,
    transform_vector,
    SpatialPose,
    quaternion_to_yaw,
)
from uraeus.rnea.algorithms import get_qdt1_from_udt0, get_udt0_from_qdt1

np.set_printoptions(precision=3)


def get_permutation_matrix(final_index, initial_index):
    n = len(final_index)
    permutation_matrix = np.zeros((n, n), dtype=int)
    permutation_matrix[final_index, initial_index] = 1
    # permutation_matrix[permuted_vector, np.arange(n)] = 1
    return permutation_matrix


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


@unittest.skip
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

        model = Model(tree)

        model.fk = lambda x: np.array([0, 0, -x * k, 0, 0, 0])
        model.fc = lambda v: np.array([0, 0, -v * c, 0, 0, 0])

        self.multibody_system = model
        return model


@unittest.skip
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

        return ydt2_true, np.array(
            [ydt2_test[2], ydt2_test[2 + 6], ydt2_test[2 + 12], ydt2_test[2 + 6 + 12]]
        )

    def _evaluate_test_forward_dynamics(self, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        self.test_system.forces_map["m1"]["global"]["spring"] = (
            self.multibody_system.fk(qdt0[2])
        )
        self.test_system.forces_map["m1"]["global"]["damper"] = (
            self.multibody_system.fc(qdt1[2])
        )

        self.test_system.forces_map["m2"]["global"]["spring"] = (
            self.multibody_system.fk(qdt0[2 + 6])
        )
        self.test_system.forces_map["m2"]["global"]["damper"] = (
            self.multibody_system.fc(qdt1[2 + 6])
        )

        qdt2 = self.test_system.forward_dynamics_pass(qdt0, qdt1, np.zeros_like(qdt1))

        return np.array([*qdt1, *qdt2])

    def _evaluate_true_forward_dynamics(self, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        self.true_system.forces_map["m1"]["global"]["spring"] = (
            self.multibody_system.fk(qdt0[0])
        )
        self.true_system.forces_map["m1"]["global"]["damper"] = (
            self.multibody_system.fc(qdt1[0])
        )

        self.true_system.forces_map["m2"]["global"]["spring"] = (
            self.multibody_system.fk(qdt0[1])
        )
        self.true_system.forces_map["m2"]["global"]["damper"] = (
            self.multibody_system.fc(qdt1[1])
        )

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


class TestKinematics(unittest.TestCase):

    def setUp(self):
        self._build_multibody_system()
        self._build_true_system()

    def _build_multibody_system(self):
        tree = MultiBodyTree("one_body_system")

        m1_data = RigidBodyData(
            np.array([0, 0, 0.3]), np.array([1, 0, 0, 0]), 1, np.eye(3)
        )
        j1_data = JointConfigInputs(np.array([0, 0, 0.3]), np.array([0, 0, 1]), None)

        tree.add_joint("j1", "ground", "body", m1_data, FreeJoint, j1_data)

        model = Model(tree)

        self.multibody_system = model
        return model

    def _build_true_system(self):
        tree = MultiBodyTree("one_body_system")

        m1_data = RigidBodyData(
            np.array([0, 0, 0.3]), np.array([1, 0, 0, 0]), 1, np.eye(3)
        )
        j1_data = JointConfigInputs(np.array([0, 0, 0.3]), np.array([0, 0, 1]), None)

        # tree.add_joint("j1", "ground", "body", m1_data, FreeJoint, j1_data)
        tree = emulate_free_joint_v2(tree, "ground", "body", m1_data, j1_data)

        model = Model(tree)

        self.true_system = model
        return model

    @unittest.skip
    def test_kinematics_pos_variables(self):
        x = 100
        y = 50
        z = 0
        phi = 0
        theta = 0
        psi = np.radians(90)
        qdt0 = np.array([x, y, z, phi, theta, psi])
        qdt1 = np.array([1, 0, 0, 0, 0, 0])
        qdt2 = np.array([1, 0, 0, 0, 0, 0])
        bodies_kin, joints_kin = self.multibody_system.forward_kinematics_pass(
            qdt0, qdt1, qdt2
        )
        body_kin = self.multibody_system.get_body_kinematics("body", bodies_kin)
        print(f"psi = {np.rad2deg(psi)}")
        print(f"joints_kin = {joints_kin}")
        print(f"body_kin.p_GB.r = {body_kin.p_GB.r}")
        print(f"body_kin.p_BG.r = {body_kin.p_BG.r}")
        print(f"body_kin.p_GB.q = {quaternion_to_dcm(body_kin.p_GB.q)}")
        print(f"body_kin.p_BG.q = {quaternion_to_dcm(body_kin.p_BG.q)}")
        print(f"body_kin.v_B = {body_kin.v_B}")
        print(f"body_kin.v_G = {body_kin.v_G}")
        print(f"body_kin.p_GB.yaw = {quaternion_to_yaw(body_kin.p_GB.q)}")
        print(f"body_kin.p_BG.yaw = {quaternion_to_yaw(body_kin.p_BG.q)}")
        print(
            f"quaternion_to_yaw(body_kin.p_GB.q) = {quaternion_to_dcm(quaternion_from_euler_angles(0,0,quaternion_to_yaw(body_kin.p_GB.q)))}"
        )

        print("")

    @unittest.skip
    def test_inverse_dynamics_call(self):
        x = 100
        y = 50
        z = 0
        phi = 0
        theta = np.radians(0)
        psi = np.radians(90)
        qdt0 = np.array([x, y, z, phi, theta, psi])
        qdt1 = np.array([0, 0, 0, 0, 0, 0])
        qdt2 = np.array([1, 0, 0, 0, 0, 0])

        gravity_force_G = np.array([0, 0, -9.81, 0, 0, 0])
        applied_force_G = np.array([0, 0, 0, 0, 0, 0])
        applied_force_F = np.array([0, 0, 0, 0, 0, 0])
        tau = np.array([0, 0, 0, 0, 0, 0])

        self.multibody_system.forces_map["body"]["local"][
            "applied_force_F"
        ] = applied_force_F
        self.multibody_system.forces_map["body"]["global"][
            "applied_force_G"
        ] = applied_force_G

        id_results = self.multibody_system.inverse_dynamics_pass(qdt0, qdt1, qdt2)

        bodies_kin = id_results.bodies_kinematics
        joints_kin = id_results.joints_kinematics
        joints_frc = id_results.joints_forces

        body_kin = self.multibody_system.get_body_kinematics("body", bodies_kin)
        print(f"psi = {np.rad2deg(psi)}")
        # print(f"joints_kin = {joints_kin}")
        # print(f"body_kin.p_GB.r = {body_kin.p_GB.r}")
        # print(f"body_kin.p_BG.r = {body_kin.p_BG.r}")
        # print(f"body_kin.p_GB.q = {quaternion_to_dcm(body_kin.p_GB.q)}")
        # print(f"body_kin.p_BG.q = {quaternion_to_dcm(body_kin.p_BG.q)}")

        print("\nVelocities:\n----------------------------------------------")
        print(f"body_kin.v_B = {body_kin.v_B}")
        print(f"body_kin.v_G = {body_kin.v_G}")
        # print(f"body_kin.p_GB.yaw = {quaternion_to_yaw(body_kin.p_GB.q)}")
        # print(f"body_kin.p_BG.yaw = {quaternion_to_yaw(body_kin.p_BG.q)}")
        # print(
        #     f"quaternion_to_yaw(body_kin.p_GB.q) = {quaternion_to_dcm(quaternion_from_euler_angles(0,0,quaternion_to_yaw(body_kin.p_GB.q)))}"
        # )

        print("\nAccelerations:\n----------------------------------------------")
        print(f"body_kin.a_B = {body_kin.a_B}")
        print(f"body_kin.a_G = {body_kin.a_G}")

        print(
            "\nForces Inverse Dynamics:\n----------------------------------------------"
        )
        print(f"joints_frc = {joints_frc}")
        print(f"id_results.tau = {id_results.tau}")

        print("")

    # @unittest.skip
    def test_joint_generalized_forces(self):
        x = 0
        y = 0
        z = 0
        phi = 0
        theta = np.radians(0)
        psi = np.radians(45)
        qdt0 = np.array([x, y, z, phi, theta, psi])
        qdt1 = np.array([0, 0, 0, 0, 0, 0])
        # qdt2 = np.array([1, 0, 0, 0, 0, 0])

        gravity_force_G = np.array([0, 0, -0, 0, 0, 0])
        applied_force_G = np.array([0, 0, 0, 0, 0, 0])
        applied_force_F = np.array([0, 0, 0, 0, 0, 0])
        tau = np.array([10, 0, 0, 0, 0, 0])

        true_system = self.true_system
        test_system = self.multibody_system

        forces_map = self.true_system.forces_map

        forces_map["body"]["local"]["applied_force_F"] = applied_force_F
        forces_map["body"]["global"]["applied_force_G"] = applied_force_G
        forces_map["body"]["global"]["gravity"] = gravity_force_G

        Q = get_permutation_matrix(np.array([5, 4, 3, 0, 1, 2]), np.arange(6))

        # print("\nForward pass (True System)\n-------------\n")
        # self._evaluate_forward_pass(
        #     true_system, Q.T @ qdt0, Q.T @ qdt1, forces_map, Q.T @ tau
        # )

        # print("\nForward pass (Test System)\n-------------\n")
        # self._evaluate_forward_pass(test_system, qdt0, qdt1, forces_map, tau)

        self._do_time_stepping(test_system, qdt0, qdt1, forces_map, tau)

    def _evaluate_forward_pass(self, model: Model, qdt0, qdt1, forces_map, tau):
        model.forces_map.update(**forces_map)

        qdt2_fd = model.forward_dynamics_pass(qdt0, qdt1, tau)
        print("\n------\n")

        id_results = model.inverse_dynamics_pass(qdt0, qdt1, qdt2_fd)

        bodies_kin = id_results.bodies_kinematics
        joints_kin = id_results.joints_kinematics
        joints_frc = id_results.joints_forces
        generalized_forces = id_results.tau

        body_kin = model.get_body_kinematics("body", bodies_kin)

        def forces_func(model, qdt0, qdt1, qdt2, t):
            return tau, forces_map

        print(f"joints_kin = {joints_kin}")
        print(f"body_kin.p_GB.r = {body_kin.p_GB.r}")
        print(f"body_kin.p_BG.r = {body_kin.p_BG.r}")
        print(f"body_kin.p_GB.q = {quaternion_to_dcm(body_kin.p_GB.q)}")
        print(f"body_kin.p_BG.q = {quaternion_to_dcm(body_kin.p_BG.q)}")

        print("\nVelocities:\n----------------------------------------------")
        print(f"body_kin.v_B = {body_kin.v_B}")
        print(f"body_kin.v_G = {body_kin.v_G}")
        # print(f"body_kin.p_GB.yaw = {quaternion_to_yaw(body_kin.p_GB.q)}")
        # print(f"body_kin.p_BG.yaw = {quaternion_to_yaw(body_kin.p_BG.q)}")
        # print(
        #     f"quaternion_to_yaw(body_kin.p_GB.q) = {quaternion_to_dcm(quaternion_from_euler_angles(0,0,quaternion_to_yaw(body_kin.p_GB.q)))}"
        # )

        print("\nAccelerations:\n----------------------------------------------")
        print(f"body_kin.a_B = {body_kin.a_B}")
        print(f"body_kin.a_G = {body_kin.a_G}")

        print(
            "\nForces Inverse Dynamics:\n----------------------------------------------"
        )
        print(f"joints_frc = {joints_frc}")
        print(f"generalized_forces = {generalized_forces}")

        print(
            "\nAccelerations Forward Dynamics:\n----------------------------------------------"
        )
        print(f"generalized_accelerations = {qdt2_fd}")
        print(f"model.ssode = {model.ssode(0, np.array([*qdt0, *qdt1]), forces_func)}")

        # print(
        #     "get_qdt1_from_udt0 = ", get_qdt1_from_udt0(model.tree_data, qdt0, qdt2_fd)
        # )

        print("")

    def _do_time_stepping(self, model: Model, qdt0, qdt1, forces_map, tau):
        model.forces_map.update(**forces_map)

        def forces_func(model, qdt0, qdt1, qdt2, t):
            return tau, forces_map

        ydt0 = np.array([*qdt0, *qdt1])

        res = integrator(
            model, ydt0, (0, 1), lambda t, ydt0: model.ssode(t, ydt0, forces_func)
        )

        # print("\nVelocities:\n----------------------------------------------")
        # print(f"body_kin.v_B = {body_kin.v_B}")
        # print(f"body_kin.v_G = {body_kin.v_G}")

        # print("\nAccelerations:\n----------------------------------------------")
        # print(f"body_kin.a_B = {body_kin.a_B}")
        # print(f"body_kin.a_G = {body_kin.a_G}")

        # print(
        #     "\nForces Inverse Dynamics:\n----------------------------------------------"
        # )
        # print(f"joints_frc = {joints_frc}")
        # print(f"generalized_forces = {generalized_forces}")

        # print(
        #     "\nAccelerations Forward Dynamics:\n----------------------------------------------"
        # )
        # print(f"generalized_accelerations = {qdt2_fd}")
        # print(f"model.ssode = {model.ssode(0, np.array([*qdt0, *qdt1]), forces_func)}")

        # # print(
        # #     "get_qdt1_from_udt0 = ", get_qdt1_from_udt0(model.tree_data, qdt0, qdt2_fd)
        # # )

        # print("")

    @unittest.skip
    def test_kinematics_pos(self):
        l = 5
        # psi = np.linspace(0, 2 * np.pi, 100)
        # x = l * np.cos(psi)
        # y = l * np.sin(psi)
        # z = 0
        # phi = 0
        # theta = 0
        # # psi = np.radians(45)
        # qdt0 = np.array([x, y, z, phi, theta, psi])
        # qdt1 = np.array([1, 0, 0, 0, 0, 0])
        # qdt2 = np.array([1, 0, 0, 0, 0, 0])

        # x_axis = np.array([1, 0, 0])
        # y_axis = np.array([0, 1, 0])
        # z_axis = np.array([0, 0, 1])

        # q1 = quaternion_from_axis_angle(psi, np.array([0, 0, 1]))
        # q2 = quaternion_from_euler_angles(0, 0, psi)

        for psi in np.linspace(0, 2 * np.pi, 100):
            x = l * np.cos(psi)
            y = l * np.sin(psi)
            z = 0
            phi = 0
            theta = 0
            qdt0 = np.array([x, y, z, phi, theta, psi])
            qdt1 = np.array([1, 0, 0, 0, 0, 0])
            qdt2 = np.array([1, 0, 0, 0, 0, 0])
            bodies_kin, joints_kin = self.multibody_system.forward_kinematics_pass(
                qdt0, qdt1, qdt2
            )
            body_kin = self.multibody_system.get_body_kinematics("body", bodies_kin)
            print(f"psi = {np.rad2deg(psi)}")
            print(f"joints_kin = {joints_kin}")
            print(f"body_kin.p_GB.r = {body_kin.p_GB.r}")
            print(f"body_kin.p_BG.r = {body_kin.p_BG.r}")
            print(f"body_kin.p_GB.q = {quaternion_to_dcm(body_kin.p_GB.q)}")
            print(f"body_kin.p_BG.q = {quaternion_to_dcm(body_kin.p_BG.q)}")
            print(f"body_kin.v_B = {body_kin.v_B}")
            print(f"body_kin.v_G = {body_kin.v_G}")
            print("")

        # print(f"q1 = {q1}")
        # print(f"q2 = {q2}")
        # print(f"q1_dcm = \n{quaternion_to_dcm(q1)}")
        # print(f"q2_dcm = \n{quaternion_to_dcm(q2)}")
        # print(f"q1 @ x_axis = {transform_vector(q1, x_axis)}")

        # bodies_kin, joints_kin = self.multibody_system.forward_kinematics_pass(
        #     qdt0, qdt1, qdt2
        # )
        # body_kin = self.multibody_system.get_body_kinematics("body", bodies_kin)

        # print(f"\nqdt0 = {qdt0}")
        # print(f"body_kin.p_GB.r = {body_kin.p_GB.r}")
        # print(f"body_kin.p_BG.r = {body_kin.p_BG.r}")
        # print(f"body_kin.p_GB.dcm = \n{quaternion_to_dcm(body_kin.p_GB.q)}")
        # print(f"body_kin.p_BG.dcm = \n{quaternion_to_dcm(body_kin.p_BG.q)}")
        # # print(f"qdt1 = {qdt1}")
        # # print(f"qdt2 = {qdt2}")
        # print(f"body_kin = {body_kin}")

        # np.testing.assert_allclose(body_kin.p_GB.r, qdt0[:3])


@unittest.skip
class TestSpatialPose(unittest.TestCase):

    def test_pose_transformation(self):
        print("")
        angle = np.radians(45)
        axis = np.array([0, 0, 1])
        q1 = quaternion_from_axis_angle(angle, axis)

        p_0 = SpatialPose.Identity()

        p_01 = SpatialPose(np.array([5, 0, 0]), p_0.q)
        p_12 = SpatialPose(np.array([0, 5, 0]), p_0.q)
        p_23 = SpatialPose(np.array([0, 0, 5]), p_0.q)

        p_03 = p_23 @ p_12 @ p_01
        print(f"p_03 = {p_03}")
        desired_r = np.array([5, 5, 5])
        actual_r = p_03.r

        np.testing.assert_allclose(actual_r, desired_r)

    def test_combined_transformation(self):
        """
                                            x
                                              \
                                               \       
                                                /f1
                                               / 
                                               y

                                     x
                                       \ 
                                        \       
                                        /f2&3
                                       / 
                                      y

                                                 ^ x
                                                 |
                                                 |
                                        y <------f0
        """
        print("")
        angle = np.radians(45)
        axis = np.array([0, 0, 1])
        q1 = quaternion_from_axis_angle(angle, axis)

        # p_0 is the identity pose, representing the fixed inertial-frame (ground)
        p_0 = SpatialPose.Identity()

        # P_01 -> applies clockwise rotation of 45 deg around z-axis and
        # translation of 5 units along parent (p_0) x-axis.
        p_01 = SpatialPose(np.array([5, 0, 0]), q1)

        # P_12 -> applies no rotation, but applies translation of
        # 5 units along parent (p_01) y-axis.
        p_12 = SpatialPose(np.array([0, 5, 0]), p_0.q)

        # P_23 -> applies no rotation, but applies translation of
        # 5 units along parent (p_12) z-axis.
        p_23 = SpatialPose(np.array([0, 0, 5]), p_0.q)

        # Final translation should be:
        # - translate 5 units along p_0 x-axis
        # - translate another 5 units along p_01 y-axis == 5 * p_01.j_direction
        #   == 5 * [-np.sin(angle), p.cos(angle)]
        # - translate another 5 unit units along p_12 z-axis
        # --> final position is [5 + (5 * -np.sin(angle)), (5 * np.cos(angle)), 5]

        # Final orientation should be:
        # - rotate 45 deg around p_0 z-axis
        # - apply no more rotation
        # --> final orientation of frame 3 relative to 0 is 45 deg around
        #     p_0 z-axis

        p_03 = p_23 @ p_12 @ p_01

        desired_r = np.array([5 + (5 * -np.sin(angle)), (5 * np.cos(angle)), 5])
        actual_r = p_03.r

        desired_q = q1
        actual_q = p_03.q

        print(f"p_03 = {p_03}")
        print(f"desired_r = {desired_r}")

        np.testing.assert_allclose(actual_r, desired_r)
        np.testing.assert_allclose(actual_q, desired_q)


class IntgRes(NamedTuple):
    time_history: np.ndarray
    qdt0_history: np.ndarray
    qdt1_history: np.ndarray
    qdt2_history: np.ndarray


def integrator(
    model: Model, ydt0: np.ndarray, t_span: tuple[float, float], ssode
) -> IntgRes:
    t0, t_bound = t_span
    # ssode = model.ssode
    # stepper = integrate.RK45(ssode, t0, ydt0, t_bound)
    stepper = integrate.BDF(ssode, t0, ydt0, t_bound)

    time_history = []
    qdt0_history = []
    qdt1_history = []
    qdt2_history = []

    while stepper.status == "running":
        stepper.step()
        y = stepper.y
        ydt1 = ssode(stepper.t, y)
        qdt0, udt0 = y.reshape(2, -1)
        qdt1, udt1 = ydt1.reshape(2, -1)

        # # udt0 = get_udt0_from_qdt1(model.tree_data, qdt0, qdt1)
        # print(f"ydt0 = {y}")
        # print(f"ydt1 = {ydt1}\n")

        # qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
        #     model,
        #     (qdt0_fd, qdt1_fd, qdt2_fd),
        #     steering_function(construct_inputs_dict(stepper.t)["steering_input"]),
        # )

        log_output_to_consol(model.tree_data.qdt0_names, qdt0, udt0, udt1, stepper.t)

        time_history.append(stepper.t)
        qdt0_history.append(qdt0)
        qdt1_history.append(udt0)
        qdt2_history.append(udt1)

    # log_output_to_consol(model.tree_data.qdt0_names, qdt0, udt0, udt1, stepper.t)

    res = IntgRes(
        time_history=np.array(time_history),
        qdt0_history=np.array(qdt0_history),
        qdt1_history=np.array(qdt1_history),
        qdt2_history=np.array(qdt2_history),
    )

    return res


def log_output_to_consol(
    coordinates_map, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray, t: float
):
    # states = {
    #     "time": t,
    #     "z": qdt0[coordinates_map.free_joint.z],
    #     "pitch": qdt0[coordinates_map.free_joint.theta],
    #     "acc_x": qdt2[coordinates_map.free_joint.x] / 9.81,
    #     "acc_y": qdt2[coordinates_map.free_joint.y] / 9.81,
    #     "velx": qdt1[coordinates_map.free_joint.x] * 3.6,
    #     "vely": qdt1[coordinates_map.free_joint.y] * 3.6,
    #     "yaw_dt0": qdt0[coordinates_map.free_joint.psi],
    #     "yaw_dt1": qdt1[coordinates_map.free_joint.psi],
    #     "yaw_dt2": qdt2[coordinates_map.free_joint.psi],
    # }

    states = {
        "time": t,
        "x": qdt0[0],
        "y": qdt0[1],
        "z": qdt0[2],
        "pitch": qdt0[4],
        "acc_x": qdt2[0],
        "acc_y": qdt2[1],
        "velx": qdt1[0],
        "vely": qdt1[1],
        "yaw_dt0": qdt0[5],
        "yaw_dt1": qdt1[5],
        "yaw_dt2": qdt2[5],
    }

    values = [f"{i: 08.5f}" for i in states.values()]
    fields = [f"{i: >{len(v)}}" for i, v in zip(states.keys(), values)]
    # print("|".join([f"{i:>8}" for i in states.keys()]))
    # print("|".join([f"{i: 08.5f}" for i in states.values()]))
    print("|".join(fields))
    print("|".join(values))
    print("")


if __name__ == "__main__":

    unittest.main()
