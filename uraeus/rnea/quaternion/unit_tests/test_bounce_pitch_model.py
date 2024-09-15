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

np.set_printoptions(precision=3)


class AnalyticalBouncePitchModel(object):

    def __init__(self, mass, I, k1, k2, c1, c2, l, l1):

        self.mass = mass
        self.I = I
        self.k1 = k1
        self.c1 = c1
        self.k2 = k2
        self.c2 = c2
        self.l = l
        self.l1 = l1
        self.l2 = l - l1

    def ssode(self, t, y):
        """ODE for mass-spring-damper system

        Parameters
        ----------
        t : time
        """
        qdt0, qdt1 = np.split(y, 2)

        z_dt0, theta_dt0 = qdt0
        z_dt1, theta_dt1 = qdt1

        m2 = (self.l1 / self.l) * self.mass
        m1 = self.mass - m2

        F1 = (
            -self.k1 * (z_dt0 - self.l1 * theta_dt0)
            - self.c1 * (z_dt1 - self.l1 * theta_dt1)
            + (0.9 * m1 * 9.81)
        )
        F2 = (
            -self.k2 * (z_dt0 + self.l2 * theta_dt0)
            - self.c2 * (z_dt1 + self.l2 * theta_dt1)
            + (0.8 * m2 * 9.81)
        )
        Fg = -self.mass * 9.81

        T1 = F1 * self.l1
        T2 = F2 * self.l2

        z_dt2 = (1 / self.mass) * -(-F1 - F2 - Fg)
        theta_dt2 = (1 / self.I) * (-T1 + T2)
        qdt2 = np.array([z_dt2, theta_dt2])

        print("Analytical:")
        print(f"    F1 = {F1}")
        print(f"    F2 = {F2}")
        print(f"    T1 = {T1}")
        print(f"    T2 = {T2}")
        print(f"    qdt2 = {qdt2}")

        return np.array([*qdt1, *qdt2])


class BouncePitchModelTest(unittest.TestCase):

    multibody_system: Model

    def setUp(self):

        self.mass = 250
        self.m1 = 0.45 * self.mass
        self.m2 = self.mass - self.m1
        self.l = 1.6
        self.l2 = (self.m1 / self.mass) * self.l
        self.l1 = self.l - self.l2

        # wn1 = (1 / (2 * np.pi)) * 1
        self.wn1 = 2
        self.k1 = self.wn1**2 * self.m1
        self.eta1 = 0.5
        self.c1 = self.eta1 * (2 * self.m1 * self.wn1)

        # wn2 = (1 / (2 * np.pi)) * 1
        self.wn2 = 2
        self.k2 = self.wn2**2 * self.m2
        self.eta2 = 0.5
        self.c2 = self.eta2 * (2 * self.m2 * self.wn2)

        self.I_theta = 150

        self.analytical_model = AnalyticalBouncePitchModel(
            self.mass,
            self.I_theta,
            self.k1,
            self.k2,
            self.c1,
            self.c2,
            self.l,
            self.l1,
        )

        self.multibody_system = self._build_multibody_system(
            self.mass,
            self.I_theta,
            self.k1,
            self.k2,
            self.c1,
            self.c2,
            self.l,
            self.l1,
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

        qdt2_true = self._evaluate_analytical_forward_dynamics(t, ydt0)
        qdt2_test = self._evaluate_multibody_forward_dynamics(t, ydt0)

        print("true_sol = ", qdt2_true)
        print("test_sol = ", qdt2_test)
        print("")
        return qdt2_true, qdt2_test

    def _evaluate_true_forward_dynamics(self, t, ydt0):
        return self.analytical_model.ssode(t, ydt0)

    def _evaluate_test_forward_dynamics(self, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)

        f1_x = qdt0[2] - self.l1 * qdt0[4]
        f1_v = qdt1[2] - self.l1 * qdt1[4]

        f2_x = qdt0[2] + self.l2 * qdt0[4]
        f2_v = qdt1[2] + self.l2 * qdt1[4]

        # print(
        #     f"f1_x = {f1_x}, qdt0[2] = {qdt0[2]}, self.l1 * qdt0[4] = {self.l1 * qdt0[4]}"
        # )
        # print(
        #     f"f2_x = {f2_x}, qdt0[2] = {qdt0[2]}, self.l2 * qdt0[4] = {self.l2 * qdt0[4]}"
        # )

        F1 = self.multibody_system.F1(f1_x, f1_v, (0.9 * self.m1 * 9.81))
        F2 = self.multibody_system.F2(f2_x, f2_v, (0.8 * self.m2 * 9.81))

        self.multibody_system.forces_map["m1"]["F1"] = F1
        self.multibody_system.forces_map["m1"]["F2"] = F2

        qdt2 = self.multibody_system.forward_dynamics_pass(
            qdt0, qdt1, np.zeros_like(qdt1)
        )
        bodies_kinematics, joints_kinematics = (
            self.multibody_system.forward_kinematics_pass(qdt0, qdt1, qdt2)
        )
        m1_kin = self.multibody_system.get_body_kinematics("m1", bodies_kinematics)

        print("MultiBody:")
        print(f"    F1 = {F1[2]}")
        print(f"    F2 = {F2[2]}")
        print(f"    T1 = {F1[4]}")
        print(f"    T2 = {F2[4]}")
        print(f"    qdt2 = {qdt2}")
        # print(f"    m1.p_GB = {m1_kin.p_GB}")
        print(f"    m1.a_G = {m1_kin.a_G}")
        # print(f"    m1.a_B = {m1_kin.a_B}")

        return np.array([*qdt1, *qdt2])

    def _build_multibody_system(self, mass, I, k1, k2, c1, c2, l, l1) -> Model:
        tree = MultiBodyTree("bounce_pitch")

        m1_data = RigidBodyData(
            np.array([0, 0, 0]),
            np.array([1, 0, 0, 0]),
            mass,
            # np.diag([1e5, I, 1e5]),
            np.diag([1, I, 1]),
        )
        j1_data = JointConfigInputs(np.array([0, 0, 0]), np.array([0, 0, 1]), None)

        tree.add_joint("j1", "ground", "m1", m1_data, FreeJoint, j1_data)
        # tree = emulate_free_joint(tree, "ground", "m1", m1_data)

        model = Model(tree)

        def F1(x, v, preload):
            fs = -k1 * x
            fd = -c1 * v
            f_total = fs + fd + preload
            # f_total = fs + preload
            return np.array([0, 0, f_total, 0, -f_total * self.l1, 0])

        def F2(x, v, preload):
            fs = -k2 * x
            fd = -c2 * v
            f_total = fs + fd + preload
            return np.array([0, 0, f_total, 0, f_total * self.l2, 0])

        model.F1 = F1
        model.F2 = F2

        self.multibody_system = model
        return model

    # def _build_multibody_system_forces_function(self):

    #     def extarnal_forces(qdt0, qdt1):

    #         bodies_kinematics, joints_kinematics = (
    #             self.multibody_system.forward_kinematics_pass(qdt0, qdt1, 0 * qdt1)
    #         )
    #         m1_kin = self.multibody_system.get_body_kinematics("m1", bodies_kinematics)


if __name__ == "__main__":

    # unittest.main()
    import scipy.integrate as integrate
    import matplotlib.pyplot as plt

    def simulate(ssode, ydt0, t_end):

        time_history = []
        qdt0_history = []
        qdt1_history = []
        qdt2_history = []

        stepper = integrate.BDF(ssode, 0, ydt0, t_end)
        # stepper = integrate.DOP853(ssode, 0, ydt0, t_end)
        while stepper.status == "running":
            y = stepper.y
            ydt1 = ssode(stepper.t, y)
            qdt0, qdt1 = y.reshape(2, -1)
            _, qdt2 = ydt1.reshape(2, -1)

            print(stepper.t)
            print(qdt0)

            time_history.append(stepper.t)
            qdt0_history.append(qdt0)
            qdt1_history.append(qdt1)
            qdt2_history.append(qdt2)
            stepper.step()

        return time_history, qdt0_history, qdt1_history, qdt2_history

    # mass = 250
    # m1 = 0.45 * mass
    # m2 = mass - m1
    # l = 1.6
    # l2 = (m1 / mass) * l
    # l1 = l - l2

    # # wn1 = (1 / (2 * np.pi)) * 1
    # wn1 = 2
    # k1 = wn1**2 * m1
    # eta1 = 0.5
    # c1 = eta1 * (2 * m1 * wn1)

    # # wn2 = (1 / (2 * np.pi)) * 1
    # wn2 = 2
    # k2 = wn2**2 * m2
    # eta2 = 0.5
    # c2 = eta2 * (2 * m2 * wn2)

    # I_theta = 150

    # print(f"k1 = {k1}")
    # print(f"c1 = {c1}")
    # print(f"k2 = {k2}")
    # print(f"c2 = {c2}")

    # analytical_model = AnalyticalBouncePitchModel(
    #     mass,
    #     I_theta,
    #     k1,
    #     k1,
    #     c1,
    #     c1,
    #     l,
    #     l1,
    # )
    test = BouncePitchModelTest()
    test.setUp()

    z_dt0 = -0.5
    theta_dt0 = np.radians(7)
    z_dt1 = 2
    theta_dt1 = np.radians(20)

    test._evaluate_true_forward_dynamics(
        0, np.array([z_dt0, theta_dt0, z_dt1, theta_dt1])
    )
    test._evaluate_test_forward_dynamics(
        0, np.array([0, 0, z_dt0, 0, theta_dt0, 0, 0, 0, z_dt1, 0, theta_dt1, 0])
    )

    true_t, true_qdt0, true_qdt1, true_qdt2 = simulate(
        test._evaluate_true_forward_dynamics, np.array([0, 0, 0, 0]), 5
    )

    test_t, test_qdt0, test_qdt1, test_qdt2 = simulate(
        test._evaluate_test_forward_dynamics, np.zeros((12,)), 5
    )

    plt.figure()
    plt.plot(true_t, np.array(true_qdt0)[:, 0])
    plt.plot(true_t, np.rad2deg(np.array(true_qdt0)[:, 1]))
    plt.grid()

    plt.figure()
    plt.plot(test_t, np.array(test_qdt0)[:, 2])
    plt.plot(test_t, np.rad2deg(np.array(test_qdt0)[:, 4]))
    plt.grid()

    plt.show()
