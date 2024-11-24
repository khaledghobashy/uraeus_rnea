import unittest

import numpy as np
import jax.numpy as jnp
import jax

from uraeus.rnea.bodies import RigidBodyData
from uraeus.rnea.joints import JointConfigInputs, RevoluteJoint
from uraeus.rnea.multibody_models import MultiBodyTree, Model
from uraeus.rnea.algorithms import IDCallRes
from uraeus.rnea.models.analytic_double_pendulum import (
    AnalyticDoublePendulum,
)
from uraeus.rnea.tree_traversals import extract_mobilizer_forces


class DoublePendulumTest(unittest.TestCase):

    l1: float
    l2: float
    multibody_system: Model

    def setUp(self):

        self.l1 = 7
        self.l2 = 13

        self.m1 = 4
        self.m2 = 2

        self.theta1_dt0_f = lambda t: 2 * jnp.sin(t)
        self.theta2_dt0_f = lambda t: 1.5 * jnp.sin(2 * t)

        self.theta1_dt1_f = jax.jacfwd(self.theta1_dt0_f)
        self.theta2_dt1_f = jax.jacfwd(self.theta2_dt0_f)
        self.theta1_dt2_f = jax.jacfwd(self.theta1_dt1_f)
        self.theta2_dt2_f = jax.jacfwd(self.theta2_dt1_f)

        self.multibody_system = self._build_multibody_system(
            self.l1, self.l2, self.m1, self.m2
        )
        self.analytic_system = AnalyticDoublePendulum(
            self.l1, self.l2, self.m1, self.m2
        )

    def test_forward_kinematics(self):
        time_array = np.linspace(0, 2 * np.pi, 100)

        test_sol, true_sol = zip(
            *[self._evaluate_forward_kinematics(t) for t in time_array]
        )

        test_pose_G, test_vel_G, test_acc_G = zip(*test_sol)
        true_pose_G, true_vel_G, true_acc_G = zip(*true_sol)

        np.testing.assert_almost_equal(test_pose_G, true_pose_G)
        np.testing.assert_almost_equal(test_vel_G, true_vel_G)
        np.testing.assert_almost_equal(np.array(true_acc_G), test_acc_G)

    def test_inverse_dynamics(self):
        time_array = np.linspace(0, 2 * np.pi, 100)

        test_sol, true_sol = zip(
            *[self._evaluate_inverse_dynamics(t) for t in time_array]
        )

        np.testing.assert_almost_equal(true_sol, test_sol)

    def test_forward_dynamics(self):
        time_array = np.linspace(0, 2 * np.pi, 100)

        true_sol, test_sol = zip(
            *[self._evaluate_forward_dynamics(t) for t in time_array]
        )
        np.testing.assert_almost_equal(true_sol, test_sol)

    def _evaluate_forward_dynamics(self, t):

        qdt0 = np.array([self.theta1_dt0_f(t), self.theta2_dt0_f(t)])
        qdt1 = np.array([self.theta1_dt1_f(t), self.theta2_dt1_f(t)])
        qdt2 = np.array([self.theta1_dt2_f(t), self.theta2_dt2_f(t)])

        res: IDCallRes = self.multibody_system.inverse_dynamics_pass(qdt0, qdt1, qdt2)
        tau = res.tau

        qdt2_test = self.multibody_system.forward_dynamics_call(qdt0, qdt1, tau)

        return qdt2, qdt2_test

    def _evaluate_forward_kinematics(self, t):
        test_sol = self._evaluate_multibody_forward_kinematics(t)
        true_sol = self._evaluate_analytical_forward_kinematics(t)
        return test_sol, true_sol

    def _evaluate_inverse_dynamics(self, t):
        test_sol = self._evaluate_multibody_inverse_dynamics(t)
        true_sol = self._evaluate_analytical_inverse_dynamics(t)
        return test_sol, true_sol

    def _evaluate_multibody_forward_kinematics(self, t):

        qdt0 = np.array([self.theta1_dt0_f(t), self.theta2_dt0_f(t)])
        qdt1 = np.array([self.theta1_dt1_f(t), self.theta2_dt1_f(t)])
        qdt2 = np.array([self.theta1_dt2_f(t), self.theta2_dt2_f(t)])

        bodies_kinematics, _ = self.multibody_system.forward_kinematics_pass(
            qdt0, qdt1, qdt2
        )

        l1_kin = self.multibody_system.get_body_kinematics("l1", bodies_kinematics)
        l2_kin = self.multibody_system.get_body_kinematics("l2", bodies_kinematics)

        l1_r_y = l1_kin.p_GB.r[1]
        l1_r_z = l1_kin.p_GB.r[2]

        l2_r_y = l2_kin.p_GB.r[1]
        l2_r_z = l2_kin.p_GB.r[2]

        l1_v_y = l1_kin.v_G[1]
        l1_v_z = l1_kin.v_G[2]

        l2_v_y = l2_kin.v_G[1]
        l2_v_z = l2_kin.v_G[2]

        l1_a_y = l1_kin.a_G[1]
        l1_a_z = l1_kin.a_G[2]

        l2_a_y = l2_kin.a_G[1]
        l2_a_z = l2_kin.a_G[2]

        sys_kinematics = (
            ((l1_r_y, l1_r_z), (l2_r_y, l2_r_z)),
            ((l1_v_y, l1_v_z), (l2_v_y, l2_v_z)),
            ((l1_a_y, l1_a_z), (l2_a_y, l2_a_z)),
        )

        return sys_kinematics

    def _evaluate_analytical_forward_kinematics(self, t):
        qdt0 = np.array([self.theta1_dt0_f(t), self.theta2_dt0_f(t)])
        qdt1 = np.array([self.theta1_dt1_f(t), self.theta2_dt1_f(t)])
        qdt2 = np.array([self.theta1_dt2_f(t), self.theta2_dt2_f(t)])

        return self.analytic_system.evaluate_kinematics(qdt0, qdt1, qdt2)

    def _evaluate_analytical_inverse_dynamics(self, t):
        qdt0 = np.array([self.theta1_dt0_f(t), self.theta2_dt0_f(t)])
        qdt1 = np.array([self.theta1_dt1_f(t), self.theta2_dt1_f(t)])
        qdt2 = np.array([self.theta1_dt2_f(t), self.theta2_dt2_f(t)])
        forces = self.analytic_system.evaluate_inverse_dynamics(qdt0, qdt1, qdt2)
        return forces

    def _evaluate_multibody_inverse_dynamics(self, t):
        qdt0 = np.array([self.theta1_dt0_f(t), self.theta2_dt0_f(t)])
        qdt1 = np.array([self.theta1_dt1_f(t), self.theta2_dt1_f(t)])
        qdt2 = np.array([self.theta1_dt2_f(t), self.theta2_dt2_f(t)])

        model = self.multibody_system
        inverse_dyn_res = model.inverse_dynamics_pass(qdt0, qdt1, qdt2)

        taus, bodies_kinematics, joints_kinematics, joints_forces = inverse_dyn_res
        joints_frames = (
            model.topology.joints["j1"].joint_data.frames,
            model.topology.joints["j2"].joint_data.frames,
        )
        mobilizer_forces = extract_mobilizer_forces(
            joints_forces,
            joints_frames,
            joints_kinematics,
            bodies_kinematics[1:],
        )

        j1_forces, j2_forces = mobilizer_forces
        j1_fi_S, j1_fc_S, j1_fa_S, j1_fc_G, j1_tau = j1_forces
        j2_fi_S, j2_fc_S, j2_fa_S, j2_fc_G, j2_tau = j2_forces

        forces = (
            -np.array([*j1_fc_G[1:3], j1_tau[0]]),
            -np.array([*j2_fc_G[1:3], j2_tau[0]]),
        )

        return forces

    def _evaluate_multibody_forward_dynamics(self, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        qdt2 = self.multibody_system.forward_dynamics_pass(
            qdt0, qdt1, np.zeros_like(qdt1)
        )

        return np.array([*qdt1, *qdt2])

    def _build_multibody_system(self, l1, l2, m1, m2) -> Model:
        tree = MultiBodyTree("pendulum")

        l1_data = RigidBodyData(
            np.array([0, 0, -l1]), np.array([1, 0, 0, 0]), m1, 0 * np.eye(3)
        )
        l2_data = RigidBodyData(
            np.array([0, 0, -l2 - l1]), np.array([1, 0, 0, 0]), m2, 0 * np.eye(3)
        )
        j1_data = JointConfigInputs(np.array([0, 0, 0]), np.array([1, 0, 0]), None)
        j2_data = JointConfigInputs(np.array([0, 0, -l1]), np.array([1, 0, 0]), None)

        tree.add_joint("j1", "ground", "l1", l1_data, RevoluteJoint, j1_data)
        tree.add_joint("j2", "l1", "l2", l2_data, RevoluteJoint, j2_data)

        return Model(tree)


if __name__ == "__main__":

    unittest.main()
    import scipy.integrate as integrate
    import matplotlib.pyplot as plt

    def evaluate_multibody_forward_dynamics(model, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        qdt2 = model.forward_dynamics_pass(qdt0, qdt1, np.zeros_like(qdt1))
        return np.array([*qdt1, *qdt2])

    def simulate(ssode, ydt0, t_end):

        time_history = []
        qdt0_history = []
        qdt1_history = []
        qdt2_history = []

        stepper = integrate.BDF(ssode, 0, ydt0, t_end)
        while stepper.status == "running":
            y = stepper.y
            ydt1 = ssode(stepper.t, y)
            qdt0, qdt1 = y.reshape(2, -1)
            _, qdt2 = ydt1.reshape(2, -1)

            time_history.append(stepper.t)
            qdt0_history.append(qdt0)
            qdt1_history.append(qdt1)
            qdt2_history.append(qdt2)
            stepper.step()

        return time_history, qdt0_history, qdt1_history, qdt2_history

    multibody_model = DoublePendulumTest()._build_multibody_system(5, 5, 1, 1)
    test_t, test_qdt0, test_qdt1, test_qdt2 = simulate(
        lambda t, y: evaluate_multibody_forward_dynamics(multibody_model, t, y),
        np.array([np.pi / 2, 0, 0, 0]),
        5,
    )

    bodies_kinematics = [
        multibody_model.forward_kinematics_pass(
            test_qdt0[i], test_qdt1[i], test_qdt2[i]
        )[0]
        for i in range(len(test_t))
    ]

    l1_kin = [
        multibody_model.get_body_kinematics("l1", bodies)
        for bodies in bodies_kinematics
    ]
    l2_kin = [
        multibody_model.get_body_kinematics("l2", bodies)
        for bodies in bodies_kinematics
    ]

    l1_p = [kin.p_GB for kin in l1_kin]
    l2_p = [kin.p_GB for kin in l2_kin]
    l1_v = [kin.v_G for kin in l1_kin]
    l2_v = [kin.v_G for kin in l2_kin]

    plt.figure()
    plt.plot(test_t, [q[0] for q in test_qdt0])
    plt.plot(test_t, [q[1] + q[0] for q in test_qdt0])
    plt.grid()

    plt.show()
