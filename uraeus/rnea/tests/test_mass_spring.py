import unittest

import numpy as np
import jax.numpy as jnp
import jax

from uraeus.rnea.bodies import RigidBodyData
from uraeus.rnea.joints import JointConfigInputs, TranslationalJoint
from uraeus.rnea.topologies import MultiBodyTree
from uraeus.rnea.multibody_models import Model


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

        qdt2_true = self._evaluate_analytical_forward_dynamics(t, ydt0)
        qdt2_test = self._evaluate_multibody_forward_dynamics(t, ydt0)

        return qdt2_true, qdt2_test

    def _evaluate_analytical_forward_dynamics(self, t, ydt0):
        return self.analytical_model.ssode(t, ydt0)

    def _evaluate_multibody_forward_dynamics(self, t, ydt0):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        self.multibody_system.forces_map["m1"]["global"]["spring"] = (
            self.multibody_system.fk(qdt0[0])
        )
        self.multibody_system.forces_map["m1"]["global"]["damper"] = (
            self.multibody_system.fc(qdt1[0])
        )

        qdt2 = self.multibody_system.forward_dynamics_call(
            qdt0, qdt1, np.zeros_like(qdt1)
        )

        return np.array([*qdt1, *qdt2])

    def _build_multibody_system(self, m1, k, c) -> Model:
        tree = MultiBodyTree("mass_spring")

        m1_data = RigidBodyData(
            np.array([0, 0, 1]), np.array([1, 0, 0, 0]), m1, 0 * np.eye(3)
        )
        j1_data = JointConfigInputs(np.array([0, 0, 0]), np.array([0, 0, 1]), None)

        tree.add_joint("j1", "ground", "m1", m1_data, TranslationalJoint, j1_data)

        model = Model(tree)

        model.fk = lambda x: np.array([0, 0, -x * k, 0, 0, 0])
        model.fc = lambda v: np.array([0, 0, -v * c, 0, 0, 0])

        self.multibody_system = model
        return model


if __name__ == "__main__":

    unittest.main()
    import scipy.integrate as integrate
    import matplotlib.pyplot as plt

    def simulate(ssode, ydt0, t_end):

        time_history = []
        qdt0_history = []
        qdt1_history = []
        qdt2_history = []

        ydt0 = np.array([0, 0])
        stepper = integrate.BDF(ssode, 0, ydt0, 10)
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

    analytical_model = AnalyticalMassSpringDamper(10, 10, 10)
    true_t, true_qdt0, true_qdt1, true_qdt2 = simulate(
        analytical_model.ssode, np.array([0, 0]), 10
    )

    mutlibody_model = MassSpringDamperTest()
    mutlibody_model._build_multibody_system(10, 10, 10)
    test_t, test_qdt0, test_qdt1, test_qdt2 = simulate(
        mutlibody_model._evaluate_multibody_forward_dynamics, np.array([0, 0]), 10
    )

    plt.figure()
    plt.plot(true_t, true_qdt0)
    plt.plot(test_t, test_qdt0)
    plt.grid()
    plt.show()
