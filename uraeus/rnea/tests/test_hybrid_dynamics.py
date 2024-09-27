import unittest
import logging
import typing

import numpy as np
import jax.numpy as jnp
import jax

from uraeus.utils.logging import construct_logger
from uraeus.rnea.bodies import RigidBodyData
from uraeus.rnea.joints import JointConfigInputs, RevoluteJoint
from uraeus.rnea.topologies import MultiBodyTree, HybridModel

logger = construct_logger(__name__, logging.DEBUG)


class DoublePendulum(object):

    def __init__(self):
        tree = MultiBodyTree("double_pendulum")

        l1_length = 1
        l1_mass = 10

        l2_length = 1
        l2_mass = 10

        l1_data = RigidBodyData(
            np.array([0, 0, -l1_length]), np.array([1, 0, 0, 0]), l1_mass, 0 * np.eye(3)
        )
        l2_data = RigidBodyData(
            np.array([0, 0, -l2_length - l1_length]),
            np.array([1, 0, 0, 0]),
            l2_mass,
            0 * np.eye(3),
        )
        j1_data = JointConfigInputs(np.array([0, 0, 0]), np.array([1, 0, 0]), None)
        j2_data = JointConfigInputs(
            np.array([0, 0, -l1_length]), np.array([1, 0, 0]), None
        )

        tree.add_joint("j1", "ground", "l1", l1_data, RevoluteJoint, j1_data)
        tree.add_joint("j2", "l1", "l2", l2_data, RevoluteJoint, j2_data)

        self.model = HybridModel(tree, [1])

    def forward_dynamics_call(
        self,
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        qdt2_id: np.ndarray,
        tau_fd: np.ndarray,
    ) -> np.ndarray:
        qdt2_fd = self.model.forward_dynamics_call(
            qdt0,
            qdt1,
            qdt2_id,
            tau_fd,
        )
        return qdt2_fd

    def ssode(
        self, t, ydt0, forces_func: typing.Callable, motion_func: typing.Callable
    ) -> np.ndarray:
        return self.model.ssode(t, ydt0, forces_func, motion_func)


if __name__ == "__main__":

    from scipy import integrate
    import matplotlib.pyplot as plt

    model = DoublePendulum()

    class MotionFunctions(typing.NamedTuple):
        qdt0_f = lambda t: 0 * jnp.sin(t)
        qdt1_f = jax.jacfwd(qdt0_f)
        qdt2_f = jax.jacfwd(qdt1_f)

    def motion_func(t):
        qdt0 = MotionFunctions.qdt0_f(t)
        qdt1 = MotionFunctions.qdt1_f(t)
        qdt2 = MotionFunctions.qdt2_f(t)
        return np.array([qdt0]), np.array([qdt1]), np.array([qdt2])

    logger.debug(motion_func(0.0))

    def force_func(model: HybridModel, qdt0, qdt1, qdt2, t):
        return np.zeros((2,)), model.forces_map

    ssode = lambda t, ydt0: model.ssode(t, ydt0, force_func, motion_func)

    def simulate(ssode, ydt0, t_end):

        time_history = []
        qdt0_history = []
        qdt1_history = []
        qdt2_history = []

        stepper = integrate.BDF(ssode, 0.0, ydt0, t_end)
        while stepper.status == "running":
            y = stepper.y
            ydt1 = ssode(stepper.t, y)
            qdt0, qdt1 = y.reshape(2, -1)
            _, qdt2 = ydt1.reshape(2, -1)

            logger.debug(f"time = {stepper.t}")
            logger.debug(f"qdt0 = {qdt0}")
            logger.debug(f"qdt1 = {qdt1}")
            logger.debug(f"qdt2 = {qdt2}")

            time_history.append(stepper.t)
            qdt0_history.append(qdt0)
            qdt1_history.append(qdt1)
            qdt2_history.append(qdt2)
            stepper.step()

        return time_history, qdt0_history, qdt1_history, qdt2_history

    test_t, test_qdt0, test_qdt1, test_qdt2 = simulate(
        ssode, np.array([np.pi / 2, 0]), 2 * np.pi
    )

    plt.figure()
    plt.plot(test_t, [q[0] for q in test_qdt0])
    # plt.plot(test_t, [q[1] + q[0] for q in test_qdt0])
    plt.grid()

    plt.show()
