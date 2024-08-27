import unittest
import functools
import itertools
from typing import Callable, Any, List, Dict, Tuple

import numpy as np
import jax.numpy as jnp
import jax


from uraeus.rnea.quaternion.bodies import RigidBodyData, BodyKinematics
from uraeus.rnea.quaternion.joints import (
    JointConfigInputs,
    RevoluteJoint,
    JointKinematics,
)
from uraeus.rnea.quaternion.topologies import (
    MultiBodyTree,
    construct_multibodydata,
    MultiBodyData,
    base_to_tip,
)
from uraeus.rnea.quaternion.algorithms import (
    split_coordinates,
    IDCallRes,
    inverse_dynamics_call,
)
from uraeus.rnea.quaternion.models.analytic_double_pendulum import (
    analytic_system,
    inverse_dynamics,
    AnalyticDoublePendulum,
)
from uraeus.rnea.quaternion.tree_traversals import extract_mobilizer_forces


class Model(object):
    topology: MultiBodyTree
    forces_map: Dict[str, Dict[str, np.ndarray]]
    tree_data: MultiBodyData

    def __init__(self, topology: MultiBodyTree):
        self.topology = topology
        gravity = np.array([0, 0, -9.81, 0, 0, 0])
        self.forces_map = {
            b.name: {"gravity": b.I @ gravity} for b in self.topology.bodies.values()
        }

        self.tree_data = construct_multibodydata(topology)
        self.bodies_idx = {b: i for i, b in enumerate(self.topology.tree.nodes)}

    def get_body_kinematics(
        self, name: str, bodies_kinematics: List[BodyKinematics]
    ) -> BodyKinematics:
        return bodies_kinematics[self.bodies_idx[name]]

    def forward_kinematics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> Tuple[BodyKinematics, JointKinematics]:
        coordinates = split_coordinates(self.tree_data.qdt0_idx, qdt0, qdt1, qdt2)

        bodies_kinematics, joints_kinematics = base_to_tip(
            self.tree_data.joints, coordinates, self.tree_data.forward_traversal
        )

        return bodies_kinematics, joints_kinematics

    def inverse_dynamics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> IDCallRes:
        forces = [
            list(forces_dict.values()) for name, forces_dict in self.forces_map.items()
        ]
        # print(forces)
        res = inverse_dynamics_call(self.tree_data, forces, qdt0, qdt1, qdt2)
        return res


class DoublePendulumTest(unittest.TestCase):

    l1: float
    l2: float
    multibody_system: Model

    def setUp(self):

        self.l1 = 7
        self.l2 = 13

        self.m1 = 1
        self.m2 = 1

        # self.theta1_dt0_f = lambda t: np.radians(45)
        # self.theta2_dt0_f = lambda t: np.radians(-90)

        self.theta1_dt0_f = lambda t: t
        self.theta2_dt0_f = lambda t: t

        # self.theta1_dt0_f = lambda t: 2 * jnp.sin(t)
        # self.theta2_dt0_f = lambda t: 1.5 * jnp.sin(2 * t)

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

    @unittest.skip
    def test_forward_kinematics(self):
        time_array = np.linspace(0, 2 * np.pi, 100)

        test_sol, true_sol = zip(
            *[self._evaluate_forward_kinematics(t) for t in time_array]
        )

        test_pose_G, test_vel_G, test_acc_G = zip(*test_sol)
        true_pose_G, true_vel_G, true_acc_G = zip(*true_sol)

        np.testing.assert_almost_equal(true_pose_G, test_pose_G)
        np.testing.assert_almost_equal(true_vel_G, test_vel_G)
        np.testing.assert_almost_equal(true_acc_G, test_acc_G)

    # @unittest.skip
    def test_inverse_dynamics(self):
        time_array = np.linspace(0, 2 * np.pi, 100)

        test_sol, true_sol = zip(
            *[self._evaluate_inverse_dynamics(t) for t in time_array]
        )

        # test_sol, true_sol = self._evaluate_inverse_dynamics(time_array[29])

        np.testing.assert_almost_equal(true_sol, test_sol)

    def _evaluate_forward_kinematics(self, t):
        test_sol = self._evaluate_multibody_forward_kinematics(t)
        true_sol = self._evaluate_analytical_forward_kinematics(t)
        return test_sol, true_sol

    def _evaluate_inverse_dynamics(self, t):
        test_sol = self._evaluate_multibody_inverse_dynamics(t)
        true_sol = self._evaluate_analytical_inverse_dynamics(t)
        print("\n")
        print(true_sol, "\n", test_sol)
        print("")
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

    # def _evaluate_analytical_forward_kinematics(self, t):
    #     analytical_system = lambda t: analytic_system(
    #         self.l1, self.l2, self.theta1_dt0_f, self.theta2_dt0_f, t
    #     )
    #     return analytical_system(t)
    # def _evaluate_analytical_inverse_dynamics(self, t):
    #     return inverse_dynamics(
    #         self.l1, self.l2, self.m1, self.m2, self.theta1_dt0_f, self.theta2_dt0_f, t
    #     )

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
            np.array([*j1_fc_G[1:3], j1_tau[0]]),
            np.array([*j2_fc_G[1:3], j2_tau[0]]),
        )

        return forces

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
