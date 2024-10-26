import unittest

import numpy as np

from uraeus.rnea.bodies import RigidBodyData
from uraeus.rnea.joints import JointConfigInputs, RevoluteJoint
from uraeus.rnea.multibody_models import MultiBodyTree, Model
from uraeus.rnea.graphs import Tree
from uraeus.rnea.algorithms import (
    CompositeInertiaMatrixOperations,
    JointInertiaMatrixOperations,
)


class DoublePendulumTest(unittest.TestCase):

    l1: float
    l2: float
    multibody_system: Model

    def setUp(self):

        self.multibody_system = self._build_multibody_tree()

    def test_chained_system(self):

        qdt0 = np.zeros((self.multibody_system.dof,))
        qdt1 = np.zeros((self.multibody_system.dof,))
        qdt2 = np.zeros((self.multibody_system.dof,))

        bodies_kinematics, joints_kinematics = (
            self.multibody_system.forward_kinematics_pass(qdt0, qdt1, qdt2)
        )

        H_crba = CompositeInertiaMatrixOperations.construct_H(
            self.multibody_system.tree_data, joints_kinematics, qdt0
        )
        H_joints_inertia = JointInertiaMatrixOperations.construct_H(
            self.multibody_system.tree_data, joints_kinematics, qdt0
        )
        np.testing.assert_allclose(H_crba, H_joints_inertia)

    def _build_multibody_tree(self):

        tree = Tree("tree_revolute", "ground")
        tree.add_edge("ground", "l1")
        tree.add_edge("ground", "l2")
        tree.add_edge("l1", "l3")
        tree.add_edge("l2", "l4")
        tree.add_edge("l2", "l5")
        tree.add_edge("l3", "l6")
        tree.add_edge("l3", "l7")
        tree.add_edge("l5", "l8")
        tree.add_edge("l8", "l9")
        tree.add_edge("l8", "l10")

        n_nodes = len(tree.nodes)
        masses = np.arange(1, 2 * (n_nodes + 1), 2)
        lengths = np.arange(1, n_nodes + 1)

        topology = MultiBodyTree("tree_revolute")

        joints_names = ["j%s" % i for i, _ in enumerate(tree.edges, 1)]
        print(joints_names)

        # print(lengths)

        bodies_data = {
            node: RigidBodyData(
                np.array([0, 0, -sum(lengths[:i]) - lengths[i]]),
                np.array([1, 0, 0, 0]),
                masses[i],
                3 * np.eye(3),
            )
            for i, node in enumerate(list(tree.nodes)[1:])
        }
        joints_data = [
            JointConfigInputs(
                np.array([0, 0, -sum(lengths[:i])]), np.array([1, 0, 0]), None
            )
            for i in np.arange(n_nodes - 1)
        ]
        print(bodies_data.keys())

        for j, (p, s), j_data in zip(joints_names, tree.edges, joints_data):
            print((p, s))
            topology.add_joint(j, p, s, bodies_data[s], RevoluteJoint, j_data)

        model = Model(topology)

        return model

    def _build_multibody_chain(self, chain_length: int):
        tree = MultiBodyTree("pendulums")

        masses = np.arange(1, 2 * (chain_length + 1), 2)
        lengths = np.arange(1, chain_length + 1)

        bodies_names = ["ground"] + ["l%s" % i for i in np.arange(1, chain_length + 1)]
        joints_names = ["j%s" % i for i in np.arange(1, chain_length + 1)]

        print(lengths)

        bodies_data = [
            RigidBodyData(
                np.array([0, 0, -sum(lengths[:i]) - lengths[i]]),
                np.array([1, 0, 0, 0]),
                masses[i],
                3 * np.eye(3),
            )
            for i in np.arange(chain_length)
        ]
        joints_data = [
            JointConfigInputs(
                np.array([0, 0, -sum(lengths[:i])]), np.array([1, 0, 0]), None
            )
            for i in np.arange(chain_length)
        ]

        for i in np.arange(chain_length):
            print(i)
            tree.add_joint(
                joints_names[i],
                bodies_names[i],
                bodies_names[i + 1],
                bodies_data[i],
                RevoluteJoint,
                joints_data[i],
            )

        model = Model(tree)

        return model


if __name__ == "__main__":

    unittest.main()
