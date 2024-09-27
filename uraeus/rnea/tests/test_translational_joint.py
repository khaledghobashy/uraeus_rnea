import unittest

import numpy as np
import jax.numpy as jnp
import jax

from uraeus.rnea.bodies import RigidBodyData
from uraeus.rnea.joints import JointConfigInputs, TranslationalJoint
from uraeus.rnea.topologies import MultiBodyTree, Model
from uraeus.rnea.tree_traversals import extract_mobilizer_forces


def build_multibody_system_v01(
    motion_axis: np.ndarray, joint_location: np.ndarray, link_location: np.ndarray
):

    tree = MultiBodyTree("mass_spring")

    l1_data = RigidBodyData(link_location, np.array([1, 0, 0, 0]), 1, np.eye(3))
    j1_data = JointConfigInputs(joint_location, motion_axis, None)

    tree.add_joint("j1", "ground", "l1", l1_data, TranslationalJoint, j1_data)

    model = Model(tree)

    # model.fk = lambda x: np.array([0, 0, -x * k, 0, 0, 0])
    # model.fc = lambda v: np.array([0, 0, -v * c, 0, 0, 0])

    return model


def build_multibody_system_v02(
    motion_axis_1: np.ndarray,
    joint_location_1: np.ndarray,
    link_location_1: np.ndarray,
    motion_axis_2: np.ndarray,
    joint_location_2: np.ndarray,
    link_location_2: np.ndarray,
):

    tree = MultiBodyTree("v02")

    l1_data = RigidBodyData(link_location_1, np.array([1, 0, 0, 0]), 1, np.eye(3))
    l2_data = RigidBodyData(link_location_2, np.array([1, 0, 0, 0]), 1, np.eye(3))

    j1_data = JointConfigInputs(joint_location_1, motion_axis_1, None)
    j2_data = JointConfigInputs(joint_location_2, motion_axis_2, None)

    tree.add_joint("j1", "ground", "l1", l1_data, TranslationalJoint, j1_data)
    tree.add_joint("j2", "l1", "l2", l2_data, TranslationalJoint, j2_data)

    model = Model(tree)

    return model


class TranslationalJointTest(unittest.TestCase):

    def test_vertical_orientation_kinematics(self):
        qdt0 = np.array([1])
        qdt1 = np.array([1])
        qdt2 = np.array([1])

        motion_axis = np.array([0, 0, 1])
        joint_location = np.array([0, 0, 0])
        link_location = np.array([0, 0, 0])

        model = build_multibody_system_v01(motion_axis, joint_location, link_location)
        bodies_kin, joints_kin = model.forward_kinematics_pass(qdt0, qdt1, qdt2)

        l1_kin = model.get_body_kinematics("l1", bodies_kin)

        np.testing.assert_almost_equal(l1_kin.p_GB.r, np.array([0, 0, 1]))
        np.testing.assert_almost_equal(l1_kin.p_GB.q, np.array([1, 0, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.v_G, np.array([0, 0, 1, 0, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.a_G, np.array([0, 0, 1, 0, 0, 0]))

    def test_lateral_orientation_kinematics(self):
        qdt0 = np.array([1])
        qdt1 = np.array([1])
        qdt2 = np.array([1])

        motion_axis = np.array([0, 1, 0])
        joint_location = np.array([0, 0, 0])
        link_location = np.array([0, 0, 0])

        model = build_multibody_system_v01(motion_axis, joint_location, link_location)
        bodies_kin, joints_kin = model.forward_kinematics_pass(qdt0, qdt1, qdt2)

        l1_kin = model.get_body_kinematics("l1", bodies_kin)

        np.testing.assert_almost_equal(l1_kin.p_GB.r, np.array([0, 1, 0]))
        np.testing.assert_almost_equal(l1_kin.p_GB.q, np.array([1, 0, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.v_G, np.array([0, 1, 0, 0, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.a_G, np.array([0, 1, 0, 0, 0, 0]))

    def test_longitudinal_orientation_kinematics(self):
        qdt0 = np.array([1])
        qdt1 = np.array([1])
        qdt2 = np.array([1])

        motion_axis = np.array([1, 0, 0])
        joint_location = np.array([0, 0, 0])
        link_location = np.array([0, 0, 0])

        model = build_multibody_system_v01(motion_axis, joint_location, link_location)
        bodies_kin, joints_kin = model.forward_kinematics_pass(qdt0, qdt1, qdt2)

        l1_kin = model.get_body_kinematics("l1", bodies_kin)

        np.testing.assert_almost_equal(l1_kin.p_GB.r, np.array([1, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.p_GB.q, np.array([1, 0, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.v_G, np.array([1, 0, 0, 0, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.a_G, np.array([1, 0, 0, 0, 0, 0]))

    def test_two_links_orientation_kinematics(self):

        motion_axis_1 = np.array([1, 0, 0])
        joint_location_1 = np.array([0, 0, 1])
        link_location_1 = np.array([0, 0, 1])

        motion_axis_2 = np.array([0, 0, 1])
        joint_location_2 = np.array([0, 0, 1])
        link_location_2 = np.array([0, 0, 1])

        model = build_multibody_system_v02(
            motion_axis_1,
            joint_location_1,
            link_location_1,
            motion_axis_2,
            joint_location_2,
            link_location_2,
        )

        qdt0 = np.array([1, 1])
        qdt1 = np.array([1, 1])
        qdt2 = np.array([1, 1])

        bodies_kin, joints_kin = model.forward_kinematics_pass(qdt0, qdt1, qdt2)

        l1_kin = model.get_body_kinematics("l1", bodies_kin)
        l2_kin = model.get_body_kinematics("l2", bodies_kin)

        np.testing.assert_almost_equal(l1_kin.p_GB.r, np.array([1, 0, 1]))
        np.testing.assert_almost_equal(l1_kin.p_GB.q, np.array([1, 0, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.v_G, np.array([1, 0, 0, 0, 0, 0]))
        np.testing.assert_almost_equal(l1_kin.a_G, np.array([1, 0, 0, 0, 0, 0]))

        np.testing.assert_almost_equal(l2_kin.p_GB.r, np.array([1, 0, 1 + 1]))
        np.testing.assert_almost_equal(l2_kin.p_GB.q, np.array([1, 0, 0, 0]))
        np.testing.assert_almost_equal(l2_kin.v_G, np.array([1, 0, 1, 0, 0, 0]))
        np.testing.assert_almost_equal(l2_kin.a_G, np.array([1, 0, 1, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
