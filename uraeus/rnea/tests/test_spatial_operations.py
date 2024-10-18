import unittest

import numpy as np
from uraeus.rnea.spatial_algebra import (
    rot_x,
    rot_y,
    rot_z,
    yaw_pitch_roll_intrinsic_rotation,
    quaternion_from_axis_angle,
    quaternion_from_euler_angles,
    quaternion_to_dcm,
    transform_vector,
    skew_M,
    SpatialPose,
)


class TestTransformations(unittest.TestCase):

    def test_intrinsic_yaw_pitch_roll(self):
        alpha, beta, gamma = np.random.random((3,))
        true_pose = rot_z(gamma) @ rot_y(beta) @ rot_x(alpha)
        test_pose = yaw_pitch_roll_intrinsic_rotation(gamma, beta, alpha)

        np.testing.assert_allclose(true_pose, test_pose)

    def test_quaternion_from_euler_angles(self):
        alpha, beta, gamma = np.random.random((3,))
        true_pose = rot_z(gamma) @ rot_y(beta) @ rot_x(alpha)
        test_pose = quaternion_to_dcm(quaternion_from_euler_angles(alpha, beta, gamma))

        np.testing.assert_allclose(true_pose, test_pose)

    def test_quaternion_from_axis_angle(self):
        alpha, beta, gamma = np.random.random((3,))
        true_pose = rot_x(alpha)
        test_pose = quaternion_to_dcm(
            quaternion_from_axis_angle(alpha, np.array([1, 0, 0]))
        )
        np.testing.assert_allclose(true_pose, test_pose)

        true_pose = rot_y(beta)
        test_pose = quaternion_to_dcm(
            quaternion_from_axis_angle(beta, np.array([0, 1, 0]))
        )
        np.testing.assert_allclose(true_pose, test_pose)

        true_pose = rot_z(gamma)
        test_pose = quaternion_to_dcm(
            quaternion_from_axis_angle(gamma, np.array([0, 0, 1]))
        )
        np.testing.assert_allclose(true_pose, test_pose)

    def test_transform_vector(self):

        v_A = np.random.random((3,))
        alpha, beta, gamma = np.random.random((3,))
        E_AB = yaw_pitch_roll_intrinsic_rotation(gamma, beta, alpha)
        true_v_B = E_AB @ v_A

        q_AB = quaternion_from_euler_angles(alpha, beta, gamma)
        test_v_B = transform_vector(q_AB, v_A)

        np.testing.assert_allclose(true_v_B, test_v_B, atol=1e-7)

    def test_skew(self):
        true_matrix = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        test_matrix = skew_M @ np.array([1, 2, 3])
        np.testing.assert_allclose(true_matrix, test_matrix, atol=1e-7)

    def test_SpatialPose_orientation_transform(self):
        alpha_AB, beta_AB, gamma_AB = np.random.random((3,))
        E_AB = yaw_pitch_roll_intrinsic_rotation(gamma_AB, beta_AB, alpha_AB)
        q_AB = quaternion_from_euler_angles(alpha_AB, beta_AB, gamma_AB)

        alpha_BC, beta_BC, gamma_BC = np.random.random((3,))
        E_BC = yaw_pitch_roll_intrinsic_rotation(gamma_BC, beta_BC, alpha_BC)
        q_BC = quaternion_from_euler_angles(alpha_BC, beta_BC, gamma_BC)

        p_AB = SpatialPose(np.zeros((3,)), q_AB)
        p_BC = SpatialPose(np.zeros((3,)), q_BC)

        E_AC = E_AB @ E_BC
        p_AC = p_AB @ p_BC

        np.testing.assert_allclose(quaternion_to_dcm(p_AC.q), E_AC)
        np.testing.assert_allclose(quaternion_to_dcm(p_AC.inv().q), E_AC.T)

    def test_SpatialPose_location_transform(self):
        alpha_AB, beta_AB, gamma_AB = np.random.random((3,))
        v1_A = 5 * np.random.random((3,))
        E_AB = yaw_pitch_roll_intrinsic_rotation(gamma_AB, beta_AB, alpha_AB)
        q_AB = quaternion_from_euler_angles(alpha_AB, beta_AB, gamma_AB)

        alpha_BC, beta_BC, gamma_BC = np.random.random((3,))
        v2_B = 7 * np.random.random((3,))
        E_BC = yaw_pitch_roll_intrinsic_rotation(gamma_BC, beta_BC, alpha_BC)
        q_BC = quaternion_from_euler_angles(alpha_BC, beta_BC, gamma_BC)

        E_AC = E_BC @ E_AB
        v3_C = (E_AC @ v1_A) + (E_BC @ v2_B)
        v3_A = v1_A + (E_AB.T @ v2_B)

        p_AB = SpatialPose(v1_A, q_AB)
        p_BC = SpatialPose(v2_B, q_BC)
        p_AC = p_BC @ p_AB

        np.testing.assert_allclose(quaternion_to_dcm(p_AC.q), E_AC)
        np.testing.assert_allclose(p_AC.r, v3_A)
        np.testing.assert_allclose(p_AC.inv().r, -v3_C)


if __name__ == "__main__":
    unittest.main()
