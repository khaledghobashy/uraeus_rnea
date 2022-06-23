""" _ """

from typing import Callable, NamedTuple
import numpy as np

from multibody.spatial_algebra import (
    rot_x,
    rot_y,
    rot_z,
    skew_matrix,
    spatial_motion_transformation,
)

from multibody.motion_equations import (
    MotionEquations,
    RevolutePolynomials,
    TranslationalPolynomials,
    FreePolynomials,
    PlanarPolynomials,
)


class MobilizerKinematics(NamedTuple):
    X_FM: np.ndarray
    S_FM: np.ndarray
    v_J: np.ndarray
    a_J: np.ndarray


class MobilizerForces(NamedTuple):
    fi_S: np.ndarray
    fc_S: np.ndarray
    fa_S: np.ndarray
    fc_G: np.ndarray
    tau: np.ndarray


class AbstractMobilizer(NamedTuple):
    def X_FM(self, qdt0: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def W_FM_dt0(self, qdt0: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def W_FM_dt1(self, W_FM_dt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        raise NotImplementedError


class CustomMobilizer(AbstractMobilizer):

    polynomials: MotionEquations

    def X_FM(self, qdt0: np.ndarray) -> np.ndarray:
        pose_dt0 = self.polynomials.pose_polynomials(qdt0)
        orientation, location = np.split(pose_dt0, 2)
        phi, theta, psi = orientation
        R_FM = rot_z(psi) @ rot_y(theta) @ rot_x(phi)
        return spatial_motion_transformation(R_FM, -R_FM.T @ location)

    def W_FM_dt0(self, qdt0: np.ndarray) -> np.ndarray:

        # Getting the position-level spatial coordinates
        pose_dt0 = self.polynomials.pose_polynomials(qdt0)

        # Getting the orientaion vector from the 6D spatial vector
        orientation, _ = np.split(pose_dt0, 2)
        phi, theta, psi = orientation

        R_y = rot_y(theta)
        R_z = rot_z(psi)

        # z-column where the 1st rotation, `psi`, took place, in frame F
        z_col = np.array([0, 0, 1])
        # y-column where the 2nd rotation, `theta`, took place, in frame F
        y_col = R_z @ np.array([0, 1, 0])
        # x-column where the 3rd rotation, `phi`, took place, in frame F
        x_col = R_z @ R_y @ np.array([1, 0, 0])

        # transformation matrix from orientation-parameters 1st derivative to
        # the relative angular velocities.
        # (The unit vectors where the rotations took place around,
        # expressed in frame F)

        W_FM_dt0 = np.column_stack([x_col, y_col, z_col])
        return W_FM_dt0

    def W_FM_dt1(self, W_FM_dt0: np.ndarray, pose_dt1: np.ndarray) -> np.ndarray:

        x_col_dt0, y_col_dt0, z_col_dt0 = np.hsplit(W_FM_dt0, 3)

        orientation_dt1, _ = np.split(pose_dt1, 2)
        phi_dt1, theta_dt1, psi_dt1 = orientation_dt1

        omega_1 = psi_dt1 * z_col_dt0
        omega_2 = omega_1 + (theta_dt1 * y_col_dt0)

        z_col_dt1 = np.zeros((3,))
        y_col_dt1 = skew_matrix(omega_1.flatten()) @ y_col_dt0
        x_col_dt1 = skew_matrix(omega_2.flatten()) @ x_col_dt0

        W_FM_dt1 = np.column_stack([x_col_dt1, y_col_dt1, z_col_dt1])

        return W_FM_dt1

    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:

        pose_jacobian_dt0 = self.polynomials.pose_jacobian_dt0(qdt0)
        pose_dt1 = pose_jacobian_dt0 @ qdt1
        orientation_dt1, location_dt1 = np.split(pose_dt1, 2)

        angular_vel = self.W_FM_dt0(qdt0) @ orientation_dt1

        return np.hstack([angular_vel, location_dt1])

    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:
        pose_jacobian_dt0 = self.polynomials.pose_jacobian_dt0(qdt0)
        pose_jacobian_dt1 = self.polynomials.pose_jacobian_dt1(qdt0, qdt1)

        pose_dt1 = pose_jacobian_dt0 @ qdt1
        pose_dt2 = (pose_jacobian_dt0 @ qdt2) + (pose_jacobian_dt1 @ qdt1)

        orientation_dt1, location_dt1 = np.split(pose_dt1, 2)
        orientation_dt2, location_dt2 = np.split(pose_dt2, 2)

        W_FM_dt0 = self.W_FM_dt0(qdt0)
        W_FM_dt1 = self.W_FM_dt0(W_FM_dt0, pose_dt1)

        angular_acc = (W_FM_dt0 @ orientation_dt2) + (W_FM_dt1 @ orientation_dt1)

        return np.hstack([angular_acc, location_dt2])

    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:

        W_FM_dt0 = self.W_FM_dt0(qdt0)
        A_FM_dt0 = np.eye(3)
        pose_jacobian_dt0 = self.polynomials.pose_jacobian_dt0(qdt0)

        print(W_FM_dt0, pose_jacobian_dt0[:0], pose_jacobian_dt0[:3])
        S_FM = np.vstack(
            [W_FM_dt0 @ pose_jacobian_dt0[:3], A_FM_dt0 @ pose_jacobian_dt0[3:]]
        )
        return S_FM

    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:

        pose_jacobian_dt0 = self.polynomials.pose_jacobian_dt0(qdt0)
        pose_jacobian_dt1 = self.polynomials.pose_jacobian_dt1(qdt0, qdt1)

        pose_dt0 = self.polynomials.pose_polynomials(qdt0)
        pose_dt1 = pose_jacobian_dt0 @ qdt1
        pose_dt2 = (pose_jacobian_dt0 @ qdt2) + (pose_jacobian_dt1 @ qdt1)

        W_FM_dt0 = self.W_FM_dt0(qdt0)
        W_FM_dt1 = self.W_FM_dt1(W_FM_dt0, pose_dt1)

        # position-level evaluations
        orientation_dt0, location_dt0 = np.split(pose_dt0, 2)
        phi, theta, psi = orientation_dt0
        R_FM = rot_z(psi) @ rot_y(theta) @ rot_x(phi)
        X_FM = spatial_motion_transformation(R_FM, -R_FM.T @ location_dt0)
        S_FM = np.vstack([W_FM_dt0 @ pose_jacobian_dt0[:3], pose_jacobian_dt0[3:]])

        # velocity-level evaluations
        orientation_dt1, location_dt1 = np.split(pose_dt1, 2)
        angular_vel = W_FM_dt0 @ orientation_dt1
        spatial_vel = np.hstack([angular_vel, location_dt1])

        # acceleration-level evaluations
        orientation_dt2, location_dt2 = np.split(pose_dt2, 2)
        angular_acc = (W_FM_dt1 @ orientation_dt1) + (W_FM_dt0 @ orientation_dt2)
        spatial_acc = np.hstack([angular_acc, location_dt2])

        kinematics = MobilizerKinematics(X_FM, S_FM, spatial_vel, spatial_acc)
        return kinematics


class RevoluteMobilizer(CustomMobilizer):

    nj = 1
    polynomials: MotionEquations = RevolutePolynomials


class TranslationalMobilizer(CustomMobilizer):

    nj = 1
    polynomials: MotionEquations = TranslationalPolynomials


class PlanarMobilizer(CustomMobilizer):

    nj = 3
    polynomials: MotionEquations = PlanarPolynomials


class FreeMobilizer(CustomMobilizer):

    nj = 6
    polynomials: MotionEquations = FreePolynomials
