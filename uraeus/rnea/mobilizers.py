""" _ """

from typing import Callable, NamedTuple
import numpy as np

from uraeus.rnea.spatial_algebra import (
    rot_x,
    rot_y,
    rot_z,
    skew_matrix,
    spatial_motion_rotation,
    spatial_motion_transformation,
    spatial_motion_translation,
)

from uraeus.rnea.motion_equations import (
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

    nj: int = None

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

        R_x = rot_x(phi)
        R_y = rot_y(theta)
        # R_z = rot_z(psi)

        # z-column where the 1st rotation, `psi`, took place, in frame F
        # z_col = np.array([0, 0, 1])
        # y-column where the 2nd rotation, `theta`, took place, in frame F
        # y_col = R_z @ np.array([0, 1, 0])
        # x-column where the 3rd rotation, `phi`, took place, in frame F
        # x_col = R_z @ R_y @ np.array([1, 0, 0])

        # transformation matrix from orientation-parameters 1st derivative to
        # the relative angular velocities.
        # (The unit vectors where the rotations took place around,
        # expressed in frame F)

        a1 = np.array([1, 0, 0])
        a2 = R_x @ np.array([0, 1, 0])
        a3 = R_x @ R_y @ np.array([0, 0, 1])

        # W_FM_dt0 = np.column_stack([x_col, y_col, z_col])
        W_FM_dt0 = np.column_stack([a1, a2, a3])
        return W_FM_dt0

    def W_FM_dt1(self, W_FM_dt0: np.ndarray, pose_dt1: np.ndarray) -> np.ndarray:

        x_col_dt0, y_col_dt0, z_col_dt0 = np.hsplit(W_FM_dt0, 3)

        orientation_dt1, _ = np.split(pose_dt1, 2)
        phi_dt1, theta_dt1, psi_dt1 = orientation_dt1

        # omega_1 = psi_dt1 * z_col_dt0
        # omega_2 = omega_1 + (theta_dt1 * y_col_dt0)

        # z_col_dt1 = np.zeros((3,))
        # y_col_dt1 = skew_matrix(omega_1.flatten()) @ y_col_dt0
        # x_col_dt1 = skew_matrix(omega_2.flatten()) @ x_col_dt0

        # W_FM_dt1 = np.column_stack([x_col_dt1, y_col_dt1, z_col_dt1])

        # omega_1 = W_FM_dt0 @ np.array([phi_dt1, 0, 0])
        # omega_2 = W_FM_dt0 @ np.array([phi_dt1, theta_dt1, 0])
        omega_1 = phi_dt1 * x_col_dt0
        omega_2 = omega_1 + (theta_dt1 * y_col_dt0)

        x_col_dt1 = np.zeros((3,))
        y_col_dt1 = skew_matrix(omega_1.flatten()) @ y_col_dt0
        z_col_dt1 = skew_matrix(omega_2.flatten()) @ z_col_dt0

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
        W_FM_dt1 = self.W_FM_dt1(W_FM_dt0, pose_dt1)

        angular_acc = (W_FM_dt0 @ orientation_dt2) + (W_FM_dt1 @ orientation_dt1)

        return np.hstack([angular_acc, location_dt2])

    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:

        W_FM_dt0 = self.W_FM_dt0(qdt0)
        A_FM_dt0 = np.eye(3)
        pose_jacobian_dt0 = self.polynomials.pose_jacobian_dt0(qdt0)

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
        orientation_dt0, location_dt0 = pose_dt0.reshape(2, -1)
        phi, theta, psi = orientation_dt0
        R_FM = rot_z(psi) @ rot_y(theta) @ rot_x(phi)
        X_FM = spatial_motion_transformation(R_FM, -R_FM.T @ location_dt0)
        S_FM = np.vstack([W_FM_dt0 @ pose_jacobian_dt0[:3], pose_jacobian_dt0[3:]])

        # velocity-level evaluations
        orientation_dt1, location_dt1 = pose_dt1.reshape(2, -1)
        angular_vel = W_FM_dt0 @ orientation_dt1
        spatial_vel = np.hstack([angular_vel, location_dt1])

        # acceleration-level evaluations
        orientation_dt2, location_dt2 = pose_dt2.reshape(2, -1)
        angular_acc = (W_FM_dt1 @ orientation_dt1) + (W_FM_dt0 @ orientation_dt2)
        spatial_acc = np.hstack([angular_acc, location_dt2])

        kinematics = MobilizerKinematics(X_FM, S_FM, spatial_vel, spatial_acc)
        return kinematics


class RevoluteMobilizer(CustomMobilizer):

    nj = 1
    polynomials: MotionEquations = RevolutePolynomials

    def X_FM(self, qdt0: np.ndarray) -> np.ndarray:
        psi_dt0 = qdt0[0]
        return spatial_motion_rotation(rot_z(psi_dt0))

    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        return np.array([0, 0, 1, 0, 0, 0])[:, None]

    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        psi_dt1 = qdt1[0]
        return np.array([0, 0, psi_dt1, 0, 0, 0])

    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:
        psi_dt2 = qdt2[0]
        return np.array([0, 0, psi_dt2, 0, 0, 0])

    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        X_FM = self.X_FM(qdt0)
        S_FM = self.S_FM(qdt0)
        v_J = self.v_J(qdt0, qdt1)
        a_J = self.a_J(qdt0, qdt1, qdt2)
        return MobilizerKinematics(X_FM, S_FM, v_J, a_J)


class TranslationalMobilizer(CustomMobilizer):

    nj = 1
    polynomials: MotionEquations = TranslationalPolynomials

    def X_FM(self, qdt0: np.ndarray) -> np.ndarray:
        z_dt0 = qdt0[0]
        return spatial_motion_translation(np.array([0, 0, -z_dt0]))

    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        return np.array([0, 0, 0, 0, 0, 1])[:, None]

    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        z_dt1 = qdt1[0]
        return np.array([0, 0, 0, 0, 0, z_dt1])

    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:
        z_dt2 = qdt2[0]
        return np.array([0, 0, 0, 0, 0, z_dt2])

    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        X_FM = self.X_FM(qdt0)
        S_FM = self.S_FM(qdt0)
        v_J = self.v_J(qdt0, qdt1)
        a_J = self.a_J(qdt0, qdt1, qdt2)
        return MobilizerKinematics(X_FM, S_FM, v_J, a_J)


class PlanarMobilizer(CustomMobilizer):

    nj = 3
    polynomials: MotionEquations = PlanarPolynomials


class FreeMobilizer(CustomMobilizer):

    nj = 6
    polynomials: MotionEquations = FreePolynomials
