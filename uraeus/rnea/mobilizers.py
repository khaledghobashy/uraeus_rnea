""" _ """

from typing import NamedTuple
from functools import partial

import jax

import jax.numpy as jnp
import numpy as np

from uraeus.rnea.spatial_algebra import (
    SpatialPose,
    rot_x,
    rot_y,
    rot_z,
    quaternion_from_euler_angles,
    quaternion_inverse,
    quaternion_from_axis_angle,
    transform_screw,
    skew_M,
    yaw_pitch_roll_intrinsic_rotation,
)

from uraeus.rnea.motion_equations import (
    MotionEquations,
    RevolutePolynomials,
    CylindricalPolynomials,
    TranslationalPolynomials,
    FreePolynomials,
    PlanarPolynomials,
)


class MobilizerKinematics(NamedTuple):
    """
    Represents the kinematics of a mobilizer.

    Attributes
    ----------
    p_FM : SpatialPose
        The pose of the mobilizer, transforming motion form F to M.
    S_FM : np.ndarray
        The motion subspace matrix of the mobilizer.
    v_J : np.ndarray
        The mobilizer spatial velocity screw, expressed in the M frame.
    a_J : np.ndarray
        The mobilizer spatial acceleration screw, expressed in the M frame.
    """

    p_FM: SpatialPose
    S_FM: np.ndarray
    v_J: np.ndarray
    a_J: np.ndarray


class MobilizerForces(NamedTuple):
    """
    Represents the forces acting at a mobilizer.

    Attributes
    ----------
    fi_S : np.ndarray
        Internal force in the mobilizer, in the (S)uccessor frame.
    fc_S : np.ndarray
        Constraint force in the mobilizer, in the (S)uccessor frame.
    fa_S : np.ndarray
        Applied force in the mobilizer, in the (S)uccessor frame.
    fc_G : np.ndarray
        Constraint force, in the (G)lobal frame.
    tau : np.ndarray
        Generalized forces at the mobilizer.
    """

    fi_S: np.ndarray
    fc_S: np.ndarray
    fa_S: np.ndarray
    fc_G: np.ndarray
    tau: np.ndarray


class AbstractMobilizer(NamedTuple):
    nj: int = None

    def p_FM(self, qdt0: np.ndarray) -> SpatialPose:
        """Evaluates the SpatialPose, p_FM, that applies motion (translation
        and rotation) from the F (fixed) frame to the M (moving) frame.
        This can be used to transform local-vectors from F to M.

        Parameters
        ----------
        qdt0 : np.ndarray
            Mobilizer's position variables.

        Returns
        -------
        SpatialPose
            Returns the SpatialPose, p_FM, that applies motion (translation
            and rotation) from the F (fixed) frame to the M (moving) frame.

        Raises
        ------
        NotImplementedError
            Abstract class.
        """
        raise NotImplementedError

    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        """Evaluates the Motion subspace matrix of the mobilizer, S.
        This maps the joint's velocity variables into 6D spatial velocity screw.

        Parameters
        ----------
        qdt0 : np.ndarray
            Mobilizer's position variables.

        Returns
        -------
        np.ndarray
            Motion subspace matrix of the mobilizer, S.

        Raises
        ------
        NotImplementedError
            Method is not defined in concrete sub-class.
        """
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

    def N(self, qdt0: np.ndarray) -> np.ndarray:
        return np.eye(self.nj)


class CustomMobilizer(AbstractMobilizer):
    polynomials: MotionEquations

    @partial(jax.jit, static_argnums=(0,))
    def p_FM(self, qdt0: np.ndarray) -> SpatialPose:

        # Evaluate the spatial pose from the given pose_polynomials from the given
        # joint's generalized position variables, qdt0
        pose_dt0 = self.polynomials.pose_polynomials(qdt0)
        location, orientation = jnp.split(pose_dt0, 2)
        phi_dt0, theta_dt0, psi_dt0 = orientation

        # Constructing a quaternion from the given euler angles.
        # This gives the rotation quaternion needed to rotate frame M relative
        # to frame F, with the given angles. This q transforms/rotate vectors
        # from the F frame to M frame, and represents F frame expressed in M
        # frame
        q = quaternion_from_euler_angles(phi_dt0, theta_dt0, psi_dt0)
        p_FM = SpatialPose(location, q)
        return p_FM

    @partial(jax.jit, static_argnums=(0,))
    def W_FM_dt0(self, qdt0: np.ndarray) -> np.ndarray:
        # Getting the position-level spatial coordinates
        pose_dt0 = self.polynomials.pose_polynomials(qdt0)

        # Getting the orientaion vector from the 6D spatial vector
        _, orientation = jnp.split(pose_dt0, 2)
        phi_dt0, theta_dt0, psi_dt0 = orientation

        R_x = rot_x(phi_dt0)
        R_y = rot_y(theta_dt0)
        R_z = rot_z(psi_dt0)

        a1 = np.array([0, 0, 1])
        a2 = R_z @ np.array([0, 1, 0])
        a3 = R_z @ R_y @ np.array([1, 0, 0])

        W_FM_dt0 = jnp.column_stack([a3, a2, a1])
        return W_FM_dt0

    @partial(jax.jit, static_argnums=(0,))
    def W_FM_dt1(self, W_FM_dt0: np.ndarray, pose_dt1: np.ndarray) -> np.ndarray:
        x_col_dt0, y_col_dt0, z_col_dt0 = [a.flatten() for a in jnp.hsplit(W_FM_dt0, 3)]

        _, orientation_dt1 = jnp.split(pose_dt1, 2)
        phi_dt1, theta_dt1, psi_dt1 = orientation_dt1

        omega_1 = phi_dt1 * x_col_dt0
        omega_2 = omega_1 + (theta_dt1 * y_col_dt0)

        omega_1 = psi_dt1 * z_col_dt0
        omega_2 = omega_1 + (theta_dt1 * y_col_dt0)

        x_col_dt1 = (skew_M @ omega_2) @ z_col_dt0
        y_col_dt1 = (skew_M @ omega_1) @ y_col_dt0
        z_col_dt1 = np.zeros((3,))

        W_FM_dt1 = jnp.column_stack([x_col_dt1, y_col_dt1, z_col_dt1])

        return W_FM_dt1

    @partial(jax.jit, static_argnums=(0,))
    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        """Evaluate the joint's sub-space matrix given the joint's position
        variables.
        The S matrix maps the joint's velocity variables into the joint's
        relative spatial velocity, expressed in the M frame.
        For spherical motion, the corresponding sub-matrix of the S matrix,
        represents the coordinates in M frame, of the axes about which the
        rotations take place.

        Following rotation order, psi_dt0(z)->theta_dt0(y)->phi_dt0(x), we can
        interpret the S matrix as follows:
        The third column is for the first rotation around z-axis of the F frame,
        the second is where the y axis was after the first rotation, and the
        first is where the x axis was after the first two rotations.

        This sub-matrix can be derived following Eqn (7.39), p.372 A.Shabana's
        Computational dynamics 3rd edition (2010).
                                    Ȧ˙A.T = ̃ω
        where:
            A = rot_x(phi_dt0) @ rot_y(theta_dt0) @ rot_z(psi_dt0)
            Ȧ = dA/dt
            ῶ = skew-symmetric matrix for of the spatial angular velocities

        Using this, with Eqn 7.70:
                                    ω = G . θ˙
        where θ˙ is the joint's generalized velocity variables (euler angles in
        this case).
        Using this equation, we can evaluate G, which is the same as the
        sub-matrix of S derived above.

        Parameters
        ----------
        qdt0 : np.ndarray
            The joint's generalized position variables.

        Returns
        -------
        np.ndarray
            The motion subspace matrix of the joint.
        """
        W_FM_dt0 = self.W_FM_dt0(qdt0)
        A_FM_dt0 = np.eye(3)
        pose_jacobian_dt0 = self.polynomials.pose_jacobian_dt0(qdt0)

        S_FM = jnp.vstack(
            [A_FM_dt0 @ pose_jacobian_dt0[:3], W_FM_dt0 @ pose_jacobian_dt0[3:]]
        )
        return S_FM

    @partial(jax.jit, static_argnums=(0,))
    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        pose_jacobian_dt0 = self.polynomials.pose_jacobian_dt0(qdt0)
        pose_dt1 = pose_jacobian_dt0 @ qdt1
        location_dt1, orientation_dt1 = jnp.split(pose_dt1, 2)

        angular_vel = self.W_FM_dt0(qdt0) @ orientation_dt1

        return jnp.hstack([location_dt1, angular_vel])

    @partial(jax.jit, static_argnums=(0,))
    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:
        pose_jacobian_dt0 = self.polynomials.pose_jacobian_dt0(qdt0)
        pose_jacobian_dt1 = self.polynomials.pose_jacobian_dt1(qdt0, qdt1)

        pose_dt1 = pose_jacobian_dt0 @ qdt1
        pose_dt2 = (pose_jacobian_dt0 @ qdt2) + (pose_jacobian_dt1 @ qdt1)

        location_dt1, orientation_dt1 = jnp.split(pose_dt1, 2)
        location_dt2, orientation_dt2 = jnp.split(pose_dt2, 2)

        W_FM_dt0 = self.W_FM_dt0(qdt0)
        W_FM_dt1 = self.W_FM_dt1(W_FM_dt0, pose_dt1)

        angular_acc = (W_FM_dt0 @ orientation_dt2) + (W_FM_dt1 @ orientation_dt1)

        return jnp.hstack([location_dt2, angular_acc])

    @partial(jax.jit, static_argnums=(0,))
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
        location_dt0, orientation_dt0 = pose_dt0.reshape(2, -1)
        phi_dt0, theta_dt0, psi_dt0 = orientation_dt0
        q_FM = quaternion_from_euler_angles(phi_dt0, theta_dt0, psi_dt0)
        p_FM = SpatialPose(location_dt0, q_FM)
        S_FM = jnp.vstack([pose_jacobian_dt0[:3], W_FM_dt0 @ pose_jacobian_dt0[3:]])

        # velocity-level evaluations
        location_dt1, orientation_dt1 = pose_dt1.reshape(2, -1)
        angular_vel = W_FM_dt0 @ orientation_dt1
        spatial_vel = transform_screw(p_FM, jnp.hstack([location_dt1, angular_vel]))

        # acceleration-level evaluations
        location_dt2, orientation_dt2 = pose_dt2.reshape(2, -1)
        angular_acc = (W_FM_dt1 @ orientation_dt1) + (W_FM_dt0 @ orientation_dt2)
        spatial_acc = transform_screw(p_FM, jnp.hstack([location_dt2, angular_acc]))

        kinematics = MobilizerKinematics(p_FM, S_FM, spatial_vel, spatial_acc)

        return kinematics


class RevoluteMobilizer(CustomMobilizer):
    nj = 1
    polynomials: MotionEquations = RevolutePolynomials

    @partial(jax.jit, static_argnums=(0,))
    def p_FM(self, qdt0: np.ndarray) -> np.ndarray:
        psi_dt0 = qdt0[0]
        # Using the negative of the angle to get the F frame in M frame, as
        # using the positive would do a rotation from F frame to M frame, which
        # represents the M frame in F frame
        q_FM = quaternion_from_axis_angle(-psi_dt0, np.array([0, 0, 1]))
        p_FM = SpatialPose(np.zeros((3,)), q_FM)
        return p_FM

    @partial(jax.jit, static_argnums=(0,))
    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        return np.array([0, 0, 0, 0, 0, 1])[:, None]

    @partial(jax.jit, static_argnums=(0,))
    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        psi_dt1 = qdt1[0]
        return jnp.array([0, 0, 0, 0, 0, psi_dt1])

    @partial(jax.jit, static_argnums=(0,))
    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:
        psi_dt2 = qdt2[0]
        return jnp.array([0, 0, 0, 0, 0, psi_dt2])

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        p_FM = self.p_FM(qdt0)
        S_FM = self.S_FM(qdt0)
        v_J = self.v_J(qdt0, qdt1)
        a_J = self.a_J(qdt0, qdt1, qdt2)
        return MobilizerKinematics(p_FM, S_FM, v_J, a_J)


class TranslationalMobilizer(CustomMobilizer):
    nj = 1
    polynomials: MotionEquations = TranslationalPolynomials

    @partial(jax.jit, static_argnums=(0,))
    def p_FM(self, qdt0: np.ndarray) -> np.ndarray:
        z_dt0 = qdt0[0]
        p_FM = SpatialPose(jnp.array([0, 0, z_dt0]), jnp.array([1, 0, 0, 0]))
        return p_FM

    @partial(jax.jit, static_argnums=(0,))
    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        return np.array([0, 0, 1, 0, 0, 0])[:, None]

    @partial(jax.jit, static_argnums=(0,))
    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        z_dt1 = qdt1[0]
        return jnp.array([0, 0, z_dt1, 0, 0, 0])

    @partial(jax.jit, static_argnums=(0,))
    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:
        z_dt2 = qdt2[0]
        return jnp.array([0, 0, z_dt2, 0, 0, 0])

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        p_FM = self.p_FM(qdt0)
        S_FM = self.S_FM(qdt0)
        v_J = self.v_J(qdt0, qdt1)
        a_J = self.a_J(qdt0, qdt1, qdt2)
        return MobilizerKinematics(p_FM, S_FM, v_J, a_J)


class CylindricalMobilizer(CustomMobilizer):
    nj = 2
    polynomials: MotionEquations = CylindricalPolynomials

    @partial(jax.jit, static_argnums=(0,))
    def p_FM(self, qdt0: np.ndarray) -> np.ndarray:
        z_dt0, psi_dt0 = qdt0
        # Using the negative of the angle to get the F frame in M frame, as
        # using the positive would do a rotation from F frame to M frame, which
        # represents the M frame in F frame
        q_FM = quaternion_from_axis_angle(-psi_dt0, np.array([0, 0, 1]))
        p_FM = SpatialPose(jnp.array([0, 0, z_dt0]), q_FM)
        return p_FM

    @partial(jax.jit, static_argnums=(0,))
    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        return np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]]).T

    @partial(jax.jit, static_argnums=(0,))
    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        z_dt1, psi_dt1 = qdt1
        return jnp.array([0, 0, z_dt1, 0, 0, psi_dt1])

    @partial(jax.jit, static_argnums=(0,))
    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:
        z_dt2, psi_dt2 = qdt2
        return jnp.array([0, 0, z_dt2, 0, 0, psi_dt2])

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        p_FM = self.p_FM(qdt0)
        S_FM = self.S_FM(qdt0)
        v_J = self.v_J(qdt0, qdt1)
        a_J = self.a_J(qdt0, qdt1, qdt2)
        return MobilizerKinematics(p_FM, S_FM, v_J, a_J)


class PlanarMobilizer(CustomMobilizer):
    nj = 3
    polynomials: MotionEquations = PlanarPolynomials


class FreeMobilizer(CustomMobilizer):
    nj = 6
    polynomials: MotionEquations = FreePolynomials

    @partial(jax.jit, static_argnums=(0,))
    def p_FM(self, qdt0: np.ndarray) -> np.ndarray:
        location, orientation = jnp.split(qdt0, 2)
        q_FM = quaternion_inverse(quaternion_from_euler_angles(*orientation))
        p_FM = SpatialPose(location, q_FM)
        return p_FM

    @partial(jax.jit, static_argnums=(0,))
    def S_FM(self, qdt0: np.ndarray) -> np.ndarray:
        # Getting the orientaion vector from the 6D spatial vector
        _, orientation = jnp.split(qdt0, 2)
        phi_dt0, theta_dt0, _ = orientation

        S_M_rot = jnp.array(
            [
                [1, 0, jnp.sin(theta_dt0)],
                [0, jnp.cos(phi_dt0), -jnp.sin(phi_dt0) * jnp.cos(theta_dt0)],
                [0, jnp.sin(phi_dt0), jnp.cos(phi_dt0) * jnp.cos(theta_dt0)],
            ]
        )

        S_M = jnp.vstack(
            [
                jnp.hstack([np.eye(3), np.zeros((3, 3))]),
                jnp.hstack([np.zeros((3, 3)), S_M_rot]),
            ]
        )

        return S_M

    @partial(jax.jit, static_argnums=(0,))
    def v_J(self, qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        # Note: Translational velocities of mobilities are expressed in
        vj_M = self.S_FM(qdt0) @ qdt1
        return vj_M

    @partial(jax.jit, static_argnums=(0,))
    def N(self, qdt0: np.ndarray) -> np.ndarray:
        _, orientation_dt0 = jnp.split(qdt0, 2)

        phi_dt0, theta_dt0, psi_dt0 = orientation_dt0
        E_MF = yaw_pitch_roll_intrinsic_rotation(psi_dt0, theta_dt0, phi_dt0)
        N = jnp.vstack(
            [
                jnp.hstack([E_MF, np.zeros((3, 3))]),
                jnp.hstack([np.zeros((3, 3)), np.eye(3)]),
            ]
        )
        return N

    @partial(jax.jit, static_argnums=(0,))
    def a_J(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray) -> np.ndarray:

        _, orientation_dt0 = jnp.split(qdt0, 2)
        _, orientation_dt1 = jnp.split(qdt1, 2)
        location_dt2, orientation_dt2 = jnp.split(qdt2, 2)

        S_M = self.S_FM(qdt0)
        S_M_rot = S_M[3:, 3:]

        phi_dt0, theta_dt0, psi_dt0 = orientation_dt0
        phi_dt1, theta_dt1, psi_dt1 = orientation_dt1

        W_FM_dt1 = jnp.array(
            [
                [0, 0, theta_dt1 * jnp.cos(theta_dt0)],
                [
                    0,
                    -phi_dt1 * jnp.sin(phi_dt0),
                    -phi_dt1 * jnp.cos(phi_dt0) * jnp.cos(theta_dt0)
                    + theta_dt1 * jnp.sin(phi_dt0) * jnp.sin(theta_dt0),
                ],
                [
                    0,
                    phi_dt1 * jnp.cos(phi_dt0),
                    -phi_dt1 * jnp.sin(phi_dt0) * jnp.cos(theta_dt0)
                    - theta_dt1 * jnp.sin(theta_dt0) * jnp.cos(phi_dt0),
                ],
            ]
        )

        angular_acc = (S_M_rot @ orientation_dt2) + (W_FM_dt1 @ orientation_dt1)

        return jnp.hstack([location_dt2, angular_acc])

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        p_FM = self.p_FM(qdt0)
        S_FM = self.S_FM(qdt0)
        v_J = self.v_J(qdt0, qdt1)
        a_J = self.a_J(qdt0, qdt1, qdt2)
        return MobilizerKinematics(p_FM, S_FM, v_J, a_J)
