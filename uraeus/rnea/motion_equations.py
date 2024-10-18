from typing import NamedTuple, Callable

import jax

import jax.numpy as jnp
import numpy as np


class MotionEquations(NamedTuple):
    """
    Represents the motion equations for a joint.

    Attributes
    ----------
    nj : int
        Number of joints.
    pose_polynomials : Callable[[np.ndarray], np.ndarray]
        Function to compute pose polynomials.
    pose_jacobian_dt0 : Callable[[np.ndarray], np.ndarray]
        Function to compute the pose Jacobian at time t0.
    pose_jacobian_dt1 : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function to compute the pose Jacobian at time t1.
    """

    nj: int
    pose_polynomials: Callable[[np.ndarray], np.ndarray]
    pose_jacobian_dt0: Callable[[np.ndarray], np.ndarray]
    pose_jacobian_dt1: Callable[[np.ndarray, np.ndarray], np.ndarray]


class MotionEquationsMeta(type):
    """
    Metaclass for motion equations, ensuring required fields are implemented.
    """

    _required_fields = {
        "nj",
        "pose_polynomials",
        "pose_jacobian_dt0",
        "pose_jacobian_dt1",
    }

    def __new__(cls, class_name, bases, attrs):
        if class_name == "AbstractMotionEquations":
            return type.__new__(cls, class_name, bases, attrs)

        for attr in cls._required_fields:
            if attr not in attrs:
                raise NotImplementedError(f"'{attr}' should be implemented")

        equations_class = super().__new__(cls, class_name, bases, attrs)
        kwargs = {k: getattr(equations_class, k) for k in cls._required_fields}
        return MotionEquations(**kwargs)


class AbstractMotionEquations(object, metaclass=MotionEquationsMeta):
    """
    Abstract base class for motion equations.
    """

    nj: int

    @staticmethod
    def pose_polynomials(qdt0: np.ndarray) -> np.ndarray:
        """
        Computes pose polynomials.

        Parameters
        ----------
        qdt0 : np.ndarray
            Joint coordinates.

        Returns
        -------
        np.ndarray
            Pose polynomials.
        """
        pass

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray) -> np.ndarray:
        """
        Computes the pose Jacobian at pose qdt0.

        Parameters
        ----------
        qdt0 : np.ndarray
            Joint coordinates.

        Returns
        -------
        np.ndarray
            Pose Jacobian at pose qdt0.
        """

        pass

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray) -> np.ndarray:
        """
        Computes the pose Jacobian derivative at Joint pose, qdt0, and velocity
        qdt1.

        Parameters
        ----------
        qdt0 : np.ndarray
            Joint coordinates.
        qdt1 : np.ndarray
            Rate of change of qdt0, qdt1.

        Returns
        -------
        np.ndarray
            Pose Jacobian derivative.
        """
        pass


class RevolutePolynomials(AbstractMotionEquations):
    """
    Motion equations for a revolute joint.
    """

    nj = 1

    @staticmethod
    def pose_polynomials(qdt0: np.ndarray):
        psi = qdt0[0]
        pose_states = jnp.array([0, 0, 0, 0, 0, psi])
        return pose_states

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray):
        pose_states_jacobian = np.array([0, 0, 0, 0, 0, 1])[:, None]
        return pose_states_jacobian

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray):
        pose_states_jacobian_dt1 = np.zeros((6, 1))
        return pose_states_jacobian_dt1


class TranslationalPolynomials(AbstractMotionEquations):
    """
    Motion equations for a translational joint.
    """

    nj = 1

    @staticmethod
    def pose_polynomials(qd0: np.ndarray):
        z = qd0[0]
        pose_states = jnp.array([0, 0, z, 0, 0, 0])
        return pose_states

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray):
        pose_states_jacobian = np.array([0, 0, 1, 0, 0, 0])[:, None]
        return pose_states_jacobian

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray):
        pose_states_jacobian_dt1 = np.zeros((6, 1))
        return pose_states_jacobian_dt1


class CylindricalPolynomials(AbstractMotionEquations):
    """
    Motion equations for a cylindrical joint.
    """

    nj = 1

    @staticmethod
    def pose_polynomials(qdt0: np.ndarray):
        z, psi = qdt0
        pose_states = jnp.array([0, 0, z, 0, 0, psi])
        return pose_states

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray):
        pose_states_jacobian = np.array([0, 0, 1, 0, 0, 1])[:, None]
        return pose_states_jacobian

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray):
        pose_states_jacobian_dt1 = np.zeros((6, 1))
        return pose_states_jacobian_dt1


class PlanarPolynomials(AbstractMotionEquations):
    """
    Motion equations for a Planar joint.
    """

    nj = 3

    @staticmethod
    def pose_polynomials(qdt0: np.ndarray):
        x, y, psi = qdt0
        pose_states = jnp.array([x, y, 0, 0, 0, psi])
        return pose_states

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray):
        pose_states_jacobian = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ]
        )
        return pose_states_jacobian

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray):
        pose_states_jacobian_dt1 = np.zeros((6, 3))
        return pose_states_jacobian_dt1


class FreePolynomials(AbstractMotionEquations):
    """
    Motion equations for a Free joint.
    """

    nj = 6

    @staticmethod
    def pose_polynomials(qdt0: np.ndarray):
        x, y, z, phi, theta, psi = qdt0
        pose_states = jnp.array([x, y, z, phi, theta, psi])
        return pose_states

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray):
        pose_states_jacobian = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        return pose_states_jacobian

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray):
        pose_states_jacobian_dt1 = np.zeros((6, 6))
        return pose_states_jacobian_dt1


def construct_motion_jacobians(
    pose_polynomials: Callable[[np.ndarray], np.ndarray]
) -> tuple[
    Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]
]:
    """
    Constructs the Jacobians for motion equations based on pose polynomials.

    Parameters
    ----------
    pose_polynomials : Callable[[np.ndarray], np.ndarray]
        Function to compute pose polynomials.

    Returns
    -------
    tuple
        A tuple containing two functions:
        - pose_jacobian_dt0: Callable[[np.ndarray], np.ndarray]
          Function to compute the pose Jacobian.
        - pose_jacobian_dt1: Callable[[np.ndarray, np.ndarray], np.ndarray]
          Function to compute the pose Jacobian derivative.
    """
    pose_jacobian_dt0 = jax.jit(jax.jacfwd(pose_polynomials))

    def pose_jacobian_dt0_mul_qdt1(qd0, qd1):
        return pose_jacobian_dt0(qd0) @ qd1

    pose_jacobian_dt1 = jax.jit(jax.jacfwd(pose_jacobian_dt0_mul_qdt1))

    return pose_jacobian_dt0, pose_jacobian_dt1
