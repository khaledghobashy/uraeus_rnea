from typing import NamedTuple, Callable, Tuple

# import jax
import numpy as np

# import numpy as np


def construct_motion_jacobians(
    pose_polynomials: Callable[[np.ndarray], np.ndarray]
) -> Tuple[
    Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]
]:
    """_summary_

    Parameters
    ----------
    pose_polynomials : Callable[[np.ndarray], np.ndarray]
        _description_

    Returns
    -------
    Tuple[ Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray] ]
        _description_
    """

    pose_jacobian_dt0 = jax.jit(jax.jacfwd(pose_polynomials))

    def pose_jacobian_dt0_mul_qdt1(qd0, qd1):
        return pose_jacobian_dt0(qd0) @ qd1

    pose_jacobian_dt1 = jax.jit(jax.jacfwd(pose_jacobian_dt0_mul_qdt1))

    return pose_jacobian_dt0, pose_jacobian_dt1


class MotionEquations(NamedTuple):

    nj: int
    pose_polynomials: Callable[[np.ndarray], np.ndarray]
    pose_jacobian_dt0: Callable[[np.ndarray], np.ndarray]
    pose_jacobian_dt1: Callable[[np.ndarray, np.ndarray], np.ndarray]


class MotionEquationsMeta(type):

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

    nj: int

    @staticmethod
    def pose_polynomials(qd0: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def pose_jacobian_dt0(qd0: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def pose_jacobian_dt1(qd0: np.ndarray, qd1: np.ndarray) -> np.ndarray:
        pass


class RevolutePolynomials(AbstractMotionEquations):

    nj = 1

    @staticmethod
    def pose_polynomials(qd0: np.ndarray):
        psi = qd0[0]
        pose_states = np.array([0, 0, psi, 0, 0, 0])
        return pose_states

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray):
        pose_states_jacobian = np.array([0, 0, 1, 0, 0, 0])[:, None]
        return pose_states_jacobian

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray):
        pose_states_jacobian_dt1 = np.zeros((6, 1))
        return pose_states_jacobian_dt1


class TranslationalPolynomials(AbstractMotionEquations):

    nj = 1

    @staticmethod
    def pose_polynomials(qd0: np.ndarray):
        z = qd0[0]
        pose_states = np.array([0, 0, 0, 0, 0, z])
        return pose_states

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray):
        pose_states_jacobian = np.array([0, 0, 0, 0, 0, 1])[:, None]
        return pose_states_jacobian

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray):
        pose_states_jacobian_dt1 = np.zeros((6, 1))
        return pose_states_jacobian_dt1


class PlanarPolynomials(AbstractMotionEquations):

    nj = 3

    @staticmethod
    def pose_polynomials(qdt0: np.ndarray):
        psi, x, y = qdt0
        pose_states = np.array([0, 0, psi, x, y, 0])
        return pose_states

    @staticmethod
    def pose_jacobian_dt0(qdt0: np.ndarray):
        pose_states_jacobian = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        return pose_states_jacobian

    @staticmethod
    def pose_jacobian_dt1(qdt0: np.ndarray, qdt1: np.ndarray):
        pose_states_jacobian_dt1 = np.zeros((6, 3))
        return pose_states_jacobian_dt1


class FreePolynomials(AbstractMotionEquations):

    nj = 6

    @staticmethod
    def pose_polynomials(qdt0: np.ndarray):
        phi, theta, psi, x, y, z = qdt0
        pose_states = np.array([phi, theta, psi, x, y, z])
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
