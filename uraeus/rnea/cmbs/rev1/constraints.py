from typing import NamedTuple, Callable
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from uraeus.rnea.cmbs.rev1.spatial_algebra import A, B


class BaseVector(Enum):
    i = 0
    j = 1
    k = 2


class ConstraintConstants(NamedTuple):
    u_i: np.ndarray
    u_j: np.ndarray
    M_i: np.ndarray
    M_j: np.ndarray


class SphericalConstraintConstants(NamedTuple):
    u_i: np.ndarray
    u_j: np.ndarray


class CoordinateConstraintConstants(NamedTuple):
    u_i: np.ndarray
    u_j: np.ndarray
    index: int


class DP1ConstraintConstants(NamedTuple):
    v1_i: np.ndarray
    v2_j: np.ndarray


class DP2ConstraintConstants(NamedTuple):
    u_i: np.ndarray
    u_j: np.ndarray
    v1_i: np.ndarray


class AngleConstraintConstants(NamedTuple):
    v1_i: np.ndarray
    v2_j: np.ndarray
    v3_i: np.ndarray


def at_point_constraint(
    qdt0_i: np.ndarray, qdt0_j: np.ndarray, ubar_i: np.ndarray, ubar_j: np.ndarray
) -> np.ndarray:
    ri, pi = jnp.split(qdt0_i, [3])
    rj, pj = jnp.split(qdt0_j, [3])

    Ai = A(pi)
    Aj = A(pj)

    u_i = ri + Ai @ ubar_i
    u_j = rj + Aj @ ubar_j

    residual = u_i - u_j

    return residual


def dp1_constraint(
    qdt0_i: np.ndarray, qdt0_j: np.ndarray, v1bar_i: np.ndarray, v2bar_j: np.ndarray
) -> np.ndarray:
    _, pi = jnp.split(qdt0_i, [3])
    _, pj = jnp.split(qdt0_j, [3])

    Ai = A(pi)
    Aj = A(pj)

    v1_i = Ai @ v1bar_i
    v2_j = Aj @ v2bar_j

    residual = v1_i @ v2_j

    return residual


def dp2_constraint(
    qdt0_i: np.ndarray,
    qdt0_j: np.ndarray,
    v1bar_i: np.ndarray,
    ubar_i: np.ndarray,
    ubar_j: np.ndarray,
) -> np.ndarray:
    _, pi = jnp.split(qdt0_i, [3])

    dij = at_point_constraint(qdt0_i, qdt0_j, ubar_i, ubar_j)

    Ai = A(pi)

    v1_i = Ai @ v1bar_i

    residual = v1_i @ dij

    return residual


class AbstractConstraintEquations(NamedTuple):
    nc: int = None

    @staticmethod
    def pos_constraint(
        constants: ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def vel_constraint(
        constants: ConstraintConstants, qdt0_i: np.ndarray, qdt0_j: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def acc_constraint(
        constants: ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def jacobians(
        constants: ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def construct_constants(self, constraint_constants: ConstraintConstants):
        raise NotImplementedError


class SphericalConstraint(AbstractConstraintEquations):
    nc: int = 3

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        constants: SphericalConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        ubar_i = constants.u_i
        ubar_j = constants.u_j

        residual = at_point_constraint(qdt0_i, qdt0_j, ubar_i, ubar_j)

        return residual

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        constants: SphericalConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        residual = np.zeros((3,))
        return residual

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        constants: SphericalConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        # rdt0_i, pdt0_i = np.split(qdt0_i, [3])
        # rdt0_j, pdt0_j = np.split(qdt0_j, [3])
        rdt1_i, pdt1_i = jnp.split(qdt1_i, [3])
        rdt1_j, pdt1_j = jnp.split(qdt1_j, [3])
        u_i = constants.u_i
        u_j = constants.u_j
        residual = (B(pdt1_i, u_i) @ pdt1_i) - (B(pdt1_j, u_j) @ pdt1_j)
        return residual

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def jacobians(
        constants: SphericalConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_i = constants.u_i
        u_j = constants.u_j
        ri, pi = jnp.split(qdt0_i, [3])
        rj, pj = jnp.split(qdt0_j, [3])
        I = np.eye(3)
        jacobian_i = jnp.hstack((I, B(pi, u_i)))
        jacobian_j = -jnp.hstack((I, B(pj, u_j)))
        return jacobian_i, jacobian_j

    @staticmethod
    def construct_constants(constraint_constants: ConstraintConstants):
        return SphericalConstraintConstants(
            constraint_constants.u_i, constraint_constants.u_j
        )


class DP1Constraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1, v2) on two different bodies
    (body_i, body_j) to be perpendicular at all time by setting their
    dot-product to zero.

    Parameters
    ----------

    Notes
    -----
    TODO

    """

    v1_i_index: BaseVector
    v2_j_index: BaseVector
    nc: int = 1

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        constants: DP1ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        v1_i = constants.v1_i
        v2_j = constants.v2_j

        ri, pi = jnp.split(qdt0_i, [3])
        rj, pj = jnp.split(qdt0_j, [3])

        Ai = A(pi)
        Aj = A(pj)

        residual = (Ai @ v1_i).T @ (Aj @ v2_j)

        return jnp.array([residual])

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        constants: DP1ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        residual = np.array([0])
        return residual

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        constants: DP1ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        v1_i = constants.v1_i
        v2_j = constants.v2_j

        rdt0_i, pdt0_i = np.split(qdt0_i, [3])
        rdt0_j, pdt0_j = np.split(qdt0_j, [3])
        rdt1_i, pdt1_i = np.split(qdt1_i, [3])
        rdt1_j, pdt1_j = np.split(qdt1_j, [3])

        Ai = A(pdt0_i)
        Aj = A(pdt0_j)

        v1 = Ai @ v1_i
        v2 = Aj @ v2_j

        residual = (
            v1.T @ B(pdt1_j, v2_j) @ pdt1_j
            + v2.T @ B(pdt1_i, v1_i) @ pdt1_i
            + 2 * (B(pdt0_i, v1_i) @ pdt1_i).T @ (B(pdt0_j, v2_j) @ pdt1_j)
        )
        return jnp.array([residual])

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def jacobians(
        constants: DP1ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> tuple[np.ndarray, np.ndarray]:
        v1_i = constants.v1_i
        v2_j = constants.v2_j
        Z = np.zeros((3,))

        rdt0_i, pdt0_i = jnp.split(qdt0_i, [3])
        rdt0_j, pdt0_j = jnp.split(qdt0_j, [3])

        Ai = A(pdt0_i)
        Aj = A(pdt0_j)

        v1 = Ai @ v1_i
        v2 = Aj @ v2_j

        jacobian_i = jnp.hstack((Z, v2 @ B(pdt0_i, v1_i)))
        jacobian_j = jnp.hstack((Z, v1 @ B(pdt0_j, v2_j)))
        return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return DP1ConstraintConstants(
            constraint_constants.M_i[:, self.v1_i_index],
            constraint_constants.M_j[:, self.v2_j_index],
        )


class DP2Constraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1, v2) on two different bodies
    (body_i, body_j) to be perpendicular at all time by setting their
    dot-product to zero.

    Parameters
    ----------

    Notes
    -----
    TODO

    """

    v1_i_index: int
    nc: int = 1

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        constants: DP2ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        v1_i = constants.v1_i
        u_i = constants.u_i
        u_j = constants.u_j

        ri, pi = jnp.split(qdt0_i, [3])
        rj, pj = jnp.split(qdt0_j, [3])

        Ai = A(pi)
        Aj = A(pj)

        v1 = Ai @ v1_i

        dij = (ri + Ai @ u_i) - (rj + Aj @ u_j)

        residual = v1.T @ dij

        return jnp.array([residual])

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        constants: DP2ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        residual = np.array([0])
        return residual

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        constants: DP2ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        v1_i = constants.v1_i
        u_i = constants.u_i
        u_j = constants.u_j

        rdt0_i, pdt0_i = np.split(qdt0_i, [3])
        rdt0_j, pdt0_j = np.split(qdt0_j, [3])
        rdt1_i, pdt1_i = np.split(qdt1_i, [3])
        rdt1_j, pdt1_j = np.split(qdt1_j, [3])

        Ai = A(pdt0_i)
        Aj = A(pdt0_j)

        v1 = Ai @ v1_i

        dij_dt0 = (rdt0_i + Ai @ u_i) - (rdt0_j + Aj @ u_j)
        dij_dt1 = (rdt1_i + B(pdt0_i, u_i) @ pdt1_i) - (
            rdt1_j + B(pdt0_j, u_j) @ pdt1_j
        )

        residual = (
            v1 @ (B(pdt1_i, u_i) @ pdt1_i - B(pdt1_j, u_j) @ pdt1_j)
            + dij_dt0 @ B(pdt1_i, v1_i) @ pdt1_i
            + 2 * (B(pdt0_i, v1_i) @ pdt1_i) @ dij_dt1
        )
        return jnp.array([residual])

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def jacobians(
        constants: DP2ConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_i = constants.u_i
        u_j = constants.u_j
        v1_i = constants.v1_i

        rdt0_i, pdt0_i = np.split(qdt0_i, [3])
        rdt0_j, pdt0_j = np.split(qdt0_j, [3])

        Ai = A(pdt0_i)
        Aj = A(pdt0_j)

        v1 = Ai @ v1_i

        dij_dt0 = (rdt0_i + Ai @ u_i) - (rdt0_j + Aj @ u_j)

        jacobian_i = jnp.stack(
            (v1, dij_dt0.T @ B(pdt0_i, v1_i) + v1.T @ B(pdt0_i, u_i))
        )
        jacobian_j = jnp.stack((-v1.T, -v1.T @ B(pdt0_j, u_j)))

        return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return DP2ConstraintConstants(
            constraint_constants.u_i,
            constraint_constants.u_j,
            constraint_constants.M_i[:, self.v1_i_index],
        )


class DistanceConstraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1, v2) on two different bodies
    (body_i, body_j) to be perpendicular at all time by setting their
    dot-product to zero.

    Parameters
    ----------

    Notes
    -----
    TODO

    """

    nc: int = 1

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def pos_constraint(
        constants: DP2ConstraintConstants,
        actuation: Callable[[float], float],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        u_i = constants.u_i
        u_j = constants.u_j
        v1_i = constants.v1_i

        ri, pi = jnp.split(qdt0_i, [3])
        rj, pj = jnp.split(qdt0_j, [3])

        Ai = A(pi)
        Aj = A(pj)

        v1 = Ai @ v1_i
        dij = (ri + Ai @ u_i) - (rj + Aj @ u_j)

        distance = actuation(t)
        residual = (v1.T @ dij) - distance
        # jax.debug.print("Distance = {}", distance)
        # jax.debug.print("residual = {}", residual)
        # jax.debug.print("residual = {}", residual)

        return jnp.array([residual])

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def vel_constraint(
        constants: DP2ConstraintConstants,
        actuation: Callable[[float], float],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        residual = np.array([0])
        return residual

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def acc_constraint(
        constants: DP2ConstraintConstants,
        actuation: Callable[[float], float],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        u_i = constants.u_i
        u_j = constants.u_j

        rdt0_i, pdt0_i = jnp.split(qdt0_i, [3])
        rdt0_j, pdt0_j = jnp.split(qdt0_j, [3])
        rdt1_i, pdt1_i = jnp.split(qdt1_i, [3])
        rdt1_j, pdt1_j = jnp.split(qdt1_j, [3])

        Ai = A(pdt0_i)
        Aj = A(pdt0_j)

        dij_dt0 = (rdt0_i + Ai @ u_i) - (rdt0_j + Aj @ u_j)
        dij_dt1 = (rdt1_i + B(pdt0_i, u_i) @ pdt1_i) - (
            rdt1_j + B(pdt0_j, u_j) @ pdt1_j
        )

        residual = 2 * dij_dt0 @ (B(pdt1_i, u_i) @ pdt1_i - B(pdt1_j, u_j) @ pdt1_j) + (
            2 * dij_dt1 @ dij_dt1
        )
        return jnp.array([residual])

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def jacobians(
        constants: DP2ConstraintConstants,
        actuation: Callable[[float], float],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_i = constants.u_i
        u_j = constants.u_j

        rdt0_i, pdt0_i = jnp.split(qdt0_i, [3])
        rdt0_j, pdt0_j = jnp.split(qdt0_j, [3])

        Ai = A(pdt0_i)
        Aj = A(pdt0_j)

        dij_dt0 = (rdt0_i + Ai @ u_i) - (rdt0_j + Aj @ u_j)

        jacobian_i = jnp.stack((2 * dij_dt0 @ np.eye(3), 2 * dij_dt0 @ B(pdt0_i, u_i)))
        jacobian_j = jnp.stack(
            (-2 * dij_dt0 @ np.eye(3), -2 * dij_dt0 @ B(pdt0_j, u_j))
        )

        return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return DP2ConstraintConstants(
            constraint_constants.u_i,
            constraint_constants.u_j,
            constraint_constants.M_i[:, 2],
        )


class AngleConstraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1, v2) on two different bodies
    (body_i, body_j) to be perpendicular at all time by setting their
    dot-product to zero.

    Parameters
    ----------

    Notes
    -----
    TODO

    """

    v1_i_index: BaseVector
    v2_j_index: BaseVector
    v3_i_index: BaseVector
    nc: int = 1

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def pos_constraint(
        constants: AngleConstraintConstants,
        actuation: Callable[[float], float],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        v1_i = constants.v1_i
        v2_j = constants.v2_j
        v3_i = constants.v3_i

        _, pi = jnp.split(qdt0_i, [3])
        _, pj = jnp.split(qdt0_j, [3])

        Ai = A(pi)
        Aj = A(pj)

        v1 = Ai @ v1_i
        v2 = Aj @ v2_j
        v3 = Ai @ v3_i

        theta = actuation(t)
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        residual = (v3.T @ v2) * c - (v1.T @ v2) * s

        return jnp.array([residual])

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def vel_constraint(
        constants: AngleConstraintConstants,
        actuation: Callable[[float], float],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        residual = np.array([0])
        return residual

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def acc_constraint(
        constants: AngleConstraintConstants,
        actuation: Callable[[float], float],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        v1_i = constants.v1_i
        v2_j = constants.v2_j
        v3_i = constants.v3_i

        _, pdt0_i = np.split(qdt0_i, [3])
        _, pdt0_j = np.split(qdt0_j, [3])
        _, pdt1_i = np.split(qdt1_i, [3])
        _, pdt1_j = np.split(qdt1_j, [3])

        Ai = A(pdt0_i)
        Aj = A(pdt0_j)

        v1 = Ai @ v1_i
        v2 = Aj @ v2_j
        v3 = Ai @ v3_i

        theta = actuation(t)
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        residual = -(
            (c * v3.T - s * v1.T) @ B(pdt1_j, v2_j) @ pdt1_j
            + v2.T @ (c * B(pdt1_i, v3_i) - s * B(pdt1_i, v1_i)) @ pdt1_i
            + 2
            * (c * B(pdt0_i, v3_i) @ pdt1_i - s * B(pdt0_i, v1_i) @ pdt1_i).T
            @ (B(pdt0_j, v2_j) @ pdt1_j)
        )
        return jnp.array([residual])

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def jacobians(
        constants: AngleConstraintConstants,
        actuation: Callable[[float], float],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        v1_i = constants.v1_i
        v2_j = constants.v2_j
        v3_i = constants.v3_i

        Z = np.zeros((1, 3))

        rdt0_i, pdt0_i = jnp.split(qdt0_i, [3])
        rdt0_j, pdt0_j = jnp.split(qdt0_j, [3])

        Ai = A(pdt0_i)
        Aj = A(pdt0_j)

        v1 = Ai @ v1_i
        v2 = Aj @ v2_j
        v3 = Ai @ v3_i

        theta = actuation(t)
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        jacobian_i = jnp.hstack((Z, v2.T @ (c * B(pdt0_i, v3_i) - s * B(pdt0_i, v1_i))))
        jacobian_j = jnp.hstack((Z, (c * v3.T - s * v1.T) * B(pdt0_j, v2_j)))

        return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return AngleConstraintConstants(
            constraint_constants.M_i[:, self.v1_i_index],
            constraint_constants.M_j[:, self.v2_j_index],
            constraint_constants.M_i[:, self.v3_i_index],
        )


class CoordConstraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1, v2) on two different bodies
    (body_i, body_j) to be perpendicular at all time by setting their
    dot-product to zero.

    Parameters
    ----------

    Notes
    -----
    TODO

    """

    coordinate_index: int
    nc: int = 1

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        constants: CoordinateConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        u_i = constants.u_i
        u_j = constants.u_j
        index = constants.index

        ri, pi = jnp.split(qdt0_i, [3])
        rj, pj = jnp.split(qdt0_j, [3])

        Ai = A(pi)
        Aj = A(pj)

        residual = (ri + Ai @ u_i) - (rj + Aj @ u_j)

        return np.array(residual[index])

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        constants: CoordinateConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        residual = np.array([0])
        return residual

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        constants: CoordinateConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        # rdt0_i, pdt0_i = np.split(qdt0_i, [3])
        # rdt0_j, pdt0_j = np.split(qdt0_j, [3])
        rdt1_i, pdt1_i = np.split(qdt1_i, [3])
        rdt1_j, pdt1_j = np.split(qdt1_j, [3])
        u_i = constants.u_i
        u_j = constants.u_j
        index = constants.index
        residual = (B(pdt1_i, u_i) @ pdt1_i) - (B(pdt1_j, u_j) @ pdt1_j)
        return np.array(residual[index])

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def jacobians(
        constants: CoordinateConstraintConstants,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_i = constants.u_i
        u_j = constants.u_j
        index = constants.index
        ri, pi = jnp.split(qdt0_i, [3])
        rj, pj = jnp.split(qdt0_j, [3])
        I = np.eye(3)
        jacobian_i = jnp.hstack((I, B(pi, u_i)))[index, :]
        jacobian_j = -jnp.hstack((I, B(pj, u_j)))[index, :]
        return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return CoordinateConstraintConstants(
            constraint_constants.u_i,
            constraint_constants.u_j,
            self.coordinate_index,
        )
