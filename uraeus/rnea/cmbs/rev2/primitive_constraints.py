"""Module for common primitive multibody constraints' functions, e.g. 
spherical, dot-product, ... etc.

Operations are tailored for Euler-parameters (quaternions) orientation 
representation.
"""
from typing import NamedTuple, Callable, ClassVar
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from uraeus.rnea.cmbs.rev2.spatial_algebra import A, B


class ConstraintConstants(NamedTuple):
    u_P: np.ndarray
    u_S: np.ndarray
    R_J1P: np.ndarray
    R_J2S: np.ndarray


class SphericalConstraintConstants(NamedTuple):
    u_P: np.ndarray
    u_S: np.ndarray


class DP1ConstraintConstants(NamedTuple):
    v1_P: np.ndarray
    v2_S: np.ndarray


class DP2ConstraintConstants(NamedTuple):
    u_P: np.ndarray
    u_S: np.ndarray
    v1_P: np.ndarray


class AngleConstraintConstants(NamedTuple):
    v1_P: np.ndarray
    v2_S: np.ndarray
    v3_P: np.ndarray


@dataclass
class AbstractConstraintEquations(object):
    nc: ClassVar[int]

    @staticmethod
    def pos_constraint(
        constants: ConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        *args,
    ):
        pass

    @staticmethod
    def vel_constraint(
        constants: ConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        *args,
    ):
        pass

    @staticmethod
    def acc_constraint(
        constants: ConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        qdt1P_G: np.ndarray,
        qdt1S_G: np.ndarray,
        *args,
    ):
        pass


class SphericalConstraint(NamedTuple):
    nc: int = 3

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        constants: SphericalConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        *args,
    ) -> np.ndarray:
        u_P, u_S = constants

        rdt0P_G, pdt0P_G = jnp.split(qdt0P_G, [3])
        rdt0S_G, pdt0S_G = jnp.split(qdt0S_G, [3])

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        uP_G = rdt0P_G + R_PG @ u_P
        uS_G = rdt0S_G + R_SG @ u_S

        residual = uP_G - uS_G

        return residual

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        constants: SphericalConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        *args,
    ) -> np.ndarray:
        residual = np.zeros((3,))
        return residual

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        constants: SphericalConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        qdt1P_G: np.ndarray,
        qdt1S_G: np.ndarray,
        *args,
    ) -> np.ndarray:
        u_P, u_S = constants

        _, pdt0P_G = jnp.split(qdt0P_G, [3])
        _, pdt0S_G = jnp.split(qdt0S_G, [3])
        _, pdt1P_G = jnp.split(qdt1P_G, [3])
        _, pdt1S_G = jnp.split(qdt1S_G, [3])

        residual = (B(pdt0P_G, u_P) @ pdt1P_G) - (B(pdt0S_G, u_S) @ pdt1S_G)

        return residual

    # @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    # def jacobians(
    #     constants: SphericalConstraintConstants,
    #     qdt0P_G: np.ndarray,
    #     qdt0S_G: np.ndarray,
    #     *args,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     u_P = constants.u_P
    #     u_S = constants.u_S
    #     ri, pdt0P_G = jnp.split(qdt0P_G, [3])
    #     rj, pdt0S_G = jnp.split(qdt0S_G, [3])
    #     I = np.eye(3)
    #     jacobian_i = jnp.hstack((I, B(pdt0P_G, u_P)))
    #     jacobian_j = -jnp.hstack((I, B(pdt0S_G, u_S)))
    #     return jacobian_i, jacobian_j

    @staticmethod
    def construct_constants(constraint_constants: ConstraintConstants):
        return SphericalConstraintConstants(
            constraint_constants.u_P, constraint_constants.u_S
        )


class DP1Constraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1_G, v2_G) on two different bodies
    (body_i, body_j) to be perpendicular at all time by setting their
    dot-product to zero.

    Parameters
    ----------

    Notes
    -----
    TODO

    """

    v1_P_index: int
    v2_S_index: int

    nc: int = 1

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        constants: DP1ConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        *args,
    ) -> np.ndarray:
        v1_P, v2_S = constants

        _, pdt0P_G = jnp.split(qdt0P_G, [3])
        _, pdt0S_G = jnp.split(qdt0S_G, [3])

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        residual = (R_PG @ v1_P) @ (R_SG @ v2_S)

        return jnp.array([residual])

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        constants: DP1ConstraintConstants, qdt0P_G: np.ndarray, qdt0S_G: np.ndarray
    ) -> np.ndarray:
        residual = np.array([0])
        return residual

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        constants: DP1ConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        qdt1P_G: np.ndarray,
        qdt1S_G: np.ndarray,
        *args,
    ) -> np.ndarray:
        v1_P, v2_S = constants

        _, pdt0P_G = jnp.split(qdt0P_G, [3])
        _, pdt0S_G = jnp.split(qdt0S_G, [3])
        _, pdt1P_G = jnp.split(qdt1P_G, [3])
        _, pdt1S_G = jnp.split(qdt1S_G, [3])

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        v1_G = R_PG @ v1_P
        v2_G = R_SG @ v2_S

        residual = (
            v1_G.T @ B(pdt1S_G, v2_S) @ pdt1S_G
            + v2_G.T @ B(pdt1P_G, v1_P) @ pdt1P_G
            + 2 * (B(pdt0P_G, v1_P) @ pdt1P_G).T @ (B(pdt0S_G, v2_S) @ pdt1S_G)
        )
        return jnp.array([residual])

    # @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    # def jacobians(
    #     constants: DP1ConstraintConstants,
    #     qdt0P_G: np.ndarray,
    #     qdt0S_G: np.ndarray,
    #     *args,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     v1_P = constants.v1_P
    #     v2_S = constants.v2_S
    #     Z = np.zeros((3,))

    #     rdt0P_G, pdt0P_G = jnp.split(qdt0P_G, [3])
    #     rdt0P_S, pdt0S_G = jnp.split(qdt0S_G, [3])

    #     R_PG = A(pdt0P_G)
    #     R_SG = A(pdt0S_G)

    #     v1_G = R_PG @ v1_P
    #     v2_G = R_SG @ v2_S

    #     jacobian_i = jnp.hstack((Z, v2_G @ B(pdt0P_G, v1_P)))
    #     jacobian_j = jnp.hstack((Z, v1_G @ B(pdt0S_G, v2_S)))
    #     return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return DP1ConstraintConstants(
            constraint_constants.R_J1P[:, self.v1_P_index],
            constraint_constants.R_J2S[:, self.v2_S_index],
        )


class DP2Constraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1_G, v2_G) on two different bodies
    (body_i, body_j) to be perpendicular at all time by setting their
    dot-product to zero.

    Parameters
    ----------

    Notes
    -----
    TODO

    """

    v1_P_index: int
    nc: int = 1

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        constants: DP2ConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        *args,
    ) -> np.ndarray:
        u_P, u_S, v1_P = constants

        rdt0P_G, pdt0P_G = jnp.split(qdt0P_G, [3])
        rdt0S_G, pdt0S_G = jnp.split(qdt0S_G, [3])

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        v1_G = R_PG @ v1_P

        dij = (rdt0P_G + R_PG @ u_P) - (rdt0S_G + R_SG @ u_S)

        residual = v1_G.T @ dij

        return jnp.array([residual])

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        constants: DP2ConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        *args,
    ) -> np.ndarray:
        residual = np.array([0])
        return residual

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        constants: DP2ConstraintConstants,
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        qdt1P_G: np.ndarray,
        qdt1S_G: np.ndarray,
        *args,
    ) -> np.ndarray:
        u_P, u_S, v1_P = constants

        rdt0P_G, pdt0P_G = jnp.split(qdt0P_G, [3])
        rdt0P_S, pdt0S_G = jnp.split(qdt0S_G, [3])
        rdt1P_G, pdt1P_G = jnp.split(qdt1P_G, [3])
        rdt1P_S, pdt1S_G = jnp.split(qdt1S_G, [3])

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        v1_G = R_PG @ v1_P

        dij_dt0 = (rdt0P_G + R_PG @ u_P) - (rdt0P_S + R_SG @ u_S)
        dij_dt1 = (rdt1P_G + B(pdt0P_G, u_P) @ pdt1P_G) - (
            rdt1P_S + B(pdt0S_G, u_S) @ pdt1S_G
        )

        residual = (
            v1_G @ (B(pdt1P_G, u_P) @ pdt1P_G - B(pdt1S_G, u_S) @ pdt1S_G)
            + dij_dt0 @ B(pdt1P_G, v1_P) @ pdt1P_G
            + 2 * (B(pdt0P_G, v1_P) @ pdt1P_G) @ dij_dt1
        )
        return jnp.array([residual])

    # @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    # def jacobians(
    #     constants: DP2ConstraintConstants,
    #     qdt0P_G: np.ndarray,
    #     qdt0S_G: np.ndarray,
    #     t: float,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     u_P = constants.u_P
    #     u_S = constants.u_S
    #     v1_P = constants.v1_P

    #     rdt0P_G, pdt0P_G = np.split(qdt0P_G, [3])
    #     rdt0P_S, pdt0S_G = np.split(qdt0S_G, [3])

    #     R_PG = A(pdt0P_G)
    #     R_SG = A(pdt0S_G)

    #     v1_G = R_PG @ v1_P

    #     dij_dt0 = (rdt0P_G + R_PG @ u_P) - (rdt0P_S + R_SG @ u_S)

    #     jacobian_i = jnp.stack(
    #         (v1_G, dij_dt0.T @ B(pdt0P_G, v1_P) + v1_G.T @ B(pdt0P_G, u_P))
    #     )
    #     jacobian_j = jnp.stack((-v1_G.T, -v1_G.T @ B(pdt0S_G, u_S)))

    #     return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return DP2ConstraintConstants(
            constraint_constants.u_P,
            constraint_constants.u_S,
            constraint_constants.R_J1P[:, self.v1_P_index],
        )


class DistanceConstraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1_G, v2_G) on two different bodies
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
    # @partial(jax.jit, static_argnums=(0, 1))
    def pos_constraint(
        constants: DP2ConstraintConstants,
        actuation: Callable[[float], float],
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        t: float,
        *args,
    ) -> np.ndarray:
        residual = DP2Constraint.pos_constraint(
            constants, qdt0P_G, qdt0S_G
        ) - actuation(t)

        return residual

    @staticmethod
    # @partial(jax.jit, static_argnums=(0, 1))
    def vel_constraint(
        constants: DP2ConstraintConstants,
        actuation: Callable[[float], float],
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        t: float,
        *args,
    ) -> np.ndarray:
        actuation_dt1 = jax.grad(actuation)

        residual = jnp.array([-actuation_dt1(t)])
        return residual

    @staticmethod
    # @partial(jax.jit, static_argnums=(0, 1))
    def acc_constraint(
        constants: DP2ConstraintConstants,
        actuation: Callable[[float], float],
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        qdt1P_G: np.ndarray,
        qdt1S_G: np.ndarray,
        t: float,
        *args,
    ) -> np.ndarray:
        actuation_dt2 = jax.grad(jax.grad(actuation))

        residual = DP2Constraint.acc_constraint(
            constants, qdt0P_G, qdt0S_G, qdt1P_G, qdt1S_G
        )
        return residual - actuation_dt2(t)
        # u_P, u_S, v1_P = constants

        # rdt0P_G, pdt0P_G = jnp.split(qdt0P_G, [3])
        # rdt0P_S, pdt0S_G = jnp.split(qdt0S_G, [3])
        # rdt1P_G, pdt1P_G = jnp.split(qdt1P_G, [3])
        # rdt1P_S, pdt1S_G = jnp.split(qdt1S_G, [3])

        # R_PG = A(pdt0P_G)
        # R_SG = A(pdt0S_G)

        # dij_dt0 = (rdt0P_G + R_PG @ u_P) - (rdt0P_S + R_SG @ u_S)
        # dij_dt1 = (rdt1P_G + B(pdt0P_G, u_P) @ pdt1P_G) - (
        #     rdt1P_S + B(pdt0S_G, u_S) @ pdt1S_G
        # )

        # residual = 2 * dij_dt0 @ (
        #     B(pdt1P_G, u_P) @ pdt1P_G - B(pdt1S_G, u_S) @ pdt1S_G
        # ) + (2 * dij_dt1 @ dij_dt1)

        # return jnp.array([residual])

    @staticmethod
    # @partial(jax.jit, static_argnums=(0, 1))
    def jacobians(
        constants: DP2ConstraintConstants,
        actuation: Callable[[float], float],
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        t: float,
        *args,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_P, u_S, v1_P = constants

        rdt0P_G, pdt0P_G = jnp.split(qdt0P_G, [3])
        rdt0P_S, pdt0S_G = jnp.split(qdt0S_G, [3])

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        dij_dt0 = (rdt0P_G + R_PG @ u_P) - (rdt0P_S + R_SG @ u_S)

        jacobian_i = jnp.stack((2 * dij_dt0 @ np.eye(3), 2 * dij_dt0 @ B(pdt0P_G, u_P)))
        jacobian_j = jnp.stack(
            (-2 * dij_dt0 @ np.eye(3), -2 * dij_dt0 @ B(pdt0S_G, u_S))
        )

        return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return DP2ConstraintConstants(
            constraint_constants.u_P,
            constraint_constants.u_S,
            constraint_constants.R_J1P[:, 2],
        )


class AngleConstraint(NamedTuple):
    """A primitive dot-product Constraint.
    This constraint enforces two vectors (v1_G, v2_G) on two different bodies
    (body_i, body_j) to be perpendicular at all time by setting their
    dot-product to zero.

    Parameters
    ----------

    Notes
    -----
    TODO

    """

    v1_P_index: int
    v2_S_index: int
    v3_P_index: int

    nc: int = 1

    @staticmethod
    # @partial(jax.jit, static_argnums=(0, 1))
    def pos_constraint(
        constants: AngleConstraintConstants,
        actuation: Callable[[float], float],
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        t: float,
        *args,
    ) -> np.ndarray:
        v1_P, v2_S, v3_P = constants

        _, pdt0P_G = np.split(qdt0P_G, [3])
        _, pdt0S_G = np.split(qdt0S_G, [3])

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        v1_G = R_PG @ v1_P
        v2_G = R_SG @ v2_S
        v3_G = R_PG @ v3_P

        theta = actuation(t)
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        residual = (v3_G.T @ v2_G) * c - (v1_G.T @ v2_G) * s

        return jnp.array([residual])

    @staticmethod
    # @partial(jax.jit, static_argnums=(0, 1))
    def vel_constraint(
        constants: AngleConstraintConstants,
        actuation: Callable[[float], float],
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        t: float,
        *args,
    ) -> np.ndarray:
        actuation_dt1 = jax.grad(actuation)

        residual = jnp.array([-actuation_dt1(t)])
        return residual

    @staticmethod
    # @partial(jax.jit, static_argnums=(0, 1))
    def acc_constraint(
        constants: AngleConstraintConstants,
        actuation: Callable[[float], float],
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        qdt1P_G: np.ndarray,
        qdt1S_G: np.ndarray,
        t: float,
        *args,
    ) -> np.ndarray:
        v1_P, v2_S, v3_P = constants

        _, pdt0P_G = np.split(qdt0P_G, [3])
        _, pdt0S_G = np.split(qdt0S_G, [3])
        _, pdt1P_G = np.split(qdt1P_G, [3])
        _, pdt1S_G = np.split(qdt1S_G, [3])

        actuation_dt2 = jax.grad(jax.grad(actuation))

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        v1_G = R_PG @ v1_P
        v2_G = R_SG @ v2_S
        v3_G = R_PG @ v3_P

        theta = actuation(t)
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        residual = -(
            (c * v3_G.T - s * v1_G.T) @ B(pdt1S_G, v2_S) @ pdt1S_G
            + v2_G.T @ (c * B(pdt1P_G, v3_P) - s * B(pdt1P_G, v1_P)) @ pdt1P_G
            + 2
            * (c * B(pdt0P_G, v3_P) @ pdt1P_G - s * B(pdt0P_G, v1_P) @ pdt1P_G).T
            @ (B(pdt0S_G, v2_S) @ pdt1S_G)
        )
        return jnp.array([residual - actuation_dt2(t)])

    @staticmethod
    # @partial(jax.jit, static_argnums=(0, 1))
    def jacobians(
        constants: AngleConstraintConstants,
        actuation: Callable[[float], float],
        qdt0P_G: np.ndarray,
        qdt0S_G: np.ndarray,
        t: float,
        *args,
    ) -> tuple[np.ndarray, np.ndarray]:
        v1_P, v2_S, v3_P = constants

        Z = np.zeros((1, 3))

        rdt0P_G, pdt0P_G = jnp.split(qdt0P_G, [3])
        rdt0P_S, pdt0S_G = jnp.split(qdt0S_G, [3])

        R_PG = A(pdt0P_G)
        R_SG = A(pdt0S_G)

        v1_G = R_PG @ v1_P
        v2_G = R_SG @ v2_S
        v3_G = R_PG @ v3_P

        theta = actuation(t)
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        jacobian_i = jnp.hstack(
            (Z, v2_G.T @ (c * B(pdt0P_G, v3_P) - s * B(pdt0P_G, v1_P)))
        )
        jacobian_j = jnp.hstack((Z, (c * v3_G.T - s * v1_G.T) * B(pdt0S_G, v2_S)))

        return jacobian_i, jacobian_j

    def construct_constants(
        self,
        constraint_constants: ConstraintConstants,
    ):
        return AngleConstraintConstants(
            constraint_constants.R_J1P[:, self.v1_P_index],
            constraint_constants.R_J2S[:, self.v2_S_index],
            constraint_constants.R_J1P[:, self.v3_P_index],
        )


if __name__ == "__main__":
    qdt0P_G = np.arange(7, dtype=np.float64)
    u = np.array([0.0, 0, 0])
    print(jax.make_jaxpr(SphericalConstraint.pos_constraint)((u, u), qdt0P_G, qdt0P_G))


# class CoordConstraint(NamedTuple):
#     """A primitive dot-product Constraint.
#     This constraint enforces two vectors (v1_G, v2_G) on two different bodies
#     (body_i, body_j) to be perpendicular at all time by setting their
#     dot-product to zero.

#     Parameters
#     ----------

#     Notes
#     -----
#     TODO

#     """

#     coordinate_index: int
#     nc: int = 1

#     @staticmethod
#     @partial(jax.jit, static_argnums=(0,))
#     def pos_constraint(
#         constants: CoordinateConstraintConstants,
#         qdt0P_G: np.ndarray,
#         qdt0S_G: np.ndarray,
#         *args,
#     ) -> np.ndarray:
#         u_P = constants.u_P
#         u_S = constants.u_S
#         index = constants.index

#         ri, pdt0P_G = jnp.split(qdt0P_G, [3])
#         rj, pdt0S_G = jnp.split(qdt0S_G, [3])

#         R_PG = A(pdt0P_G)
#         R_SG = A(pdt0S_G)

#         residual = (ri + R_PG @ u_P) - (rj + R_SG @ u_S)

#         return np.array(residual[index])

#     @staticmethod
#     @partial(jax.jit, static_argnums=(0,))
#     def vel_constraint(
#         constants: CoordinateConstraintConstants,
#         qdt0P_G: np.ndarray,
#         qdt0S_G: np.ndarray,
#         *args,
#     ) -> np.ndarray:
#         residual = np.array([0])
#         return residual

#     @staticmethod
#     @partial(jax.jit, static_argnums=(0,))
#     def acc_constraint(
#         constants: CoordinateConstraintConstants,
#         qdt0P_G: np.ndarray,
#         qdt0S_G: np.ndarray,
#         qdt1P_G: np.ndarray,
#         qdt1S_G: np.ndarray,
#         *args,
#     ) -> np.ndarray:
#         # rdt0P_G, pdt0P_G = np.split(qdt0P_G, [3])
#         # rdt0P_S, pdt0S_G = np.split(qdt0S_G, [3])
#         rdt1P_G, pdt1P_G = np.split(qdt1P_G, [3])
#         rdt1P_S, pdt1S_G = np.split(qdt1S_G, [3])
#         u_P = constants.u_P
#         u_S = constants.u_S
#         index = constants.index
#         residual = (B(pdt1P_G, u_P) @ pdt1P_G) - (B(pdt1S_G, u_S) @ pdt1S_G)
#         return np.array(residual[index])

#     @staticmethod
#     @partial(jax.jit, static_argnums=(0,))
#     def jacobians(
#         constants: CoordinateConstraintConstants,
#         qdt0P_G: np.ndarray,
#         qdt0S_G: np.ndarray,
#         *args,
#     ) -> tuple[np.ndarray, np.ndarray]:
#         u_P = constants.u_P
#         u_S = constants.u_S
#         index = constants.index
#         ri, pdt0P_G = jnp.split(qdt0P_G, [3])
#         rj, pdt0S_G = jnp.split(qdt0S_G, [3])
#         I = np.eye(3)
#         jacobian_i = jnp.hstack((I, B(pdt0P_G, u_P)))[index, :]
#         jacobian_j = -jnp.hstack((I, B(pdt0S_G, u_S)))[index, :]
#         return jacobian_i, jacobian_j

#     def construct_constants(
#         self,
#         constraint_constants: ConstraintConstants,
#     ):
#         return CoordinateConstraintConstants(
#             constraint_constants.u_P,
#             constraint_constants.u_S,
#             self.coordinate_index,
#         )


# def at_point_constraint(
#     qdt0P_G: np.ndarray,
#     qdt0S_G: np.ndarray,
#     u_P: np.ndarray,
#     u_S: np.ndarray,
# ) -> np.ndarray:
#     ri, pdt0P_G = jnp.split(qdt0P_G, [3])
#     rj, pdt0S_G = jnp.split(qdt0S_G, [3])

#     R_PG = A(pdt0P_G)
#     R_SG = A(pdt0S_G)

#     u_P = ri + R_PG @ u_P
#     u_S = rj + R_SG @ u_S

#     residual = u_P - u_S

#     return residual


# def dp1_constraint(
#     qdt0P_G: np.ndarray,
#     qdt0S_G: np.ndarray,
#     v1_P: np.ndarray,
#     v2_S: np.ndarray,
# ) -> np.ndarray:
#     _, pdt0P_G = jnp.split(qdt0P_G, [3])
#     _, pdt0S_G = jnp.split(qdt0S_G, [3])

#     R_PG = A(pdt0P_G)
#     R_SG = A(pdt0S_G)

#     v1_P = R_PG @ v1_P
#     v2_S = R_SG @ v2_S

#     residual = v1_P @ v2_S

#     return residual


# def dp2_constraint(
#     qdt0P_G: np.ndarray,
#     qdt0S_G: np.ndarray,
#     v1_P: np.ndarray,
#     u_P: np.ndarray,
#     u_S: np.ndarray,
# ) -> np.ndarray:
#     _, pdt0P_G = jnp.split(qdt0P_G, [3])

#     dij = at_point_constraint(qdt0P_G, qdt0S_G, u_P, u_S)

#     R_PG = A(pdt0P_G)

#     v1_P = R_PG @ v1_P

#     residual = v1_P @ dij

#     return residual
