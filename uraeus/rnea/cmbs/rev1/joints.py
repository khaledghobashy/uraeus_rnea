from typing import NamedTuple, Callable, Any, Protocol, ClassVar
from functools import partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

from uraeus.rnea.cmbs.rev1.constraints import (
    SphericalConstraint,
    DP1Constraint,
    DP2Constraint,
    AngleConstraint,
    DistanceConstraint,
    ConstraintConstants,
    AbstractConstraintEquations,
)

from uraeus.rnea.cmbs.rev1.bodies import RigidBodyData
from uraeus.rnea.cmbs.rev1.spatial_algebra import A, B, skew_matrix, triad


class JointConfigInputs(NamedTuple):
    pos: np.ndarray
    z_axis: np.ndarray
    x_axis: np.ndarray = None
    z_axis_j: np.ndarray = None


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.sqrt(v @ v)


def construct_joints_constant(
    joint_inputs: JointConfigInputs, body_i: RigidBodyData, body_j: RigidBodyData
):
    u_i = A(body_i.orientation).T @ (joint_inputs.pos - body_i.location)
    u_j = A(body_j.orientation).T @ (joint_inputs.pos - body_j.location)

    z_axis_i = joint_inputs.z_axis
    x_axis_i = joint_inputs.x_axis
    frame_i = triad(z_axis_i, x_axis_i)

    z_axis_j = (
        joint_inputs.z_axis_j
        if joint_inputs.z_axis_j is not None
        else joint_inputs.z_axis
    )
    x_axis_j = (
        frame_i[:, 1] if joint_inputs.z_axis_j is not None else joint_inputs.x_axis
    )
    frame_j = triad(z_axis_j, x_axis_j)

    M_i = A(body_i.orientation).T @ frame_i
    M_j = A(body_j.orientation).T @ frame_j

    return ConstraintConstants(u_i, u_j, M_i, M_j)


class PosConstraintCallable(Protocol):
    def __call__(
        self, qdt0_i: np.ndarray, qdt0_j: np.ndarray, *args: ...
    ) -> np.ndarray:
        ...


class VelConstraintCallable(Protocol):
    def __call__(
        self, qdt0_i: np.ndarray, qdt0_j: np.ndarray, *args: ...
    ) -> np.ndarray:
        ...


class AccConstraintCallable(Protocol):
    def __call__(
        self,
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        *args: ...,
    ) -> np.ndarray:
        ...


@dataclass
class FunctionalJoint(object):
    name: str
    nc: int
    pos_constraint: PosConstraintCallable = None
    vel_constraint: VelConstraintCallable = None
    acc_constraint: AccConstraintCallable = None

    def __hash__(self):
        return hash(self.name)


@dataclass
class AbstractJoint(object):
    nc: ClassVar[int] = 0
    equations: ClassVar[tuple[AbstractConstraintEquations, ...]] = None

    name: str
    body_i: RigidBodyData
    body_j: RigidBodyData
    joint_config: JointConfigInputs

    def __hash__(self):
        return hash(self.name)

    @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        self,
        constants: tuple[ConstraintConstants, ...],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        equations = (eq.pos_constraint for eq in self.equations)
        # constants = (eq.co for eq in self.equations)
        residual = jnp.concatenate(
            tuple(eq(cons, qdt0_i, qdt0_j) for eq, cons in zip(equations, constants))
        )
        return residual

    @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        self,
        constants: tuple[ConstraintConstants, ...],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        equations = (eq.vel_constraint for eq in self.equations)
        residual = jnp.concatenate(
            tuple(eq(cons, qdt0_i, qdt0_j) for eq, cons in zip(equations, constants))
        )
        return residual

    @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        self,
        constants: tuple[ConstraintConstants, ...],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        *args,
    ) -> np.ndarray:
        equations = (eq.acc_constraint for eq in self.equations)
        residual = jnp.concatenate(
            tuple(
                eq(cons, qdt0_i, qdt0_j, qdt1_i, qdt1_j)
                for eq, cons in zip(equations, constants)
            )
        )
        return residual

    # @partial(jax.jit, static_argnums=(0,))
    def jacobians(
        self,
        constants: tuple[ConstraintConstants, ...],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        *args,
    ) -> tuple[np.ndarray, np.ndarray]:
        equations = (eq.jacobians for eq in self.equations)
        jacobians = tuple(
            eq(qdt0_i, qdt0_j, *cons) for eq, cons in zip(equations, constants)
        )

        jacs_i, jacs_j = zip(*jacobians)

        return jnp.vstack(jacs_i), jnp.vstack(jacs_j)


class RevoluteJoint(AbstractJoint):
    nc = 5
    equations = (
        SphericalConstraint(),
        DP1Constraint(0, 2),
        DP1Constraint(1, 2),
    )


class SphericalJoint(AbstractJoint):
    nc = 3
    equations = (SphericalConstraint(),)


class CylindricalJoint(AbstractJoint):
    nc = 4
    equations = (
        DP1Constraint(0, 2),
        DP1Constraint(1, 2),
        DP2Constraint(0),
        DP2Constraint(1),
    )


class UniversalJoint(AbstractJoint):
    nc = 4
    equations = (
        SphericalConstraint(),
        DP1Constraint(0, 0),
    )


@dataclass
class AbstractJointActuator(object):
    nc: ClassVar[int] = None
    equations: ClassVar[tuple[AbstractConstraintEquations, ...]] = None

    name: str
    joint: AbstractJoint
    driver: Callable[[float], float]

    def __hash__(self):
        return hash(self.name)

    @partial(jax.jit, static_argnums=(0,))
    def pos_constraint(
        self,
        constants: tuple[ConstraintConstants, ...],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        equations = (eq.pos_constraint for eq in self.equations)
        residual = jnp.concatenate(
            tuple(
                eq(
                    cons,
                    self.driver,
                    qdt0_i,
                    qdt0_j,
                    t,
                )
                for eq, cons in zip(equations, constants)
            )
        )
        return residual

    @partial(jax.jit, static_argnums=(0,))
    def vel_constraint(
        self,
        constants: tuple[ConstraintConstants, ...],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        equations = (eq.vel_constraint for eq in self.equations)
        residual = jnp.concatenate(
            tuple(
                eq(cons, self.driver, qdt0_i, qdt0_j, t)
                for eq, cons in zip(equations, constants)
            )
        )
        return residual

    @partial(jax.jit, static_argnums=(0,))
    def acc_constraint(
        self,
        constants: tuple[ConstraintConstants, ...],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        qdt1_i: np.ndarray,
        qdt1_j: np.ndarray,
        t: float,
    ) -> np.ndarray:
        equations = (eq.acc_constraint for eq in self.equations)
        residual = jnp.concatenate(
            tuple(
                eq(cons, self.driver, qdt0_i, qdt0_j, qdt1_i, qdt1_j, t)
                for eq, cons in zip(equations, constants)
            )
        )
        return residual

    @partial(jax.jit, static_argnums=(0,))
    def jacobians(
        self,
        constants: tuple[ConstraintConstants, ...],
        qdt0_i: np.ndarray,
        qdt0_j: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        equations = (eq.jacobians for eq in self.equations)
        jacobians = tuple(
            eq(cons, self.driver, qdt0_i, qdt0_j, t)
            for eq, cons in zip(equations, constants)
        )

        jacs_i, jacs_j = zip(*jacobians)

        return jnp.vstack(jacs_i), jnp.vstack(jacs_j)


class RotationActuator(AbstractJointActuator):
    nc = 1
    equations = (AngleConstraint(0, 0, 1),)


class TranslationActuator(AbstractJointActuator):
    nc = 1
    equations = (DistanceConstraint(),)


def construct_joint(
    joint_instance: AbstractJoint | AbstractJointActuator,
) -> FunctionalJoint:
    # 1. construct constant vectors and frames out from joint_input
    # and bodies i and j
    print(joint_instance.name)
    if isinstance(joint_instance, AbstractJointActuator):
        joint_constants = construct_joints_constant(
            joint_instance.joint.joint_config,
            joint_instance.joint.body_i,
            joint_instance.joint.body_j,
        )
    else:
        joint_constants = construct_joints_constant(
            joint_instance.joint_config,
            joint_instance.body_i,
            joint_instance.body_j,
        )

    constraint_constants = tuple(
        constraint.construct_constants(joint_constants)
        for constraint in joint_instance.equations
    )
    # 2. Construct joint object from joint_type and constants

    pos_eq = jax.jit(partial(joint_instance.pos_constraint, constraint_constants))
    vel_eq = jax.jit(partial(joint_instance.vel_constraint, constraint_constants))
    acc_eq = jax.jit(partial(joint_instance.acc_constraint, constraint_constants))
    # jac_eq = jax.jit(partial(joint_instance.jacobians, constraint_constants))

    func_joint = FunctionalJoint(
        name=joint_instance.name,
        nc=joint_instance.nc,
        pos_constraint=pos_eq,
        vel_constraint=vel_eq,
        acc_constraint=acc_eq,
    )

    return func_joint


# def construct_distance_actuator(
#     parent_joint: CompositeConstraints, driver: Callable[[float], float]
# ):
#     def append_equation(func):
#         def pos_eq(
#             constants: tuple[ConstraintConstants, ...],
#             qdt0_i: np.ndarray,
#             qdt0_j: np.ndarray,
#             t: float,
#         ):
#             return DP2Constraint.pos_constraint(
#                 qdt0_i,
#                 qdt0_j,
#             )


if __name__ == "__main__":
    from scipy.optimize import fsolve, minimize

    @partial(jax.jit, static_argnums=(0,))
    def func(f: Callable[[float], float], v: np.ndarray):
        return 5 * f(v)

    def f(v):
        return v @ v

    print(func(f, np.array([2, 2, 2])))

    # @partial(jax.jit, static_argnums=(2,))
    def system_constraints(qdt0, t, joints: tuple[AbstractJoint, ...]):
        res1 = jnp.hstack(
            tuple(j.pos_constraint(qdt0[:7], qdt0[7:], t) for j in joints)
        )
        pdt0_i = qdt0[3:7]
        pdt0_j = qdt0[10:]
        ground_cons = jnp.hstack((qdt0[:3], pdt0_i - np.array([1, 0, 0, 0])))
        # print(ground_cons)
        res2 = jnp.hstack(
            (
                ground_cons,
                jnp.array([pdt0_j.T @ pdt0_j - 1]),
            )
        )
        return jnp.hstack((res1, res2))

    consttraint_jacobian = partial(jax.jit, static_argnums=(2,))(
        jax.jacfwd(system_constraints)
    )

    def static_equilibrium(
        qdt0, t: float, joints: tuple[AbstractJoint, ...]
    ) -> np.ndarray:
        x0 = qdt0
        # x = minimize(system_constraints, x0, args=(t, joints), method="BFGS", tol=1e-3)
        x = fsolve(
            system_constraints, x0, args=(t, joints), fprime=consttraint_jacobian
        )
        return x

    body_i = RigidBodyData()
    body_j = RigidBodyData(
        location=np.array([0, 5, 0]), mass=1, inertia_tensor=np.eye(3)
    )

    joint_inputs = JointConfigInputs(
        pos=np.array([0, 0, 0]), z_axis=np.array([0, 0, 1]), x_axis=np.array([1, 0, 0])
    )
    # joint_constant = construct_joints_constant(joint_inputs, body_i, body_j)

    rev_joint = RevoluteJoint("rev", body_i, body_j, joint_inputs)
    ang_joint = RotationActuator("ang", rev_joint, lambda t: t / 2)

    qdt0 = np.hstack([np.hstack([b.location, b.orientation]) for b in (body_i, body_j)])

    func_joints = (construct_joint(rev_joint), construct_joint(ang_joint))

    print(func_joints[0].pos_constraint(qdt0[:7], qdt0[7:]))
    print(func_joints[1].pos_constraint(qdt0[:7], qdt0[7:], 1))

    print(system_constraints(qdt0, np.pi / 2, func_joints))
    print(static_equilibrium(qdt0, np.pi / 2, func_joints))

# print(qdt0)
# rev_res = rev_joint.pos_constraint(
#     np.hstack([body_i.location, body_i.orientation]),
#     np.hstack([body_j.location * 2, body_j.orientation]),
# )

# print(rev_res)

# ang_res = ang_joint.pos_constraint(
#     np.hstack([body_i.location, body_i.orientation]),
#     np.hstack([body_j.location * 2, body_j.orientation]),
#     0,
# )
# print(ang_res)
# class JointMetaConstructor(type):
#     _required_fields = {
#         "constraint_equations",
#     }

#     def __new__(cls, class_name, bases, attrs):
#         if class_name == "AbstractJoint":
#             return type.__new__(cls, class_name, bases, attrs)

#         for attr in cls._required_fields:
#             if attr not in attrs:
#                 raise NotImplementedError(f"'{attr}' should be implemented")

#         constraint_equations:tuple[SphericalConstraint] = attrs["constraint_equations"]

#         nc = sum([e.nc for e in constraint_equations])


#         equations_class = super().__new__(cls, class_name, bases, attrs)
#         kwargs = {k: getattr(equations_class, k) for k in cls._required_fields}
#         return MotionEquations(**kwargs)


# def __new__(mcls, name, bases, attrs):
#     # getting the joint/actuator vector equations stored as a class
#     # attribute in the constructed class
#     vector_equations = attrs["vector_equations"]

#     # number of vector constraint equations
#     nve = len(vector_equations)
#     # number of scalar constraint equations
#     nc = sum([e.nc for e in vector_equations])

#     # defining the construct method of the concrete classes
#     def construct(self):
#         self._create_equations_lists()
#         self._construct_actuation_functions()

#         # calling the construct method of each algebraic equation to
#         # construct the equations between the constrained bodies. This is
#         # specific for each joint class
#         for e in vector_equations:
#             e.construct(self)

#         # call the `abstract_joint._construct` method to construct common
#         # instance attributes.
#         self._construct()

#     # updating the concrete class methods and members
#     attrs["construct"] = construct
#     attrs["nve"] = nve
#     attrs["nc"] = nc
#     attrs["n"] = 0

#     cls_ = super(joint_constructor, mcls).__new__(mcls, name, tuple(bases), attrs)

#     return cls_
