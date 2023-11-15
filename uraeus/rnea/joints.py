from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, NamedTuple, Set, Tuple, Type
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from uraeus.rnea.motion_equations import MotionEquations, construct_motion_jacobians
from uraeus.rnea.spatial_algebra import (
    spatial_motion_transformation,
    spatial_transform_transpose,
    skew_matrix,
)
from uraeus.rnea.mobilizers import (
    MobilizerForces,
    MobilizerKinematics,
    AbstractMobilizer,
    CustomMobilizer,
    FreeMobilizer,
    RevoluteMobilizer,
    TranslationalMobilizer,
    PlanarMobilizer,
)

from uraeus.rnea.bodies import RigidBody


class JointFrames(NamedTuple):
    X_SM: np.ndarray
    X_PF: np.ndarray


class JointKinematics(NamedTuple):
    X_FM: np.ndarray
    X_SP: np.ndarray
    X_PS: np.ndarray
    S_FM: np.ndarray
    v_J: np.ndarray
    a_J: np.ndarray


class JointVariables(NamedTuple):
    kinematics: JointKinematics
    forces: MobilizerForces


class StatesNames(NamedTuple):
    pos_states: List[str]
    vel_states: List[str]
    acc_states: List[str]


class JointConfigInputs(NamedTuple):
    pos: np.ndarray
    z_axis: np.ndarray
    x_axis: np.ndarray


class JointData(NamedTuple):
    name: str
    predecessor: RigidBody
    successor: RigidBody
    frames: JointFrames
    state_name: StatesNames


def construct_state_names(name: str, coordinates_names: List[str]) -> StatesNames:
    pos_states = [f"{name}_{coordinate}_dt0" for coordinate in coordinates_names]
    vel_states = [f"{name}_{coordinate}_dt1" for coordinate in coordinates_names]
    acc_states = [f"{name}_{coordinate}_dt2" for coordinate in coordinates_names]
    state_names = StatesNames(pos_states, vel_states, acc_states)
    return state_names


class AbstractJoint(NamedTuple):
    nj: int
    mobilizer: AbstractMobilizer
    coordinates_names: List[str]


class JointInstance(NamedTuple):
    joint_data: JointData
    joint_type: AbstractJoint

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        mobilizer_kinematics = self.joint_type.mobilizer.evaluate_kinematics(
            qdt0, qdt1, qdt2
        )
        joint_kinematics = evaluate_joint_kinematics(
            mobilizer_kinematics, self.joint_data.frames
        )
        return joint_kinematics


RevoluteJoint = AbstractJoint(
    nj=1,
    mobilizer=RevoluteMobilizer(),
    coordinates_names=["psi"],
)


TranslationalJoint = AbstractJoint(
    nj=1,
    mobilizer=TranslationalMobilizer(),
    coordinates_names=["z"],
)


PlanarJoint = AbstractJoint(
    nj=3,
    mobilizer=PlanarMobilizer(),
    coordinates_names=["psi", "x", "y"],
)


FreeJoint = AbstractJoint(
    nj=6,
    mobilizer=FreeMobilizer(),
    coordinates_names=["phi", "theta", "psi", "x", "y", "z"],
)


class FunctionalJoint(NamedTuple):
    nj: int
    mobilizer: AbstractMobilizer
    frames: JointFrames

    # @partial(jax.jit, static_argnums=(0,))
    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> JointKinematics:
        mobilizer_kinematics = self.mobilizer.evaluate_kinematics(qdt0, qdt1, qdt2)
        joint_kinematics = evaluate_joint_kinematics(mobilizer_kinematics, self.frames)
        return joint_kinematics

    def __hash__(self):
        return hash(self.__class__.__name__)


def construct_functional_joint(joint: JointInstance) -> FunctionalJoint:
    joint = FunctionalJoint(
        joint.joint_type.nj,
        joint.joint_type.mobilizer,
        joint.joint_data.frames,
    )
    return joint


def construct_joint_instance(
    joint_type: AbstractJoint,
    name: str,
    predecessor: RigidBody,
    successor: RigidBody,
    joint_frames: JointFrames,
) -> JointInstance:
    state_names = construct_state_names(name, joint_type.coordinates_names)
    joint_data = JointData(name, predecessor, successor, joint_frames, state_names)
    joint_instance = JointInstance(joint_data, joint_type)
    return joint_instance


@jax.jit
def evaluate_joint_kinematics(
    mobilizer_kinematics: MobilizerKinematics,
    joint_frames: JointFrames,
) -> JointKinematics:
    X_SM = joint_frames.X_SM
    X_PF = joint_frames.X_PF

    X_FM, S_FM, v_J, a_J = mobilizer_kinematics

    X_PS = X_PF @ X_FM @ spatial_transform_transpose(X_SM)
    X_SP = spatial_transform_transpose(X_PS)

    v_J = X_SM @ v_J

    a_J = X_SM @ a_J

    kinematics = JointKinematics(X_FM, X_SP, X_PS, S_FM, v_J, a_J)

    return kinematics


def construct_custom_joint(
    cls_name: str,
    pose_polynomials: Callable[[np.ndarray], np.ndarray],
    nj: int,
    coordinates_names: List[str],
) -> Type[AbstractJoint]:
    pose_jacobian_dt0, pose_jacobian_dt1 = construct_motion_jacobians(pose_polynomials)
    polynomials = MotionEquations(
        nj, pose_polynomials, pose_jacobian_dt0, pose_jacobian_dt1
    )

    mobilizer = type(
        f"{cls_name}Mobilizer", (CustomMobilizer,), {"polynomials": polynomials}
    )

    joint_class = type(
        f"{cls_name}Joint",
        (AbstractJoint,),
        {"nj": nj, "mobilizer": mobilizer(), "coordinates_names": coordinates_names},
    )

    return joint_class


def initialize_joint(
    location: np.ndarray,
    z_axis: np.ndarray,
    x_axis: np.ndarray,
    P_X_BG: np.ndarray,
    S_X_BG: np.ndarray,
) -> JointFrames:
    R_GJ = triad(z_axis, x_axis)

    X_GJ = spatial_motion_transformation(R_GJ, R_GJ.T @ -location)

    X_PF = P_X_BG @ X_GJ
    X_SM = S_X_BG @ X_GJ

    return JointFrames(X_SM, X_PF)


def orthogonal_vector(v: np.ndarray):
    x, y, z = v

    v1 = np.array([y, -x, 0])
    v2 = np.array([-z, 0, x])

    v3 = (5 * v1) + (9 * v2)

    u = v3 / np.linalg.norm(v3)

    return u


def triad(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    k = v1 / np.linalg.norm(v1)
    if v2 is not None:
        i = v2 / np.linalg.norm(v2)
    else:
        i = orthogonal_vector(k)

    j = skew_matrix(k) @ i
    j = j / np.linalg.norm(j)

    R = np.vstack([i, j, k]).T

    return R
