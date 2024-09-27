from typing import Callable, NamedTuple, Type
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from uraeus.rnea.motion_equations import (
    MotionEquations,
    construct_motion_jacobians,
)
from uraeus.rnea.spatial_algebra import (
    SpatialPose,
    skew_M,
    quaternion_from_axis_angle,
    transform_vector,
    transform_screw,
    quaternion_inverse,
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
    """
    Represents the frames of a joint.

    Attributes
    ----------
    p_SM : SpatialPose
        The pose of the successor frame (S) in the moving frame (M).
    p_PF : SpatialPose
        The pose of the predecessor frame (P) in the fixed frame (F).
    """

    p_SM: SpatialPose
    p_PF: SpatialPose


class JointKinematics(NamedTuple):
    """
    Represents the kinematics of a joint.

    Attributes
    ----------
    p_FM : SpatialPose
        The pose of the fixed frame (F) in the moving frame (M).
    p_SP : SpatialPose
        The pose of the successor frame (S) in the predecessor frame (P).
    p_PS : SpatialPose
        The pose of the predecessor frame (P) in the successor frame (S).
    S_FM : np.ndarray
        The screw matrix of the joint.
    v_J : np.ndarray
        The joint velocity.
    a_J : np.ndarray
        The joint acceleration.
    """

    p_FM: SpatialPose
    p_SP: SpatialPose
    p_PS: SpatialPose
    S_FM: np.ndarray
    v_J: np.ndarray
    a_J: np.ndarray


class JointVariables(NamedTuple):
    """
    Represents the variables of a joint, including kinematics and forces.

    Attributes
    ----------
    kinematics : JointKinematics
        The kinematics of the joint.
    forces : MobilizerForces
        The forces acting on the joint.
    """

    kinematics: JointKinematics
    forces: MobilizerForces


class StatesNames(NamedTuple):
    pos_states: list[str]
    vel_states: list[str]
    acc_states: list[str]


class JointConfigInputs(NamedTuple):
    """
    Represents the configuration inputs for a joint.

    Attributes
    ----------
    pos : np.ndarray
        The position vector of the joint.
    z_axis : np.ndarray
        The z-axis vector of the joint.
    x_axis : np.ndarray
        The x-axis vector of the joint.
    """

    pos: np.ndarray
    z_axis: np.ndarray
    x_axis: np.ndarray


class JointData(NamedTuple):
    """
    Represents the data associated with a joint.

    Attributes
    ----------
    name : str
        The name of the joint.
    predecessor : RigidBody
        The predecessor rigid body.
    successor : RigidBody
        The successor rigid body.
    frames : JointFrames
        The frames of the joint.
    state_name : StatesNames
        The state names associated with the joint.
    """

    name: str
    predecessor: RigidBody
    successor: RigidBody
    frames: JointFrames
    state_name: StatesNames


def construct_state_names(name: str, coordinates_names: list[str]) -> StatesNames:
    pos_states = [f"{name}_{coordinate}_dt0" for coordinate in coordinates_names]
    vel_states = [f"{name}_{coordinate}_dt1" for coordinate in coordinates_names]
    acc_states = [f"{name}_{coordinate}_dt2" for coordinate in coordinates_names]
    state_names = StatesNames(pos_states, vel_states, acc_states)
    return state_names


class AbstractJoint(NamedTuple):
    nj: int
    mobilizer: AbstractMobilizer
    coordinates_names: list[str]


class JointInstance(NamedTuple):
    """
    Represents an instance of a joint with its data and type.

    Attributes
    ----------
    joint_data : JointData
        The data associated with the joint.
    joint_type : AbstractJoint
        The type of the joint.
    """

    joint_data: JointData
    joint_type: AbstractJoint

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_kinematics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> MobilizerKinematics:
        """
        Evaluates the kinematics of the joint.

        Parameters
        ----------
        qdt0 : np.ndarray
            Joint coordinates pose.
        qdt1 : np.ndarray
            Joint coordinates velocity.
        qdt2 : np.ndarray
            Joint coordinates acceleration.

        Returns
        -------
        MobilizerKinematics
            The kinematics of the joint.
        """
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
    coordinates_names=["x", "y", "psi"],
)


FreeJoint = AbstractJoint(
    nj=6,
    mobilizer=FreeMobilizer(),
    coordinates_names=["x", "y", "z", "phi", "theta", "psi"],
)


class FunctionalJoint(NamedTuple):
    nj: int
    mobilizer: AbstractMobilizer
    frames: JointFrames

    @partial(jax.jit, static_argnums=(0,))
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


# @jax.jit
def evaluate_joint_kinematics(
    mobilizer_kinematics: MobilizerKinematics,
    joint_frames: JointFrames,
) -> JointKinematics:
    p_SM = joint_frames.p_SM
    p_PF = joint_frames.p_PF

    p_FM, S_FM, v_J, a_J = mobilizer_kinematics

    p_PS = p_SM.inv() @ p_FM @ p_PF
    p_SP = p_PS.inv()

    v_J = transform_screw(p_SM.inv(), v_J)
    a_J = transform_screw(p_SM.inv(), a_J)

    kinematics = JointKinematics(p_FM, p_SP, p_PS, S_FM, v_J, a_J)

    return kinematics


def construct_custom_joint(
    cls_name: str,
    pose_polynomials: Callable[[np.ndarray], np.ndarray],
    nj: int,
    coordinates_names: list[str],
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
    p_GP: SpatialPose,
    p_GS: SpatialPose,
) -> JointFrames:

    z_axis_G = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis_G, z_axis)
    if np.linalg.norm(rot_axis) != 0:
        angle = np.arccos(z_axis_G @ z_axis)
        q_JG = quaternion_from_axis_angle(-angle, rot_axis)
    else:
        q_JG = np.array([1, 0, 0, 0])

    p_JG = SpatialPose(transform_vector(quaternion_inverse(q_JG), -location), q_JG)

    p_FP = p_GP @ p_JG
    p_MS = p_GS @ p_JG

    return JointFrames(p_MS.inv(), p_FP.inv())


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

    j = skew_M @ k @ i
    j = j / np.linalg.norm(j)

    R = np.vstack([i, j, k]).T

    return R
