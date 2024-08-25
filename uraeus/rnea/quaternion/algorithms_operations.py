from typing import List, Tuple

import jax

import jax.numpy as jnp
import numpy as np

from uraeus.rnea.quaternion.spatial_algebra import (
    skew_M,
    SpatialPose,
    quaternion_to_dcm,
    transform_screw,
    transform_vector,
    express_screw,
)
from uraeus.rnea.quaternion.bodies import BodyKinematics
from uraeus.rnea.quaternion.joints import (
    JointKinematics,
    JointFrames,
)
from uraeus.rnea.quaternion.mobilizers import MobilizerForces


# @jax.jit
def evaluate_successor_kinematics(
    predecessor_kin: BodyKinematics,
    joint_kin: JointKinematics,
) -> BodyKinematics:
    # p_GB = predecessor_kin.p_GB @ joint_kin.p_PS
    p_GB = joint_kin.p_PS @ predecessor_kin.p_GB
    p_BG = p_GB.inv()

    # jax.debug.print("{r}", r=p_GB)

    v_B = transform_screw(joint_kin.p_PS, predecessor_kin.v_B) + joint_kin.v_J
    v_GB = express_screw(p_BG, v_B)

    # jax.debug.print("joint_kin.v_J = {x}", x=joint_kin.v_J)
    # jax.debug.print("predecessor_kin.v_B = {x}", x=predecessor_kin.v_B)
    # jax.debug.print(
    #     "transform_screw(joint_kin.p_PS, predecessor_kin.v_B) = {x}",
    #     x=transform_screw(joint_kin.p_PS, predecessor_kin.v_B),
    # )
    # jax.debug.print("v_B = {x}", x=v_B)
    # jax.debug.print("\n")

    a_B = (
        transform_screw(joint_kin.p_PS, predecessor_kin.a_B)
        + joint_kin.a_J
        + spatial_cross(v_B, joint_kin.v_J)
    )

    v_s0 = translational_spatial_vector(v_B)
    a_GB = express_screw(p_BG, (a_B - spatial_cross(v_s0, v_B)))

    successor_kin = BodyKinematics(p_GB, p_BG, v_B, a_B, v_GB, a_GB)
    return successor_kin


@jax.jit
def spatial_cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:

    v1_v, v1_w = jnp.split(v1, 2)
    v2_v, v2_w = jnp.split(v2, 2)

    v3_v = (skew_M @ v1_v @ v2_w) + (skew_M @ v1_w @ v2_v)
    v3_w = (skew_M @ v1_w) @ v2_w

    return -jnp.array([*v3_v, *v3_w])


# @jax.jit
def evaluate_joint_inertia_force(
    successor_kin: BodyKinematics,
    successor_I: np.ndarray,
    external_forces: List[np.ndarray],
) -> np.ndarray:
    # fb_S = (successor_I @ successor_kin.a_B) + (
    #     motion_to_force_transform(spatial_skew(successor_kin.v_B))
    #     @ (successor_I @ successor_kin.v_B)
    # )
    # R_BG = get_orientation_matrix_from_transformation(successor_kin.X_BG)
    # E_BG = spatial_motion_rotation(R_BG)
    # fe_S = motion_to_force_transform(E_BG) @ sum(external_forces, np.zeros((6,)))

    # f = fb_S - fe_S

    return np.zeros((6,))


# @jax.jit
def construct_mobilizer_force(
    fi_S: np.ndarray,
    joint_frames: JointFrames,
    joint_kin: JointKinematics,
    successor_kin: BodyKinematics,
) -> MobilizerForces:
    fc_S, fa_S, tau = extract_force_components(fi_S, joint_frames, joint_kin)
    E_GB = quaternion_to_dcm(successor_kin.p_GB)
    fc_G = E_GB @ fc_S

    return MobilizerForces(fi_S, fc_S, fa_S, fc_G, tau)


# @jax.jit
def extract_force_components(
    fi_S: np.ndarray, joint_frames: JointFrames, joint_kin: JointKinematics
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_SM = joint_frames.p_SM
    p_MS = p_SM.inv()
    E_SM = quaternion_to_dcm(p_MS)

    # fi_M = motion_to_force_transform(X_MS) @ fi_S
    fi_M = np.zeros((6,))
    tau = joint_kin.S_FM.T @ fi_M

    fa_M = joint_kin.S_FM @ tau
    fc_M = fi_M - fa_M

    fc_S = E_SM @ fc_M
    fa_S = E_SM @ fa_M

    return fc_S, fa_S, tau


# @jax.jit
def translational_spatial_vector(v: np.ndarray) -> np.ndarray:
    rotational_part = np.zeros((3,))
    translational_part, _ = v.reshape(2, -1)
    return jnp.hstack([translational_part, rotational_part])


# =============================================================================
# Obselete Code
# =============================================================================

# def evaluate_joint_forces(
#     successor_I: np.ndarray,
#     successor_kin: BodyKinematics,
#     joint_kin: JointKinematics,
#     joint_frames: JointFrames,
#     out_joint: List[JointVariables],
#     external_forces: List[np.ndarray],
# ):

#     fb_S = (successor_I @ successor_kin.a_B) + (
#         motion_to_force_transform(spatial_skew(successor_kin.v_B))
#         @ (successor_I @ successor_kin.v_B)
#     )

#     E_BG = spatial_motion_rotation(
#         get_orientation_matrix_from_transformation(successor_kin.X_BG)
#     )
#     fe_S = E_BG @ sum(external_forces, np.zeros((6,)))

#     out_joints_forces = [
#         motion_to_force_transform(joint.kinematics.X_PS) @ joint.forces.fi_S
#         for joint in out_joint
#     ]

#     fj_S = sum(out_joints_forces, np.zeros((6,)))

#     fi_S = fb_S - fe_S + fj_S

#     fc_S, fa_S, tau = extract_force_components(fi_S, joint_frames, joint_kin)

#     E_GB = spatial_motion_rotation(
#         get_orientation_matrix_from_transformation(successor_kin.X_GB)
#     )
#     fc_G = E_GB @ fc_S

#     return MobilizerForces(fi_S, fc_S, fa_S, fc_G, tau)
