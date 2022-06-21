from typing import List, Tuple

import jax
import numpy as np

from multibody.spatial_algebra import (
    spatial_motion_rotation,
    spatial_transform_transpose,
    spatial_skew,
    get_pose_from_transformation,
    get_orientation_matrix_from_transformation,
    motion_to_force_transform,
    cross,
)
from multibody.bodies import BodyKinematics
from multibody.joints import AbstractJoint, JointKinematics, JointFrames, JointVariables
from multibody.mobilizers import MobilizerForces


def evaluate_successor_kinematics(
    predecessor_kin: BodyKinematics,
    joint_kin: JointKinematics,
) -> BodyKinematics:

    X_GB = predecessor_kin.X_GB @ joint_kin.X_PS

    X_BG = spatial_transform_transpose(X_GB)

    v_B = (joint_kin.X_SP @ predecessor_kin.v_B) + joint_kin.v_J

    a_B = (
        (joint_kin.X_SP @ predecessor_kin.a_B)
        + joint_kin.a_J
        + cross(v_B, joint_kin.v_J)
    )

    R_GB = get_orientation_matrix_from_transformation(X_GB)

    p_GB = get_pose_from_transformation(X_GB)
    v_GB = spatial_motion_rotation(R_GB) @ v_B

    v_s0 = translational_spatial_vector(v_B)
    a_GB = spatial_motion_rotation(R_GB) @ (a_B - cross(v_s0, v_B))

    successor_kin = BodyKinematics(X_BG, X_GB, p_GB, R_GB, v_B, a_B, v_GB, a_GB)
    return successor_kin


def evaluate_joint_forces(
    successor_I: np.ndarray,
    successor_kin: BodyKinematics,
    joint_kin: JointKinematics,
    joint_frames: JointFrames,
    out_joint: List[AbstractJoint],
    external_forces: List[np.ndarray],
):

    fb_S = (successor_I @ successor_kin.a_B) + (
        motion_to_force_transform(spatial_skew(successor_kin.v_B))
        @ (successor_I @ successor_kin.v_B)
    )

    E_BG = spatial_motion_rotation(
        get_orientation_matrix_from_transformation(successor_kin.X_BG)
    )
    fe_S = E_BG @ sum(external_forces, np.zeros((6,)))

    out_joints_forces = [
        motion_to_force_transform(joint.kinematics.X_PS) @ joint.forces.fi_S
        for joint in out_joint
    ]

    fj_S = sum(out_joints_forces, np.zeros((6,)))

    fi_S = fb_S - fe_S + fj_S

    fc_S, fa_S, tau = extract_force_components(fi_S, joint_frames, joint_kin)

    E_GB = spatial_motion_rotation(
        get_orientation_matrix_from_transformation(successor_kin.X_GB)
    )
    fc_G = E_GB @ fc_S

    return MobilizerForces(fi_S, fc_S, fa_S, tau, fc_G)


def extract_force_components(
    fi_S: np.ndarray, joint_frames: JointFrames, joint_kin: JointKinematics
):
    X_SM = joint_frames.X_SM
    X_MS = spatial_transform_transpose(X_SM)
    fi_M = motion_to_force_transform(X_MS) @ fi_S

    tau = joint_kin.S_FM.T @ fi_M

    fa_M = joint_kin.S_FM @ tau

    fc_M = fi_M - fa_M

    E_SM = spatial_motion_rotation(get_orientation_matrix_from_transformation(X_SM))

    fc_S = E_SM @ fc_M
    fa_S = E_SM @ fa_M

    return fc_S, fa_S, tau


def translational_spatial_vector(v: np.ndarray) -> np.ndarray:
    rotational_part = np.zeros((3,))
    _, translational_part = np.split(v, 2)
    return np.hstack([rotational_part, translational_part])
