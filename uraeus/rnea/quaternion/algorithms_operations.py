from typing import List, Tuple

import jax

import jax.numpy as jnp
import numpy as np

from uraeus.rnea.quaternion.spatial_algebra import (
    skew_M,
    transform_screw,
    express_screw,
    transform_screw_force,
)
from uraeus.rnea.quaternion.bodies import BodyKinematics
from uraeus.rnea.quaternion.joints import (
    JointKinematics,
    JointFrames,
)
from uraeus.rnea.quaternion.mobilizers import MobilizerForces


@jax.jit
def evaluate_successor_kinematics(
    predecessor_kin: BodyKinematics,
    joint_kin: JointKinematics,
) -> BodyKinematics:
    p_BG = predecessor_kin.p_BG @ joint_kin.p_SP
    p_GB = p_BG.inv()

    v_B = transform_screw(joint_kin.p_PS, predecessor_kin.v_B) + joint_kin.v_J
    v_GB = express_screw(p_BG, v_B)

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


@jax.jit
def force_spatial_cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:

    v1_v, v1_w = jnp.split(v1, 2)
    v2_v, v2_w = jnp.split(v2, 2)

    v3_v = skew_M @ v1_w @ v2_v
    v3_w = (skew_M @ v1_w @ v2_w) + (skew_M @ v1_v @ v2_v)

    return jnp.array([*v3_v, *v3_w])


@jax.jit
def screw_cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:

    v1_v, v1_w = jnp.split(v1, 2)
    v2_v, v2_w = jnp.split(v2, 2)

    v3_v = skew_M @ v1_v @ v2_v
    v3_w = (skew_M @ v1_v @ v2_w) + (skew_M @ v1_w @ v2_v)

    return jnp.array([*v3_v, *v3_w])


@jax.jit
def evaluate_joint_inertia_force(
    successor_kin: BodyKinematics,
    successor_I: np.ndarray,
    external_forces: tuple[list[np.ndarray], list[np.ndarray]],
) -> np.ndarray:

    # inertia forces from direct accelerations
    fi_S_qdt2 = successor_I @ successor_kin.a_B

    # inertia forces from rotational velocity
    fi_S_qdt1 = -force_spatial_cross(
        successor_kin.v_B, (successor_I @ successor_kin.v_B)
    )
    # Total inertia forces
    fi_S = fi_S_qdt2 + fi_S_qdt1

    global_external_forces, local_external_forces = external_forces

    g_fe_S = express_screw(
        successor_kin.p_GB, sum(global_external_forces, np.zeros((6,)))
    )
    l_fe_S = sum(local_external_forces, np.zeros((6,)))

    fe_S = g_fe_S + l_fe_S
    fb_S = fi_S - fe_S
    return fb_S


@jax.jit
def construct_mobilizer_force(
    fi_S: np.ndarray,
    joint_frames: JointFrames,
    joint_kin: JointKinematics,
    successor_kin: BodyKinematics,
) -> MobilizerForces:
    fc_S, fa_S, tau = extract_force_components(fi_S, joint_frames, joint_kin)
    fc_G = express_screw(successor_kin.p_BG, fc_S)
    return MobilizerForces(fi_S, fc_S, fa_S, fc_G, tau)


@jax.jit
def extract_force_components(
    fi_S: np.ndarray, joint_frames: JointFrames, joint_kin: JointKinematics
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_SM = joint_frames.p_SM
    p_MS = p_SM.inv()

    fi_M = transform_screw_force(p_SM, fi_S)
    tau = joint_kin.S_FM.T @ fi_M

    fa_M = joint_kin.S_FM @ tau
    fc_M = fi_M - fa_M

    fc_S = express_screw(p_MS, fc_M)
    fa_S = express_screw(p_MS, fa_M)

    return fc_S, fa_S, tau


@jax.jit
def translational_spatial_vector(v: np.ndarray) -> np.ndarray:
    rotational_part = np.zeros((3,))
    translational_part, _ = v.reshape(2, -1)
    return jnp.hstack([translational_part, rotational_part])
