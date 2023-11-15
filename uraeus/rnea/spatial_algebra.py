from typing import Tuple

import jax
import numpy as np
import jax.numpy as jnp

from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_traceback_filtering", "off")


@jax.jit
def vsplit(arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split an 2D array `arr` into two equally sized sections vertically.
    This mimics the `jnp.vspilt(arr, 2)`, but uses a smart `reshape` trick,
    avoiding expensive copy operations.

    Parameters
    ----------
    arr : jnp.ndarray
        2D numpy array

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        A tuple of the two
    """
    top_half, low_half = arr.reshape(2, -1, arr.shape[-1])
    return top_half, low_half


@jax.jit
def hsplit(arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split an 2D array `arr` into two equally sized sections horizontally.
    This mimics the `jnp.hspilt(arr, 2)`, but uses a smart `.reshape` trick,
    avoiding expensive copy operations.

    Parameters
    ----------
    arr : jnp.ndarray
        2D numpy array

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        A tuple of the two
    """
    top_half, low_half = arr.T.reshape(2, -1, arr.shape[-1])
    return top_half.T, low_half.T


@jax.jit
def skew_matrix(v: jnp.ndarray) -> jnp.ndarray:
    """Create a skew-matrix out of the given cartesian vector.

    Parameters
    ----------
    v : np.ndarray
        A (3,) numpy array representing a cartesian vector.

    Returns
    -------
    np.ndarray
        A (3, 3) np.array
    """

    x, y, z = v
    mat = jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return mat


@jax.jit
def spatial_skew(v: jnp.ndarray) -> jnp.ndarray:
    orient_elements, trans_elements = v.reshape(2, -1)

    b00 = skew_matrix(orient_elements)
    b01 = np.zeros((3, 3))
    b10 = skew_matrix(trans_elements)
    b11 = b00

    result = jnp.vstack([jnp.hstack([b00, b01]), jnp.hstack([b10, b11])])
    return result


@jax.jit
def cross(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    return spatial_skew(v1) @ v2


@jax.jit
def rot_x(theta: float) -> jnp.ndarray:
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    mat = jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return mat


@jax.jit
def rot_y(theta: float) -> jnp.ndarray:
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    mat = jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return mat


@jax.jit
def rot_z(theta: float) -> jnp.ndarray:
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    mat = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return mat


@jax.jit
def spatial_motion_translation(p_PS: jnp.ndarray) -> jnp.ndarray:
    X_PS = jnp.vstack(
        [
            jnp.hstack([np.eye(3), np.zeros((3, 3))]),
            jnp.hstack([-skew_matrix(p_PS), np.eye(3)]),
        ]
    )
    return X_PS


@jax.jit
def spatial_motion_rotation(R_PS: jnp.ndarray) -> jnp.ndarray:
    X_PS = jnp.vstack(
        [
            jnp.hstack([R_PS, np.zeros((3, 3))]),
            jnp.hstack([np.zeros((3, 3)), R_PS]),
        ]
    )
    return X_PS


@jax.jit
def spatial_motion_transformation(R_PS: jnp.ndarray, p_PS: jnp.ndarray) -> jnp.ndarray:
    X_PS = jnp.vstack(
        [
            jnp.hstack([R_PS, np.zeros((3, 3))]),
            jnp.hstack([-R_PS @ skew_matrix(p_PS), R_PS]),
        ]
    )
    return X_PS


@jax.jit
def spatial_force_transformation(R_PS: jnp.ndarray, p_PS: jnp.ndarray) -> jnp.ndarray:
    X_PS = jnp.vstack(
        [
            jnp.hstack([R_PS, -R_PS @ skew_matrix(p_PS)]),
            jnp.hstack([np.zeros((3, 3)), R_PS]),
        ]
    )
    return X_PS


@jax.jit
def motion_to_force_transform(X_PS: jnp.ndarray) -> jnp.ndarray:
    left_half, right_half = hsplit(X_PS)
    b00, b10 = vsplit(left_half)
    b01, b11 = vsplit(right_half)

    X_f_PS = jnp.vstack(
        [
            jnp.hstack([b00, b10]),
            jnp.hstack([b01, b11]),
        ]
    )

    return X_f_PS


@jax.jit
def spatial_transform_transpose(X_PS: jnp.ndarray) -> jnp.ndarray:
    left_half, right_half = hsplit(X_PS)
    b00, b10 = vsplit(left_half)
    b01, b11 = vsplit(right_half)
    X_SP = jnp.vstack(
        [
            jnp.hstack([b00.T, b01.T]),
            jnp.hstack([b10.T, b11.T]),
        ]
    )

    return X_SP


@jax.jit
def vector_from_skew(skew_m: jnp.ndarray) -> jnp.ndarray:
    x = skew_m[2, 1]
    y = skew_m[0, 2]
    z = skew_m[1, 0]

    v = jnp.array([x, y, z])

    return v


@jax.jit
def get_position_from_transformation(X_PS: jnp.ndarray) -> jnp.ndarray:
    left_half, _ = hsplit(X_PS)
    b00, b10 = vsplit(left_half)

    skewed_matrix = b00.T @ b10
    p_PS = vector_from_skew(-skewed_matrix)

    return p_PS


@jax.jit
def get_euler_angles_from_rotation(R_PS: jnp.ndarray) -> jnp.ndarray:
    r00, r01, r02 = R_PS[0, :]
    r10, r11, r12 = R_PS[1, :]
    r20, r21, r22 = R_PS[2, :]

    theta_x = jnp.arctan2(-r12, r22)
    theta_y = jnp.arctan2(r02, jnp.sqrt(r12**2 + r22**2))
    theta_z = jnp.arctan2(-r01, r00)

    euler_angles = jnp.array([theta_x, theta_y, theta_z])

    return euler_angles


@jax.jit
def get_euler_angles_from_transformation(X_PS: jnp.ndarray) -> jnp.ndarray:
    left_half, _ = hsplit(X_PS)
    b00, _ = vsplit(left_half)

    e_PS = get_euler_angles_from_rotation(b00)

    return b00


@jax.jit
def get_pose_from_transformation(X_PS: jnp.ndarray) -> jnp.ndarray:
    left_half, _ = hsplit(X_PS)
    b00, b10 = vsplit(left_half)

    skewed_matrix = b00.T @ b10
    p_PS = vector_from_skew(-skewed_matrix)
    r_PS = -b00 @ p_PS

    e_PS = get_euler_angles_from_rotation(b00)

    return jnp.hstack([e_PS, r_PS])


@jax.jit
def get_orientation_matrix_from_transformation(X_PS: jnp.ndarray) -> jnp.ndarray:
    left_half, _ = hsplit(X_PS)
    R_PS, _ = vsplit(left_half)

    return R_PS
