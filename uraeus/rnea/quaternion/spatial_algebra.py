from __future__ import annotations
from typing import Tuple
import itertools

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


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


def levi_cevita_tensor(len: int) -> np.ndarray:
    """Transformation tensor that maps vectors into skew-symmetric
    matrices

    Parameters
    ----------
    len : int
        Vector length

    Returns
    -------
    np.ndarray
        Array with `len` dimensions
    """
    arr = np.zeros(tuple([len for _ in range(len)]))
    for x in itertools.permutations(tuple(range(len))):
        mat = np.zeros((len, len), dtype=np.int32)
        for i, j in zip(range(len), x):
            mat[i, j] = 1
        arr[x] = int(np.linalg.det(mat))
    return arr


skew_M = levi_cevita_tensor(3)


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
def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.

    Args:
        Q0 (np.ndarray): A 4-element array containing the first quaternion (q01, q11, q21, q31).
        Q1 (np.ndarray): A 4-element array containing the second quaternion (q02, q12, q22, q32).

    Returns:
        np.ndarray: A 4-element array containing the final quaternion (q03, q13, q23, q33).
    """
    q1_w, q1_x, q1_y, q1_z = q1
    q2_w, q2_x, q2_y, q2_z = q2

    t0 = q1_w * q2_w - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z
    t1 = q1_w * q2_x + q1_x * q2_w + q1_y * q2_z - q1_z * q2_y
    t2 = q1_w * q2_y - q1_x * q2_z + q1_y * q2_w + q1_z * q2_x
    t3 = q1_w * q2_z + q1_x * q2_y - q1_y * q2_x + q1_z * q2_w

    final_quaternion = jnp.array([t0, t1, t2, t3])
    return normalize(final_quaternion)


@jax.jit
def transform_vector(pdt0F_G: np.ndarray, u_F: np.ndarray):
    w, v = jnp.split(pdt0F_G, [1])
    # w = w[0]

    uF_G = (
        (w**2 * u_F)
        + (2 * w * (skew_M @ v @ u_F))
        + (2 * (v @ u_F) * v)
        - ((v @ v) * u_F)
    )
    return uF_G


@register_pytree_node_class
class SpatialPose(object):

    # reference location expressed in self
    r: np.ndarray
    # reference orientation as quaternion
    q: np.ndarray

    def __init__(self, r: np.ndarray, q: np.ndarray):
        self.r = r
        self.q = q

    def __matmul__(self, other):
        """Return a SpatialPose that describes other (P) in self (S)

        self = p_BC
        other = p_AB
        new = p_AC = p_BC @ p_AB

        Parameters
        ----------
        other : _type_
            _description_
        """

        new_q = quaternion_multiply(other.q, self.q)
        # transforms from P -> S
        self_r_in_other = transform_vector(other.inv().q, self.r)
        new_r_in_self = self_r_in_other + other.r
        new_pose = SpatialPose(new_r_in_self, new_q)

        return new_pose

    def inv(self):
        q_inv = quaternion_inverse(self.q)
        p_inv = SpatialPose(-transform_vector(self.q, self.r), q_inv)
        return p_inv

    def __repr__(self) -> str:
        return f"r({self.r}), q({self.q})"

    def tree_flatten(self):
        return ((self.r, self.q), None)

    @classmethod
    def tree_unflatten(cls, aux_data, args):
        return cls(*args)

    @staticmethod
    def Identity() -> SpatialPose:
        return SpatialPose(np.zeros(3), np.array([1, 0, 0, 0]))


@register_pytree_node_class
class SpatialScrew(object):

    r: np.ndarray
    w: np.ndarray

    def __init__(self, r: np.ndarray, w: np.ndarray):
        self.r = r
        self.w = w

    def tree_flatten(self):
        return ((self.r, self.w), None)

    @classmethod
    def tree_unflatten(cls, aux_data, args):
        return cls(*args)


@jax.jit
def transform_screw(pose: SpatialPose, screw: np.ndarray) -> np.ndarray:
    v, w = jnp.split(screw, 2)
    new_w = transform_vector(pose.q, w)
    new_v = transform_vector(pose.q, v + (skew_M @ pose.r) @ w)
    new_screw = jnp.array([*new_v, *new_w])
    return new_screw


@jax.jit
def transform_screw_force(pose: SpatialPose, screw: np.ndarray) -> np.ndarray:
    force, torque = jnp.split(screw, 2)
    new_w = transform_vector(pose.q, torque) + transform_vector(
        pose.q, ((skew_M @ pose.r) @ force)
    )
    new_v = transform_vector(pose.q, force)

    new_screw = jnp.array([*new_v, *new_w])
    return new_screw


@jax.jit
def express_screw(pose: SpatialPose, screw: np.ndarray) -> np.ndarray:
    v, w = jnp.split(screw, 2)
    new_w = transform_vector(pose.q, w)
    new_v = transform_vector(pose.q, v)
    new_screw = jnp.array([*new_v, *new_w])
    return new_screw


@jax.jit
def normalize(v):
    return v / jnp.sqrt(v @ v)


def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler angles (in radians) to a quaternion.

    Args:
        roll (float): Rotation around the x-axis (roll angle).
        pitch (float): Rotation around the y-axis (pitch angle).
        yaw (float): Rotation around the z-axis (yaw angle).

    Returns:
        np.ndarray: A 4-element array representing the quaternion (w, x, y, z).
    """
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return normalize(jnp.array([qw, qx, qy, qz]))


def dcm_to_quaternion(dcm):
    """
    Converts a Direction Cosine Matrix (DCM) to a quaternion.

    Args:
        dcm (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: A 4-element array representing the quaternion (w, x, y, z).
    """
    trace = jnp.trace(dcm)
    if trace > 0:
        S = 2 * jnp.sqrt(trace + 1)
        qw = 0.25 * S
        qx = (dcm[2, 1] - dcm[1, 2]) / S
        qy = (dcm[0, 2] - dcm[2, 0]) / S
        qz = (dcm[1, 0] - dcm[0, 1]) / S
    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
        S = 2 * jnp.sqrt(1 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
        qw = (dcm[2, 1] - dcm[1, 2]) / S
        qx = 0.25 * S
        qy = (dcm[0, 1] + dcm[1, 0]) / S
        qz = (dcm[0, 2] + dcm[2, 0]) / S
    elif dcm[1, 1] > dcm[2, 2]:
        S = 2 * jnp.sqrt(1 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
        qw = (dcm[0, 2] - dcm[2, 0]) / S
        qx = (dcm[0, 1] + dcm[1, 0]) / S
        qy = 0.25 * S
        qz = (dcm[1, 2] + dcm[2, 1]) / S
    else:
        S = 2 * jnp.sqrt(1 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
        qw = (dcm[1, 0] - dcm[0, 1]) / S
        qx = (dcm[0, 2] + dcm[2, 0]) / S
        qy = (dcm[1, 2] + dcm[2, 1]) / S
        qz = 0.25 * S

    q = jnp.array([qw, qx, qy, qz])
    return q / jnp.linalg.norm(q)


@jax.jit
def quaternion_from_axis_angle(angle: float, axis: np.ndarray):
    axis = normalize(axis)
    c = jnp.cos(0.5 * angle)
    s = jnp.sin(0.5 * angle)
    return jnp.array([c, *(s * axis)])


@jax.jit
def quaternion_inverse(q: np.ndarray):
    return jnp.array([q[0], *(-q[1:])])


@jax.jit
def E(p: np.ndarray) -> np.ndarray:
    """A property matrix of euler parameters. Mostly used to transform between the
    cartesian angular velocity of body and the euler-parameters time derivative
    in the global coordinate system.

    Parameters
    ----------
    p : np.ndarray
        Euler parameters array of shape (4,)

    Returns
    -------
    np.ndarray
        E matrix of shape (3,4)

        m = np.array([
            [-e1, e0,-e3, e2],
            [-e2, e3, e0,-e1],
            [-e3,-e2, e1, e0],
            ])
    """
    e0, e = jnp.split(p, [1])
    I = np.eye(3)
    m = jnp.hstack((-e[:, None], (e0 * I) + skew_M @ e))
    return m


@jax.jit
def G(p: np.ndarray) -> np.ndarray:
    """A property matrix of euler parameters. Mostly used to transform between the
    cartesian angular velocity of body and the euler-parameters time derivative
    in the body coordinate system.

    Note: This is half the G_bar given in Shabana's book

    Parameters
    ----------
    p : np.ndarray
        Euler parameters array of shape (4,)

    Returns
    -------
    np.ndarray
        G matrix of shape (3,4)
    """
    e0, e = jnp.split(p, [1])
    I = np.eye(3)
    m = jnp.hstack((-e[:, None], (e0 * I) - skew_M @ e))
    return m


@jax.jit
def quaternion_to_dcm(p: np.ndarray) -> np.ndarray:
    """Transformation matrix as a function of euler parameters
    Note: The matrix is defined as a product of the two special matrices
    of euler parameters, the E and G matrices. This function is faster.

    Parameters
    ----------
    p : np.ndarray
        Euler parameters array of shape (4,)

    Returns
    -------
    np.ndarray
        Transformation matrix of shape (3,3)
    """
    m = E(p) @ G(p).T
    return m
