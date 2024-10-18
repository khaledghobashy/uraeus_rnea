"""Module for common spatial multibody math operations, e.g. 
transformation, rotations, ... etc.

Operations are tailored for Euler-parameters (quaternions) orientation 
representation.
"""

import itertools
import functools
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


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
    return final_quaternion


# class Quaternion(object):

#     # def __init__(self, w:float, x:float, y:float, z:float):
#     def __init__(self, *args):
#         if len(args) == 1:

#             self.q = jnp.array(*args)

#         elif len(args) == 4:
#             self.q = jnp.array(args)

#         self.dtype = np.dtypes.Float64DType

#     def inv(self):
#         return Quaternion(self.q[0], *(-self.q[1:]))

#     def __mul__(self, other):
#         if isinstance(other, Quaternion):
#             return Quaternion(self.q * other.q)
#         else:
#             return Quaternion(self.q * other)

#     def __rmul__(self, other):
#         return other * self.q

#     def __pow__(self, n):
#         return self.q**n

#     def __repr__(self):
#         return self.q.__repr__()

#     def __matmul__(self, other):
#         if isinstance(other, Quaternion):
#             return Quaternion(quaternion_multiply(self.q, other.q))
#         else:
#             if other.size == 4:
#                 return Quaternion(quaternion_multiply(self.q, other))
#             elif other.size == 3:
#                 new_other = jnp.array([0, *other])
#                 return Quaternion(quaternion_multiply(self.q, new_other))
#             else:
#                 raise ValueError("Non valid value!")


# def transform_vector(pdt0F_G: np.ndarray, u_F: np.ndarray):
#     pdt0F_G_inv = jnp.array([pdt0F_G[0], *(-pdt0F_G[1:])])
#     uF_G = quaternion_multiply(
#         pdt0F_G, quaternion_multiply(jnp.array([0, *u_F]), pdt0F_G_inv)
#     )
#     return uF_G[1:]


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
def skew_matrix(v: np.ndarray) -> np.ndarray:
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

    return skew_M @ v


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
    m = jnp.hstack((-e[:, None], (e0 * I) + skew_matrix(e)))
    return m


# @jax.jit
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
    # e0 = p[0]
    # e = p[1:]
    e0, e = jnp.split(p, [1])
    I = np.eye(3)
    m = jnp.hstack((-e[:, None], (e0 * I) - skew_matrix(e)))
    return m


# @jax.jit
def A(p: np.ndarray) -> np.ndarray:
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


@jax.jit
def B(p: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    This matrix represents the variation of the body orientation with respect
    to the change in euler parameters. This can be thought as the jacobian of
    the A.dot(a), where A is the transformation matrix in terms of euler
    parameters.

    Parameters
    ----------
    p : np.ndarray
        Euler parameters array of shape (4,)

    a : np.ndarray
        Local vector defined in the given euler-parameters frame, shape (3,)

    Returns
    -------
    np.ndarray
        Jacobian of `A(p) @ a`, shape (3, 4)
    """
    I = np.eye(3, dtype=np.float64)

    e0, e = jnp.split(p, [1])
    a_s = skew_matrix(a)
    e_s = skew_matrix(e)

    a_v = a[:, None]
    e_v = e[:, None]

    m0 = (e0 * I) + e_s
    m1 = m0 @ a_v
    m2 = (e_v @ a_v.T) - (m0 @ a_s)

    m = 2 * jnp.hstack((m1, m2))

    return m


def orthogonal_vector(v: np.ndarray) -> np.ndarray:
    """Generate an arbitrary vector, `u`, that is normal to the given vector `v`,
    so that `v.T @ u` evaluates to 0.

    Parameters
    ----------
    v : np.ndarray
        Vector of shape (3,)

    Returns
    -------
    np.ndarray
        Vector of shape (3,), that is normal to `v`

    """
    x, y, z = v

    v1 = np.array([y, -x, 0])
    v2 = np.array([-z, 0, x])

    v3 = (5 * v1) + (9 * v2)

    u = v3 / np.linalg.norm(v3)

    return u


def triad(v1: np.ndarray, v2: Optional[np.ndarray] = None) -> np.ndarray:
    """Create a (3, 3) orthonormal array that represents a given spatial
    reference frame, where the z-axis is oriented along the given `v1`
    vector, and x-axis is oriented along the given `v2` vector, if given.

    Parameters
    ----------
    v1 : np.ndarray
        Orientation of z-axis of the reference frame

    v2 : np.ndarray, optional
        Orientation of x-axis of the reference frame, by default None

    Returns
    -------
    np.ndarray
        Orthonormal (3, 3) array.
    """
    k = v1 / np.linalg.norm(v1)

    if v2 is not None:
        i = v2 / np.linalg.norm(v2)
    else:
        i = orthogonal_vector(k)

    j = skew_matrix(k) @ i
    j = j / np.linalg.norm(j)

    R = np.vstack([i, j, k]).T

    return R
