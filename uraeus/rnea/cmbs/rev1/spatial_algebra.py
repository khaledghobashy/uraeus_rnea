"""Module for common spatial multibody math operations, e.g. 
transformation, rotations, ... etc.

Operations are tailored for Euler-parameters (quaternions) orientation 
representation.
"""
import itertools
import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_traceback_filtering", "off")


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

    return v @ skew_M


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
