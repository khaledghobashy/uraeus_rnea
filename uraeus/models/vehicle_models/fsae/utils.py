import jax
import jax.numpy as jnp
import numpy as np

from uraeus.rnea.joints import construct_custom_joint


@jax.jit
def right_pose_polynomials(qdt0: np.ndarray) -> np.ndarray:
    z = qdt0[0] * 1e3

    phi = (
        (-2.5065e-9 * z**3)
        + (1.8870e-6 * z**2)
        + (3.9619e-4 * z**1)
        + (-4.2353e-5 * z**0)
    )

    theta = (
        (-2.5065e-9 * z**3)
        + (1.8870e-6 * z**2)
        + (3.9619e-4 * z**1)
        + (-4.2353e-5 * z**0)
    )

    psi = (
        (-2.5065e-9 * z**3)
        + (1.8870e-6 * z**2)
        + (3.9619e-4 * z**1)
        + (-4.2353e-5 * z**0)
    )

    x = (2.5462e-5 * z**2) + (-7.98277e-2 * z) + (-3.3937e-3)
    y = (1.3162e-5 * z**3) + (-1.2945e-3 * z**2) + (1.8655e-1 * z**1) + (1.475e-2)

    pose_states = jnp.array([0, 0, 0, 0, 0, z * 1e-3])
    return pose_states


# @jax.jit
def left_pose_polynomials(qdt0: np.ndarray) -> np.ndarray:
    z = qdt0[0] * 1e3

    phi = (
        (-2.5065e-9 * z**3)
        + (1.8870e-6 * z**2)
        + (3.9619e-4 * z**1)
        + (-4.2353e-5 * z**0)
    )

    theta = (
        (-2.5065e-9 * z**3)
        + (1.8870e-6 * z**2)
        + (3.9619e-4 * z**1)
        + (-4.2353e-5 * z**0)
    )

    psi = (
        (-2.5065e-9 * z**3)
        + (1.8870e-6 * z**2)
        + (3.9619e-4 * z**1)
        + (-4.2353e-5 * z**0)
    )

    x = (2.5462e-5 * z**2) + (-7.98277e-2 * z) + (-3.3937e-3)
    y = (1.3162e-5 * z**3) + (-1.2945e-3 * z**2) + (1.8655e-1 * z**1) + (1.475e-2)

    pose_states = jnp.array([0, 0, 0, 0, 0, z * 1e-3])
    return pose_states


RightSuspensionJoint = construct_custom_joint(
    "RightSuspensionJoint", right_pose_polynomials, 1, ["z"]
)
LeftSuspensionJoint = construct_custom_joint(
    "LeftSuspensionJoint", left_pose_polynomials, 1, ["z"]
)
