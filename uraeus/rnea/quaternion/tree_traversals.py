from functools import reduce, partial

import jax

import jax.numpy as jnp
import numpy as np

from uraeus.rnea.quaternion.bodies import (
    BodyKinematics,
    get_initialized_body_kinematics,
)
from uraeus.rnea.quaternion.joints import (
    FunctionalJoint,
    JointFrames,
    JointKinematics,
)
from uraeus.rnea.quaternion.mobilizers import MobilizerForces
from uraeus.rnea.quaternion.algorithms_operations import (
    construct_mobilizer_force,
    evaluate_joint_inertia_force,
    evaluate_successor_kinematics,
)
from uraeus.rnea.quaternion.spatial_algebra import transform_screw_force, express_screw
from uraeus.rnea.quaternion.graphs import (
    accumulate_leaf_to_root,
    accumulate_root_to_leaf,
)

# type definition fro repetitive type-hints
SystemForces = list[tuple[list[np.ndarray], list[np.ndarray]]]


@partial(jax.jit, static_argnums=(0,))
def eval_joints_kinematics(
    joints: tuple[FunctionalJoint, ...], coordinates: tuple[tuple[np.ndarray, ...], ...]
):
    """
    Evaluates the kinematics of a set of joints given their coordinates.

    Parameters
    ----------
    joints : tuple of FunctionalJoint
        A tuple containing the joints to evaluate.
    coordinates : tuple of tuple of np.ndarray
        A tuple containing the coordinates for each joint.

    Returns
    -------
    tuple
        A tuple containing the evaluated kinematics for each joint.
    """
    new_kin = tuple(
        j.evaluate_kinematics(*coords) for j, coords in zip(joints, coordinates)
    )
    return new_kin


def edge_force_func(
    successor_force: np.ndarray,
    transforms: list[np.ndarray],
    out_forces: list[np.ndarray],
):
    """
    Computes the resultant force at an edge/joint by transforming and summing
    the output forces.

    Parameters
    ----------
    successor_force : np.ndarray
        The force applied by the successor as a 6-element array.
    transforms : list of np.ndarray
        A list of transformation matrices.
    out_forces : list of np.ndarray
        A list of output forces to be transformed and summed.

    Returns
    -------
    np.ndarray
        The resultant force as a 6-element array.
    """
    out_forces_S = sum(
        map(transform_screw_force, transforms, out_forces), np.zeros((6,))
    )
    return successor_force + out_forces_S


root_to_leaf = accumulate_root_to_leaf(
    get_initialized_body_kinematics(np.zeros((3,)), np.array([1, 0, 0, 0])),
    evaluate_successor_kinematics,
)


joints_forces_accumulator = accumulate_leaf_to_root(edge_force_func)


@partial(jax.jit, static_argnums=(0, 2))
def base_to_tip(
    joints: tuple[FunctionalJoint, ...],
    joints_coordinates: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    traversal_order: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[BodyKinematics], tuple[JointKinematics]]:
    joints_kinematics = eval_joints_kinematics(joints, joints_coordinates)
    bodies_kinematics = root_to_leaf(traversal_order, joints_kinematics)

    return (bodies_kinematics, joints_kinematics)


def tip_to_base(
    joints_kinematics: list[JointKinematics],
    traversal_order: list[tuple[int, list[int]]],
    bodies_kinematics: list[BodyKinematics],
    bodies_inertias: list[np.ndarray],
    external_forces: SystemForces,
) -> list[np.ndarray]:
    # Evaluate inertia forces and external forces on bodies
    bodies_forces = list(
        map(
            evaluate_joint_inertia_force,
            bodies_kinematics,
            bodies_inertias,
            external_forces,
        )
    )

    # Extract joints' transforms from joints' kinematics
    forces_transforms = [j.p_SP for j in reversed(joints_kinematics)]

    # Traverse the tree tip-to-base and Evaluate joints' forces
    joints_forces = list(
        reversed(
            joints_forces_accumulator(bodies_forces, forces_transforms, traversal_order)
        )
    )
    return joints_forces


dot = jax.vmap(jnp.dot)
# dot = np.dot


@jax.jit
def evaluate_tau(
    joints_frames: list[JointFrames],
    joints_kinematics: list[JointKinematics],
    joints_forces: list[np.ndarray],
) -> np.ndarray:
    forces_transforms_p_SM = [j.p_SM for j in joints_frames]
    fi_Ms = list(map(transform_screw_force, forces_transforms_p_SM, joints_forces))
    taus = map(jnp.dot, [j.S_FM.T for j in joints_kinematics], fi_Ms)
    tau = jnp.hstack(list(taus))
    return tau


def extract_mobilizer_forces(
    joints_forces: list[np.ndarray],
    joints_frames: list[JointFrames],
    joints_kinematics: list[JointKinematics],
    bodies_kinematics: list[BodyKinematics],
):
    force_instances = list(
        map(
            construct_mobilizer_force,
            joints_forces,
            joints_frames,
            joints_kinematics,
            bodies_kinematics,
        )
    )
    return force_instances


def eval_bodies_forces(
    bodies_kinematics: list[BodyKinematics],
    bodies_inertias: list[np.ndarray],
    external_forces: SystemForces,
):
    forces = list(
        map(
            evaluate_joint_inertia_force,
            bodies_kinematics,
            bodies_inertias,
            external_forces,
        )
    )
    return forces


def extract_state_vectors(
    system_kinematics: list[BodyKinematics],
) -> tuple[np.ndarray, ...]:
    pos_vector = np.hstack([b.p_GB for b in system_kinematics])
    vel_vector = np.hstack([b.v_G for b in system_kinematics])
    acc_vector = np.hstack([b.a_G for b in system_kinematics])

    return (pos_vector, vel_vector, acc_vector)


def extract_generalized_forces(joints_forces: list[MobilizerForces]) -> np.ndarray:
    tau_vector = np.hstack([j.tau for j in joints_forces])
    return tau_vector


def extract_reaction_forces(joints_forces: list[MobilizerForces]) -> np.ndarray:
    rct_vector = np.hstack([j.fc_G for j in joints_forces])
    return rct_vector
