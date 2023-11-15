from functools import reduce, partial
from typing import Any, Callable, Iterable, List, NamedTuple, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

from uraeus.rnea.bodies import BodyKinematics, get_initialized_body_kinematics
from uraeus.rnea.joints import (
    FunctionalJoint,
    JointFrames,
    JointKinematics,
)
from uraeus.rnea.mobilizers import MobilizerForces
from uraeus.rnea.algorithms_operations import (
    construct_mobilizer_force,
    evaluate_joint_inertia_force,
    evaluate_successor_kinematics,
)
from uraeus.rnea.spatial_algebra import (
    motion_to_force_transform,
    spatial_transform_transpose,
)
from uraeus.rnea.graphs import (
    accumulate_leaf_to_root,
    accumulate_root_to_leaf,
)


@partial(jax.jit, static_argnums=(0,))
def eval_joints_kinematics(
    joints: tuple[FunctionalJoint, ...], coordinates: tuple[tuple[np.ndarray, ...], ...]
):
    new_kin = tuple(
        j.evaluate_kinematics(*coords) for j, coords in zip(joints, coordinates)
    )
    # new_kin = []
    # for i in range(len(joints)):
    #     qdt0, qdt1, qdt2 = coordinates[i]
    #     new_kin.append(joints[i].evaluate_kinematics(qdt0, qdt1, qdt2))
    return new_kin


def edge_force_func(
    successor_force: np.ndarray,
    transforms: List[np.ndarray],
    out_forces: List[np.ndarray],
):
    return successor_force + sum(map(jnp.dot, transforms, out_forces), np.zeros((6,)))


root_to_leaf = accumulate_root_to_leaf(
    get_initialized_body_kinematics(np.zeros((3,)), np.eye(3)),
    evaluate_successor_kinematics,
)


joints_forces_accumulator = accumulate_leaf_to_root(edge_force_func)


@partial(jax.jit, static_argnums=(0, 2))
def base_to_tip(
    joints: Tuple[FunctionalJoint, ...],
    joints_coordinates: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    traversal_order: Tuple[Tuple[int, int, int], ...],
) -> Tuple[Tuple[BodyKinematics], Tuple[JointKinematics]]:
    joints_kinematics = eval_joints_kinematics(joints, joints_coordinates)
    bodies_kinematics = root_to_leaf(traversal_order, joints_kinematics)

    return (bodies_kinematics, joints_kinematics)


def tip_to_base(
    joints_kinematics: List[JointKinematics],
    traversal_order: List[Tuple[int, List[int]]],
    bodies_kinematics: List[BodyKinematics],
    bodies_inertias: List[np.ndarray],
    external_forces: List[List[np.ndarray]],
) -> List[np.ndarray]:
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
    forces_transforms = [
        motion_to_force_transform(j.X_PS) for j in reversed(joints_kinematics)
    ]

    # Traverse the tree tip-to-base and Evaluate joints' forces
    joints_forces = list(
        reversed(
            joints_forces_accumulator(bodies_forces, forces_transforms, traversal_order)
        )
    )

    return joints_forces


dot = jax.vmap(jnp.dot)


@jax.jit
def evaluate_tau(
    joints_frames: List[JointFrames],
    joints_kinematics: List[JointKinematics],
    joints_forces: List[np.ndarray],
) -> np.ndarray:
    forces_transforms_X_MS = map(
        motion_to_force_transform,
        map(spatial_transform_transpose, [j.X_SM for j in joints_frames]),
    )
    # fi_Ms = map(jnp.dot, forces_transforms_X_MS, joints_forces)
    fi_Ms = dot(jnp.stack(list(forces_transforms_X_MS)), jnp.stack(joints_forces))
    taus = map(jnp.dot, [j.S_FM.T for j in joints_kinematics], fi_Ms)
    # taus = dot(jnp.stack([j.S_FM.T for j in joints_kinematics]), fi_Ms)
    tau = jnp.hstack(list(taus))
    # tau = np.hstack([(j.S_FM.T @ fi_M) for j, fi_M in zip(joints_kinematics, fi_Ms)])

    return tau


def extract_mobilizer_forces(
    joints_forces: List[np.ndarray],
    joints_frames: List[JointFrames],
    joints_kinematics: List[JointKinematics],
    bodies_kinematics: List[BodyKinematics],
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
    bodies_kinematics: List[BodyKinematics],
    bodies_inertias: List[np.ndarray],
    external_forces: List[List[np.ndarray]],
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
    system_kinematics: List[BodyKinematics],
) -> Tuple[np.ndarray, ...]:
    pos_vector = np.hstack([b.p_GB for b in system_kinematics])
    vel_vector = np.hstack([b.v_G for b in system_kinematics])
    acc_vector = np.hstack([b.a_G for b in system_kinematics])

    return (pos_vector, vel_vector, acc_vector)


def extract_generalized_forces(joints_forces: List[MobilizerForces]) -> np.ndarray:
    tau_vector = np.hstack([j.tau for j in joints_forces])
    return tau_vector


def extract_reaction_forces(joints_forces: List[MobilizerForces]) -> np.ndarray:
    rct_vector = np.hstack([j.fc_G for j in joints_forces])
    return rct_vector


# =============================================================================
# Obselete Code
# =============================================================================
# def root_to_leaf1(
#     joints_kinematics: List[JointKinematics],
#     traversal_order: List[Tuple[int, int, int]],
# ):
#     bodies_kin = [
#         get_initialized_body_kinematics(
#             np.zeros((3,)),
#             np.eye(3),
#         )
#     ]

#     for _, joint_index, predecessor_index in traversal_order:
#         suc_kin = evaluate_successor_kinematics(
#             bodies_kin[predecessor_index], joints_kinematics[joint_index]
#         )
#         bodies_kin.append(suc_kin)
#     return bodies_kin

# def leaf_to_root1(
#     bodies_forces: List[np.ndarray],
#     forces_transforms: List[np.ndarray],
#     traversal_order: List[Tuple[int, List[int]]],
# ) -> List[np.ndarray]:

#     forces = []
#     for successor_index, sub_joints in traversal_order[:-1]:
#         sub_joints_X_PS = [forces_transforms[i] for i in sub_joints]
#         sub_joints_fi_S = [forces[i] for i in sub_joints]
#         sub_joints_forces = sum(
#             map(np.dot, sub_joints_X_PS, sub_joints_fi_S), np.zeros((6,))
#         )
#         joint_force = bodies_forces[successor_index] + sub_joints_forces
#         forces.append(joint_force)

#     return forces
