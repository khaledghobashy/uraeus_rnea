import logging
from functools import partial
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from uraeus.utils.logging import construct_logger
from uraeus.rnea.bodies import BodyKinematics
from uraeus.rnea.joints import JointKinematics, FunctionalJoint
from uraeus.rnea.spatial_algebra import (
    SpatialPose,
    express_screw,
    transform_screw,
    transform_screw_force,
    skew_M,
    quaternion_to_dcm,
    transform_spatial_inertia,
    SpatialInertia,
    transform_vector,
)

from uraeus.rnea.graphs import accumulate_root_to_leaf, accumulate_leaf_to_root
from uraeus.rnea.tree_traversals import (
    base_to_tip,
    tip_to_base,
    evaluate_tau,
    joints_forces_accumulator,
    SystemForces,
)
from uraeus.rnea.topologies import MultiBodyData, HybridDynamicsData

logger = construct_logger(__name__, logging.DEBUG)


@partial(jax.jit, static_argnums=(0,))
def split_coordinates(
    idx: tuple[int], qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    coordinates = tuple(
        (qdt0[i:j], qdt1[i:j], qdt2[i:j]) for (i, j) in zip(idx[:-1], idx[1:])
    )
    return coordinates


def get_qdt1_from_udt0(
    tree_data: MultiBodyData, qdt0: np.ndarray, udt0: np.ndarray
) -> np.ndarray:
    qdt0s, udt0s, _ = zip(*split_coordinates(tree_data.qdt0_idx, qdt0, udt0, qdt0))
    qdt1 = [
        j.mobilizer.N(qdt0_j) @ udt0_j
        for j, qdt0_j, udt0_j in zip(tree_data.joints, qdt0s, udt0s)
    ]
    return jnp.hstack(qdt1)


def get_udt0_from_qdt1(
    tree_data: MultiBodyData, qdt0: np.ndarray, udt0: np.ndarray
) -> np.ndarray:
    qdt0s, udt0s, _ = zip(*split_coordinates(tree_data.qdt0_idx, qdt0, udt0, qdt0))
    qdt1 = [
        j.mobilizer.N(qdt0_j).T @ udt0_j
        for j, qdt0_j, udt0_j in zip(tree_data.joints, qdt0s, udt0s)
    ]
    return jnp.hstack(qdt1)


class IDCallRes(NamedTuple):
    tau: np.ndarray
    bodies_kinematics: list[BodyKinematics]
    joints_kinematics: list[JointKinematics]
    joints_forces: list[np.ndarray]


@partial(jax.jit, static_argnums=(0,))
def inverse_dynamics_call(
    tree_data: MultiBodyData,
    external_forces: SystemForces,
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    qdt2: np.ndarray,
) -> IDCallRes:
    func_joints = tree_data.joints
    forward_traversal = tree_data.graph_data.base_to_tip
    backward_traversal = tree_data.graph_data.adjacency_list
    qdt0_idx = tree_data.qdt0_idx

    joints_coordinates = split_coordinates(qdt0_idx, qdt0, qdt1, qdt2)
    joints_frames = tuple(j.frames for j in tree_data.joints)

    bodies_kin, joints_kin = base_to_tip(
        joints=func_joints,
        joints_coordinates=joints_coordinates,
        traversal_order=forward_traversal,
    )

    joints_forces = tip_to_base(
        joints_kinematics=joints_kin,
        traversal_order=backward_traversal,
        bodies_kinematics=bodies_kin,
        bodies_inertias=tree_data.bodies_inertias,
        external_forces=external_forces,
    )
    tau = evaluate_tau(joints_frames, joints_kin, joints_forces)

    return IDCallRes(tau, bodies_kin, joints_kin, joints_forces)


@partial(jax.jit, static_argnums=(0,))
def evaluate_C(
    tree_data: MultiBodyData,
    external_forces: SystemForces,
    qdt0: np.ndarray,
    qdt1: np.ndarray,
) -> IDCallRes:
    qdt2 = jnp.zeros_like(qdt1)
    res = inverse_dynamics_call(tree_data, external_forces, qdt0, qdt1, qdt2)
    return res


@partial(jax.jit, static_argnums=(0,))
def forward_dynamics_call(
    tree_data: MultiBodyData,
    external_forces: SystemForces,
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    C, _, joints_kin, _ = evaluate_C(tree_data, external_forces, qdt0, qdt1)
    H = JointInertiaMatrixOperations.construct_H(tree_data, joints_kin, qdt0)
    rhs = tau - C
    qdt2 = jnp.linalg.solve(H, rhs)
    return qdt2


@jax.jit
def eval_successor_acc(
    predecessor_acc: np.ndarray, joint_kin: JointKinematics
) -> np.ndarray:
    a_B = transform_screw(joint_kin.p_PS, predecessor_acc) + joint_kin.a_J
    return a_B


node_acceleration_accumulator = accumulate_root_to_leaf(
    np.zeros((6,)),
    eval_successor_acc,
)


class JointInertiaMatrixOperations(NamedTuple):
    @classmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def construct_H(
        cls,
        tree_data: MultiBodyData,
        joints_kin: list[JointKinematics],
        qdt0: np.ndarray,
    ):
        booleans = np.eye(len(qdt0))
        new_kins = [
            cls.construct_new_acc(tree_data, joints_kin, qdt0, delta)
            for delta in booleans
        ]
        H_columns = [cls.traverse(tree_data, j_kin) for j_kin in new_kins]
        return jnp.column_stack(H_columns)

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def construct_new_acc(
        tree_data: MultiBodyData,
        joints_kin: list[JointKinematics],
        qdt0: np.ndarray,
        qdt2: np.ndarray,
    ) -> list[np.ndarray]:
        coordinates = split_coordinates(
            tree_data.qdt0_idx, qdt0, jnp.zeros_like(qdt0), qdt2
        )
        a_J_mob = [j.mobilizer.a_J(*qs) for j, qs in zip(tree_data.joints, coordinates)]
        a_J_jnt = [
            transform_screw(j.frames.p_MS, a_J)
            for j, a_J in zip(tree_data.joints, a_J_mob)
        ]
        new_kin = [
            JointKinematics(*kin[:-1], a_J) for kin, a_J in zip(joints_kin, a_J_jnt)
        ]
        return new_kin

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def traverse(
        tree_data: MultiBodyData,
        joints_kin: list[JointKinematics],
    ):
        forward_traversal = tree_data.graph_data.base_to_tip
        backward_traversal = tree_data.graph_data.adjacency_list
        joints_frames = tuple(j.frames for j in tree_data.joints)
        forces_transforms = tuple(j.p_SP for j in joints_kin)

        bodies_acc = node_acceleration_accumulator(forward_traversal, joints_kin)
        bodies_forces = tuple(map(jnp.dot, tree_data.bodies_inertias, bodies_acc))
        forces = joints_forces_accumulator(
            bodies_forces, forces_transforms, backward_traversal
        )

        tau = evaluate_tau(joints_frames, joints_kin, forces)
        return tau


class HybridDynamics(object):
    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def evaluate_C(
        hybrid_data: HybridDynamicsData,
        external_forces: SystemForces,
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        qdt2_id: np.ndarray,
    ) -> IDCallRes:
        """Evaluate C' following Eqn (9.3), page 173. Physically, C' is the force
        required to impart zero acceleration to each forward-dynamics joint and
        the given acceleration to each inverse-dynamics joint.
        C' = ID(qdt0, qdt1, Q.T @ [0, qdt2_fd].T)

        Parameters
        ----------
        hybrid_data : HybridDynamicsData
            Container for the permutation matrix, Q,  and other attributes
            relevant for the hybrid-system
        external_forces : SystemForces
            External forces applied on the system.
        qdt0 : np.ndarray
            Position coordinates of the system joints (inverse and forward)
        qdt1 : np.ndarray
            Velocity coordinates of the system joints (inverse and forward)
        qdt2_id : np.ndarray
            Acceleration coordinates of the inverse-dynamics joints

        Returns
        -------
        IDCallRes
            Inverse-dynamics return object

        References
        ----------
        Featherstone 2008 - Rigid-body dynamics algorithms
        """
        n_fd = hybrid_data.n_fd
        Q = hybrid_data.permutation_matrix
        qdt2 = Q.T @ jnp.hstack([np.zeros((n_fd,)), qdt2_id])

        return inverse_dynamics_call(
            hybrid_data.tree_data, external_forces, qdt0, qdt1, qdt2
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def forward_dynamics_call(
        cls,
        hybrid_data: HybridDynamicsData,
        external_forces: SystemForces,
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        qdt2_id: np.ndarray,
        tau_fd: np.ndarray,
    ) -> np.ndarray:
        """Solving for the generalized accelerations of the forward-dynamics
        joints, qdt2_fd, following Eqn (9.2), page 173, using the provided
        algorithm:
            1- Calculate C', using Eqn (9.3)
            2- Calculate H11.
            3- Solve H11 @ qdt2_fd = tau_fd - C'fd for qdt2_fd

        Parameters
        ----------
        hybrid_data : HybridDynamicsData
            Container for the permutation matrix, Q,  and other attributes
            relevant for the hybrid-system
        external_forces : SystemForces
            External forces applied on the system.
        qdt0 : np.ndarray
            Position coordinates of the system joints (inverse and forward)
        qdt1 : np.ndarray
            Velocity coordinates of the system joints (inverse and forward)
        qdt2_id : np.ndarray
            Acceleration coordinates of the inverse-dynamics joints
        tau_fd : np.ndarray
            Applied generalized forces to the forward-dynamics joints

        Returns
        -------
        np.ndarray
            Generalized accelerations for the forward-dynamics joints
        """
        n_fd = hybrid_data.n_fd
        Q = hybrid_data.permutation_matrix

        C, _, joints_kin, _ = cls.evaluate_C(
            hybrid_data, external_forces, qdt0, qdt1, qdt2_id
        )
        # H = JointInertiaMatrixOperations.construct_H(
        #     hybrid_data.tree_data, joints_kin, qdt0
        # )
        H = CompositeInertiaMatrixOperations.construct_H(
            hybrid_data.tree_data, joints_kin, qdt0
        )
        H_fd = (Q @ H @ Q.T)[:n_fd, :n_fd]
        C_fd = (Q @ C)[:n_fd]

        rhs = tau_fd - C_fd
        qdt2_fd = jnp.linalg.solve(H_fd, rhs)
        return qdt2_fd


def _helper(predecessor_p_GB: SpatialPose, joint: JointKinematics):
    p_GB = joint.p_PS @ predecessor_p_GB
    return p_GB


_bodies_config_func = accumulate_root_to_leaf(
    SpatialPose(np.array([0, 0, 0]), np.array([1, 0, 0, 0])), _helper
)


@partial(jax.jit, static_argnums=(0,))
def ext_forces_to_gen_forces(
    tree_data: MultiBodyData,
    joints_kin: tuple[JointKinematics, ...],
    ext_forces: SystemForces,
):
    bodies_p_GB = _bodies_config_func(tree_data.graph_data.base_to_tip, joints_kin)
    bodies_fe_S_g = map(
        express_screw,
        bodies_p_GB,
        [sum(forces[0], jnp.zeros((6,))) for forces in ext_forces],
    )
    bodies_fe_S_l = [sum(forces[1], jnp.zeros((6,))) for forces in ext_forces]
    bodies_fe_S = [f1 + f2 for f1, f2 in zip(bodies_fe_S_g, bodies_fe_S_l)]
    forces_transforms = [j.p_SP for j in joints_kin]
    joints_forces = list(
        (
            joints_forces_accumulator(
                list(bodies_fe_S),
                forces_transforms,
                tree_data.graph_data.adjacency_list,
            )
        )
    )

    joints_frames = [j.frames for j in tree_data.joints]

    tau = evaluate_tau(joints_frames, joints_kin, joints_forces)
    return tau


def accumulate_spatial_inertia(
    predecessor_inertia: SpatialInertia,
    successors_transforms: list[np.ndarray],
    successors_inertia: list[SpatialInertia],
):
    children_inertias_S = sum(
        map(transform_spatial_inertia, successors_transforms, successors_inertia),
        SpatialInertia.Identity(),
    )
    return predecessor_inertia + children_inertias_S


inertia_accumulator = accumulate_leaf_to_root(accumulate_spatial_inertia)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return jnp.allclose(a, a.T, rtol=rtol, atol=atol)


def spatial_inertia_to_matrix(inertia: SpatialInertia):
    coupling_term = inertia.m * (skew_M @ inertia.d)
    spatial_inertia_matrix = jnp.vstack(
        [
            jnp.hstack([inertia.m * np.eye(3), -coupling_term]),
            jnp.hstack([coupling_term, inertia.J]),
        ]
    )
    return spatial_inertia_matrix


def transform_hing_matrix(p_AB: SpatialPose, S_A: np.ndarray):
    v, w = jnp.split(S_A, 2)
    new_v = transform_vector(p_AB.q, v + (-skew_M @ p_AB.r) @ w)
    new_w = transform_vector(p_AB.q, w)
    new_screw = jnp.array([*new_v, *new_w])
    return new_screw


@partial(jax.jit, static_argnums=(0,))
def assemble_inertia_matrix(
    indices: tuple[tuple[slice, slice, int]],
    diagonal_blocks: list[np.ndarray],
    values: list[np.ndarray],
):
    diagonal_matrix = jax.scipy.linalg.block_diag(*diagonal_blocks)
    lower_matrix = jnp.zeros(diagonal_matrix.shape)
    for (u_slice, v_slice, _), sub_block in zip(indices, values):
        v_slice = slice(v_slice.start, v_slice.start + sub_block.shape[1])
        lower_matrix = lower_matrix.at[u_slice, v_slice].set(
            sub_block, unique_indices=True
        )
        # matrix = matrix.at[v_slice, u_slice].set(sub_block.T, unique_indices=True)
    return diagonal_matrix + lower_matrix + lower_matrix.T


def scan(f, init, xs):
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, ys


def off_diagonal_force_accumulator(
    carry_force: np.ndarray, inputs: tuple[SpatialPose, np.ndarray]
):
    p_SP, s_S = inputs
    Fs = transform_screw_force(p_SP, carry_force)
    H = Fs.T @ s_S
    return Fs, H


class CompositeInertiaMatrixOperations(NamedTuple):
    @classmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def construct_H(
        cls,
        tree_data: MultiBodyData,
        joints_kin: list[JointKinematics],
        qdt0: np.ndarray,
    ):
        # Notes:
        # Assumptions:
        # 1. All bodies are given some SpatialInertia that has body mass, m,
        # inertia tensor, J, evaluated at center of mass, and offset-vector, d
        # that represents the relative position of center-of-mass to the body's
        # reference-frame origin.
        # Computations:
        # 1. Starting from the leaves of the multibody tree, we want to
        # accumulate the composite-rigid-body inertia, Ic, traversing down to the
        # root, in which the Ic of any given body i, represents the inertia of
        # the sub-tree rooted at body i, as if it is one big rigid body
        # constructed out of a set of rigid bodies.
        # 2. Assuming each body in the tree has its inertia defined in the body
        # reference frame.
        # 3. The inertia has to be re-expressed from the successor (S) frame to
        # its parent's/predecessor (P) frame, then compute a new Ic_P for the
        # parent body, as a composition of two rigid-bodies inertia, evaluated
        # at the parent's (P) reference origin.

        transforms = [j.p_SP for j in joints_kin]
        composite_inertias = inertia_accumulator(
            tree_data.spatial_inertias,
            transforms,
            tree_data.graph_data.adjacency_list,
        )

        inertia_matrices = list(map(spatial_inertia_to_matrix, composite_inertias))

        joints_hing_matrices = [
            transform_hing_matrix(f.frames.p_MS, j.S_FM)
            for f, j in zip(tree_data.joints, joints_kin)
        ]

        Fs = jax.tree.map(jnp.dot, inertia_matrices, joints_hing_matrices)

        diagonal_inertias = [S.T @ F for S, F in zip(joints_hing_matrices, Fs)]

        paths = tree_data.graph_data.nodes_to_root_paths
        off_diagonal_inertias = []
        for path in paths:
            initial_force = Fs[path[0] - 1]
            joints_indices = [j - 1 for j in path[:-1]]
            hinges_indices = [j - 1 for j in path[1:]]
            joints_transform = [joints_kin[j].p_SP for j in joints_indices]
            hinges_matrices = [joints_hing_matrices[j] for j in hinges_indices]
            transforms = list(zip(joints_transform, hinges_matrices))

            _, accumulated_forces = scan(
                off_diagonal_force_accumulator, initial_force, transforms
            )

            off_diagonal_inertias.extend(accumulated_forces)

        H = assemble_inertia_matrix(
            tree_data.sparsity_pattern, diagonal_inertias, off_diagonal_inertias
        )

        return H
