from functools import partial
from operator import sub
from typing import Iterable, List, NamedTuple, Tuple

import jax

import jax.numpy as jnp
import numpy as np

from uraeus.rnea.quaternion.bodies import BodyKinematics
from uraeus.rnea.quaternion.joints import JointKinematics, FunctionalJoint
from uraeus.rnea.quaternion.spatial_algebra import (
    SpatialPose,
    express_screw,
    transform_screw,
)

from uraeus.rnea.quaternion.graphs import accumulate_root_to_leaf
from uraeus.rnea.quaternion.tree_traversals import (
    base_to_tip,
    tip_to_base,
    evaluate_tau,
    joints_forces_accumulator,
)


@partial(jax.jit, static_argnums=(0,))
def split_coordinates(
    idx: Tuple[int], qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    coordinates = tuple(
        (qdt0[i:j], qdt1[i:j], qdt2[i:j]) for (i, j) in zip(idx[:-1], idx[1:])
    )
    return coordinates


class MultiBodyData(NamedTuple):
    joints: Tuple[FunctionalJoint]
    bodies_inertias: List[np.ndarray]
    forward_traversal: Tuple[Tuple[int, int, int], ...]
    backward_traversal: List[Tuple[int, List[int]]]
    qdt0_idx: Tuple[int]
    qdt1_idx: Tuple[int]

    def __hash__(self):
        return hash(self.__class__.__name__)


class HybridDynamicsData(NamedTuple):
    tree_data: MultiBodyData
    permutation_matrix: np.ndarray
    n_fd: int


class IDCallRes(NamedTuple):
    tau: np.ndarray
    bodies_kinematics: List[BodyKinematics]
    joints_kinematics: List[JointKinematics]
    joints_forces: List[np.ndarray]


@partial(jax.jit, static_argnums=(0,))
def inverse_dynamics_call(
    tree_data: MultiBodyData,
    external_forces: List[List[np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    qdt2: np.ndarray,
) -> IDCallRes:
    func_joints = tree_data.joints
    forward_traversal = tree_data.forward_traversal
    backward_traversal = tree_data.backward_traversal
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
    external_forces: List[List[np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
) -> IDCallRes:
    qdt2 = jnp.zeros_like(qdt1)
    res = inverse_dynamics_call(tree_data, external_forces, qdt0, qdt1, qdt2)
    return res


@partial(jax.jit, static_argnums=(0,))
def forward_dynamics_call(
    tree_data: MultiBodyData,
    external_forces: List[List[np.ndarray]],
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
        joints_kin: List[JointKinematics],
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
        joints_kin: List[JointKinematics],
        qdt0: np.ndarray,
        qdt2: np.ndarray,
    ) -> List[np.ndarray]:
        coordinates = split_coordinates(
            tree_data.qdt0_idx, qdt0, jnp.zeros_like(qdt0), qdt2
        )
        a_J_mob = [j.mobilizer.a_J(*qs) for j, qs in zip(tree_data.joints, coordinates)]
        a_J_jnt = [
            transform_screw(j.frames.p_SM.inv(), a_J)
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
        joints_kin: List[JointKinematics],
    ):
        forward_traversal = tree_data.forward_traversal
        backward_traversal = tree_data.backward_traversal
        joints_frames = tuple(j.frames for j in tree_data.joints)
        forces_transforms = tuple(j.p_SP for j in reversed(joints_kin))

        bodies_acc = node_acceleration_accumulator(forward_traversal, joints_kin)
        bodies_forces = tuple(map(jnp.dot, tree_data.bodies_inertias, bodies_acc))
        forces = joints_forces_accumulator(
            bodies_forces, forces_transforms, backward_traversal
        )
        forces = tuple(reversed(forces))

        tau = evaluate_tau(joints_frames, joints_kin, forces)
        return tau


class HybridDynamics(object):
    def evaluate_C(
        self,
        hybrid_data: HybridDynamicsData,
        external_forces: List[List[np.ndarray]],
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        qdt2_id: np.ndarray,
    ) -> IDCallRes:
        n_fd = hybrid_data.n_fd
        Q = hybrid_data.permutation_matrix
        qdt2 = Q.T @ np.hstack([np.zeros((n_fd,)), qdt2_id])

        return inverse_dynamics_call(
            hybrid_data.tree_data, external_forces, qdt0, qdt1, qdt2
        )

    def forward_dynamics_call(
        self,
        hybrid_data: HybridDynamicsData,
        external_forces: List[List[np.ndarray]],
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        qdt2_id: np.ndarray,
        tau_fd: np.ndarray,
    ) -> np.ndarray:
        n_fd = hybrid_data.n_fd
        Q = hybrid_data.permutation_matrix

        C, _, joints_kin, _ = self.evaluate_C(
            hybrid_data, external_forces, qdt0, qdt1, qdt2_id
        )
        H = JointInertiaMatrixOperations.construct_H(
            hybrid_data.tree_data, joints_kin, qdt0
        )

        H_fd = (Q @ H @ Q.T)[:n_fd, :n_fd]
        C_fd = (Q @ C)[:n_fd]

        rhs = tau_fd - C_fd
        qdt2_fd = np.linalg.solve(H_fd, rhs)
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
    joints_kin: Tuple[JointKinematics, ...],
    ext_forces: Tuple[Tuple[np.ndarray, ...], ...],
):
    bodies_p_GB = _bodies_config_func(tree_data.forward_traversal, joints_kin)
    bodies_fe_S = map(
        express_screw,
        bodies_p_GB,
        [sum(forces, np.zeros((6,))) for forces in ext_forces],
    )
    forces_transforms = [(j.p_SP) for j in reversed(joints_kin)]
    joints_forces = list(
        reversed(
            joints_forces_accumulator(
                list(bodies_fe_S), forces_transforms, tree_data.backward_traversal
            )
        )
    )

    joints_frames = [j.frames for j in tree_data.joints]

    tau = evaluate_tau(joints_frames, joints_kin, joints_forces)
    return tau
