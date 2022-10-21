from functools import partial
from itertools import repeat
from operator import sub
from typing import Iterable, List, Dict, NamedTuple, Tuple

import numpy as np
from uraeus.rnea.bodies import BodyKinematics

from uraeus.rnea.joints import (
    JointKinematics,
)
from uraeus.rnea.spatial_algebra import motion_to_force_transform
from uraeus.rnea.topologies import MultiBodyData
from uraeus.rnea.graphs import accumulate_root_to_leaf
from uraeus.rnea.tree_traversals import (
    base_to_tip,
    evaluate_tau,
    joints_forces_accumulator,
    tip_to_base,
)


def split(arr: np.ndarray, idx: np.ndarray):
    res = [arr[i:j] for (i, j) in zip(idx[:-1], idx[1:])]
    return res


def split_coordinates(
    idx: np.ndarray, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    coordinates = [
        (qdt0[i:j], qdt1[i:j], qdt2[i:j]) for (i, j) in zip(idx[:-1], idx[1:])
    ]
    return coordinates


class IDCallRes(NamedTuple):
    tau: np.ndarray
    bodies_kinematics: List[BodyKinematics]
    joints_kinematics: List[JointKinematics]
    joints_forces: List[np.ndarray]


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
    joints_frames = [j.frames for j in tree_data.joints]

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


def evaluate_C(
    tree_data: MultiBodyData,
    external_forces: List[List[np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
) -> IDCallRes:

    qdt2 = np.zeros_like(qdt1)
    res = inverse_dynamics_call(tree_data, external_forces, qdt0, qdt1, qdt2)
    return res


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
    qdt2 = np.linalg.solve(H, rhs)
    return qdt2


def eval_successor_acc(
    predecessor_acc: np.ndarray, joint_kin: JointKinematics
) -> np.ndarray:
    a_B = (joint_kin.X_SP @ predecessor_acc) + joint_kin.a_J
    return a_B


node_acceleration_accumulator = accumulate_root_to_leaf(
    np.zeros((6,)),
    eval_successor_acc,
)


class JointInertiaMatrixOperations(NamedTuple):
    @classmethod
    def construct_H(
        cls,
        tree_data: MultiBodyData,
        joints_kin: List[JointKinematics],
        qdt0,
    ):
        booleans = np.eye(len(qdt0))
        new_kins = [
            cls.construct_new_acc(tree_data, joints_kin, qdt0, delta)
            for delta in booleans
        ]
        H_columns = [cls.traverse(tree_data, j_kin) for j_kin in new_kins]
        return np.column_stack(H_columns)

    @staticmethod
    def construct_new_acc(
        tree_data: MultiBodyData,
        joints_kin: List[JointKinematics],
        qdt0: np.ndarray,
        qdt2: np.ndarray,
    ) -> List[np.ndarray]:
        coordinates = split_coordinates(
            tree_data.qdt0_idx, qdt0, np.zeros_like(qdt0), qdt2
        )
        a_J_mob = [j.mobilizer.a_J(*qs) for j, qs in zip(tree_data.joints, coordinates)]
        a_J_jnt = [j.frames.X_SM @ a_J for j, a_J in zip(tree_data.joints, a_J_mob)]
        new_kin = [
            JointKinematics(*kin[:-1], a_J) for kin, a_J in zip(joints_kin, a_J_jnt)
        ]
        return new_kin

    @staticmethod
    def traverse(
        tree_data: MultiBodyData,
        joints_kin: List[JointKinematics],
    ):
        forward_traversal = tree_data.forward_traversal
        backward_traversal = tree_data.backward_traversal
        joints_frames = [j.frames for j in tree_data.joints]
        forces_transforms = [
            motion_to_force_transform(j.X_PS) for j in reversed(joints_kin)
        ]

        bodies_acc = node_acceleration_accumulator(joints_kin, forward_traversal)
        bodies_forces = list(map(np.dot, tree_data.bodies_inertias, bodies_acc))
        forces = joints_forces_accumulator(
            bodies_forces, forces_transforms, backward_traversal
        )
        forces = reversed(forces)

        tau = evaluate_tau(joints_frames, joints_kin, forces)
        return tau


# =============================================================================
# Obselete
# =============================================================================
def evaluate_H(
    tree_data: MultiBodyData,
    external_forces: List[List[np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    C_vec: np.ndarray,
) -> np.ndarray:

    boolean_deltas = np.eye(len(qdt0))
    partial_func = partial(
        inverse_dynamics_call,
        tree_data,
        external_forces,
        qdt0,
        qdt1,
    )
    # H_columns = map(partial_func, boolean_deltas)
    # H_columns = map(sub, H_columns, repeat(C_vec, len(qdt0)))
    # H_matrix = np.column_stack(list(H_columns))

    H_columns = [
        (inverse_dynamics_call(tree_data, external_forces, qdt0, qdt1, col).tau - C_vec)
        for col in boolean_deltas
    ]
    H_matrix = np.column_stack(H_columns)

    return H_matrix


# def evaluate_H2(
#     tree_data: MultiBodyData,
#     qdt0: np.ndarray,
# ) -> np.ndarray:

#     ext_forces = [[] for _ in qdt0]
#     boolean_deltas = np.eye(len(qdt0))
#     partial_func = partial(
#         inverse_dynamics_call,
#         tree_data,
#         ext_forces,
#         qdt0,
#         np.zeros_like(qdt0),
#     )
#     H_columns = map(partial_func, boolean_deltas)
#     H_matrix = np.column_stack(list(H_columns))

#     return H_matrix


# def forward_dynamics_call(
#     tree_data: MultiBodyData,
#     external_forces: List[List[np.ndarray]],
#     qdt0: np.ndarray,
#     qdt1: np.ndarray,
#     tau: np.ndarray,
# ) -> np.ndarray:

#     C = evaluate_C(tree_data, external_forces, qdt0, qdt1)
#     H = evaluate_H(tree_data, external_forces, qdt0, qdt1, C)

#     rhs = tau - C
#     qdt2 = np.linalg.solve(H, rhs)
#     return qdt2
