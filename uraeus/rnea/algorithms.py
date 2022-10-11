from typing import Iterable, List, Dict, NamedTuple, Tuple

import numpy as np

from uraeus.rnea.joints import (
    AbstractJoint,
    JointInstance,
    FunctionalJoint,
    construct_functional_joint,
)
from uraeus.rnea.tree_traversals import (
    base_to_tip,
    tip_to_base,
    extract_generalized_forces,
)


def split_joints_coordinates(sections: np.ndarray, q: np.ndarray):
    return np.split(q, sections[:-1])


class MultiBodyData(NamedTuple):

    joints: List[FunctionalJoint]
    bodies_inertias: List[np.ndarray]
    forward_traversal: List[Tuple[int, int, int]]
    backward_traversal: List[Tuple[int, List[int]]]
    qdt0_indicies: np.ndarray


def inverse_dynamics_call(
    tree_data: MultiBodyData,
    external_forces: List[List[np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    qdt2: np.ndarray,
) -> np.ndarray:

    func_joints = tree_data.joints
    forward_traversal = tree_data.forward_traversal
    backward_traversal = tree_data.backward_traversal

    qdt0_indicies = tree_data.qdt0_indicies
    qdt0_sections = split_joints_coordinates(qdt0_indicies, qdt0)
    qdt1_sections = split_joints_coordinates(qdt0_indicies, qdt1)
    qdt2_sections = split_joints_coordinates(qdt0_indicies, qdt2)
    joints_coordinates = list(zip(qdt0_sections, qdt1_sections, qdt2_sections))

    bodies_kin, joints_kin = base_to_tip(
        func_joints, joints_coordinates, forward_traversal
    )

    joints_forces = tip_to_base(
        joints=func_joints,
        joints_kinematics=joints_kin,
        dependency_order=backward_traversal,
        bodies_kinematics=bodies_kin,
        bodies_inertias=tree_data.bodies_inertias,
        external_forces=external_forces,
    )
    tau = extract_generalized_forces(joints_forces)

    return tau


def evaluate_C(
    tree_data: MultiBodyData,
    external_forces: List[List[np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
) -> np.ndarray:

    qdt2 = np.zeros_like(qdt1)
    return inverse_dynamics_call(tree_data, external_forces, qdt0, qdt1, qdt2)


def evaluate_H(
    tree_data: MultiBodyData,
    external_forces: List[List[np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    C_vec: np.ndarray,
) -> np.ndarray:

    boolean_deltas = np.eye(len(qdt0))
    H_columns = [
        (inverse_dynamics_call(tree_data, external_forces, qdt0, qdt1, col) - C_vec)
        for col in boolean_deltas
    ]

    H_matrix = np.column_stack(H_columns)
    return H_columns


def forward_dynamics_call(
    tree_data: MultiBodyData,
    external_forces: List[List[np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:

    C = evaluate_C(tree_data, external_forces, qdt0, qdt1)
    H = evaluate_H(tree_data, external_forces, qdt0, qdt1, C)

    rhs = tau - C

    qdt2 = np.linalg.solve(H, rhs)

    return qdt2
