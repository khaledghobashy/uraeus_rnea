from typing import List, Dict, NamedTuple

import numpy as np

from multibody.joints import AbstractJoint
from multibody.tree_traversals import (
    base_to_tip,
    tip_to_base,
    extract_generalized_forces,
)


def inverse_dynamics_call(
    joints: List[AbstractJoint],
    adj_list: Dict[str, List[AbstractJoint]],
    forces_map: Dict[str, Dict[str, np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    qdt2: np.ndarray,
) -> np.ndarray:

    boides_kin, joints_kin = base_to_tip(joints, qdt0, qdt1, qdt2)

    joints_forces = tip_to_base(joints, joints_kin, adj_list, forces_map)
    tau = extract_generalized_forces(joints_forces)

    return tau


def evaluate_C(
    joints: List[AbstractJoint],
    adj_list: Dict[str, List[AbstractJoint]],
    forces_map: Dict[str, Dict[str, np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
) -> np.ndarray:

    qdt2 = np.zeros_like(qdt1)
    return inverse_dynamics_call(joints, adj_list, forces_map, qdt0, qdt1, qdt2)


def evaluate_H(
    joints: List[AbstractJoint],
    adj_list: Dict[str, List[AbstractJoint]],
    forces_map: Dict[str, Dict[str, np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    C_vec: np.ndarray,
) -> np.ndarray:

    boolean_deltas = np.eye(len(qdt0))
    H_columns = [
        (inverse_dynamics_call(joints, adj_list, forces_map, qdt0, qdt1, col) - C_vec)
        for col in boolean_deltas
    ]

    H_matrix = np.column_stack(H_columns)
    return H_columns


def forward_dynamics_call(
    joints: List[AbstractJoint],
    adj_list: Dict[str, List[AbstractJoint]],
    forces_map: Dict[str, Dict[str, np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:

    C = evaluate_C(joints, adj_list, forces_map, qdt0, qdt1)
    H = evaluate_H(joints, adj_list, forces_map, qdt0, qdt1, C)

    rhs = tau - C

    qdt2 = np.linalg.solve(H, rhs)

    return qdt2
