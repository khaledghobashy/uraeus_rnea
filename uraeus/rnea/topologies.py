from functools import reduce
from typing import Dict, List, NamedTuple, Tuple

from uraeus.rnea.bodies import RigidBody, RigidBodyData
from uraeus.rnea.graphs import Graph, Tree, contstruct_traversal_orders
from uraeus.rnea.joints import (
    AbstractJoint,
    FunctionalJoint,
    JointConfigInputs,
    JointInstance,
    construct_functional_joint,
    construct_joint_instance,
    initialize_joint,
)

import numpy as np


class MultiBodyGraph(object):
    name: str
    graph: Graph
    bodies: Dict[str, RigidBody]
    joints: Dict[str, JointInstance]

    def __init__(self, name: str):
        self.name = name
        self.graph = Graph(self.name)

        self.bodies = {}
        self.joints = {}

    @property
    def nc(self):
        return sum(j[0].nc for j in self.joints.values())

    @property
    def dof(self):
        return (6 * (len(self.bodies) - 1)) - self.nc

    def add_body(self, body_name: str, body_data):
        assert body_name not in self.bodies
        self.bodies[body_name] = body_data

    def add_joint(
        self,
        joint_name: str,
        body_i: RigidBodyData,
        body_j: RigidBodyData,
        joint_type,
        joint_config,
    ):
        assert body_i in self.bodies
        assert body_j in self.bodies
        assert joint_name not in self.joints
        self.graph.add_edge(body_i, body_j)
        self.joints[joint_name] = (
            joint_type,
            joint_config,
            self.bodies[body_i],
            self.bodies[body_j],
        )


class MultiBodyTree(object):
    name: str
    tree: Tree
    bodies: Dict[str, RigidBody]
    joints: Dict[str, JointInstance]

    def __init__(self, name: str):
        self.name = name
        self.tree = Tree(self.name, root="ground")

        self.bodies = {"ground": RigidBody("ground", RigidBodyData())}
        self.joints = {}

    @property
    def dof(self) -> int:
        return sum(j.joint_type.nj for j in self.joints.values())

    def add_joint(
        self,
        joint_name: str,
        predecessor: str,
        successor: str,
        succ_data: RigidBodyData,
        joint_type: AbstractJoint,
        joint_data: JointConfigInputs,
    ) -> None:
        if self.check_if_joint_exists(joint_name):
            raise ValueError(f"Joint '{joint_name}' already exists!")

        self.tree.add_edge(predecessor, successor)

        pred_body = self.get_body(predecessor)
        succ_body = RigidBody(successor, succ_data)

        joint_frames = initialize_joint(
            joint_data.pos,
            joint_data.z_axis,
            joint_data.x_axis,
            pred_body.kinematics.X_BG,
            succ_body.kinematics.X_BG,
        )

        joint = construct_joint_instance(
            joint_type=joint_type,
            name=joint_name,
            predecessor=pred_body,
            successor=succ_body,
            joint_frames=joint_frames,
        )

        self.bodies[successor] = succ_body
        self.joints[joint_name] = joint

        return

    def get_body(self, name: str) -> RigidBody:
        return self.bodies[name]

    def construct_bodies_coordinates_names(self) -> List[str]:
        coordinates = ["phi", "theta", "psi", "x", "y", "z"]
        namer = lambda l, name: l + [f"{name}.{c}" for c in coordinates]
        bodies_coordinates = reduce(namer, self.bodies.keys(), [])
        return bodies_coordinates

    def check_if_joint_exists(self, joint_name: str) -> bool:
        return joint_name in self.joints


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


def construct_permutation_matrix(dof: int, id_indices: List[int]) -> np.ndarray:
    permutation = [i for i in range(dof) if i not in id_indices]
    permutation += id_indices
    mat = np.zeros((dof, dof))
    mat[np.arange(0, dof), permutation] = 1
    return mat


def construct_hybriddynamics_data(
    tree_data: MultiBodyData, id_coordintaes: np.ndarray
) -> HybridDynamicsData:
    n_dof = sum([j.nj for j in tree_data.joints])
    n_id = len(id_coordintaes)
    n_fd = n_dof - n_id
    Q = construct_permutation_matrix(n_dof, id_coordintaes)

    data = HybridDynamicsData(
        tree_data=tree_data,
        permutation_matrix=Q,
        n_fd=n_fd,
    )
    return data


def construct_multibodydata(topology: MultiBodyTree) -> MultiBodyData:
    func_joints = tuple(map(construct_functional_joint, topology.joints.values()))
    bodies_inertias = [b.I for b in topology.bodies.values()]
    forward_traversal, backward_traversal = contstruct_traversal_orders(topology.tree)
    qdt0_idx = [0] + list(np.cumsum([j.nj for j in func_joints]))
    qdt1_idx = qdt0_idx  # Equal each other for now. Later could be different.

    data = MultiBodyData(
        joints=func_joints,
        bodies_inertias=bodies_inertias,
        forward_traversal=forward_traversal,
        backward_traversal=backward_traversal,
        qdt0_idx=tuple(qdt0_idx),
        qdt1_idx=tuple(qdt1_idx),
    )

    return data


# =============================================================================
# Obselete Code
# =============================================================================

# def construct_out_joints_map(
#     model_tree: MultiBodyTree,
# ) -> Dict[str, List[JointInstance]]:

#     out_joints = {b: [] for b in model_tree.bodies}
#     for j in model_tree.joints.values():
#         out_joints[j.joint_data.predecessor.name].append(j)
#     return out_joints
