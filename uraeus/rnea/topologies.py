from functools import reduce
from typing import Callable, Any, NamedTuple
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx

from uraeus.rnea.spatial_algebra import SpatialInertia
from uraeus.rnea.bodies import RigidBody, RigidBodyData
from uraeus.rnea.graphs import Graph, Tree, extract_graph_data, GraphConnectivity

from uraeus.rnea.joints import (
    AbstractJoint,
    JointConfigInputs,
    JointInstance,
    construct_functional_joint,
    construct_joint_instance,
    initialize_joint,
    FunctionalJoint,
)


class MultiBodyData(NamedTuple):
    n: int
    joints: tuple[FunctionalJoint]
    bodies_inertias: list[np.ndarray]
    graph_data: GraphConnectivity
    sparsity_pattern: tuple[tuple[slice, slice, int], ...]
    qdt0_idx: tuple[int, ...]
    qdt1_idx: tuple[int, ...]
    qdt0_names: NamedTuple
    spatial_inertias: tuple[SpatialInertia, ...] = None

    def __hash__(self):
        return hash(self.__class__.__name__)


class HybridDynamicsData(NamedTuple):
    tree_data: MultiBodyData
    permutation_matrix: np.ndarray
    n_fd: int

    def __hash__(self):
        return hash(self.__class__.__name__)


class MultiBodyGraph(object):
    name: str
    graph: Graph
    bodies: dict[str, RigidBody]
    joints: dict[str, JointInstance]

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
    bodies: dict[str, RigidBody]
    joints: dict[str, JointInstance]

    def __init__(self, name: str):
        self.name = name
        self.tree = Tree(self.name, root="ground")

        self.bodies = {"ground": RigidBody("ground", RigidBodyData())}
        self.joints = {}

    @property
    def dof(self) -> int:
        return sum([e[-1] for e in self.tree.edges(data="n")])

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

        self.tree.add_edge(predecessor, successor, n=joint_type.nj, name=joint_name)

        pred_body = self.get_body(predecessor)
        succ_body = RigidBody(successor, succ_data)

        joint_frames = initialize_joint(
            joint_data.pos,
            joint_data.z_axis,
            joint_data.x_axis,
            pred_body.kinematics.p_GB,
            succ_body.kinematics.p_GB,
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

    def construct_bodies_coordinates_names(self) -> list[str]:
        coordinates = ["x", "y", "z", "phi", "theta", "psi"]
        namer = lambda l, name: l + [f"{name}.{c}" for c in coordinates]
        bodies_coordinates = reduce(namer, self.bodies.keys(), [])
        return bodies_coordinates

    def check_if_joint_exists(self, joint_name: str) -> bool:
        return joint_name in self.joints


def construct_permutation_matrix(dof: int, id_indices: list[int]) -> np.ndarray:
    permutation = [i for i in range(dof) if i not in id_indices]
    permutation += id_indices
    mat = np.zeros((dof, dof))
    mat[np.arange(0, dof), permutation] = 1
    return mat


def construct_hybrid_dynamics_data(
    tree_data: MultiBodyData, id_coordinates: np.ndarray
) -> HybridDynamicsData:
    n_dof = sum([j.nj for j in tree_data.joints])
    n_id = len(id_coordinates)
    n_fd = n_dof - n_id
    Q = construct_permutation_matrix(n_dof, id_coordinates)

    data = HybridDynamicsData(
        tree_data=tree_data,
        permutation_matrix=Q,
        n_fd=n_fd,
    )
    return data


def create_namedtuple_inplace(name: str, field_dict: dict[str, Any]) -> namedtuple:
    return namedtuple(name, field_dict.keys())(*field_dict.values())


def construct_multibody_data(topology: MultiBodyTree) -> MultiBodyData:
    func_joints = tuple(map(construct_functional_joint, topology.joints.values()))
    bodies_inertias = [b.I for b in topology.bodies.values()]
    graph_data = extract_graph_data(topology.tree)
    qdt0_idx = [0] + list(np.cumsum([j.nj for j in func_joints]))
    qdt1_idx = qdt0_idx  # Equal each other for now. Later could be different.

    numbered_graph = nx.convert_node_labels_to_integers(topology.tree.nxgraph)
    edges_with_n = list(numbered_graph.edges(data="n", default=1))

    sparsity_pattern = construct_sparsity_pattern(edges_with_n)

    states_tuple = namedtuple("States", topology.joints.keys())
    qdt0_names = states_tuple(
        **{
            name: create_namedtuple_inplace(
                name, dict(zip(joint.joint_type.coordinates_names, indices))
            )
            for (name, joint), indices in zip(
                topology.joints.items(),
                np.split(np.arange(0, topology.dof), qdt0_idx)[1:],
            )
        },
    )

    spatial_inertias = tuple(
        SpatialInertia(b.body_data.mass, b.body_data.inertia_tensor, np.zeros((3,)))
        for b in topology.bodies.values()
    )

    data = MultiBodyData(
        n=topology.dof,
        joints=func_joints,
        bodies_inertias=bodies_inertias,
        graph_data=graph_data,
        sparsity_pattern=sparsity_pattern,
        qdt0_idx=tuple(qdt0_idx),
        qdt1_idx=tuple(qdt1_idx),
        qdt0_names=qdt0_names,
        spatial_inertias=spatial_inertias,
    )

    return data


def construct_sparsity_pattern(edges_list: list[tuple[int, int, int]]):
    blocks = []
    for p_index, s_index, block_size in reversed(edges_list):
        moving_p = p_index
        while moving_p != 0:
            u_index = slice(s_index - 1, s_index - 1 + block_size)
            v_index = slice(moving_p - 1, moving_p - 1 + block_size)
            blocks.append((u_index, v_index, block_size))
            moving_p, _, block_size = edges_list[moving_p - 1]

    return tuple(blocks)
