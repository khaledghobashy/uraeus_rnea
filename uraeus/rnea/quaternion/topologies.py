from functools import reduce
from typing import Callable

import numpy as np

from uraeus.rnea.quaternion.bodies import RigidBody, RigidBodyData, BodyKinematics
from uraeus.rnea.quaternion.tree_traversals import base_to_tip, SystemForces
from uraeus.rnea.quaternion.graphs import Graph, Tree, construct_traversal_orders

from uraeus.rnea.quaternion.joints import (
    AbstractJoint,
    JointConfigInputs,
    JointInstance,
    construct_functional_joint,
    construct_joint_instance,
    initialize_joint,
    JointKinematics,
)

from uraeus.rnea.quaternion.algorithms import (
    split_coordinates,
    MultiBodyData,
    HybridDynamicsData,
    IDCallRes,
    inverse_dynamics_call,
    forward_dynamics_call,
)

Forcesdict = dict[str, dict[str, dict[str, np.ndarray]]]


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
        coordinates = ["phi", "theta", "psi", "x", "y", "z"]
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
    forward_traversal, backward_traversal = construct_traversal_orders(topology.tree)
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


class Model(object):
    topology: MultiBodyTree
    forces_map: Forcesdict
    tree_data: MultiBodyData

    def __init__(self, topology: MultiBodyTree):
        self.topology = topology
        gravity = np.array([0, 0, -9.81, 0, 0, 0])
        self.forces_map = {
            b.name: {"global": {"gravity": b.I @ gravity}, "local": dict()}
            for b in self.topology.bodies.values()
        }

        self.tree_data = construct_multibodydata(topology)
        self.bodies_idx = {b: i for i, b in enumerate(self.topology.tree.nodes)}

    def get_body_kinematics(
        self, name: str, bodies_kinematics: list[BodyKinematics]
    ) -> BodyKinematics:
        return bodies_kinematics[self.bodies_idx[name]]

    def forward_kinematics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> tuple[BodyKinematics, JointKinematics]:
        coordinates = split_coordinates(self.tree_data.qdt0_idx, qdt0, qdt1, qdt2)

        bodies_kinematics, joints_kinematics = base_to_tip(
            self.tree_data.joints, coordinates, self.tree_data.forward_traversal
        )

        return bodies_kinematics, joints_kinematics

    def inverse_dynamics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> IDCallRes:
        forces = construct_system_forces_from_dict(self.forces_map)
        res = inverse_dynamics_call(self.tree_data, forces, qdt0, qdt1, qdt2)
        return res

    def forward_dynamics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, tau: np.ndarray
    ):
        forces = construct_system_forces_from_dict(self.forces_map)
        qdt2 = forward_dynamics_call(self.tree_data, forces, qdt0, qdt1, tau)
        return qdt2

    def ssode(
        self,
        t: float,
        ydt0: np.ndarray,
        forces_func: Callable,
    ):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        tau, self.forces_map = forces_func(self, qdt0, qdt1, 0 * qdt1, t)
        qdt2 = self.forward_dynamics_pass(qdt0, qdt1, tau)
        return np.hstack([qdt1, qdt2])


def _convert_body_forces_dict_to_list(forces_dict: dict[str, dict[str, np.ndarray]]):
    forces = list(forces_dict.values()) if len(forces_dict) > 0 else []
    return forces


def construct_system_forces_from_dict(external_forces: Forcesdict) -> SystemForces:

    system_forces = [
        (
            _convert_body_forces_dict_to_list(forces_dict["global"]),
            _convert_body_forces_dict_to_list(forces_dict["local"]),
        )
        for forces_dict in external_forces.values()
    ]
    # print(system_forces)
    return system_forces
