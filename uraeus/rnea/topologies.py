from typing import Dict, List, NamedTuple, Optional
from collections import defaultdict
from functools import reduce

from uraeus.rnea.bodies import RigidBody, RigidBodyData
from uraeus.rnea.joints import (
    AbstractJoint,
    JointData,
    JointConfigInputs,
    JointInstance,
    construct_functional_joint,
    construct_joint_instance,
    initialize_joint,
)


class Graph(object):
    def __init__(self, name: str):

        self.name = name
        self.adj_list = defaultdict(list)

    def add_edge(self, predecessor: str, successor: str) -> None:
        self.adj_list[predecessor].append(successor)
        self.adj_list[successor] = []


class Tree(object):
    def __init__(self, name: str, root: Optional[str] = "root"):

        self.graph = Graph(name)
        self.root = root
        self.graph.adj_list[self.root] = []

    @property
    def adj_list(self):
        return self.graph.adj_list

    def add_edge(self, predecessor: str, successor: str) -> None:
        if not self.check_if_node_exists(predecessor):
            raise ValueError(f"Node '{predecessor}' is not in the tree!")

        if self.check_if_node_exists(successor):
            raise ValueError(f"Cannot add node '{successor}', as it already exists!")

        self.graph.add_edge(predecessor, successor)

    def check_if_node_exists(self, node: str) -> bool:
        return node in self.adj_list


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


def construct_out_joints_map(
    model_tree: MultiBodyTree,
) -> Dict[str, List[JointInstance]]:

    out_joints = {b: [] for b in model_tree.bodies}
    for j in model_tree.joints.values():
        out_joints[j.joint_data.predecessor.name].append(j)
    return out_joints


def contstruct_traversal_order(tree: MultiBodyTree):
    bodies_indicies = {b: i for i, b in enumerate(tree.bodies)}
    joints = [
        (
            j,
            bodies_indicies[j.joint_data.predecessor.name],
            bodies_indicies[j.joint_data.successor.name],
        )
        for j in tree.joints.values()
    ]
    joints = list(map(construct_functional_joint, joints))
    return joints
