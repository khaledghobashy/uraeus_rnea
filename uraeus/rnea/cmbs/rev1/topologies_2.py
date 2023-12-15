import functools
from typing import ClassVar

import networkx as nx
from pydantic import BaseModel
import matplotlib.pyplot as plt


class AbstractJoint(BaseModel):
    nc: ClassVar[int] = None

    name: str
    body_i: str
    body_j: str


def decorator(self, joint_type: type[AbstractJoint]):
    @functools.wraps(joint_type)
    def decorated(name: str, predecessor: str, successor: str):
        self.add_edge(joint_type, name, predecessor, successor)

    return decorated


class SphericalJoint(AbstractJoint):
    nc = 3


class UniversalJoint(AbstractJoint):
    nc = 4


class CylindricalJoint(AbstractJoint):
    nc = 4


class RevoluteJoint(AbstractJoint):
    """_summary_

    Parameters
    ----------
    AbstractJoint : _type_
        _description_
    """

    nc = 5


class AbstractMultiBodyGraph(object):
    name: str

    def __init__(self, name: str):
        self.name = name
        self.graph = nx.MultiGraph(name=name)

    def add_edge(
        self,
        joint_type: type[AbstractJoint],
        edge_name: str,
        predecessor: str,
        successor: str,
    ):
        """_summary_

        Parameters
        ----------
        joint_type : type[AbstractJoint]
            _description_
        edge_name : str
            _description_
        predecessor : str
            _description_
        successor : str
            _description_
        """
        assert (predecessor, successor, edge_name) not in self.graph.edges
        joint = joint_type(name=edge_name, body_i=predecessor, body_j=successor)
        self.graph.add_edge(
            predecessor, successor, edge_name, weight=joint_type.nc, joint=joint
        )

    def construct(self):
        return self.graph.adjacency()


class Joints(object):
    def __init__(self, topology: AbstractMultiBodyGraph):
        self.Spherical = decorator(topology, SphericalJoint)
        self.Revolute = decorator(topology, RevoluteJoint)
        self.Universal = decorator(topology, UniversalJoint)
        self.Cylindrical = decorator(topology, CylindricalJoint)


class MultiBodyGraph(AbstractMultiBodyGraph):
    def __init__(self, name: str):
        super().__init__(name)
        self.joints = Joints(self)

    @property
    def add_joint(self):
        return self.joints


def adj2int(graph: AbstractMultiBodyGraph):
    d = dict(zip(graph.graph.nodes, range(len(graph.graph.nodes))))
    edges = tuple((d[p], d[s]) for p, s, _ in graph.graph.edges)
    return edges


if __name__ == "__main__":
    graph = MultiBodyGraph("dwb")

    graph.add_joint.Revolute("rev_uca", "chassis", "uca")
    graph.add_joint.Revolute("rev_lca", "chassis", "lca")
    graph.add_joint.Spherical("sph_uca_upr", "uca", "upright")
    graph.add_joint.Spherical("sph_lca_upr", "lca", "upright")
    graph.add_joint.Spherical("sph_tie_upr", "tie", "upright")
    graph.add_joint.Universal("uni_tie_chassis", "tie", "chassis")
    graph.add_joint.Universal("uni_shock_chassis", "shock_u", "chassis")
    graph.add_joint.Universal("uni_shock_rocker", "shock_l", "rocker")
    graph.add_joint.Cylindrical("cyl_shock", "shock_u", "shock_l")
    graph.add_joint.Spherical("sph_push_uca", "uca", "push")
    graph.add_joint.Universal("uni_push_rocker", "rocker", "push")
    graph.add_joint.Revolute("rev_chassis_rocker", "rocker", "chassis")

    # print(graph.graph.edges(data="joint"))
    # print("")
    print(graph.graph.nodes)
    print(adj2int(graph))
    print("")

    print(dict(graph.construct())["chassis"])
    print(nx.cycle_basis(nx.Graph(graph.graph)))
    cycles = nx.cycle_basis(nx.Graph(graph.graph))
    print((6 * len(graph.graph.nodes)) - (graph.graph.size(weight="weight") + 6))
    for cycle in cycles:
        print(cycle)
        nc = graph.graph.subgraph(cycle).size(weight="weight")
        print(nc)
        print("")

    nx.draw(graph.graph, with_labels=True, font_weight="bold")
    plt.show()
