import functools
from typing import Callable

import networkx as nx
import matplotlib.pyplot as plt

from uraeus.rnea.cmbs.topologies.graphs import (
    AbstractTopologyGraph,
    TemplateBasedTopology,
    adj2int,
    MultiBodyData,
)
from uraeus.rnea.cmbs.topologies.bodies import AbstractBody, RigidBody, VirtualBody
from uraeus.rnea.cmbs.topologies.joints import (
    AbstractJoint,
    SphericalJoint,
    RevoluteJoint,
    CylindricalJoint,
    UniversalJoint,
)


def joint_decorator(topology: TemplateBasedTopology, joint_type: type[AbstractJoint]):
    @functools.wraps(topology.add_joint)
    def decorated(name: str, pred: str, succ: str, *args, **kwargs):
        topology.add_joint(joint_type, name, pred, succ, *args, **kwargs)

    return decorated


def body_decorator(
    topology: TemplateBasedTopology, body_type: type[AbstractBody]
) -> Callable[[str], None]:
    @functools.wraps(body_type)
    def decorated(name: str, *args, **kwargs):
        topology.add_body(body_type, name, *args, **kwargs)

    return decorated


class Joints(object):
    def __init__(self, topology: TemplateBasedTopology):
        self.Spherical = joint_decorator(topology, SphericalJoint)
        self.Revolute = joint_decorator(topology, RevoluteJoint)
        self.Universal = joint_decorator(topology, UniversalJoint)
        self.Cylindrical = joint_decorator(topology, CylindricalJoint)


class Bodies(object):
    def __init__(self, topology: TemplateBasedTopology):
        self.Rigid = body_decorator(topology, RigidBody)
        self.Virtual = body_decorator(topology, VirtualBody)


class MultiBodyGraph(object):
    def __init__(self, name: str):
        self.name = name
        self.topology = TemplateBasedTopology(name)
        self._joints = Joints(self.topology)
        self._bodies = Bodies(self.topology)

    @property
    def add_body(self):
        return self._bodies

    @property
    def add_joint(self):
        return self._joints


# def mirror_multibody_graph(graph:MultiBodyGraph, single_nodes:list[str]=[]) -> MultiBodyGraph:
#     single_nodes_set = set(single_nodes)
#     all_nodes_set = set(graph.topology.nodes)
#     symmetric_nodes = all_nodes_set.difference(single_nodes_set)
#     new_graph = TemplateBasedTopology(graph.name)
#     nodes_map = {}
#     for node in symmetric_nodes:
#         body_type = graph.nodes[node]["body_type"]
#         new_graph.add_body(body_type, node, mirror=True)
#         nodes_map

#     for edge in graph.topology.mobility_graph.edges:


if __name__ == "__main__":
    # g = nx.Graph(name="g")
    # g.add_node(AbstractBody(name="body1"), mirror="s")
    # g.add_node(AbstractBody(name="body2"), mirror="s")
    # g.add_node(AbstractBody(name="body3"), mirror="s")

    # print(g.nodes(data=True))
    # nx.draw(g, with_labels=True, font_weight="bold")
    # plt.show()
    # quit()
    import yaml

    from uraeus.rnea.cmbs.topologies.configuration import MultiBodyTopologyConfig

    graph = MultiBodyGraph("dwb")

    graph.add_body.Virtual("chassis")
    graph.add_body.Rigid("uca", mirror=True)
    graph.add_body.Rigid("lca", mirror=True)
    graph.add_body.Rigid("upright", mirror=True)
    graph.add_body.Rigid("tie", mirror=True)
    graph.add_body.Rigid("shock_u", mirror=True)
    graph.add_body.Rigid("shock_l", mirror=True)
    graph.add_body.Rigid("push", mirror=True)
    graph.add_body.Rigid("rocker", mirror=True)

    graph.add_joint.Revolute("rev_uca", "vbs_chassis", "rbl_uca")
    graph.add_joint.Revolute("rev_lca", "vbs_chassis", "rbl_lca")
    graph.add_joint.Spherical("sph_uca_upr", "rbl_uca", "rbl_upright")
    graph.add_joint.Spherical("sph_lca_upr", "rbl_lca", "rbl_upright")
    graph.add_joint.Spherical("sph_tie_upr", "rbl_tie", "rbl_upright")
    graph.add_joint.Universal("uni_tie_chassis", "rbl_tie", "vbs_chassis")
    graph.add_joint.Universal("uni_shock_chassis", "rbl_shock_u", "vbs_chassis")
    graph.add_joint.Universal("uni_shock_rocker", "rbl_shock_l", "rbl_rocker")
    graph.add_joint.Cylindrical("cyl_shock", "rbl_shock_u", "rbl_shock_l")
    graph.add_joint.Spherical("sph_push_uca", "rbl_uca", "rbl_push")
    graph.add_joint.Universal("uni_push_rocker", "rbl_rocker", "rbl_push")
    graph.add_joint.Revolute("rev_chassis_rocker", "rbl_rocker", "vbs_chassis")

    # print(graph.topology.nodes)

    model = MultiBodyData(nodes=graph.topology.nodes, edges=graph.topology.edges)
    config = MultiBodyTopologyConfig(graph.topology)
    config.construct_topology_inputs()

    # # print(model.model_dump_json(indent=4))
    # # stream =
    # with open("model_2.yaml", "w") as f:
    #     yaml.dump(model, stream=f, sort_keys=False, default_flow_style=False)

    # print(yaml.load("model_2.py", yaml.Loader))

    # import ruamel.yaml
    # from pathlib import Path
    # import sys

    # yaml = ruamel.yaml.YAML(typ="unsafe")
    # yaml.sort_base_mapping_type_on_output = False
    # yaml.register_class(MultiBodyData)
    # # yaml.register_class(AbstractBody)
    # # yaml.register_class(AbstractJoint)
    # # yaml.register_class(RevoluteJoint)
    # yaml.dump(RigidBody(name="body"), sys.stdout)
    # with open("model.yaml", "w") as f:
    #     yaml.dump(model, stream=f)

    # model2 = yaml.load(Path("model.yaml"))
    # print(model2)

    # # print(to_yaml_str(model, default_flow_style=False, indent=2))
    # # new_model = yaml.load(txt, yaml.Loader)
    # # print(new_model.nodes)
    quit()

    graph.topology.mobility_graph

    # print(graph.graph.edges(data="joint"))
    # print("")
    # print(graph.nodes)
    # print(adj2int(graph))
    # print("")

    # print(dict(graph.construct())["chassis"])
    g = graph.topology.mobility_graph
    print(nx.cycle_basis(nx.Graph(g)))
    cycles = nx.cycle_basis(nx.Graph(g))
    print((6 * len(g.nodes)) - (g.size(weight="weight") + 6))
    for cycle in cycles:
        print(cycle)
        nc = g.subgraph(cycle).size(weight="weight")
        print(nc)
        print("")

    nx.draw(graph.topology.mobility_graph, with_labels=True, font_weight="bold")
    plt.show()
