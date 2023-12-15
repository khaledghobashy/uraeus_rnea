from typing import Optional, Any

import networkx as nx
from pydantic import BaseModel

from uraeus.rnea.cmbs.topologies.joints import AbstractJoint
from uraeus.rnea.cmbs.topologies.bodies import AbstractBody


class MultiBodyData(BaseModel):
    nodes: dict[str, Any]
    edges: dict[tuple[str, str, str], Any]

    # @classmethod
    # def to_yaml(cls, representor, node):
    #     tag = getattr(cls, "yaml_tag", "!" + cls.__name__)
    #     # attrs = node.model_dump()
    #     attrs = {
    #         "nodes": node.nodes,
    #         "edges": node.edges,
    #     }
    #     return representor.represent_mapping(tag, attrs)

    # @classmethod
    # def from_yaml(cls, constructor, node):
    #     data = constructor.construct_mapping(node, deep=True)
    #     instance = cls(**data)
    #     return instance


class AbstractTopologyGraph(object):
    name: str
    mobility_graph: nx.MultiGraph

    def __init__(self, name: str):
        self.name = name
        self.mobility_graph = nx.MultiGraph(name=name)

    @property
    def nodes(self):
        return self.mobility_graph.nodes

    @property
    def edges(self):
        return self.mobility_graph.edges

    @property
    def n(self):
        g = self.mobility_graph
        n = (6 * (len(g.nodes) - 1)) - g.size(weight="weight")
        return n

    def add_body(self, body_type: type[AbstractBody], name: str):
        graph = self.mobility_graph
        assert name not in graph.nodes, f"Body {name} already exists!"
        body = body_type(name=name)
        # graph.add_node(name, body=body, body_type=body_type)
        graph.add_node(name, object=body)

    def add_joint(
        self, joint_type: type[AbstractJoint], name: str, pred: str, succ: str
    ):
        graph = self.mobility_graph
        assert pred in self.nodes, f"Body {pred} doesn't exist!"
        assert succ in self.nodes, f"Body {succ} doesn't exist!"
        assert (pred, succ, name) not in graph.edges, f"Joint `{name}` already exists!"
        pred_obj = self.nodes[pred]["object"]
        succ_obj = self.nodes[succ]["object"]
        joint = joint_type(name=name, pred=pred_obj, succ=succ_obj)
        graph.add_edge(pred, succ, name, weight=joint_type.nc, object=joint)

    def check(self):
        g = self.mobility_graph
        cycles = nx.cycle_basis(nx.Graph(g))
        for cycle in cycles:
            print(cycle)
            nc = g.subgraph(cycle).size(weight="weight")
            g.size()
            print(nc)
            print("")


class TemplateBasedTopology(AbstractTopologyGraph):
    def add_body(
        self, body_type: type[AbstractBody], name: str, mirror: Optional[bool] = False
    ):
        graph = self.mobility_graph
        if mirror:
            node_r = f"{body_type.prefix}r_{name}"
            node_l = f"{body_type.prefix}l_{name}"
            super().add_body(body_type, node_r)
            super().add_body(body_type, node_l)
            graph.nodes[node_r].update({"mirror": node_l, "align": "r"})
            graph.nodes[node_l].update({"mirror": node_r, "align": "l"})
        else:
            node_s = f"{body_type.prefix}s_{name}"
            super().add_body(body_type, node_s)
            graph.nodes[node_s].update({"mirror": node_s, "align": "s"})

    def add_joint(
        self, joint_type: type[AbstractJoint], name: str, pred: str, succ: str
    ):
        graph = self.mobility_graph

        pred_algn = graph.nodes[pred]["align"]
        succ_algn = graph.nodes[succ]["align"]

        mirrored = (pred_algn, succ_algn) != ("s", "s")

        if mirrored:
            joint_name_r = f"mcr_{joint_type.prefix}_{name}"
            joint_name_l = f"mcl_{joint_type.prefix}_{name}"

            pred_r = pred if pred_algn == "r" else graph.nodes[pred]["mirror"]
            succ_r = succ if succ_algn == "r" else graph.nodes[succ]["mirror"]
            pred_l = graph.nodes[pred_r]["mirror"]
            succ_l = graph.nodes[succ_r]["mirror"]

            super().add_joint(joint_type, joint_name_r, pred_r, succ_r)
            super().add_joint(joint_type, joint_name_l, pred_l, succ_l)

            joint_edge_r = (pred_r, succ_r, joint_name_r)
            joint_edge_l = (pred_l, succ_l, joint_name_l)
            graph.edges[joint_edge_r].update({"mirror": joint_name_l, "align": "r"})
            graph.edges[joint_edge_l].update({"mirror": joint_name_r, "align": "l"})

        else:
            joint_name_s = f"mcs_{joint_type.prefix}_{name}"
            super().add_joint(joint_type, joint_name_s, pred, succ)
            joint_edge_s = (pred, succ, joint_name_s)
            graph.edges[joint_edge_s].update({"mirror": joint_name_s, "align": "s"})


def adj2int(graph: AbstractTopologyGraph):
    d = dict(zip(graph.nodes, range(len(graph.nodes))))
    edges = tuple((d[p], d[s]) for p, s, _ in graph.edges)
    return edges
