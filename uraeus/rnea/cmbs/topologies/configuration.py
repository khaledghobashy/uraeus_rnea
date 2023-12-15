import itertools
import functools
from typing import Callable, Union, Any, Iterator

from pydantic import BaseModel, dataclasses

import networkx as nx
import matplotlib.pyplot as plt

from uraeus.rnea.cmbs.topologies.graphs import AbstractTopologyGraph

NodeView = Iterator[tuple[str, Any]]


class RelationalGraph(object):
    name: str
    graph: nx.DiGraph

    def __init__(self, name: str):
        self.name = name
        self.graph = nx.DiGraph(name=name)

    @property
    def input_nodes(self) -> NodeView:
        return self._get_input_nodes()

    @property
    def intermediate_nodes(self) -> NodeView:
        return self._get_intermediate_nodes()

    @property
    def output_nodes(self) -> NodeView:
        return self._get_output_nodes()

    def add_node(self, name: str, **kwargs):
        self.graph.add_node(name, **kwargs)

    def add_relation(self, variable: str, inputs: list[str]):
        inputs_attribute = [self._extract_name_and_attr(n) for n in inputs]
        inputs_names, nested_attributes = zip(*inputs_attribute)
        passed_attrs = [
            {"passed_attr": i if i != "" else None} for i in nested_attributes
        ]
        self._update_in_edges(variable, inputs_names, passed_attrs)

    def draw_node_dependencies(self, node: str):
        graph = self.graph
        edges = self._get_node_predecessors(node)
        sub_graph = graph.edge_subgraph(edges)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx(sub_graph, with_labels=True)
        plt.show()

    def draw_graph(self):
        plt.figure(figsize=(10, 6))
        nx.draw_circular(self.graph, with_labels=True)
        plt.show()

    def draw_symmetric_side_graph(self):
        plt.figure(figsize=(10, 6))
        nodes = list(
            map(
                lambda t: t[0],
                filter(lambda d: d[1] != "r", self.graph.nodes(data="align")),
            )
        )
        subgraph = self.graph.subgraph(nodes)
        nx.draw_circular(subgraph, with_labels=True)
        plt.show()

    def _update_in_edges(
        self, node: str, nbunch: list[str], edges_attrs: dict[str, str]
    ):
        graph = self.graph
        old_edges = list(graph.in_edges(node))
        graph.remove_edges_from(old_edges)
        new_edges = [(i, node, d) for i, d in zip(nbunch, edges_attrs)]
        graph.add_edges_from(new_edges)

    def _get_nodes_attribute(self, nodes: list[str], attribute: str) -> list[Any]:
        graph = self.graph
        sub_graph = graph.subgraph(nodes)
        attr_list = list(nx.get_node_attributes(sub_graph, attribute).values())
        return attr_list

    def _extract_name_and_attr(self, variable: str) -> tuple[str, str]:
        graph = self.graph
        splitted_attributes = variable.split(".")
        node_name = splitted_attributes[0]
        attribute_string = ".".join(splitted_attributes[1:])
        if node_name not in graph.nodes:
            raise ValueError(f"Node {node_name} is not is the graph.")
        return node_name, attribute_string

    def _get_node_predecessors(self, node: str) -> Iterator[tuple[str, str, str]]:
        graph = self.graph
        edges = reversed([e[:-1] for e in nx.edge_bfs(graph, node, "reverse")])
        return edges

    def _get_input_nodes(self) -> list[str]:
        graph = self.graph
        nodes = [i for i, d in graph.in_degree() if d == 0]
        return nodes

    def _get_output_nodes(self) -> list[str]:
        graph = self.graph
        condition = lambda i, d: d == 0 and graph.in_degree(i) != 0
        nodes = [i for i, d in graph.out_degree() if condition(i, d)]
        return nodes

    def _get_intermediate_nodes(self) -> list[str]:
        graph = self.graph
        input_nodes = self._get_input_nodes()
        output_nodes = self._get_output_nodes()
        edges = itertools.chain(*[self._get_node_predecessors(n) for n in output_nodes])
        mid_nodes = []
        for e in edges:
            node = e[0]
            if node not in mid_nodes and node not in input_nodes:
                mid_nodes.append(node)
        return mid_nodes


################################################################################
class ConfigurationGraph(RelationalGraph):
    name: str
    multibody_graph: AbstractTopologyGraph

    def __init__(self, name: str, multibody_graph: AbstractTopologyGraph):
        super().__init__(name)
        self.multibody_graph = multibody_graph
        # self.assemble_base_layer()
        # self.geometries_map = {}

    @property
    def arguments_symbols(self) -> list[str]:
        nodes = self.graph.nodes(data="lhs_value")
        return [n[-1] for n in nodes]

    @property
    def primary_arguments(self):
        return self.multibody_graph.arguments_symbols

    @property
    def primary_nodes(self) -> list[str]:
        nodes = self.graph.nodes
        primary_nodes = [n for n in nodes if nodes[n]["primary"]]
        return primary_nodes

    def add_node(self, name: str, node_type, prefix: str, mirrored: bool = False):
        if mirrored:
            node_r = f"{prefix}r_{name}"
            node_l = f"{prefix}l_{name}"

            node_r_attr_dict = self._create_node_dict(
                name=node_r, node_type=node_type, mirror=node_l, align="r"
            )
            node_l_attr_dict = self._create_node_dict(
                name=node_l, node_type=node_type, mirror=node_r, align="l"
            )
            super().add_node(node_r, **node_r_attr_dict)
            super().add_node(node_l, **node_l_attr_dict)

            # if not issubclass(node_type, Geometry):
            #     self.add_relation(Mirrored, node2, (node1,))
            node_name = node_r
        else:
            node_s = f"{prefix}s_{name}"
            node_s_attr_dict = self._create_node_dict(
                name=node_s, node_type=node_type, mirror=node_s, align="s"
            )
            super().add_node(node_s, **node_s_attr_dict)
            node_name = node_s

        return node_name

    def add_relation(
        self, relation, node: str, input_nodes: list[str], mirrored: bool = False
    ):
        self._assert_nodes_in_graph([node] + input_nodes)
        if mirrored:
            node1 = node
            args1 = input_nodes
            super().add_relation(node1, args1)
            self._update_node_rhs(node1, relation)

            node2 = self.graph.nodes[node1]["mirror"]
            args2 = [self.graph.nodes[i]["mirror"] for i in args1]
            super().add_relation(node2, args2)
            self._update_node_rhs(node2, relation)

        else:
            super().add_relation(node, input_nodes)
            self._update_node_rhs(node, relation)

    def _update_node_rhs(self, node: str, function: Any):
        self.graph.nodes[node]["function"] = function

    def _create_node_dict(
        self, name: str, node_type: Any, mirror: str, align: str
    ) -> dict[str, Any]:
        node_object = name
        function = None
        attributes_dict = {
            "lhs_value": node_object,
            "function": function,
            "mirror": mirror,
            "align": align,
            "equality": None,
            "primary": False,
        }
        return attributes_dict

    def _assert_nodes_in_graph(self, nodes: list[str]):
        if not isinstance(nodes, list):
            nodes = [nodes]
        for node in nodes:
            try:
                self.graph.nodes[node.split(".")[0]]
            except KeyError:
                raise ValueError(f"Node '{node}' is not in graph!")

    # def assemble_base_layer(self):
    #     edges_data = list(zip(*self.topology.edges(data=True)))
    #     edges_arguments = self._extract_primary_arguments(edges_data[-1])
    #     self._add_primary_nodes(edges_arguments)

    #     nodes_data = list(zip(*self.topology.nodes(data=True)))
    #     nodes_arguments = self._extract_primary_arguments(nodes_data[-1])
    #     self._add_primary_nodes(nodes_arguments)

    #     self.bodies = {n: self.topology.nodes[n] for n in self.topology.bodies}

    #     nodes = self.graph.nodes
    #     self.primary_equalities = dict(nodes(data="equality"))

    # def _add_primary_nodes(self, arguments_lists: list[str]):
    #     single_args, right_args, left_args = arguments_lists
    #     for arg in single_args:
    #         node = str(arg)
    #         self._add_primary_node(arg, mirr=node, align="s")
    #     for arg1, arg2 in zip(right_args, left_args):
    #         node1 = str(arg1)
    #         node2 = str(arg2)
    #         self._add_primary_node(arg1, mirr=node2, align="r")
    #         self._add_primary_node(arg2, mirr=node1, align="l")
    #         relation = self._get_primary_mirrored_relation(arg1)
    #         self.add_relation(relation, node2, (node1,))

    # def _add_primary_node(self, node_object, mirr="", align="s"):
    #     name = str(node_object)
    #     function = None
    #     equality = self._get_initial_equality(node_object)
    #     attributes_dict = {
    #         "lhs_value": node_object,
    #         "rhs_function": function,
    #         "mirr": mirr,
    #         "align": align,
    #         "primary": True,
    #         "equality": equality,
    #     }
    #     self.graph.add_node(name, **attributes_dict)

    # def _assign_geometry_to_body(self, body, geo, eval_inertia=True):
    #     b = self.bodies[body]["obj"]
    #     R, P, m, J = [str(getattr(b, i)) for i in "R,P,m,Jbar".split(",")]
    #     self.geometries_map[geo] = body
    #     if eval_inertia:
    #         self.add_relation(CR.Equal_to, R, ("%s.R" % geo,))
    #         self.add_relation(CR.Equal_to, P, ("%s.P" % geo,))
    #         self.add_relation(CR.Equal_to, J, ("%s.J" % geo,))
    #         self.add_relation(CR.Equal_to, m, ("%s.m" % geo,))

    # def _evaluate_nodes(self, nodes):
    #     equalities = [self._evaluate_node(n) for n in nodes]
    #     return equalities

    # @staticmethod
    # def _extract_primary_arguments(data_dict):
    #     s_args = [n["arguments_symbols"] for n in data_dict if n["align"] == "s"]
    #     r_args = [n["arguments_symbols"] for n in data_dict if n["align"] == "r"]
    #     l_args = [n["arguments_symbols"] for n in data_dict if n["align"] == "l"]
    #     arguments = [itertools.chain(*i) for i in (s_args, r_args, l_args)]
    #     return arguments

    # def assemble_equalities(self):
    #     self.input_equalities = self._evaluate_nodes(self.input_nodes)
    #     self.intermediat_equalities = self._evaluate_nodes(self.intermediat_nodes)
    #     self.output_equalities = self._evaluate_nodes(self.output_nodes)

    # def get_geometries_graph_data(self):
    #     graph = self.graph
    #     geo_graph = graph.edge_subgraph(
    #         self._get_node_predecessors(self.geometry_nodes)
    #     )

    #     input_nodes = self._get_input_nodes(geo_graph)
    #     input_equal = self._evaluate_nodes(input_nodes)

    #     mid_nodes = self._get_intermediat_nodes(geo_graph)
    #     mid_equal = self._evaluate_nodes(mid_nodes)

    #     output_nodes = self._get_output_nodes(geo_graph)
    #     output_equal = self._evaluate_nodes(output_nodes)

    #     data = {
    #         "input_nodes": input_nodes,
    #         "input_equal": input_equal,
    #         "output_nodes": output_nodes,
    #         "output_equal": mid_equal + output_equal,
    #         "geometries_map": self.geometries_map,
    #     }
    #     return data

    # def _create_inputs_dataframe(self):
    #     """ nodes  = self.graph.nodes
    #     inputs = self.input_nodes
    #     condition = lambda i:  isinstance(nodes[i]['lhs_value'], sm.MatrixSymbol)\
    #                         or isinstance(nodes[i]['lhs_value'], sm.Symbol)
    #     indecies = list(filter(condition, inputs))
    #     indecies.sort()
    #     shape = (len(indecies),4)
    #     dataframe = pd.DataFrame(np.zeros(shape),index=indecies,dtype=np.float64) """
    #     raise NotImplementedError

    # def assign_geometry_to_body(self, body, geo, eval_inertia=True, mirror=False):
    #     b1 = body
    #     g1 = geo
    #     b2 = self.bodies[body]["mirr"]
    #     g2 = self.graph.nodes[geo]["mirr"]
    #     self._assign_geometry_to_body(b1, g1, eval_inertia)
    #     if b1 != b2:
    #         self._assign_geometry_to_body(b2, g2, eval_inertia)


################################################################################
# @dataclasses.dataclass
class AbstractRelation(BaseModel):
    variable: str
    inputs: list[str]

    def __repr__(self):
        return f"{self.variable} = {self.__class__.__name__}({self.inputs})"

    def __str__(self):
        return f"{self.variable} = {self.__class__.__name__}({self.inputs})"


class Centered(AbstractRelation):
    variable: str
    inputs: list[str]


class Mirrored(AbstractRelation):
    variable: str
    inputs: str


class Oriented(AbstractRelation):
    variable: str
    inputs: list[str]


class UserInput(AbstractRelation):
    variable: str
    inputs: list[str]


class Vector(BaseModel):
    x: float = 0
    y: float = 0
    z: float = 0


def edge_relation_decorator(
    config: ConfigurationGraph, relation_type: type[AbstractRelation]
) -> Callable[[str], None]:
    @functools.wraps(config.add_relation)
    def decorated(variable: str, inputs: list[str], mirrored=False):
        config.add_relation(relation_type, variable, inputs, mirrored)

    return decorated


def node_relation_decorator(
    config: ConfigurationGraph,
    relation_type: type[AbstractRelation],
    node_type: type[Vector],
    prefix: str,
) -> Callable[[str], None]:
    if relation_type is not None:

        @functools.wraps(relation_type)
        def decorated(variable: str, inputs: list[str], mirrored=False):
            node = config.add_node(
                name=variable, node_type=node_type, prefix=prefix, mirrored=mirrored
            )
            config.add_relation(relation_type, node, inputs, mirrored)
            return node

    else:

        @functools.wraps(relation_type)
        def decorated(variable: str, mirrored=False):
            node = config.add_node(
                name=variable, node_type=node_type, prefix=prefix, mirrored=mirrored
            )
            return node

    return decorated


class VectorNodes(object):
    def __init__(self, config_instance: ConfigurationGraph):
        self.prefix = "vc"
        self.node_type = Vector

        self.Mirrored = node_relation_decorator(
            config_instance, Mirrored, self.node_type, self.prefix
        )
        self.Oriented = node_relation_decorator(
            config_instance, Oriented, self.node_type, self.prefix
        )
        self.UserInput = node_relation_decorator(
            config_instance, None, self.node_type, self.prefix
        )


class PointNodes(object):
    def __init__(self, config_instance: ConfigurationGraph):
        self.prefix = "hp"
        self.node_type = Vector

        self.Mirrored = node_relation_decorator(
            config_instance, Mirrored, self.node_type, self.prefix
        )
        self.Centered = node_relation_decorator(
            config_instance, Centered, self.node_type, self.prefix
        )
        self.UserInput = node_relation_decorator(
            config_instance, None, self.node_type, self.prefix
        )


class RelationEdges(object):
    def __init__(self, config_instance: ConfigurationGraph):
        self.Centered = edge_relation_decorator(config_instance, Centered)
        self.Mirrored = edge_relation_decorator(config_instance, Mirrored)
        self.Oriented = edge_relation_decorator(config_instance, Oriented)


class MultiBodyTopologyConfig(object):
    name: str
    multibody_graph: AbstractTopologyGraph
    config_instance: ConfigurationGraph

    def __init__(self, name: str, multibody_graph: AbstractTopologyGraph):
        self.multibody_graph = multibody_graph
        self.config_instance = ConfigurationGraph(name, multibody_graph)
        self._vector_relations = VectorNodes(self.config_instance)
        self._points_relations = PointNodes(self.config_instance)
        self._relations_edges = RelationEdges(self.config_instance)
        self._construct_topology_inputs()

    @property
    def add_relation(self):
        return self._relations_edges

    @property
    def add_point(self):
        return self._points_relations

    @property
    def add_vector(self):
        return self._vector_relations

    def draw_symmetric_side_graph(self):
        self.config_instance.draw_symmetric_side_graph()

    def _construct_topology_inputs(self):
        nodes = self.multibody_graph.nodes(data=True)
        unique_nodes = (node for node in nodes if node[1]["align"] in {"s", "l"})

        self.config_instance.graph.add_nodes_from(nodes)
        for node in unique_nodes:
            if node[1]["align"] != "s":
                self.add_relation.Mirrored(node[1]["mirror"], [node[0]])
            else:
                self.config_instance.add_node(node[0], None, "")
        # inputs = (node[1].get_config_inputs() for node in nodes)
        # inputs = filter(lambda var: True if var else False, inputs)
        # inputs = map(list, inputs)
        # flattened_inputs = functools.reduce(lambda a, b: a + b, inputs, [])
        # for var in flattened_inputs:
        #     self.add_variable(var)
        # print(list(flattened_inputs))


if __name__ == "__main__":
    from uraeus.rnea.cmbs.topologies.systems import MultiBodyGraph

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

    config = MultiBodyTopologyConfig(name="config1", multibody_graph=graph.topology)

    config.add_point.UserInput("ucaf", mirrored=True)
    config.add_point.UserInput("ucar", mirrored=True)
    config.add_point.UserInput("ucao", mirrored=True)

    config.add_point.UserInput("lcaf", mirrored=True)
    config.add_point.UserInput("lcar", mirrored=True)
    config.add_point.UserInput("lcao", mirrored=True)

    config.add_point.UserInput("tro", mirrored=True)
    config.add_point.UserInput("tri", mirrored=True)

    config.add_point.UserInput("push_uca", mirrored=True)
    config.add_point.UserInput("push_rocker", mirrored=True)

    config.add_point.UserInput("rocker_chassis", mirrored=True)
    config.add_point.UserInput("shock_rocker", mirrored=True)
    config.add_point.UserInput("shock_chassis", mirrored=True)
    config.add_point.Centered(
        "shock_mid", ["hpr_shock_rocker", "hpr_shock_chassis"], mirrored=True
    )

    config.add_relation.Centered(
        "rbr_uca.pose", ["hpr_ucaf", "hpr_ucar", "hpr_ucao"], mirrored=True
    )

    # config.draw_graph()
    config.draw_symmetric_side_graph()
