from typing import Any, Callable, Optional, NamedTuple
from collections import defaultdict
from functools import reduce, partial

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx


class Graph(object):
    adj_list: dict[str, list[str]]
    nodes: list[str]
    edges: list[tuple[str, str]]
    nxgraph: nx.DiGraph

    def __init__(self, name: str):
        self.name = name
        self.adj_list = defaultdict(list)
        self.edges = []
        self.nxgraph = nx.DiGraph

    # @property
    # def nodes(self):
    #     return self.adj_list.keys()

    @property
    def nodes(self, **kwargs):
        return self.nxgraph.nodes(**kwargs)

    @property
    def edges(self, **kwargs):
        return self.nxgraph.edges(**kwargs)

    def add_edge(self, predecessor: str, successor: str) -> None:
        self.nxgraph.add_edge(predecessor, successor)
        # self.adj_list[predecessor].append(successor)
        # self.adj_list[successor] = []
        # self.edges.append((predecessor, successor))


class Tree(object):
    adj_list: dict[str, list[str]]
    nodes: list[str]
    edges: nx.DiGraph.edges
    nxgraph: nx.DiGraph

    def __init__(self, name: str, root: Optional[str] = "root"):
        self.root = root
        self.nxgraph = nx.DiGraph(name=name)
        self.nxgraph.add_node(self.root)

    @property
    def adj_list(self):
        return self.nxgraph.adj

    @property
    def nodes(self):
        return self.nxgraph.nodes

    @property
    def edges(self):
        return self.nxgraph.edges

    def add_edge(self, predecessor: str, successor: str, **kwargs) -> None:
        if not self.has_node(predecessor):
            raise ValueError(f"Node '{predecessor}' is not in the tree!")

        if self.has_node(successor):
            raise ValueError(f"Cannot add node '{successor}', as it already exists!")

        self.nxgraph.add_edge(predecessor, successor, **kwargs)

    def has_node(self, node: str) -> bool:
        return self.nxgraph.has_node(node)


def accumulate_root_to_leaf(
    root_initial: Any,
    cumfunc: Callable[[Any, Any], Any],
) -> Callable[[list[Any], list[tuple[int, int, int]]], list[Any]]:
    @partial(jax.jit, static_argnums=(0,))
    def func(
        traversal_order: tuple[tuple[int, int, int], ...], edges_weights: tuple[Any]
    ):
        nodes_vals = [root_initial]

        for _, edge_index, predecessor_index in traversal_order:
            successor_val = cumfunc(
                nodes_vals[predecessor_index], edges_weights[edge_index]
            )
            nodes_vals.append(successor_val)
        return nodes_vals

    return partial(jax.jit(func, static_argnums=(0,)))


# def accumulate_leaf_to_root(
#     cumfunc: Callable[[Any, Any, Any], Any],
# ) -> Callable[[list[Any], list[Any], list[tuple[int, list[int]]]], list[Any]]:
#     def func(
#         nodes_weights: list[Any],
#         edges_weights: list[Any],
#         traversal_order: list[tuple[int, list[int]]],
#     ):
#         edges_cumvals = []
#         for successor_index, out_edges in traversal_order[:-1]:
#             out_edges_weights = [edges_weights[i] for i in out_edges]
#             out_edges_cumvasl = [edges_cumvals[i] for i in out_edges]
#             edge_val = cumfunc(
#                 nodes_weights[successor_index], out_edges_weights, out_edges_cumvasl
#             )
#             edges_cumvals.append(edge_val)

#         return edges_cumvals

#     return partial(jax.jit(func, static_argnums=(2,)))


# def construct_traversal_orders(tree: Tree):
#     nodes_indices = {n: i for i, n in enumerate(tree.nodes)}
#     base_to_tip = [
#         (nodes_indices[s], i, nodes_indices[p]) for i, (p, s) in enumerate(tree.edges)
#     ]
#     edges_indices = {e: i for i, e in enumerate(reversed(tree.edges))}
#     tip_to_base = [
#         (nodes_indices[node], tuple(edges_indices[(node, c)] for c in children))
#         for node, children in reversed(tree.adj_list.items())
#     ]
#     print("nodes_indices = ", nodes_indices)
#     print("edges_indices = ", edges_indices)
#     print("tree.edges = ", tree.edges)
#     print("tree.adj_list.items() = ", tree.adj_list.items())
#     print("tip_to_base = ", tip_to_base)
#     return tuple(base_to_tip), tuple(tip_to_base)


def accumulate_leaf_to_root(
    cumfunc: Callable[[Any, Any, Any], Any],
) -> Callable[[list[Any], list[Any], list[tuple[int, list[int]]]], list[Any]]:
    def func(
        nodes_weights: list[Any],
        edges_weights: list[Any],
        adjacency_list: list[tuple[int, list[int]]],
    ):
        edges_cumvals = dict()
        for successor_index, out_edges in reversed(adjacency_list):
            out_edges_weights = [edges_weights[i - 1] for i in out_edges]
            out_edges_cumvasl = [edges_cumvals[i] for i in out_edges]
            edge_val = cumfunc(
                nodes_weights[successor_index], out_edges_weights, out_edges_cumvasl
            )
            edges_cumvals[successor_index] = edge_val

        return list(reversed(edges_cumvals.values()))[1:]

    return partial(jax.jit(func, static_argnums=(2,)))


class GraphConnectivity(NamedTuple):

    adjacency_list: tuple[tuple[int, tuple[int, ...]], ...]
    edges_list: tuple[tuple[int, int], ...]
    base_to_tip: tuple[tuple[int, int, int], ...]
    nodes_to_root_paths: tuple[tuple[int, ...]]


def extract_graph_data(tree: Tree) -> GraphConnectivity:
    numbered_graph: nx.Graph = nx.convert_node_labels_to_integers(tree.nxgraph)

    base_to_tip = [(s, i, p) for i, (p, s) in enumerate(numbered_graph.edges)]
    edges_list = tuple(numbered_graph.edges)
    nodes_to_root_paths = [
        nx.shortest_path(numbered_graph, source=0, target=i)[:0:-1]
        for i in list(numbered_graph.nodes)[:0:-1]
    ]
    adjacency_list = tuple(
        (node, tuple(neighbors.keys()))
        for node, neighbors in numbered_graph.adj.items()
    )

    graph_connectivity = GraphConnectivity(
        adjacency_list=adjacency_list,
        edges_list=edges_list,
        base_to_tip=tuple(base_to_tip),
        nodes_to_root_paths=nodes_to_root_paths,
    )
    return graph_connectivity


def adj2int(graph: Graph):
    d = dict(zip(graph.nodes, range(len(graph.nodes))))
    edges = tuple((d[p], d[s]) for p, s in graph.edges)
    return edges


def edges_coordinates(
    qdt0s: tuple[np.ndarray],
    edges: tuple[tuple[int, int]],
) -> tuple[tuple[np.ndarray, np.ndarray]]:
    return tuple((qdt0s[p], qdt0s[s]) for p, s in edges)
