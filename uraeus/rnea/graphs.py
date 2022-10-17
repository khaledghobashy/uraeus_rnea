from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from collections import defaultdict
from functools import reduce

import numpy as np


class Graph(object):

    adj_list: Dict[str, List[str]]
    nodes: List[str]
    edges: List[Tuple[str, str]]

    def __init__(self, name: str):

        self.name = name
        self.adj_list = defaultdict(list)
        self.edges = []

    @property
    def nodes(self):
        return self.adj_list.keys()

    def add_edge(self, predecessor: str, successor: str) -> None:
        self.adj_list[predecessor].append(successor)
        self.adj_list[successor] = []
        self.edges.append((predecessor, successor))


class Tree(object):

    adj_list: Dict[str, List[str]]
    nodes: List[str]
    edges: List[Tuple[str, str]]

    def __init__(self, name: str, root: Optional[str] = "root"):

        self.graph = Graph(name)
        self.root = root
        self.graph.adj_list[self.root] = []

    @property
    def adj_list(self):
        return self.graph.adj_list

    @property
    def nodes(self):
        return self.adj_list.keys()

    @property
    def edges(self):
        return self.graph.edges

    def add_edge(self, predecessor: str, successor: str) -> None:
        if not self.check_if_node_exists(predecessor):
            raise ValueError(f"Node '{predecessor}' is not in the tree!")

        if self.check_if_node_exists(successor):
            raise ValueError(f"Cannot add node '{successor}', as it already exists!")

        self.graph.add_edge(predecessor, successor)

    def check_if_node_exists(self, node: str) -> bool:
        return node in self.adj_list


def accumulate_root_to_leaf(
    root_initial: Any,
    cumfunc: Callable[[Any, Any], Any],
) -> Callable[[List[Any], List[Tuple[int, int, int]]], List[Any]]:
    def func(edges_wieghts: List[Any], traversal_order: List[Tuple[int, int, int]]):
        nodes_vals = [root_initial]

        for _, edge_index, predecessor_index in traversal_order:
            successor_val = cumfunc(
                nodes_vals[predecessor_index], edges_wieghts[edge_index]
            )
            nodes_vals.append(successor_val)
        return nodes_vals

    return func


def accumulate_leaf_to_root(
    cumfunc: Callable[[Any, Any, Any], Any],
) -> Callable[[List[Any], List[Any], List[Tuple[int, List[int]]]], List[Any]]:
    def func(
        nodes_weights: List[Any],
        edges_weights: List[Any],
        traversal_order: List[Tuple[int, List[int]]],
    ):
        edges_cumvals = []
        for successor_index, out_edges in traversal_order[:-1]:
            out_edges_weights = [edges_weights[i] for i in out_edges]
            out_edges_cumvasl = [edges_cumvals[i] for i in out_edges]
            edge_val = cumfunc(
                nodes_weights[successor_index], out_edges_weights, out_edges_cumvasl
            )
            edges_cumvals.append(edge_val)

        return edges_cumvals

    return func


def contstruct_traversal_orders(tree: Tree):
    nodes_indicies = {n: i for i, n in enumerate(tree.nodes)}
    base_to_tip = [
        (nodes_indicies[s], i, nodes_indicies[p]) for i, (p, s) in enumerate(tree.edges)
    ]
    edges_indices = {e: i for i, e in enumerate(reversed(tree.edges))}
    tip_to_base = [
        (nodes_indicies[node], [edges_indices[(node, c)] for c in childern])
        for node, childern in reversed(tree.adj_list.items())
    ]
    return base_to_tip, tip_to_base
