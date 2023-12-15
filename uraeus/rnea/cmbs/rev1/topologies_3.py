from typing import NamedTuple, Callable, Dict
from functools import partial
from collections import defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel
import networkx as nx

from uraeus.rnea.cmbs.bodies import RigidBodyData
from uraeus.rnea.cmbs.joints import (
    FunctionalJoint,
    construct_joint,
    AbstractJoint,
    JointConfigInputs,
    AbstractJointActuator,
)


class Node(BaseModel):
    name: str

    def __hash__(self):
        return hash(self.name)


class Graph(object):
    adj_list: Dict[str, list[str]]
    nodes: list[str]
    edges: list[tuple[str, str]]

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


class MultiBodyGraph(object):
    name: str
    graph: Graph
    bodies: Dict[str, RigidBodyData]
    joints: Dict[str, AbstractJoint]
    actuators: Dict[str, AbstractJointActuator]

    def __init__(self, name: str):
        self.name = name
        self.graph = Graph(self.name)

        self.bodies = {}
        self.joints = {}
        self.actuators = {}
        self.joint_map = {}

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
        body_i: str,
        body_j: str,
        joint_type: AbstractJoint,
        joint_config: JointConfigInputs,
    ):
        assert body_i in self.bodies
        assert body_j in self.bodies
        assert joint_name not in self.joints

        self.graph.add_edge(body_i, body_j)
        self.joints[joint_name] = joint_type(
            name=joint_name,
            body_i=self.bodies[body_i],
            body_j=self.bodies[body_j],
            joint_config=joint_config,
        )
        self.joint_map[joint_name] = (body_i, body_j)

    def add_actuator(
        self,
        actuator_name: str,
        joint_name: str,
        actuator_type: type[AbstractJointActuator],
        driver: Callable[[float], float],
    ):
        assert joint_name in self.joints
        joint = self.joints[joint_name]

        self.graph.add_edge(*self.joint_map[joint_name])
        self.joints[actuator_name] = actuator_type(
            name=joint_name,
            joint=joint,
            driver=driver,
        )


def adj2int(graph: Graph):
    d = dict(zip(graph.nodes, range(len(graph.nodes))))
    edges = tuple((d[p], d[s]) for p, s in graph.edges)
    return edges


def edges_coordinates(
    qdt0s: tuple[np.ndarray],
    edges: tuple[tuple[int, int]],
) -> tuple[tuple[np.ndarray, np.ndarray]]:
    return tuple((qdt0s[p], qdt0s[s]) for p, s in edges)


def construct_qdt0(system: MultiBodyGraph):
    return np.hstack(
        tuple(np.hstack((b.location, b.orientation)) for b in system.bodies.values())
    )


# @dataclass
class MultiBodyGraphData(NamedTuple):
    name: str
    nb: int
    joints: tuple[FunctionalJoint, ...]
    edges_indices: tuple[tuple[int, int]]
    actuators: tuple[FunctionalJoint, ...]

    @partial(jax.jit, static_argnums=(0,))
    def pos_constraints(self, qdt0: np.ndarray, t: float) -> np.ndarray:
        name, nb, joints, edges_indices, actuators = self

        constraints = list(joints) + list(actuators)

        qdt0s = jnp.split(qdt0, nb)

        ground_cons = ground_constraint(qdt0s[0])
        peuler_cons = euler_parameters_constraints(qdt0s)

        constraints_eq = jnp.concatenate(
            tuple(
                j.pos_constraint(qdt0s[p], qdt0s[s], t)
                for j, (p, s) in zip(constraints, edges_indices)
            )
        )

        res = jnp.concatenate((ground_cons, peuler_cons, constraints_eq))
        return res

    @partial(jax.jit, static_argnums=(0,))
    def vel_constraints(self, qdt0: np.ndarray, t: float) -> np.ndarray:
        name, nb, joints, edges_indices, actuators = self

        qdt0s = jnp.split(qdt0, nb)

        constraints = list(joints) + list(actuators)

        ground_cons = np.zeros((7,))
        peuler_cons = np.zeros((nb - 1,))

        joints_cons = jnp.concatenate(
            tuple(
                j.vel_constraint(qdt0s[p], qdt0s[s], t)
                for j, (p, s) in zip(constraints, edges_indices)
            )
        )

        res = jnp.concatenate((ground_cons, peuler_cons, joints_cons))
        return res

    @partial(jax.jit, static_argnums=(0,))
    def acc_constraints(
        self, qdt0: np.ndarray, qdt1: np.ndarray, t: float
    ) -> np.ndarray:
        name, nb, joints, edges_indices, actuators = self

        constraints = list(joints) + list(actuators)

        qdt0s = jnp.split(qdt0, nb)
        qdt1s = jnp.split(qdt1, nb)
        pdt1s = tuple(q[3:] for q in qdt1s[1:])

        ground_cons = np.zeros((7,))
        peuler_cons = jnp.array(tuple(2 * (p @ p) for p in pdt1s))

        joints_cons = jnp.concatenate(
            tuple(
                j.acc_constraint(qdt0s[p], qdt0s[s], qdt1s[p], qdt1s[s], t)
                for j, (p, s) in zip(constraints, edges_indices)
            )
        )

        res = jnp.concatenate((ground_cons, peuler_cons, joints_cons))
        return res

    @property
    def jacobian(self):
        return jax.jacfwd(self.pos_constraints)

    def __hash__(self):
        return hash(self.name)


def construct_multibody_graph_data(multibody: MultiBodyGraph):
    name = multibody.name
    nb = len(multibody.graph.nodes)
    joints = tuple(construct_joint(joint) for joint in multibody.joints.values())
    actuators = tuple(construct_joint(act) for act in multibody.actuators.values())
    edges_indices = adj2int(multibody.graph)

    return MultiBodyGraphData(name, nb, joints, edges_indices, actuators=actuators)


def construct_qdt0(system: MultiBodyGraph):
    return np.concatenate(
        tuple(
            np.concatenate((b.location, b.orientation)) for b in system.bodies.values()
        )
    )


@jax.jit
def euler_parameters_constraints(qdt0s: tuple[np.ndarray]):
    pdt0s = tuple(q[3:] for q in qdt0s[1:])
    return jnp.array(tuple(((p.T @ p) - 1) for p in pdt0s))


@jax.jit
def ground_constraint(g_qdt0: np.ndarray):
    g_rdt0, g_pdt0 = np.split(g_qdt0, (3,))
    cons = jnp.hstack((g_rdt0, g_pdt0 - np.array([1, 0, 0, 0])))
    return cons


# class FunctionalEquations(NamedTuple):
#     pos_eqn: Callable[[np.ndarray, float], np.ndarray]
#     vel_eqn: Callable[[np.ndarray, float], np.ndarray]
#     acc_eqn: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
#     jac_eqn: Callable[[np.ndarray, float], np.ndarray]


# def construct_system_equations(
#     multibodydata: MultiBodyGraphData,
# ) -> FunctionalEquations:
#     pos_eqn = partial(pos_constraints, multibodydata=multibodydata)
#     vel_eqn = partial(vel_constraints, multibodydata=multibodydata)
#     acc_eqn = partial(acc_constraints, multibodydata=multibodydata)
#     jac_eqn = partial(constraint_jacobian, multibodydata=multibodydata)
#     return FunctionalEquations(pos_eqn, vel_eqn, acc_eqn, jac_eqn)


# constraint_jacobian = partial(jax.jit, static_argnums=(0,))(
#     jax.jacfwd(MultiBodyEquations.pos_constraints)
# )

# constraint_jacobian = jax.jit(jax.jacfwd(MultiBodyGraphData.pos_constraints))
# vel_eqn_dt = jax.jit(jax.jacfwd(pos_constraints, argnums=1), static_argnums=(2,))
# acc_eqn_dt = jax.jit(jax.jacfwd(vel_constraints, argnums=1), static_argnums=(2,))

# vel_eqn_dt = jax.jacfwd(MultiBodyGraphData.pos_constraints, argnums=1)
# acc_eqn_dt = jax.jacfwd(MultiBodyGraphData.vel_constraints, argnums=1)


def newton_raphson(model: MultiBodyGraphData, qdt0, t: float):
    tol = 1e-5

    # eval constraints vector "residual"
    residual = model.pos_constraints(qdt0, t)

    # eval constraints jacobian
    jacobian = model.jacobian(qdt0, t)

    # solve for delta q
    delta_q = jnp.linalg.solve(jacobian, -residual)

    iter = 0
    while np.linalg.norm(residual) >= tol:
        # print("delta_q norm = ", jnp.linalg.norm(delta_q))
        # update generalized coordinates vector q
        qdt0 = qdt0 + delta_q

        # eval constraints vector "residual"
        residual = model.pos_constraints(qdt0, t)
        # print(residual)

        # eval constraints jacobian
        # jacobian = model.jac_eqn(qdt0, t)
        jacobian = model.jacobian(qdt0, t)

        # solve for delta q
        delta_q = jnp.linalg.solve(jacobian, -residual)

        iter += 1
        if iter >= 30:
            print("Iterations Exceeded!")
            print("Couldn't Converge at t = %s!" % t)
            print("This could lead to a wrong solution!\n")
            break

    jacobian = model.jacobian(qdt0, t)

    return qdt0, jacobian
