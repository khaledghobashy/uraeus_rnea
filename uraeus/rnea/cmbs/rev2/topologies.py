from typing import NamedTuple, Callable, Dict
from functools import partial
import itertools
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

from uraeus.rnea.cmbs.rev2.bodies import RigidBodyData
from uraeus.rnea.cmbs.rev2.joints import (
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
        _, nb, joints, edges_indices, actuators = self

        constraints_functions = [
            j.pos_constraint for j in itertools.chain(joints, actuators)
        ]

        qdt0s = jnp.split(qdt0, nb)
        arguments = zip(constraints_functions, edges_indices)

        ground_error = ground_constraint(qdt0s[0])
        peuler_error = euler_parameters_constraints(qdt0s)
        joints_error = jnp.concatenate(
            tuple(func(qdt0s[p], qdt0s[s], t) for func, (p, s) in arguments)
        )

        # inputs_P = jnp.stack([qdt0s[p] for p, _ in edges_indices])
        # print(inputs_P.shape)
        # inputs_S = jnp.stack([qdt0s[s] for _, s in edges_indices])
        # index = np.arange(len(constraints_functions))

        # vmap_body = lambda i, q_P, q_S: jax.lax.switch(
        #     i, constraints_functions, q_P, q_S, t
        # )
        # vmap_func = jax.vmap(vmap_body, out_axes=0)

        # constraints_eq = vmap_func(index, inputs_P, inputs_S)

        res = jnp.concatenate((ground_error, peuler_error, joints_error))
        return res

    @partial(jax.jit, static_argnums=(0,))
    def vel_constraints(self, qdt0: np.ndarray, t: float) -> np.ndarray:
        _, nb, joints, edges_indices, actuators = self

        qdt0s = jnp.split(qdt0, nb)

        constraints = itertools.chain(joints, actuators)

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
        _, nb, joints, edges_indices, actuators = self

        constraints = itertools.chain(joints, actuators)

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
    return np.concatenate(tuple((b.qdt0_G for b in system.bodies.values())))


@jax.jit
def euler_parameters_constraints(qdt0s: tuple[np.ndarray]):
    pdt0s = jnp.vstack([q[3:] for q in qdt0s[1:]])
    func = lambda p: (p @ p) - 1
    return jax.lax.map(func, pdt0s)
    # return jnp.array(tuple(((p.T @ p) - 1) for p in pdt0s))


@jax.jit
def ground_constraint(g_qdt0: np.ndarray):
    g_rdt0, g_pdt0 = jnp.split(g_qdt0, (3,))
    cons = jnp.hstack((g_rdt0, g_pdt0 - np.array([1.0, 0.0, 0.0, 0.0])))
    return cons


def newton_step(model: MultiBodyGraphData, guess: np.ndarray, t: float) -> np.ndarray:
    error = model.pos_constraints(guess, t)
    jac = model.jacobian(guess, t)
    correction = jnp.linalg.solve(jac, -error)
    new_guess = guess + correction
    return new_guess, t


def newton_error(model: MultiBodyGraphData, guess: np.ndarray, t: float) -> float:
    return jnp.linalg.norm(model.pos_constraints(guess, t))


@partial(jax.jit, static_argnums=(0,))
def newton_raphson(model: MultiBodyGraphData, qdt0, t: float):
    tol = 1e-5

    qdt0, t = jax.lax.while_loop(
        lambda args: newton_error(model, *args) > tol,
        lambda args: newton_step(model, *args),
        (qdt0, t),
    )

    return qdt0, model.jacobian(qdt0, t)


@partial(jax.jit, static_argnums=(0,))
def kinematic_scan(
    model: MultiBodyGraphData,
    carry: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    t: float,
):
    x_p, v_p, a_p, dt = carry

    # vel_eqn_dt = jax.jacfwd(model.pos_constraints, argnums=1)
    # acc_eqn_dt = jax.jacfwd(vel_eqn_dt, argnums=1)

    guess = x_p + (v_p * dt) + (0.5 * a_p * dt**2)

    x_n, jac = newton_raphson(model, guess, t)
    v_n = jnp.linalg.solve(jac, -model.vel_constraints(x_n, t))  # - vel_eqn_dt(x_n, t),
    a_n = jnp.linalg.solve(
        jac, -model.acc_constraints(x_n, v_n, t)  # - acc_eqn_dt(x_n, t),
    )

    return (x_n, v_n, a_n, dt), (x_p, v_p, a_p)


def kinematic_sim(
    model: MultiBodyGraphData,
    x0: np.ndarray,
    t_array: np.ndarray,
):
    dt = t_array[1] - t_array[0]
    guess_states = (x0, 0 * x0, 0 * x0, dt)
    initial_states, _ = kinematic_scan(model, guess_states, 0.0)
    _, states_history = jax.lax.scan(
        lambda carry, t: kinematic_scan(model, carry, t), initial_states, t_array
    )
    return states_history


# def static_equilibrium(model: MultiBodyGraphData, qdt0, t: float) -> np.ndarray:
#     x0 = qdt0
#     x, d, success, msg = fsolve(
#         model.pos_constraints,
#         x0,
#         args=(t,),
#         fprime=model.jacobian,
#         full_output=True,
#         xtol=1e-5,
#     )
#     return x, model.jacobian(x, t)


# def minimizer(model: MultiBodyGraphData, qdt0, t: float) -> np.ndarray:
#     def objective(x: np.ndarray, t) -> float:
#         res = model.pos_constraints(x, t)
#         return jnp.sqrt(res @ res)

#     x0 = qdt0
#     x = minimize(
#         objective,
#         x0,
#         args=(t,),
#         method="BFGS",
#         # fprime=model.jacobian,
#         # full_output=True,
#     )
#     return x


# # @partial(jax.jit, static_argnums=(0,))
# def newton_raphson1(model: MultiBodyGraphData, qdt0, t: float):
#     tol = 1e-5

#     # eval constraints vector "residual"
#     residual = model.pos_constraints(qdt0, t)

#     # eval constraints jacobian
#     jacobian = model.jacobian(qdt0, t)

#     # solve for delta q
#     delta_q = jnp.linalg.solve(jacobian, -residual)

#     iter = 0

#     # def cond(dq):
#     #     return jnp.linalg.norm(dq) > tol

#     # def body(dq):
#     #     qdt0 = qdt0 + dq
#     #     res = model.pos_constraints(q, t)
#     #     j = model.jacobian(q, t)
#     #     dq = jnp.linalg.solve(qdt0, -res)
#     #     return dq

#     # delta_q = jax.lax.while_loop(cond, body, delta_q)
#     # qdt0 = qdt0 + delta_q
#     # print(qdt0)
#     # def cond(res):
#     #     return jnp.allclose(jnp.sqrt(residual @ residual), 0.00001)

#     # while jax.lax.cond(
#     #     True, cond, lambda res: False, residual
#     # ):  # jnp.sqrt(residual @ residual) >= tol:
#     error = jnp.linalg.norm(residual)

#     while error > tol:
#         # print("delta_q norm = ", jnp.linalg.norm(delta_q))
#         # update generalized coordinates vector q
#         qdt0 = qdt0 + delta_q

#         # eval constraints vector "residual"
#         residual = model.pos_constraints(qdt0, t)
#         # print(residual)

#         # eval constraints jacobian
#         # jacobian = model.jac_eqn(qdt0, t)
#         jacobian = model.jacobian(qdt0, t)

#         # solve for delta q
#         delta_q = jnp.linalg.solve(jacobian, -residual)
#         error = jnp.linalg.norm(residual)

#         iter += 1
#         if iter >= 30:
#             print("Iterations Exceeded!")
#             print("Couldn't Converge at t = %s!" % t)
#             print("This could lead to a wrong solution!\n")
#             break

#     jacobian = model.jacobian(qdt0, t)
#     print(iter)

#     return qdt0, jacobian

# # @partial(jax.jit, static_argnums=(0,))
# def kinematic_sim1(
#     model: MultiBodyGraphData,
#     x0: np.ndarray,
#     t_array: np.ndarray,
#     # solver: Callable[
#     #     [MultiBodyGraphData, np.ndarray, float], tuple[np.ndarray, np.ndarray]
#     # ] = newton_raphson,
# ):
#     qdt0s = [x0]
#     qdt1s = [0 * x0]
#     qdt2s = [0 * x0]

#     dt = t_array[1] - t_array[0]

#     vel_eqn_dt = jax.jacfwd(model.pos_constraints, argnums=1)
#     acc_eqn_dt = jax.jacfwd(vel_eqn_dt, argnums=1)

#     for i, t in enumerate(t_array):
#         # qdt0, jac = newton_raphson(model, qdt0s[-1], t)
#         guess = qdt0s[-1] + (qdt1s[-1] * dt) + (0.5 * qdt2s[-1] * dt**2)
#         qdt0, jac = newton_raphson(model, guess, t)

#         qdt1 = jnp.linalg.solve(
#             jac,
#             -model.vel_constraints(qdt0, t) - vel_eqn_dt(qdt0, t),
#         )
#         qdt2 = jnp.linalg.solve(
#             jac,
#             -model.acc_constraints(qdt0, qdt1s[-1], t) - acc_eqn_dt(qdt0, t),
#         )

#         qdt0s.append(qdt0)
#         qdt1s.append(qdt1)
#         qdt2s.append(qdt2)

#     return qdt0s[1:], qdt1s[1:], qdt2s[1:]


# if __name__ == "__main__":
#     pass
