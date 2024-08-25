from dataclasses import dataclass
from typing import NamedTuple, Dict, Tuple, List, Callable
import numpy as np
import jax.numpy as jnp
import jax

from uraeus.rnea.quaternion.spatial_algebra import quaternion_to_dcm, transform_vector
from uraeus.rnea.quaternion.bodies import RigidBody, RigidBodyData, BodyKinematics
from uraeus.rnea.quaternion.joints import (
    FreeJoint,
    JointData,
    JointConfigInputs,
    RevoluteJoint,
    TranslationalJoint,
    FunctionalJoint,
    JointKinematics,
    construct_functional_joint,
)
from uraeus.rnea.quaternion.topologies import MultiBodyTree
from uraeus.rnea.quaternion.graphs import contstruct_traversal_orders
from uraeus.rnea.quaternion.tree_traversals import (
    base_to_tip,
    extract_mobilizer_forces,
    extract_reaction_forces,
)
from uraeus.rnea.quaternion.algorithms import (
    split_coordinates,
    IDCallRes,
    inverse_dynamics_call,
)


class MultiBodyData(NamedTuple):
    joints: Tuple[FunctionalJoint]
    bodies_inertias: List[np.ndarray]
    forward_traversal: Tuple[Tuple[int, int, int], ...]
    backward_traversal: List[Tuple[int, List[int]]]
    qdt0_idx: Tuple[int]
    qdt1_idx: Tuple[int]

    def __hash__(self):
        return hash(self.__class__.__name__)


def construct_multibodydata(topology: MultiBodyTree) -> MultiBodyData:
    func_joints = tuple(map(construct_functional_joint, topology.joints.values()))
    bodies_inertias = [b.I for b in topology.bodies.values()]
    forward_traversal, backward_traversal = contstruct_traversal_orders(topology.tree)
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
    forces_map: Dict[str, Dict[str, np.ndarray]]
    tree_data: MultiBodyData

    def __init__(self, topology: MultiBodyTree):
        self.topology = topology
        gravity = np.array([0, 0, -9.81, 0, 0, 0])
        self.forces_map = {
            b.name: {"gravity": b.I @ gravity} for b in self.topology.bodies.values()
        }

        self.tree_data = construct_multibodydata(topology)
        self.bodies_idx = {b: i for i, b in enumerate(self.topology.tree.nodes)}

    def get_body_kinematics(
        self, name: str, bodies_kinematics: List[BodyKinematics]
    ) -> BodyKinematics:
        return bodies_kinematics[self.bodies_idx[name]]

    def forward_kinematics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> Tuple[BodyKinematics, JointKinematics]:
        coordinates = split_coordinates(self.tree_data.qdt0_idx, qdt0, qdt1, qdt2)

        bodies_kinematics, joints_kinematics = base_to_tip(
            self.tree_data.joints, coordinates, self.tree_data.forward_traversal
        )

        return bodies_kinematics, joints_kinematics

    def inverse_dynamics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> IDCallRes:
        forces = [
            list(forces_dict.values()) for name, forces_dict in self.forces_map.items()
        ]
        # print(forces)
        res = inverse_dynamics_call(self.tree_data, forces, qdt0, qdt1, qdt2)
        return res

    # def forward_dynamics_pass(
    #     self,
    #     qdt0: np.ndarray,
    #     qdt1: np.ndarray,
    #     tau: np.ndarray,
    #     forces_map: Dict[str, Dict[str, np.ndarray]],
    # ) -> np.ndarray:
    #     external_forces = [list(i.values()) for i in forces_map.values()]
    #     qdt2 = forward_dynamics_call(self.tree_data, external_forces, qdt0, qdt1, tau)
    #     return qdt2

    # def ssode(
    #     self,
    #     t: float,
    #     ydt0: np.ndarray,
    #     forces_func: Callable,
    # ):
    #     qdt0, qdt1 = ydt0.reshape(2, -1)

    #     bodies_kin, _ = self.forward_kinematics_pass(qdt0, qdt1, np.zeros_like(qdt1))

    #     gen_forces, ext_forces = forces_func(
    #         bodies_kin,
    #         self.bodies_idx,
    #         self.forces_map,
    #         qdt0,
    #         qdt1,
    #     )

    #     qdt2 = self.forward_dynamics_pass(qdt0, qdt1, gen_forces, ext_forces)
    #     return np.hstack([qdt1, qdt2])


if __name__ == "__main__":

    import itertools
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    tree = MultiBodyTree("pendulum")

    l1_data = RigidBodyData(np.array([0, 0, -5]), np.array([1, 0, 0, 0]), 1, np.eye(3))
    l2_data = RigidBodyData(np.array([0, 0, -10]), np.array([1, 0, 0, 0]), 1, np.eye(3))
    j1_data = JointConfigInputs(np.array([0, 0, 0]), np.array([1, 0, 0]), None)
    j2_data = JointConfigInputs(np.array([0, 0, -5]), np.array([1, 0, 0]), None)

    tree.add_joint("j1", "ground", "l1", l1_data, RevoluteJoint, j1_data)
    tree.add_joint("j2", "l1", "l2", l2_data, RevoluteJoint, j2_data)

    model = Model(tree)
    # exit()
    # bodies_kinematics, joints_kinematics = model.forward_kinematics_pass(
    #     np.array([np.radians(0), np.radians(90)]), np.array([0, 0]), np.array([0, 0])
    # )
    # # print(joints_kinematics[0])
    # l1_kin = model.get_body_kinematics("l1", bodies_kinematics)
    # l2_kin = model.get_body_kinematics("l2", bodies_kinematics)

    # print(f"l1.r_G = {transform_vector(l1_kin.p_GB.q, l1_kin.p_GB.r)}")
    # print(f"l2.r_G = {transform_vector(l2_kin.p_GB.q, l2_kin.p_GB.r)}")

    # theta1_dt0 = lambda t: jnp.radians(45) * jnp.sin(t)
    theta1_dt0 = lambda t: np.radians(0)
    theta2_dt0 = lambda t: 0

    theta1_dt1 = jax.jacfwd(theta1_dt0)
    theta2_dt1 = jax.jacfwd(theta2_dt0)

    theta1_dt2 = jax.jacfwd(theta1_dt1)
    theta2_dt2 = jax.jacfwd(theta2_dt1)

    time_array = np.linspace(0, 2 * np.pi, 100)

    inverse_dyn_res = [
        model.inverse_dynamics_pass(
            np.array([theta1_dt0(t), theta2_dt0(t)]),
            np.array([theta1_dt1(t), theta2_dt1(t)]),
            np.array([theta1_dt2(t), theta2_dt2(t)]),
        )
        for t in time_array
    ]

    taus, bodies_kinematics, joints_kinematics, joints_forces = zip(*inverse_dyn_res)

    mobilizer_forces = list(
        map(
            extract_mobilizer_forces,
            joints_forces,
            itertools.repeat(
                (
                    model.topology.joints["j1"].joint_data.frames,
                    model.topology.joints["j2"].joint_data.frames,
                )
            ),
            joints_kinematics,
            bodies_kinematics,
        )
    )

    tau1, tau2 = zip(*taus)

    j1_forces, j2_forces = zip(*mobilizer_forces)
    j1_fi_S, j1_fc_S, j1_fa_S, j1_fc_G, j1_tau = zip(*j1_forces)
    j2_fi_S, j2_fc_S, j2_fa_S, j2_fc_G, j2_tau = zip(*j2_forces)

    plt.figure()
    plt.plot(time_array, tau1)
    plt.plot(time_array, tau2)
    plt.grid()

    plt.figure()
    plt.plot(time_array, [f[2] for f in j1_fc_G])
    plt.plot(time_array, [f[2] for f in j2_fc_G])
    plt.grid()
    plt.show()

    # bodies_kinematics = [
    #     model.forward_kinematics_pass(
    #         np.array([theta1_dt0(t), theta2_dt0(t)]),
    #         # np.array([i, 0]),
    #         # np.array([i, i]),
    #         np.array([theta1_dt1(t), theta2_dt1(t)]),
    #         np.array([0, 0]),
    #     )[0]
    #     for t in time_array
    #     # for t in time_array[0:1]
    # ]

    # # print(l1_kin.p_GB)
    # # print(l2_kin.p_GB)

    # l1_pose_G = [
    #     model.get_body_kinematics("l1", bodies).p_GB for bodies in bodies_kinematics
    # ]
    # l2_pose_G = [
    #     model.get_body_kinematics("l2", bodies).p_GB for bodies in bodies_kinematics
    # ]

    # l1_v_G = [
    #     model.get_body_kinematics("l1", bodies).v_G for bodies in bodies_kinematics
    # ]
    # l2_v_G = [
    #     model.get_body_kinematics("l2", bodies).v_G for bodies in bodies_kinematics
    # ]

    # l1_a_G = [
    #     model.get_body_kinematics("l1", bodies).a_G for bodies in bodies_kinematics
    # ]
    # l2_a_G = [
    #     model.get_body_kinematics("l2", bodies).a_G for bodies in bodies_kinematics
    # ]

    # l1_r_G = [p.r for p in l1_pose_G]
    # l2_r_G = [p.r for p in l2_pose_G]

    # l1_r_y = [(0, float(r1[1]), float(r2[1])) for r1, r2 in zip(l1_r_G, l2_r_G)]
    # l1_r_z = [(0, float(r1[2]), float(r2[2])) for r1, r2 in zip(l1_r_G, l2_r_G)]

    # l1_v_y = [v[1] for v in l1_v_G]
    # l1_v_z = [v[2] for v in l1_v_G]

    # l2_v_y = [v[1] for v in l2_v_G]
    # l2_v_z = [v[2] for v in l2_v_G]

    # l1_a_y = [v[1] for v in l1_a_G]
    # l1_a_z = [v[2] for v in l1_a_G]

    # l2_a_y = [v[1] for v in l2_a_G]
    # l2_a_z = [v[2] for v in l2_a_G]

    # # fig, ax = plt.subplots()
    # # # scat = ax.scatter(l1_r_y[0], l1_r_z[0])
    # # line = ax.plot(l1_r_y[0], l1_r_z[0], label="line")[0]
    # # ax.set(xlim=[-10, -10], ylim=[-10, 10], xlabel="y", ylabel="z")
    # # ax.legend()
    # # plt.show()

    # def animate(i):
    #     print(time_array[i])
    #     plt.cla()
    #     plt.grid()
    #     plt.xlim([-10, 10])
    #     plt.ylim([-10, 10])
    #     plt.plot(l1_r_y[i], l1_r_z[i])
    #     plt.plot(l1_r_y[i], l1_r_z[i], "o")
    #     return

    # fig = plt.figure(figsize=(10, 10))
    # plt.grid()
    # ani = animation.FuncAnimation(fig, animate, frames=99, interval=50)
    # plt.show()

    # fig = plt.figure(figsize=(10, 10))
    # plt.plot(time_array, l1_v_y)
    # plt.plot(time_array, l1_v_z)
    # plt.plot(time_array, l2_v_y)
    # plt.plot(time_array, l2_v_z)
    # plt.grid()
    # plt.show()

    # fig = plt.figure(figsize=(10, 10))
    # plt.plot(time_array, l1_a_y)
    # plt.plot(time_array, l1_a_z)
    # plt.plot(time_array, l2_a_y)
    # plt.plot(time_array, l2_a_z)
    # plt.grid()
    # plt.show()
