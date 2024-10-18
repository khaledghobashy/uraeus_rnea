from functools import reduce
from typing import Callable, Any
from collections import namedtuple

import jax.numpy as jnp
import numpy as np

from uraeus.rnea.bodies import RigidBody, RigidBodyData, BodyKinematics
from uraeus.rnea.tree_traversals import base_to_tip, SystemForces
from uraeus.rnea.graphs import Graph, Tree, construct_traversal_orders

from uraeus.rnea.joints import (
    AbstractJoint,
    JointConfigInputs,
    JointInstance,
    construct_functional_joint,
    construct_joint_instance,
    initialize_joint,
    JointKinematics,
    TranslationalJoint,
    RevoluteJoint,
)

from uraeus.rnea.algorithms import (
    split_coordinates,
    MultiBodyData,
    HybridDynamicsData,
    HybridDynamics,
    IDCallRes,
    inverse_dynamics_call,
    forward_dynamics_call,
    get_qdt1_from_udt0,
)

Forcesdict = dict[str, dict[str, dict[str, np.ndarray]]]


class MultiBodyGraph(object):
    name: str
    graph: Graph
    bodies: dict[str, RigidBody]
    joints: dict[str, JointInstance]

    def __init__(self, name: str):
        self.name = name
        self.graph = Graph(self.name)

        self.bodies = {}
        self.joints = {}

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
        body_i: RigidBodyData,
        body_j: RigidBodyData,
        joint_type,
        joint_config,
    ):
        assert body_i in self.bodies
        assert body_j in self.bodies
        assert joint_name not in self.joints
        self.graph.add_edge(body_i, body_j)
        self.joints[joint_name] = (
            joint_type,
            joint_config,
            self.bodies[body_i],
            self.bodies[body_j],
        )


class MultiBodyTree(object):
    name: str
    tree: Tree
    bodies: dict[str, RigidBody]
    joints: dict[str, JointInstance]

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
            pred_body.kinematics.p_GB,
            succ_body.kinematics.p_GB,
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

    def construct_bodies_coordinates_names(self) -> list[str]:
        coordinates = ["x", "y", "z", "phi", "theta", "psi"]
        namer = lambda l, name: l + [f"{name}.{c}" for c in coordinates]
        bodies_coordinates = reduce(namer, self.bodies.keys(), [])
        return bodies_coordinates

    def check_if_joint_exists(self, joint_name: str) -> bool:
        return joint_name in self.joints


def construct_permutation_matrix(dof: int, id_indices: list[int]) -> np.ndarray:
    permutation = [i for i in range(dof) if i not in id_indices]
    permutation += id_indices
    mat = np.zeros((dof, dof))
    mat[np.arange(0, dof), permutation] = 1
    return mat


def construct_hybrid_dynamics_data(
    tree_data: MultiBodyData, id_coordinates: np.ndarray
) -> HybridDynamicsData:
    n_dof = sum([j.nj for j in tree_data.joints])
    n_id = len(id_coordinates)
    n_fd = n_dof - n_id
    Q = construct_permutation_matrix(n_dof, id_coordinates)

    data = HybridDynamicsData(
        tree_data=tree_data,
        permutation_matrix=Q,
        n_fd=n_fd,
    )
    return data


def create_namedtuple_inplace(name: str, field_dict: dict[str, Any]) -> namedtuple:
    return namedtuple(name, field_dict.keys())(*field_dict.values())


def construct_multibody_data(topology: MultiBodyTree) -> MultiBodyData:
    func_joints = tuple(map(construct_functional_joint, topology.joints.values()))
    bodies_inertias = [b.I for b in topology.bodies.values()]
    forward_traversal, backward_traversal = construct_traversal_orders(topology.tree)
    qdt0_idx = [0] + list(np.cumsum([j.nj for j in func_joints]))
    qdt1_idx = qdt0_idx  # Equal each other for now. Later could be different.

    states_tuple = namedtuple("States", topology.joints.keys())
    qdt0_names = states_tuple(
        **{
            name: create_namedtuple_inplace(
                name, dict(zip(joint.joint_type.coordinates_names, indices))
            )
            for (name, joint), indices in zip(
                topology.joints.items(),
                np.split(np.arange(0, topology.dof), qdt0_idx)[1:],
            )
        },
    )

    data = MultiBodyData(
        joints=func_joints,
        bodies_inertias=bodies_inertias,
        forward_traversal=forward_traversal,
        backward_traversal=backward_traversal,
        qdt0_idx=tuple(qdt0_idx),
        qdt1_idx=tuple(qdt1_idx),
        qdt0_names=qdt0_names,
    )

    return data


class Model(object):
    topology: MultiBodyTree
    forces_map: Forcesdict
    tree_data: MultiBodyData

    def __init__(self, topology: MultiBodyTree):
        self.topology = topology
        gravity = np.array([0, 0, -9.81, 0, 0, 0])
        self.forces_map = {
            b.name: {"global": {"gravity": b.I @ gravity}, "local": dict()}
            for b in self.topology.bodies.values()
        }

        self.tree_data = construct_multibody_data(topology)
        self.bodies_idx = {b: i for i, b in enumerate(self.topology.tree.nodes)}

    @property
    def n(self):
        return self.dof

    @property
    def dof(self):
        return self.topology.dof

    def get_body_kinematics(
        self, name: str, bodies_kinematics: list[BodyKinematics]
    ) -> BodyKinematics:
        return bodies_kinematics[self.bodies_idx[name]]

    def forward_kinematics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> tuple[BodyKinematics, JointKinematics]:
        coordinates = split_coordinates(self.tree_data.qdt0_idx, qdt0, qdt1, qdt2)

        bodies_kinematics, joints_kinematics = base_to_tip(
            self.tree_data.joints, coordinates, self.tree_data.forward_traversal
        )

        return bodies_kinematics, joints_kinematics

    def inverse_dynamics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> IDCallRes:
        forces = construct_system_forces_from_dict(self.forces_map)
        res = inverse_dynamics_call(self.tree_data, forces, qdt0, qdt1, qdt2)
        return res

    def forward_dynamics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, tau: np.ndarray
    ):
        forces = construct_system_forces_from_dict(self.forces_map)
        qdt2 = forward_dynamics_call(self.tree_data, forces, qdt0, qdt1, tau)
        return qdt2

    def ssode(self, t: float, ydt0: np.ndarray, forces_func: Callable, *args):
        qdt0, udt0 = ydt0.reshape(2, -1)
        tau, self.forces_map = forces_func(self, qdt0, udt0, np.zeros(udt0.shape), t)
        udt1 = self.forward_dynamics_pass(qdt0, udt0, tau)
        qdt1 = get_qdt1_from_udt0(self.tree_data, qdt0, udt0)
        return np.hstack([qdt1, udt1])


class HybridModel(Model):

    hybrid_dynamics_data: HybridDynamicsData

    def __init__(self, topology: MultiBodyTree, id_coordinates: np.ndarray):
        super().__init__(topology)
        self.id_coordinates = id_coordinates
        self.hybrid_dynamics_data = construct_hybrid_dynamics_data(
            self.tree_data, id_coordinates
        )

    @property
    def n(self):
        return self.dof - len(self.id_coordinates)

    def forward_dynamics_call(
        self,
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        qdt2_id: np.ndarray,
        tau_fd: np.ndarray,
    ) -> np.ndarray:
        system_forces = construct_system_forces_from_dict(self.forces_map)
        qdt2_fd = HybridDynamics().forward_dynamics_call(
            self.hybrid_dynamics_data,
            system_forces,
            qdt0,
            qdt1,
            qdt2_id,
            tau_fd,
        )
        return qdt2_fd

    def ssode(
        self,
        t: float,
        ydt0_fd: np.ndarray,
        u: dict[str, float],
        kinematic_actuation: Callable,
        dynamics_actuation: Callable,
    ) -> np.ndarray:

        n_fd = self.hybrid_dynamics_data.n_fd
        qdt0_fd, qdt1_fd = ydt0_fd.reshape(2, -1)
        qdt2_fd = np.zeros((n_fd,))

        # Evaluating the inverse-dynamics coordinates using the given
        # kinematic_actuation function.
        dt0_id, qdt1_id, qdt2_id = kinematic_actuation(self, t, ydt0_fd, u)

        # Constructing a new system-state, containing both inverse-dynamics and
        # forward-dynamics joints states, for routines which need the full system
        # state
        qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
            self, (qdt0_fd, qdt1_fd, qdt2_fd), (dt0_id, qdt1_id, qdt2_id)
        )
        ydt0 = np.hstack([qdt0, qdt1])

        # Evaluate applied external-forces using the provided callable
        tau, self.forces_map = dynamics_actuation(self, t, ydt0, u)

        # Extracting the forward-dynamics generalized forces vector.
        tau_fd = (self.hybrid_dynamics_data.permutation_matrix @ tau)[:n_fd]

        qdt2_fd = self.forward_dynamics_call(qdt0, qdt1, qdt2_id, tau_fd)

        # extracting qdt1 from udt0, assuming qdt1 != udt0
        qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
            self, (qdt0_fd, qdt1_fd, qdt2_fd), (dt0_id, qdt1_id, qdt2_id)
        )
        qdt1 = get_qdt1_from_udt0(self.tree_data, qdt0, qdt1)

        ydt1 = permute_state_coordinates(
            self.hybrid_dynamics_data.permutation_matrix,
            qdt1,
            qdt2,
            self.hybrid_dynamics_data.n_fd,
        )
        return ydt1


def permute_state_coordinates(
    permutation_matrix: np.ndarray, qdt0: np.ndarray, qdt1: np.ndarray, n_fd: int
):
    qdt0_permuted = permutation_matrix @ qdt0
    qdt1_permuted = permutation_matrix @ qdt1
    ydt0 = np.hstack([qdt0_permuted[:n_fd], qdt1_permuted[:n_fd]])
    return ydt0


def reconstruct_system_coordinates(
    model: HybridModel,
    dynamic_coordinates,
    kinematic_coordinates,
):
    qdt0_fd, qdt1_fd, qdt2_fd = dynamic_coordinates
    qdt0_id, qdt1_id, qdt2_id = kinematic_coordinates

    qdt0 = model.hybrid_dynamics_data.permutation_matrix.T @ np.array(
        [*qdt0_fd, *qdt0_id]
    )
    qdt1 = model.hybrid_dynamics_data.permutation_matrix.T @ np.array(
        [*qdt1_fd, *qdt1_id]
    )
    qdt2 = model.hybrid_dynamics_data.permutation_matrix.T @ np.array(
        [*qdt2_fd, *qdt2_id]
    )

    return qdt0, qdt1, qdt2


def _convert_body_forces_dict_to_list(forces_dict: dict[str, dict[str, np.ndarray]]):
    forces = list(forces_dict.values()) if len(forces_dict) > 0 else []
    return forces


def construct_system_forces_from_dict(external_forces: Forcesdict) -> SystemForces:

    system_forces = [
        (
            _convert_body_forces_dict_to_list(forces_dict["global"]),
            _convert_body_forces_dict_to_list(forces_dict["local"]),
        )
        for forces_dict in external_forces.values()
    ]
    return system_forces


def emulate_free_joint_v1(
    model: MultiBodyTree,
    predecessor: str,
    successor: str,
    succ_data: RigidBodyData,
    joint_config: JointConfigInputs,
) -> MultiBodyTree:
    dummy_body_data = RigidBodyData()

    x_axis_config = JointConfigInputs(
        pos=joint_config.pos, z_axis=np.array([1, 0, 0]), x_axis=None
    )
    y_axis_config = JointConfigInputs(
        pos=joint_config.pos, z_axis=np.array([0, 1, 0]), x_axis=None
    )
    z_axis_config = JointConfigInputs(
        pos=joint_config.pos, z_axis=np.array([0, 0, 1]), x_axis=None
    )

    model.add_joint(
        joint_name="x_trans",
        predecessor=predecessor,
        successor="d1",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=x_axis_config,
    )

    model.add_joint(
        joint_name="y_trans",
        predecessor="d1",
        successor="d2",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=y_axis_config,
    )

    model.add_joint(
        joint_name="z_trans",
        predecessor="d2",
        successor="d3",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=z_axis_config,
    )

    model.add_joint(
        joint_name="x_rotation",
        predecessor="d3",
        successor="d4",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=x_axis_config,
    )

    model.add_joint(
        joint_name="y_rotation",
        predecessor="d4",
        successor="d5",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=y_axis_config,
    )

    model.add_joint(
        joint_name="z_rotation",
        predecessor="d5",
        successor=successor,
        succ_data=succ_data,
        joint_type=RevoluteJoint,
        joint_data=z_axis_config,
    )

    return model


def emulate_free_joint_v2(
    model: MultiBodyTree,
    predecessor: str,
    successor: str,
    succ_data: RigidBodyData,
    joint_config: JointConfigInputs,
) -> MultiBodyTree:
    dummy_body_data = RigidBodyData()

    x_axis_config = JointConfigInputs(
        pos=joint_config.pos, z_axis=np.array([1, 0, 0]), x_axis=None
    )
    y_axis_config = JointConfigInputs(
        pos=joint_config.pos, z_axis=np.array([0, 1, 0]), x_axis=None
    )
    z_axis_config = JointConfigInputs(
        pos=joint_config.pos, z_axis=np.array([0, 0, 1]), x_axis=None
    )

    model.add_joint(
        joint_name="z_rotation",
        predecessor=predecessor,
        successor="d1",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=z_axis_config,
    )

    model.add_joint(
        joint_name="y_rotation",
        predecessor="d1",
        successor="d2",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=y_axis_config,
    )

    model.add_joint(
        joint_name="x_rotation",
        predecessor="d2",
        successor="d3",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=x_axis_config,
    )

    model.add_joint(
        joint_name="x_trans",
        predecessor="d3",
        successor="d4",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=x_axis_config,
    )

    model.add_joint(
        joint_name="y_trans",
        predecessor="d4",
        successor="d5",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=y_axis_config,
    )

    model.add_joint(
        joint_name="z_trans",
        predecessor="d5",
        successor=successor,
        succ_data=succ_data,
        joint_type=TranslationalJoint,
        joint_data=z_axis_config,
    )

    return model
