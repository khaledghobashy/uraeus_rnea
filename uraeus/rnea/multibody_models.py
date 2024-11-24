from typing import Callable, Optional, NamedTuple
import numpy as np

from uraeus.rnea.bodies import RigidBodyData, BodyKinematics
from uraeus.rnea.joints import (
    JointConfigInputs,
    JointKinematics,
    TranslationalJoint,
    RevoluteJoint,
)

from uraeus.rnea.algorithms import (
    split_coordinates,
    HybridDynamicsData,
    HybridDynamics,
    IDCallRes,
    inverse_dynamics_call,
    forward_dynamics_call,
    get_qdt1_from_udt0,
)

from uraeus.rnea.tree_traversals import base_to_tip, SystemForces
from uraeus.rnea.topologies import (
    MultiBodyTree,
    MultiBodyData,
    construct_multibody_data,
    construct_hybrid_dynamics_data,
)

Forcesdict = dict[str, dict[str, dict[str, np.ndarray]]]


class Model(object):
    topology: MultiBodyTree
    forces_map: Forcesdict
    tree_data: MultiBodyData
    is_hybrid: bool = False
    hybrid_idx: Optional[list[int]] = None
    hybrid_data: Optional[HybridDynamicsData] = None

    def __init__(self, topology: MultiBodyTree, hybrid_idx: Optional[list[int]] = None):
        self.topology = topology
        self.tree_data = construct_multibody_data(topology)

        gravity = np.array([0, 0, -9.81, 0, 0, 0])
        self.forces_map = {
            b.name: {"global": {"gravity": b.I @ gravity}, "local": dict()}
            for b in self.topology.bodies.values()
        }

        self.bodies_idx = {b: i for i, b in enumerate(self.topology.tree.nodes)}

        if hybrid_idx != None:
            self.is_hybrid = True
            self.hybrid_idx = hybrid_idx
            self.hybrid_data = self._construct_hybrid_dynamics_data(hybrid_idx)
            model_forward_equations = HybridModelForwardEquations(
                self.hybrid_data, self.forces_map
            )
            self._forward_equations = model_forward_equations
            self._forward_dynamics_call = model_forward_equations.forward_dynamics_call
            self._ssode = model_forward_equations.ssode
            # self._n = self.topology.dof - self.hybrid_data.n_fd
            self._n = self.hybrid_data.n_fd

        else:
            model_forward_equations = PureModelForwardEquations(
                self.tree_data, self.forces_map
            )
            self._forward_equations = model_forward_equations
            self._forward_dynamics_call = model_forward_equations.forward_dynamics_call
            self._ssode = model_forward_equations.ssode
            self._n = self.topology.dof

    @property
    def n(self):
        return self._n

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
            self.tree_data.joints, coordinates, self.tree_data.graph_data.base_to_tip
        )

        return bodies_kinematics, joints_kinematics

    def inverse_dynamics_pass(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ) -> IDCallRes:
        forces = construct_system_forces_from_dict(self.forces_map)
        res = inverse_dynamics_call(self.tree_data, forces, qdt0, qdt1, qdt2)
        return res

    def forward_dynamics_call(
        self,
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        tau: np.ndarray,
        qdt2_id: np.ndarray = None,
    ):
        return self._forward_dynamics_call(qdt0, qdt1, tau, qdt2_id)

    def ssode(
        self,
        t: float,
        ydt0: np.ndarray,
        u: dict[str, float],
        kinematic_actuation: Callable,
        dynamics_actuation: Callable,
    ):
        return self._ssode(self, t, ydt0, u, kinematic_actuation, dynamics_actuation)

    # def forward_dynamics_pass(
    #     self, qdt0: np.ndarray, qdt1: np.ndarray, tau: np.ndarray
    # ):
    #     forces = construct_system_forces_from_dict(self.forces_map)
    #     qdt2 = forward_dynamics_call(self.tree_data, forces, qdt0, qdt1, tau)
    #     return qdt2

    # def ssode(
    #     self,
    #     t: float,
    #     ydt0: np.ndarray,
    #     u: dict[str, float],
    #     kinematic_actuation: Callable,
    #     dynamics_actuation: Callable,
    # ):
    #     qdt0, udt0 = ydt0.reshape(2, -1)

    #     # Evaluate applied external-forces using the provided callable
    #     tau, self.forces_map = dynamics_actuation(self, t, ydt0, u)

    #     udt1 = self.forward_dynamics_pass(qdt0, udt0, tau)
    #     qdt1 = get_qdt1_from_udt0(self.tree_data, qdt0, udt0)
    #     return np.hstack([qdt1, udt1])

    def _construct_hybrid_dynamics_data(self, indices: list[int]) -> HybridDynamicsData:
        return construct_hybrid_dynamics_data(self.tree_data, indices)

    # def _forward_dynamics_pass_pure(
    #     self, qdt0: np.ndarray, qdt1: np.ndarray, tau: np.ndarray, _=None
    # ):
    #     forces = construct_system_forces_from_dict(self.forces_map)
    #     qdt2 = forward_dynamics_call(self.tree_data, forces, qdt0, qdt1, tau)
    #     return qdt2

    # def _ssode_pure(
    #     self,
    #     t: float,
    #     ydt0: np.ndarray,
    #     u: dict[str, float],
    #     kinematic_actuation: Callable,
    #     dynamics_actuation: Callable,
    # ):
    #     qdt0, udt0 = ydt0.reshape(2, -1)

    #     # Evaluate applied external-forces using the provided callable
    #     tau, self.forces_map = dynamics_actuation(self, t, ydt0, u)

    #     udt1 = self._forward_dynamics_pass_pure(qdt0, udt0, tau)
    #     qdt1 = get_qdt1_from_udt0(self.tree_data, qdt0, udt0)
    #     return np.hstack([qdt1, udt1])

    # def _forward_dynamics_pass_hybrid(
    #     self,
    #     qdt0: np.ndarray,
    #     qdt1: np.ndarray,
    #     tau_fd: np.ndarray,
    #     qdt2_id: np.ndarray,
    # ) -> np.ndarray:
    #     system_forces = construct_system_forces_from_dict(self.forces_map)
    #     qdt2_fd = HybridDynamics().forward_dynamics_call(
    #         self.hybrid_dynamics_data,
    #         system_forces,
    #         qdt0,
    #         qdt1,
    #         qdt2_id,
    #         tau_fd,
    #     )
    #     return qdt2_fd


class PureModelForwardEquations(NamedTuple):

    tree_data: MultiBodyData
    forces_map: Forcesdict

    def forward_dynamics_call(
        self, qdt0: np.ndarray, qdt1: np.ndarray, tau: np.ndarray, _=None
    ):
        forces = construct_system_forces_from_dict(self.forces_map)
        qdt2 = forward_dynamics_call(self.tree_data, forces, qdt0, qdt1, tau)
        return qdt2

    def ssode(
        self,
        model: Model,
        t: float,
        ydt0: np.ndarray,
        u: dict[str, float],
        kinematic_actuation: Callable,
        dynamics_actuation: Callable,
    ):
        qdt0, udt0 = ydt0.reshape(2, -1)

        # Evaluate applied external-forces using the provided callable
        tau, forces_map = dynamics_actuation(model, t, ydt0, u)
        self.forces_map.update(forces_map)

        udt1 = self.forward_dynamics_call(qdt0, udt0, tau)
        qdt1 = get_qdt1_from_udt0(self.tree_data, qdt0, udt0)
        return np.hstack([qdt1, udt1])


class HybridModelForwardEquations(NamedTuple):

    hybrid_data: HybridDynamicsData
    forces_map: Forcesdict

    def forward_dynamics_call(
        self,
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        tau_fd: np.ndarray,
        qdt2_id: np.ndarray,
    ) -> np.ndarray:
        system_forces = construct_system_forces_from_dict(self.forces_map)
        qdt2_fd = HybridDynamics().forward_dynamics_call(
            self.hybrid_data,
            system_forces,
            qdt0,
            qdt1,
            qdt2_id,
            tau_fd,
        )
        return qdt2_fd

    def ssode(
        self,
        model: Model,
        t: float,
        ydt0_fd: np.ndarray,
        u: dict[str, float],
        kinematic_actuation: Callable,
        dynamics_actuation: Callable,
    ) -> np.ndarray:

        tree_data = self.hybrid_data.tree_data
        n_fd = self.hybrid_data.n_fd
        permutation_matrix = self.hybrid_data.permutation_matrix

        qdt0_fd, qdt1_fd = ydt0_fd.reshape(2, -1)
        qdt2_fd = np.zeros((n_fd,))

        # Evaluating the inverse-dynamics coordinates using the given
        # kinematic_actuation function.
        dt0_id, qdt1_id, qdt2_id = kinematic_actuation(model, t, ydt0_fd, u)

        # Constructing a new system-state, containing both inverse-dynamics and
        # forward-dynamics joints states, for routines which need the full system
        # state
        qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
            permutation_matrix, (qdt0_fd, qdt1_fd, qdt2_fd), (dt0_id, qdt1_id, qdt2_id)
        )
        ydt0 = np.hstack([qdt0, qdt1])

        # Evaluate applied external-forces using the provided callable
        tau, forces_map = dynamics_actuation(model, t, ydt0, u)
        self.forces_map.update(forces_map)

        # Extracting the forward-dynamics generalized forces vector.
        tau_fd = (permutation_matrix @ tau)[:n_fd]

        qdt2_fd = self.forward_dynamics_call(qdt0, qdt1, tau_fd, qdt2_id)

        # extracting qdt1 from udt0, assuming qdt1 != udt0
        qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
            permutation_matrix, (qdt0_fd, qdt1_fd, qdt2_fd), (dt0_id, qdt1_id, qdt2_id)
        )
        qdt1 = get_qdt1_from_udt0(tree_data, qdt0, qdt1)

        ydt1 = permute_state_coordinates(permutation_matrix, qdt1, qdt2, n_fd)
        return ydt1


# class HybridModel(Model):

#     def __init__(self, topology: MultiBodyTree, hybrid_idx: np.ndarray):
#         self.topology = topology
#         self.tree_data = construct_multibody_data(topology)

#         self.is_hybrid = True
#         self.hybrid_idx = hybrid_idx
#         self.hybrid_data = construct_hybrid_dynamics_data(self.tree_data, hybrid_idx)

#     @property
#     def n(self):
#         return self.dof - len(self.id_coordinates)

#     def forward_dynamics_call(
#         self,
#         qdt0: np.ndarray,
#         qdt1: np.ndarray,
#         qdt2_id: np.ndarray,
#         tau_fd: np.ndarray,
#     ) -> np.ndarray:
#         system_forces = construct_system_forces_from_dict(self.forces_map)
#         qdt2_fd = HybridDynamics().forward_dynamics_call(
#             self.hybrid_dynamics_data,
#             system_forces,
#             qdt0,
#             qdt1,
#             qdt2_id,
#             tau_fd,
#         )
#         return qdt2_fd

#     def ssode(
#         self,
#         t: float,
#         ydt0_fd: np.ndarray,
#         u: dict[str, float],
#         kinematic_actuation: Callable,
#         dynamics_actuation: Callable,
#     ) -> np.ndarray:

#         n_fd = self.hybrid_dynamics_data.n_fd
#         qdt0_fd, qdt1_fd = ydt0_fd.reshape(2, -1)
#         qdt2_fd = np.zeros((n_fd,))

#         # Evaluating the inverse-dynamics coordinates using the given
#         # kinematic_actuation function.
#         dt0_id, qdt1_id, qdt2_id = kinematic_actuation(self, t, ydt0_fd, u)

#         # Constructing a new system-state, containing both inverse-dynamics and
#         # forward-dynamics joints states, for routines which need the full system
#         # state
#         qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
#             self, (qdt0_fd, qdt1_fd, qdt2_fd), (dt0_id, qdt1_id, qdt2_id)
#         )
#         ydt0 = np.hstack([qdt0, qdt1])

#         # Evaluate applied external-forces using the provided callable
#         tau, self.forces_map = dynamics_actuation(self, t, ydt0, u)

#         # Extracting the forward-dynamics generalized forces vector.
#         tau_fd = (self.hybrid_dynamics_data.permutation_matrix @ tau)[:n_fd]

#         qdt2_fd = self.forward_dynamics_call(qdt0, qdt1, qdt2_id, tau_fd)

#         # extracting qdt1 from udt0, assuming qdt1 != udt0
#         qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
#             self, (qdt0_fd, qdt1_fd, qdt2_fd), (dt0_id, qdt1_id, qdt2_id)
#         )
#         qdt1 = get_qdt1_from_udt0(self.tree_data, qdt0, qdt1)

#         ydt1 = permute_state_coordinates(
#             self.hybrid_dynamics_data.permutation_matrix,
#             qdt1,
#             qdt2,
#             self.hybrid_dynamics_data.n_fd,
#         )
#         return ydt1


def permute_state_coordinates(
    permutation_matrix: np.ndarray, qdt0: np.ndarray, qdt1: np.ndarray, n_fd: int
):
    qdt0_permuted = permutation_matrix @ qdt0
    qdt1_permuted = permutation_matrix @ qdt1
    ydt0 = np.hstack([qdt0_permuted[:n_fd], qdt1_permuted[:n_fd]])
    return ydt0


def reconstruct_system_coordinates(
    permutation_matrix: np.ndarray,
    dynamic_coordinates,
    kinematic_coordinates,
):
    qdt0_fd, qdt1_fd, qdt2_fd = dynamic_coordinates
    qdt0_id, qdt1_id, qdt2_id = kinematic_coordinates

    qdt0 = permutation_matrix.T @ np.array([*qdt0_fd, *qdt0_id])
    qdt1 = permutation_matrix.T @ np.array([*qdt1_fd, *qdt1_id])
    qdt2 = permutation_matrix.T @ np.array([*qdt2_fd, *qdt2_id])

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
