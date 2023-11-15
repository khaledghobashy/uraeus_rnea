from typing import Callable, Dict, List, NamedTuple, Tuple

import numpy as np

from uraeus.rnea.algorithms import (
    forward_dynamics_call,
    split_coordinates,
)
from uraeus.rnea.bodies import BodyKinematics
from uraeus.rnea.joints import (
    JointKinematics,
)
from uraeus.rnea.tree_traversals import base_to_tip
from uraeus.rnea.topologies import (
    MultiBodyTree,
    MultiBodyData,
    construct_multibodydata,
)

from .topology import VehicleData


class Model(object):
    topology: MultiBodyTree
    vehicle_data: VehicleData
    forces_map: Dict[str, Dict[str, np.ndarray]]
    tree_data: MultiBodyData

    def __init__(self, topology: MultiBodyTree, vehicle_data: VehicleData):
        self.topology = topology
        self.vehicle_data = vehicle_data
        gravity = np.array([0, 0, 0, 0, 0, -9.81])
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

    def forward_dynamics_pass(
        self,
        qdt0: np.ndarray,
        qdt1: np.ndarray,
        tau: np.ndarray,
        forces_map: Dict[str, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        external_forces = [list(i.values()) for i in forces_map.values()]
        qdt2 = forward_dynamics_call(self.tree_data, external_forces, qdt0, qdt1, tau)
        return qdt2

    def ssode(
        self,
        t: float,
        ydt0: np.ndarray,
        forces_func: Callable,
    ):
        qdt0, qdt1 = ydt0.reshape(2, -1)

        bodies_kin, _ = self.forward_kinematics_pass(qdt0, qdt1, np.zeros_like(qdt1))

        gen_forces, ext_forces = forces_func(
            bodies_kin,
            self.bodies_idx,
            self.forces_map,
            qdt0,
            qdt1,
        )

        qdt2 = self.forward_dynamics_pass(qdt0, qdt1, gen_forces, ext_forces)
        return np.hstack([qdt1, qdt2])
