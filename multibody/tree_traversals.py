from functools import reduce
from typing import Iterable, List, NamedTuple, Tuple, Dict

import numpy as np

from multibody.bodies import BodyKinematics
from multibody.joints import AbstractJoint, JointKinematics, JointVariables
from multibody.mobilizers import MobilizerForces
from multibody.algorithms_operations import (
    evaluate_successor_kinematics,
    evaluate_joint_forces,
)


def split_coordinates(
    joints: List[AbstractJoint], qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
) -> Iterable[Iterable[np.ndarray]]:
    sections = np.cumsum([j.nj for j in joints])
    coordinates = zip(*(np.split(qd, sections[:-1]) for qd in (qdt0, qdt1, qdt2)))
    return coordinates


def eval_joints_kinematics(
    joints: List[AbstractJoint], coordinates: Iterable[Iterable[np.ndarray]]
):
    new_kin = [j.evaluate_kinematics(*coords) for j, coords in zip(joints, coordinates)]
    return new_kin


def base_to_tip(
    joints: List[AbstractJoint], qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
) -> Tuple[List[BodyKinematics], List[JointKinematics]]:

    bodies_kinematics = {joints[0].predecessor.name: joints[0].predecessor.kinematics}

    joints_coordinates = split_coordinates(joints, qdt0, qdt1, qdt2)
    joints_kinematics = eval_joints_kinematics(joints, joints_coordinates)

    for joint, joint_kin in zip(joints, joints_kinematics):
        predecessor = joint.predecessor.name
        successor = joint.successor.name

        bodies_kinematics[successor] = evaluate_successor_kinematics(
            bodies_kinematics[predecessor], joint_kin
        )

    return (list(bodies_kinematics.values())[1:], joints_kinematics)


def tip_to_base(
    joints: List[AbstractJoint],
    joints_kinematics: List[JointKinematics],
    adj_joint: Dict[str, List[AbstractJoint]],
    forces_map: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, ...]:

    joints_forces = {}

    iterable = reversed(zip(joints, joints_kinematics))

    for joint, joint_kin in iterable:

        successor_name = joint.successor.name
        successor_I = joint.successor.I
        successor_kin = joint.successor.kinematics

        out_joints_variables = [
            JointVariables(j.kinematics, joints_forces[j.name])
            for j in adj_joint[successor_name]
        ]

        external_forces = list(forces_map[successor_name].values())

        forces = evaluate_joint_forces(
            successor_I,
            successor_kin,
            joint_kin,
            joint.frames,
            out_joints_variables,
            external_forces,
        )

        joints_forces[joint.name] = forces

    return list(reversed(joints_forces.values()))


def extract_state_vectors(
    system_kinematics: List[BodyKinematics],
) -> Tuple[np.ndarray, ...]:

    pos_vector = np.hstack(b.p_GB for b in system_kinematics)
    vel_vector = np.hstack(b.v_G for b in system_kinematics)
    acc_vector = np.hstack(b.a_G for b in system_kinematics)

    return (pos_vector, vel_vector, acc_vector)


def extract_generalized_forces(joints_forces: List[MobilizerForces]) -> np.ndarray:

    tau_vector = np.hstack(j.tau for j in joints_forces)
    return tau_vector


def extract_reaction_forces(joints_forces: List[MobilizerForces]) -> np.ndarray:

    rct_vector = np.hstack(j.fc_G for j in joints_forces)
    return rct_vector
