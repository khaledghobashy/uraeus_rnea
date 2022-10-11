from functools import reduce
from typing import Iterable, List, NamedTuple, Tuple, Dict

import numpy as np

from uraeus.rnea.bodies import BodyKinematics, get_initialized_body_kinematics
from uraeus.rnea.joints import (
    AbstractJoint,
    FunctionalJoint,
    JointData,
    JointInstance,
    JointKinematics,
    JointVariables,
)
from uraeus.rnea.mobilizers import MobilizerForces
from uraeus.rnea.algorithms_operations import (
    eval_joint_force_components,
    evaluate_joint_force_p1,
    evaluate_successor_kinematics,
    evaluate_joint_forces,
)
from uraeus.rnea.spatial_algebra import (
    get_orientation_matrix_from_transformation,
    motion_to_force_transform,
)


def split_coordinates(
    joints: List[FunctionalJoint], qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
) -> Iterable[Iterable[np.ndarray]]:
    sections = np.cumsum(np.array([j.nj for j in joints]))
    coordinates = zip(*(np.split(qd, sections[:-1]) for qd in (qdt0, qdt1, qdt2)))
    return coordinates


def eval_joints_kinematics(
    joints: List[FunctionalJoint], coordinates: Iterable[Iterable[np.ndarray]]
):
    new_kin = [j.evaluate_kinematics(*coords) for j, coords in zip(joints, coordinates)]
    return new_kin


def base_to_tip(
    joints: List[FunctionalJoint],
    joints_coordinates: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dependency_order: List[Tuple[int, int, int]],
) -> Tuple[List[BodyKinematics], List[JointKinematics]]:

    joints_kinematics = eval_joints_kinematics(joints, joints_coordinates)
    bodies_kinematics = root_to_leaf(joints_kinematics, dependency_order)

    return (bodies_kinematics, joints_kinematics)


def tip_to_base(
    joints: List[FunctionalJoint],
    joints_kinematics: List[JointKinematics],
    dependency_order: List[Tuple[int, List[int]]],
    bodies_kinematics: List[BodyKinematics],
    bodies_inertias: List[np.ndarray],
    external_forces: List[List[np.ndarray]],
) -> Tuple[np.ndarray, ...]:

    # Evaluate inertia forces and external forces on bodies
    bodies_forces = eval_bodies_forces(
        bodies_kinematics, bodies_inertias, external_forces
    )

    # Extract joints' transforms from joints' kinematics
    joints_transforms = [j.X_PS for j in reversed(joints_kinematics)]
    joints_frames = [j.frames for j in joints]

    # Traverse the tree tip-to-base and Evaluate joints' forces
    joints_forces = leaf_to_root(bodies_forces, joints_transforms, dependency_order)

    sorted_bodies_kin = reversed(bodies_kinematics)
    # [bodies_kinematics[i[0]] for i in dependency_order]

    args = (
        joints_forces,
        reversed(joints_frames),
        reversed(joints_kinematics),
        sorted_bodies_kin,
    )

    force_instances = reversed(list(map(eval_joint_force_components, *args)))

    return list(force_instances)


def root_to_leaf(
    joints_kinematics: List[JointKinematics],
    dependency_order: List[Tuple[int, int, int]],
):
    bodies_kin = [
        get_initialized_body_kinematics(
            np.zeros((3,)),
            np.eye(3),
        )
    ]

    for successor_index, joint_index, predecessor_index in dependency_order:
        suc_kin = evaluate_successor_kinematics(
            bodies_kin[predecessor_index], joints_kinematics[joint_index]
        )
        bodies_kin.append(suc_kin)
    return bodies_kin


def leaf_to_root(
    bodies_forces: List[np.ndarray],
    joints_transforms: List[np.ndarray],
    dependency_order: List[Tuple[int, List[int]]],
) -> List[np.ndarray]:
    joints_transforms = np.array(
        list(map(motion_to_force_transform, joints_transforms))
    )
    forces = []
    for successor_index, sub_joints in dependency_order[:-1]:
        sub_joints_X_PS = [joints_transforms[i] for i in sub_joints]
        sub_joints_fi_S = [forces[i] for i in sub_joints]
        sub_joints_forces = sum(
            map(np.dot, sub_joints_X_PS, sub_joints_fi_S), np.zeros((6,))
        )
        joint_force = bodies_forces[successor_index] + sub_joints_forces
        forces.append(joint_force)

    # print("joint_forces = ", forces[-1])
    # print("bodies_forces = ", bodies_forces[1])
    return forces


def eval_bodies_forces(
    bodies_kinematics: List[BodyKinematics],
    bodies_inertias: List[np.ndarray],
    external_forces: List[List[np.ndarray]],
):
    arguments = (bodies_kinematics, bodies_inertias, external_forces)
    return list(map(evaluate_joint_force_p1, *arguments))


def extract_state_vectors(
    system_kinematics: List[BodyKinematics],
) -> Tuple[np.ndarray, ...]:

    pos_vector = np.hstack([b.p_GB for b in system_kinematics])
    vel_vector = np.hstack([b.v_G for b in system_kinematics])
    acc_vector = np.hstack([b.a_G for b in system_kinematics])

    return (pos_vector, vel_vector, acc_vector)


def extract_generalized_forces(joints_forces: List[MobilizerForces]) -> np.ndarray:

    tau_vector = np.hstack([j.tau for j in joints_forces])
    return tau_vector


def extract_reaction_forces(joints_forces: List[MobilizerForces]) -> np.ndarray:

    rct_vector = np.hstack([j.fc_G for j in joints_forces])
    return rct_vector
