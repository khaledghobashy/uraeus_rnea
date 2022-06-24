import logging
import unittest
from typing import Tuple

import numpy as np
from numpy import assert_allclose
from scipy.misc import derivative

from multibody.bodies import RigidBodyData
from multibody.joints import RevoluteJoint, TranslationalJoint, JointData
from multibody.tree_traversals import (
    tip_to_base,
    base_to_tip,
    extract_state_vectors,
    extract_reaction_forces,
)

from multibody.algorithms import forward_dynamics_call


def ground_truth_forward_kinematics(
    qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
):

    l1 = 5
    l2 = 10

    theta1_dt0, theta2_dt0 = qdt0
    theta1_dt1, theta2_dt1 = qdt1
    theta1_dt2, theta2_dt2 = qdt2

    r1_hat = np.array([np.sin(theta1_dt0), -np.cos(theta1_dt0), 0])
    r2_hat = np.array(
        [np.sin(theta1_dt0 + theta2_dt0), -np.cos(theta1_dt0 + theta2_dt0), 0]
    )

    r1_hat_dt1 = theta1_dt1 * np.array([np.cos(theta1_dt0), np.sin(theta1_dt0), 0])
    r2_hat_dt1 = (theta1_dt1 + theta2_dt1) * np.array(
        [np.cos(theta1_dt0 + theta2_dt0), np.sin(theta1_dt0 + theta2_dt0), 0]
    )

    r1_hat_dt2 = theta1_dt2 * np.array(
        [np.cos(theta1_dt0), np.sin(theta1_dt0), 0]
    ) + theta1_dt1**2 * np.array([-np.sin(theta1_dt0), np.cos(theta1_dt0), 0])

    r2_hat_dt2 = (theta1_dt2 + theta2_dt2) * np.array(
        [np.cos(theta1_dt0 + theta2_dt0), np.sin(theta1_dt0 + theta2_dt0), 0]
    ) + (theta1_dt1 + theta2_dt1) ** 2 * np.array(
        [-np.sin(theta1_dt0 + theta2_dt0), np.cos(theta1_dt0 + theta2_dt0), 0]
    )

    r1_dt0 = (l1 / 2) * r1_hat
    r2_dt0 = (l1 * r1_hat) + ((l2 / 2) * r2_hat)

    r1_dt1 = (l1 / 2) * r1_hat_dt1
    r1_dt2 = (l1 / 2) * r1_hat_dt2

    r2_dt1 = (l1 * r1_hat_dt1) + (l2 / 2 * r2_hat_dt1)
    r2_dt2 = (l1 * r1_hat_dt2) + (l2 / 2 * r2_hat_dt2)

    link1_orientation = np.array([0, 0, np.arctan2(r1_hat[0], -r1_hat[1])])
    link1_pose = np.hstack([link1_orientation, r1_dt0])

    link1_ang_vel = np.array([0, 0, theta1_dt1])
    link1_spatial_vel = np.hstack([link1_ang_vel, r1_dt1])

    link1_ang_acc = np.array([0, 0, theta1_dt2])
    link1_spatial_acc = np.hstack([link1_ang_acc, r1_dt2])

    link2_orientation = np.array([0, 0, np.arctan2(r2_hat[0], -r2_hat[1])])
    link2_pose = np.hstack([link2_orientation, r2_dt0])

    link2_ang_vel = np.array([0, 0, theta1_dt1 + theta2_dt1])
    link2_spatial_vel = np.hstack([link2_ang_vel, r2_dt1])

    link2_ang_acc = np.array([0, 0, theta1_dt2 + theta2_dt2])
    link2_spatial_acc = np.hstack([link2_ang_acc, r2_dt2])

    pos_vector = np.hstack([link1_pose, link2_pose])
    vel_vector = np.hstack([link1_spatial_vel, link2_spatial_vel])
    acc_vector = np.hstack([link1_spatial_acc, link2_spatial_acc])

    return pos_vector, vel_vector, acc_vector


def ground_truth_inverse_dynamics(qdt0, qdt1, qdt2):

    l1 = 5
    l2 = 10

    m1 = 1
    m2 = 1

    g = 9.81

    theta1_dt0, theta2_dt0 = qdt0
    theta1_dt1, theta2_dt1 = qdt1
    theta1_dt2, theta2_dt2 = qdt2

    r1_hat = np.array([np.sin(theta1_dt0), -np.cos(theta1_dt0), 0])
    r2_hat = np.array(
        [np.sin(theta1_dt0 + theta2_dt0), -np.cos(theta1_dt0 + theta2_dt0), 0]
    )

    r1_dt0 = (l1 / 2) * r1_hat

    _, _, acc_vec = ground_truth_forward_kinematics(qdt0, qdt1, qdt2)

    link1_spatial_acc, link2_spatial_acc = np.split(acc_vec, 2)

    r1_dt2 = link1_spatial_acc[3:]
    r2_dt2 = link2_spatial_acc[3:]

    Fi2 = m2 * r2_dt2
    Fg2 = np.array([0, -m2 * g, 0])
    Fb = Fi2 - Fg2

    Ti2 = ((1 / 12) * m2 * l2**2) * (theta1_dt2 + theta2_dt2) * np.array([0, 0, 1])
    Tb = Ti2 - np.cross((l2 / 2 * r2_hat), -Fi2) - np.cross((l2 / 2 * r2_hat), Fg2)

    Fi1 = m1 * r1_dt2
    Fg1 = np.array([0, -m1 * g, 0])
    Fa = Fi1 + Fb - Fg1
    Ti1 = ((1 / 12) * m1 * l1**2) * (theta1_dt2) * np.array([0, 0, 1])
    Ta = (
        Tb
        + Ti1
        - np.cross(r1_dt0, -Fi1)
        - np.cross(r1_dt0, Fg1)
        - np.cross(l1 * r1_hat, -Fb)
    )

    reactions_at_a = np.hstack([[0, 0, 0], Fa])
    reactions_at_b = np.hstack([[0, 0, 0], Fb])
    reactions = np.hstack([reactions_at_a, reactions_at_b])

    generalized_forces = np.array([Ta[2], Tb[2]])

    return reactions, generalized_forces
