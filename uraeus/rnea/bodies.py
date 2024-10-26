from typing import NamedTuple

import numpy as np

from uraeus.rnea.spatial_algebra import SpatialPose, quaternion_inverse, SpatialInertia


class RigidBodyData(NamedTuple):

    location: np.ndarray = np.array([0.0, 0.0, 0.0])
    orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
    mass: float = 0.0
    inertia_tensor: np.ndarray = np.zeros((3, 3))
    spatial_inertia: SpatialInertia = SpatialInertia.Identity()


class BodyKinematics(NamedTuple):

    p_GB: SpatialPose
    p_BG: SpatialPose
    v_B: np.ndarray
    a_B: np.ndarray
    v_G: np.ndarray
    a_G: np.ndarray


class RigidBody(object):

    body_data: RigidBodyData
    kinematics: BodyKinematics

    def __init__(self, name: str, body_data: RigidBodyData):

        self.name = name
        self.body_data = body_data
        self.I = np.vstack(
            [
                np.hstack([body_data.mass * np.eye(3), np.zeros((3, 3))]),
                np.hstack([np.zeros((3, 3)), body_data.inertia_tensor]),
            ]
        )

        self.kinematics = get_initialized_body_kinematics(
            body_data.location, body_data.orientation
        )


def get_initialized_body_kinematics(r: np.ndarray, q_BG: np.ndarray) -> BodyKinematics:

    # r: position vector of body relative to global-origin expressed in global frame
    # q_BG transforms from body from to global frame

    p_GB = SpatialPose(r, quaternion_inverse(q_BG))
    p_BG = p_GB.inv()

    zeros = np.zeros((6,))

    kin = BodyKinematics(
        p_GB,
        p_BG,
        zeros,
        zeros,
        zeros,
        zeros,
    )

    return kin
