from typing import NamedTuple

import numpy as np

from uraeus.rnea.spatial_algebra import (
    spatial_motion_transformation,
    spatial_transform_transpose,
    get_euler_angles_from_rotation,
)


class RigidBodyData(NamedTuple):

    location: np.ndarray = np.array([0.0, 0.0, 0.0])
    orientation: np.ndarray = np.eye(3)
    mass: float = 0.0
    inertia_tensor: np.ndarray = np.zeros((3, 3))


class BodyKinematics(NamedTuple):

    X_BG: np.ndarray
    X_GB: np.ndarray
    p_GB: np.ndarray
    R_GB: np.ndarray
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
                np.hstack([body_data.inertia_tensor, np.zeros((3, 3))]),
                np.hstack([np.zeros((3, 3)), body_data.mass * np.eye(3)]),
            ]
        )

        self.kinematics = get_initialized_body_kinematics(
            body_data.location, body_data.orientation
        )


def get_initialized_body_kinematics(
    location: np.ndarray, orientation: np.ndarray
) -> BodyKinematics:

    r = orientation.T @ -location
    X_GB = spatial_motion_transformation(orientation, r)

    X_BG = spatial_transform_transpose(X_GB)

    e_GB = get_euler_angles_from_rotation(orientation)

    p_GB = np.hstack([e_GB, location])

    zeros = np.zeros((6,))

    kin = BodyKinematics(
        X_BG,
        X_GB,
        p_GB,
        orientation,
        zeros,
        zeros,
        zeros,
        zeros,
    )

    return kin
