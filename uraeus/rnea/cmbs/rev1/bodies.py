from typing import NamedTuple

import numpy as np


class RigidBodyData(NamedTuple):
    location: np.ndarray = np.array([0.0, 0.0, 0.0])
    orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
    mass: float = 0.0
    inertia_tensor: np.ndarray = np.zeros((3, 3))
