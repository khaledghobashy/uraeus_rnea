import logging
from typing import Callable, Tuple, NamedTuple

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve

from uraeus.rnea.tree_traversals import base_to_tip, eval_joints_kinematics
from uraeus.rnea.algorithms import (
    split_coordinates,
    ext_forces_to_gen_forces,
)
from uraeus.rnea.topologies import Model, construct_system_forces_from_dict

from .forces import evaluate_forces

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_eval_joints_kinematics(model: Model):
    q = np.zeros((model.topology.dof,))
    return eval_joints_kinematics(
        model.tree_data.joints, split_coordinates(model.tree_data.qdt0_idx, q, q, q)
    )


def equilibrium_func(x0: np.ndarray, model: Model, vx: float = 0) -> np.ndarray:
    qdt0 = np.array(
        [
            0,  # x
            0,  # y
            x0[0],  # z
            x0[1],  # roll
            x0[2],  # pitch
            0,  # yaw
            x0[3],
            x0[4],
            x0[5],
            x0[6],
            0,
            0,
            0,
            0,
        ]
    )

    qdt1 = np.zeros_like(qdt0)
    qdt1[0] = vx
    qdt2 = np.zeros_like(qdt1)

    coordinates = split_coordinates(model.tree_data.qdt0_idx, qdt0, qdt1, qdt2)
    _, joints_kin = base_to_tip(
        model.tree_data.joints, coordinates, model.tree_data.forward_traversal
    )

    tau, ext_forces = evaluate_forces(model, qdt0, qdt1, qdt2)

    flattened_ext_forces = construct_system_forces_from_dict(ext_forces)
    ext = ext_forces_to_gen_forces(model.tree_data, joints_kin, flattened_ext_forces)

    res = ext + tau
    res = np.array(
        [
            res[2],  # z
            res[3],  # roll
            res[4],  # pitch
            res[6],
            res[7],
            res[8],
            res[9],
        ]
    )

    return res


def static_equilibrium(model: Model, vx: float = 0) -> np.ndarray:
    x0 = np.zeros((7,))
    x = fsolve(equilibrium_func, x0, args=(model, vx))

    return x


def acceleration_sim(model: Model, v0: float = 1, tf: float = 10):
    x0 = static_equilibrium(model, v0)

    qdt0 = np.array(
        [
            0,
            0,
            x0[0],
            x0[1],
            x0[2],
            0,
            x0[3],
            x0[4],
            x0[5],
            x0[6],
            0,
            0,
            0,
            0,
        ]
    )

    qdt1 = np.zeros((model.topology.dof,))
    qdt1[0] = v0
    qdt1[10] = -v0 / model.vehicle_data.wheels_front.wc_height
    qdt1[11] = -v0 / model.vehicle_data.wheels_front.wc_height
    qdt1[12] = -v0 / model.vehicle_data.wheels_rear.wc_height
    qdt1[13] = -v0 / model.vehicle_data.wheels_rear.wc_height

    ydt0 = np.hstack([qdt0, qdt1])

    return integrator(model, ydt0, (0, tf))


def standing_sim(model: Model):
    qdt0 = np.zeros((model.topology.dof,))
    qdt1 = np.zeros((model.topology.dof,))
    ydt0 = np.hstack([qdt0, qdt1])

    return integrator(model, ydt0, (0, 10))


class IntgRes(NamedTuple):
    time_history: np.ndarray
    qdt0_history: np.ndarray
    qdt1_history: np.ndarray
    qdt2_history: np.ndarray


def acc_ssode(model: Model):
    def ssode(t: float, ydt0: np.ndarray):
        return model.ssode(t, ydt0, evaluate_forces)

    return ssode


def test_ssode(model: Model, t: float, qdt0: np.ndarray, qdt1: np.ndarray):
    ydt0 = np.hstack([qdt0, qdt1])
    return model.ssode(t, ydt0, evaluate_forces)


def integrator(
    model: Model,
    ydt0: np.ndarray,
    t_span: Tuple[float, float],
) -> IntgRes:
    t0, t_bound = t_span
    ssode = acc_ssode(model)
    stepper = integrate.BDF(ssode, t0, ydt0, t_bound)

    time_history = []
    qdt0_history = []
    qdt1_history = []
    qdt2_history = []

    coordinates_map = {
        "x": 0,
        "y": 1,
        "z": 2,
        "roll": 3,
        "pitch": 4,
        "yaw": 5,
    }

    while stepper.status == "running":
        y = stepper.y
        ydt1 = ssode(stepper.t, y)
        qdt0, qdt1 = y.reshape(2, -1)
        _, qdt2 = ydt1.reshape(2, -1)

        time_history.append(stepper.t)
        qdt0_history.append(qdt0)
        qdt1_history.append(qdt1)
        qdt2_history.append(qdt2)

        stepper.step()

        loged = {
            "time": stepper.t,
            "z": y[coordinates_map["z"]],
            "pitch": y[coordinates_map["pitch"]],
            "acc_x": ydt1[coordinates_map["x"] + 14] / 9.81,
            "acc_y": ydt1[coordinates_map["y"] + 14] / 9.81,
            "velx": ydt1[coordinates_map["x"]] * 3.6,
            "vely": ydt1[coordinates_map["y"]] * 3.6,
            "yaw_dt0": y[coordinates_map["yaw"]],
            "yaw_dt1": ydt1[coordinates_map["yaw"]],
            "yaw_dt2": ydt1[coordinates_map["yaw"] + 14],
        }

        print("|".join([f"{i:>8}" for i in loged.keys()]))
        print("|".join([f"{i: 08.5f}" for i in loged.values()]))
        print("")

    res = IntgRes(
        time_history=np.array(time_history),
        qdt0_history=np.array(qdt0_history),
        qdt1_history=np.array(qdt1_history),
        qdt2_history=np.array(qdt2_history),
    )

    return res
