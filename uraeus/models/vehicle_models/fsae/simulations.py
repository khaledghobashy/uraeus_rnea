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
from uraeus.rnea.topologies import (
    Model,
    construct_system_forces_from_dict,
    HybridModel,
    reconstruct_system_coordinates,
)
from uraeus.models.vehicle_models.fsae.external_systems import (
    evaluate_motion_inputs,
    steering_function,
    evaluate_force_inputs,
)

# from .forces import evaluate_forces

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def test_eval_joints_kinematics(model: Model):
    q = np.zeros((model.topology.dof,))
    return eval_joints_kinematics(
        model.tree_data.joints, split_coordinates(model.tree_data.qdt0_idx, q, q, q)
    )


def static_equilibrium_func(
    x0: np.ndarray, model: Model, u0: dict[str, float], vx: float = 0
) -> np.ndarray:
    qdt0_fd = np.zeros((model.n,))
    qdt1_fd = np.zeros((model.n,))
    qdt2_fd = np.zeros((model.n,))

    qdt0_id, qdt1_id, qdt2_id = steering_function(u0["steering_input"])

    qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
        model, (qdt0_fd, qdt1_fd, qdt2_fd), (qdt0_id, qdt1_id, qdt2_id)
    )

    z_index = 2
    phi_index = 3
    theta_index = 4

    qdt0[z_index] = x0[0]  # z
    qdt0[phi_index] = x0[1]  # roll
    qdt0[theta_index] = x0[2]  # pitch
    qdt0[model.tree_data.qdt0_names.fr_susp.z] = x0[3]  # susp_1
    qdt0[model.tree_data.qdt0_names.fl_susp.z] = x0[4]  # susp_1
    qdt0[model.tree_data.qdt0_names.rr_susp.z] = x0[5]  # susp_1
    qdt0[model.tree_data.qdt0_names.rl_susp.z] = x0[6]  # susp_1

    qdt1[0] = vx

    ydt0 = np.hstack([qdt0, qdt1])

    coordinates = split_coordinates(model.tree_data.qdt0_idx, qdt0, qdt1, qdt2)
    _, joints_kin = base_to_tip(
        model.tree_data.joints, coordinates, model.tree_data.forward_traversal
    )

    tau, ext_forces = evaluate_force_inputs(model, 0, ydt0, u0)
    flattened_ext_forces = construct_system_forces_from_dict(ext_forces)
    ext = ext_forces_to_gen_forces(model.tree_data, joints_kin, flattened_ext_forces)

    res = ext + tau
    res = np.array(
        [
            res[z_index],  # z
            res[phi_index],  # roll
            res[theta_index],  # pitch
            res[model.tree_data.qdt0_names.fr_susp.z],
            res[model.tree_data.qdt0_names.fl_susp.z],
            res[model.tree_data.qdt0_names.rr_susp.z],
            res[model.tree_data.qdt0_names.rl_susp.z],
        ]
    )

    return res


def solve_for_static_equilibrium(
    model: Model, u0: dict[str, float], vx: float = 0
) -> np.ndarray:
    x0 = np.zeros((7,))
    x = fsolve(static_equilibrium_func, x0, args=(model, u0, vx))

    return x


def acceleration_sim(model: Model, u0: dict[str, float], v0: float = 1, tf: float = 10):
    x0 = solve_for_static_equilibrium(model, u0, v0)

    x_index = 0
    y_index = 1
    z_index = 2
    phi_index = 3
    theta_index = 4

    qdt0 = np.zeros((model.dof,))

    # qdt0[0] = 100
    # qdt0[1] = 50
    qdt0[5] = np.radians(45)

    qdt0[z_index] = x0[0]  # z
    qdt0[phi_index] = x0[1]  # roll
    qdt0[theta_index] = x0[2]  # pitch
    qdt0[model.tree_data.qdt0_names.fr_susp.z] = x0[3]  # susp_1
    qdt0[model.tree_data.qdt0_names.fl_susp.z] = x0[4]  # susp_1
    qdt0[model.tree_data.qdt0_names.rr_susp.z] = x0[5]  # susp_1
    qdt0[model.tree_data.qdt0_names.rl_susp.z] = x0[6]  # susp_1

    qdt1 = np.zeros_like(qdt0)
    qdt1[x_index] = v0

    qdt1[model.tree_data.qdt0_names.fr_wheel_rev.psi] = (
        v0 / model.vehicle_data.wheels_front.wc_height
    )
    qdt1[model.tree_data.qdt0_names.fl_wheel_rev.psi] = (
        v0 / model.vehicle_data.wheels_front.wc_height
    )
    qdt1[model.tree_data.qdt0_names.rr_wheel_rev.psi] = (
        v0 / model.vehicle_data.wheels_rear.wc_height
    )
    qdt1[model.tree_data.qdt0_names.rl_wheel_rev.psi] = (
        v0 / model.vehicle_data.wheels_rear.wc_height
    )

    if isinstance(model, HybridModel):
        ydt0 = permute_state_coordinates(
            model.hybrid_dynamics_data.permutation_matrix,
            qdt0,
            qdt1,
            model.hybrid_dynamics_data.n_fd,
        )
    else:
        ydt0 = np.hstack([qdt0, qdt1])

    b_kin, _ = model.forward_kinematics_pass(qdt0, qdt1, np.zeros((model.dof,)))
    ch_kin = model.get_body_kinematics("chassis", b_kin)
    print(ch_kin)

    res = model.inverse_dynamics_pass(qdt0, qdt1, np.zeros((model.dof,)))
    print("joints_forces = ", res.joints_forces)
    print("tau = ", res.tau)
    # exit()

    return integrator(model, ydt0, (0, tf))


def permute_state_coordinates(
    permutation_matrix: np.ndarray, qdt0: np.ndarray, qdt1: np.ndarray, n_fd: int
):
    qdt0_permuted = permutation_matrix @ qdt0
    qdt1_permuted = permutation_matrix @ qdt1
    ydt0 = np.hstack([qdt0_permuted[:n_fd], qdt1_permuted[:n_fd]])
    return ydt0


def standing_sim(model: Model):
    qdt0 = np.zeros((model.topology.dof,))
    qdt1 = np.zeros((model.topology.dof,))
    if isinstance(model, HybridModel):
        ydt0 = permute_state_coordinates(
            model.hybrid_dynamics_data.permutation_matrix,
            qdt0,
            qdt1,
            model.hybrid_dynamics_data.n_fd,
        )
    else:
        ydt0 = np.hstack([qdt0, qdt1])

    return integrator(model, ydt0, (0, 10))


class IntgRes(NamedTuple):
    time_history: np.ndarray
    qdt0_history: np.ndarray
    qdt1_history: np.ndarray
    qdt2_history: np.ndarray


def construct_inputs_dict(t: float):
    steering = min(0.5 * np.radians(10) * (t - 1), np.radians(25)) if t > 1 else 0
    # throttle = 1 if t > 1 else 0
    throttle = 0.4

    logger.debug("steering = %s", steering)
    logger.debug("throttle = %s", throttle)
    logger.debug("time = %s", t)

    inputs = {"throttle": throttle, "steering_input": steering}
    return inputs


def acc_ssode(model: Model):
    def ssode(t: float, ydt0: np.ndarray):
        return model.ssode(
            t,
            ydt0,
            construct_inputs_dict(t),
            evaluate_motion_inputs,
            evaluate_force_inputs,
        )

    return ssode


import jax


def integrator(
    model: Model,
    ydt0: np.ndarray,
    t_span: Tuple[float, float],
) -> IntgRes:
    t0, t_bound = t_span
    ssode = acc_ssode(model)
    jac = jax.jacfwd(ssode, 1)
    # stepper = integrate.RK45(ssode, t0, ydt0, t_bound)
    stepper = integrate.BDF(ssode, t0, ydt0, t_bound)

    time_history = []
    qdt0_history = []
    qdt1_history = []
    qdt2_history = []

    while stepper.status == "running":
        stepper.step()
        y = stepper.y
        ydt1 = ssode(stepper.t, y)
        qdt0_fd, qdt1_fd = y.reshape(2, -1)
        _, qdt2_fd = ydt1.reshape(2, -1)

        qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
            model,
            (qdt0_fd, qdt1_fd, qdt2_fd),
            steering_function(construct_inputs_dict(stepper.t)["steering_input"]),
        )

        log_output_to_consol(model.tree_data.qdt0_names, qdt0, qdt1, qdt2, stepper.t)

        # print(f"qdt0 = {qdt0}")
        # print(f"qdt1 = {qdt1}")
        # print(f"qdt2 = {qdt2}\n")

        time_history.append(stepper.t)
        qdt0_history.append(qdt0)
        qdt1_history.append(qdt1)
        qdt2_history.append(qdt2)

    res = IntgRes(
        time_history=np.array(time_history),
        qdt0_history=np.array(qdt0_history),
        qdt1_history=np.array(qdt1_history),
        qdt2_history=np.array(qdt2_history),
    )

    return res


def log_output_to_consol(
    coordinates_map, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray, t: float
):
    # states = {
    #     "time": t,
    #     "z": qdt0[coordinates_map.free_joint.z],
    #     "pitch": qdt0[coordinates_map.free_joint.theta],
    #     "acc_x": qdt2[coordinates_map.free_joint.x] / 9.81,
    #     "acc_y": qdt2[coordinates_map.free_joint.y] / 9.81,
    #     "velx": qdt1[coordinates_map.free_joint.x] * 3.6,
    #     "vely": qdt1[coordinates_map.free_joint.y] * 3.6,
    #     "yaw_dt0": qdt0[coordinates_map.free_joint.psi],
    #     "yaw_dt1": qdt1[coordinates_map.free_joint.psi],
    #     "yaw_dt2": qdt2[coordinates_map.free_joint.psi],
    # }

    states = {
        "time": t,
        "z": qdt0[2],
        "pitch": qdt0[4],
        "acc_x": qdt2[0] / 9.81,
        "acc_y": qdt2[1] / 9.81,
        "velx": qdt1[0] * 3.6,
        "vely": qdt1[1] * 3.6,
        "yaw_dt0": qdt0[5],
        "yaw_dt1": qdt1[5],
        "yaw_dt2": qdt2[5],
    }

    print("|".join([f"{i:>8}" for i in states.keys()]))
    print("|".join([f"{i: 08.5f}" for i in states.values()]))
    print("")
