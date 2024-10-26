import logging
from typing import NamedTuple

import numpy as np

from uraeus.rnea.multibody_models import Model, reconstruct_system_coordinates
from uraeus.models.vehicle_models.tire_models.brush_model import (
    BrushModelParameters,
    BrushTireModel,
)
from .forces_elements import AeroForce, SimpleElectricMotor

from uraeus.utils.logging import construct_logger

logger = construct_logger(__name__, logging.DEBUG)


def construct_spring_func(stiffness: float, damping: float, preload: float):
    def spring_force(x: float, v: float):
        return (-stiffness * x) + (-damping * v) + -preload

    return spring_force


def estimate_stiffness_damping(mass: float, frequency: float, damping_ratio: float):
    stiffness = (2 * np.pi * frequency) ** 2 * mass
    damping = damping_ratio * (2 * np.sqrt(stiffness * mass))
    return stiffness, damping, 0.9 * mass * 9.81


class ForcesElements(NamedTuple):
    aero_force = AeroForce("aero", 0.3, 0.5, 1, np.array([0, 0, 0]))

    _tire_parameters = BrushModelParameters(
        mu=1.3,
        Cfk=50e3,
        Cfa=50e3,
        Cfx=150e3,
        Cfy=150e3,
        a=0.10,
        unloaded_radius=0.245,
        kz=150e3,
        cz=15e3,
        kv_low=0,
    )
    fr_tire = BrushTireModel(_tire_parameters)
    fl_tire = BrushTireModel(_tire_parameters)
    rr_tire = BrushTireModel(_tire_parameters)
    rl_tire = BrushTireModel(_tire_parameters)

    motor = SimpleElectricMotor(
        name="rl_motor",
        min_rpm=0,
        max_rpm=7000,
        min_torque=0,
        max_torque=200,
        max_power=40e3,
        reduction_ratio=1,
    )

    fr_spring = construct_spring_func(
        *estimate_stiffness_damping(250 * 0.5 * 0.5, 2.8, 0.8)
    )
    fl_spring = construct_spring_func(
        *estimate_stiffness_damping(250 * 0.5 * 0.5, 2.8, 0.8)
    )
    rr_spring = construct_spring_func(
        *estimate_stiffness_damping(250 * 0.5 * 0.5, 2.9, 0.8)
    )
    rl_spring = construct_spring_func(
        *estimate_stiffness_damping(250 * 0.5 * 0.5, 2.9, 0.8)
    )


def steering_function(steering_angle: float):
    qdt0 = np.array([steering_angle, steering_angle])
    qdt1 = np.array([0, 0])
    qdt2 = np.array([0, 0])
    return qdt0, qdt1, qdt2


def evaluate_motion_inputs(
    model: Model, t: float, ydt0: np.ndarray, u: dict[str, float]
):
    return steering_function(u["steering_input"])


def evaluate_force_inputs(
    model: Model,
    t: float,
    ydt0: np.ndarray,
    u: dict[str, float],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:

    qdt0, qdt1 = ydt0.reshape(2, -1)
    qdt2 = 0 * qdt1

    tau = np.zeros_like(qdt0)

    forces_map = model.forces_map

    bodies_kinematics, _ = model.forward_kinematics_pass(qdt0, qdt1, qdt2)
    chassis_kin = model.get_body_kinematics("chassis", bodies_kinematics)

    # logger.debug("chassis_kin = %s", chassis_kin)

    fr_wheel_kin = model.get_body_kinematics("fr_wheel", bodies_kinematics)
    fl_wheel_kin = model.get_body_kinematics("fl_wheel", bodies_kinematics)
    rr_wheel_kin = model.get_body_kinematics("rr_wheel", bodies_kinematics)
    rl_wheel_kin = model.get_body_kinematics("rl_wheel", bodies_kinematics)

    fr_tire_force = ForcesElements.fr_tire(fr_wheel_kin, t)
    fl_tire_force = ForcesElements.fl_tire(fl_wheel_kin, t)
    rr_tire_force = ForcesElements.rr_tire(rr_wheel_kin, t)
    rl_tire_force = ForcesElements.rl_tire(rl_wheel_kin, t)

    forces_map["chassis"]["local"]["aero"] = ForcesElements.aero_force(chassis_kin)
    # forces_map["chassis"]["local"]["thrust"] = np.array(
    #     [0.5 * 9.81 * 250, 0, 0, 0, 0, 0]
    # )

    forces_map["fr_wheel"]["global"]["tire"] = fr_tire_force
    forces_map["fl_wheel"]["global"]["tire"] = fl_tire_force
    forces_map["rr_wheel"]["global"]["tire"] = rr_tire_force
    forces_map["rl_wheel"]["global"]["tire"] = rl_tire_force

    tau[model.tree_data.qdt0_names.fr_susp.z] = ForcesElements.fr_spring(
        qdt0[model.tree_data.qdt0_names.fr_susp.z],
        qdt1[model.tree_data.qdt0_names.fr_susp.z],
    )
    tau[model.tree_data.qdt0_names.fl_susp.z] = ForcesElements.fl_spring(
        qdt0[model.tree_data.qdt0_names.fl_susp.z],
        qdt1[model.tree_data.qdt0_names.fl_susp.z],
    )
    tau[model.tree_data.qdt0_names.rr_susp.z] = ForcesElements.rr_spring(
        qdt0[model.tree_data.qdt0_names.rr_susp.z],
        qdt1[model.tree_data.qdt0_names.rr_susp.z],
    )
    tau[model.tree_data.qdt0_names.rl_susp.z] = ForcesElements.rl_spring(
        qdt0[model.tree_data.qdt0_names.rl_susp.z],
        qdt1[model.tree_data.qdt0_names.rl_susp.z],
    )

    throttle = u["throttle"]
    # logger.debug(f"throttle = {throttle}")

    # forces_map["rr_carier"]["local"]["thrust"] = throttle * np.array(
    #     [0.25 * 9.81 * 250, 0, 0, 0, 0, 0]
    # )
    # forces_map["rl_carier"]["local"]["thrust"] = throttle * np.array(
    #     [0.25 * 9.81 * 250, 0, 0, 0, 0, 0]
    # )

    rr_torque = ForcesElements.motor(rr_wheel_kin, throttle)
    rl_torque = ForcesElements.motor(rl_wheel_kin, throttle)
    # tau[model.tree_data.qdt0_names.rr_wheel_rev.psi] = rr_torque
    # tau[model.tree_data.qdt0_names.rl_wheel_rev.psi] = rl_torque
    tau[-2] = rr_torque
    tau[-1] = rl_torque

    # logger.debug(f"tau = {tau}")

    return tau, forces_map
