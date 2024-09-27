from typing import NamedTuple

import numpy as np

from uraeus.rnea.spatial_algebra import express_screw
from uraeus.rnea.topologies import Model
from uraeus.models.vehicle_models.tire_models.fiala_model import (
    FialaTireModel,
    FialaTireParameters,
)
from uraeus.models.vehicle_models.tire_models.brush_model import (
    BrushModelParameters,
    BrushTireModel,
)
from .forces_elements import (
    AeroForce,
    SimpleElectricMotor,
)


def construct_sping_func(stiffness: float, damping: float, preload: float):
    def spring_force(x: float, v: float):
        return (-stiffness * x) + (-damping * v) + -preload

    return spring_force


def esitmate_stiffness_damping(mass: float, frequency: float, damping_ratio: float):
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
        Cfy=1500,
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

    fr_spring = construct_sping_func(
        *esitmate_stiffness_damping(250 * 0.5 * 0.5, 2.8, 0.8)
    )
    fl_spring = construct_sping_func(
        *esitmate_stiffness_damping(250 * 0.5 * 0.5, 2.8, 0.8)
    )
    rr_spring = construct_sping_func(
        *esitmate_stiffness_damping(250 * 0.5 * 0.5, 2.9, 0.8)
    )
    rl_spring = construct_sping_func(
        *esitmate_stiffness_damping(250 * 0.5 * 0.5, 2.9, 0.8)
    )


def evaluate_forces(
    model: Model,
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    qdt2: np.ndarray,
    t: float = 0,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    tau = np.zeros_like(qdt0)

    forces_map = model.forces_map

    bodies_kinematics, _ = model.forward_kinematics_pass(qdt0, qdt1, qdt2)
    chassis_kin = model.get_body_kinematics("chassis", bodies_kinematics)

    fr_carier_kin = model.get_body_kinematics("fr_carier", bodies_kinematics)
    fl_carier_kin = model.get_body_kinematics("fl_carier", bodies_kinematics)
    rr_carier_kin = model.get_body_kinematics("rr_carier", bodies_kinematics)
    rl_carier_kin = model.get_body_kinematics("rl_carier", bodies_kinematics)

    fr_wheel_kin = model.get_body_kinematics("fr_wheel", bodies_kinematics)
    fl_wheel_kin = model.get_body_kinematics("fl_wheel", bodies_kinematics)
    rr_wheel_kin = model.get_body_kinematics("rr_wheel", bodies_kinematics)
    rl_wheel_kin = model.get_body_kinematics("rl_wheel", bodies_kinematics)

    fr_tire_force = ForcesElements.fr_tire(fr_wheel_kin, t)
    fl_tire_force = ForcesElements.fl_tire(fl_wheel_kin, t)
    rr_tire_force = ForcesElements.rr_tire(rr_wheel_kin, t)
    rl_tire_force = ForcesElements.rl_tire(rl_wheel_kin, t)

    forces_map["chassis"]["local"]["aero"] = ForcesElements.aero_force(chassis_kin)

    # forces_map["fr_wheel"]["local"]["tire"] = fr_tire_force
    # forces_map["fl_wheel"]["local"]["tire"] = fl_tire_force
    # forces_map["rr_wheel"]["local"]["tire"] = rr_tire_force
    # forces_map["rl_wheel"]["local"]["tire"] = rl_tire_force

    # forces_map["fr_wheel"]["global"]["tire"] = express_screw(
    #     fr_carier_kin.p_BG, fr_tire_force
    # )
    # forces_map["fl_wheel"]["global"]["tire"] = express_screw(
    #     fl_carier_kin.p_BG, fl_tire_force
    # )
    # forces_map["rr_wheel"]["global"]["tire"] = express_screw(
    #     rr_carier_kin.p_BG, rr_tire_force
    # )
    # forces_map["rl_wheel"]["global"]["tire"] = express_screw(
    #     rl_carier_kin.p_BG, rl_tire_force
    # )

    forces_map["fr_wheel"]["global"]["tire"] = fr_tire_force
    forces_map["fl_wheel"]["global"]["tire"] = fl_tire_force
    forces_map["rr_wheel"]["global"]["tire"] = rr_tire_force
    forces_map["rl_wheel"]["global"]["tire"] = rl_tire_force

    # thrust = 250 * 9.81 if t > 1 else 0
    # forces_map["chassis"]["thrust"] = np.array([thrust, 0, 0, 0, 0, 0])

    # rr_thrust = 250 * 0.5 * 9.81 if t > 1 else 0
    # rl_thrust = 250 * 0.5 * 9.81 if t > 1 else 0
    # forces_map["rr_carier"]["thrust"] = express_screw(
    #     rr_carier_kin.p_BG, np.array([rr_thrust, 0, 0, 0, 0, 0])
    # )
    # forces_map["rl_carier"]["thrust"] = express_screw(
    #     rl_carier_kin.p_BG, np.array([rl_thrust, 0, 0, 0, 0, 0])
    # )

    # fr_tire_force = Forces.fr_tire.Fz(fr_wheel_kin)
    # fl_tire_force = Forces.fl_tire.Fz(fl_wheel_kin)
    # rr_tire_force = Forces.rr_tire.Fz(rr_wheel_kin)
    # rl_tire_force = Forces.rl_tire.Fz(rl_wheel_kin)

    # forces_map["fr_carier"]["tire"] = fr_tire_force
    # forces_map["fl_carier"]["tire"] = fl_tire_force
    # forces_map["rr_carier"]["tire"] = rr_tire_force
    # forces_map["rl_carier"]["tire"] = rl_tire_force

    tau[6] = ForcesElements.fr_spring(qdt0[6], qdt1[6])
    tau[7] = ForcesElements.fl_spring(qdt0[7], qdt1[7])
    tau[8] = ForcesElements.rr_spring(qdt0[8], qdt1[8])
    tau[9] = ForcesElements.rl_spring(qdt0[9], qdt1[9])

    throttle = 1

    # fr_torque = Forces.motor(fr_wheel_kin, throttle)
    # fl_torque = Forces.motor(fl_wheel_kin, throttle)
    # tau[10] = fr_torque if t > 1 else 0
    # tau[11] = fl_torque if t > 1 else 0

    rr_torque = ForcesElements.motor(rr_wheel_kin, throttle)
    rl_torque = ForcesElements.motor(rl_wheel_kin, throttle)
    tau[12] = rr_torque if t > 1 else 0
    tau[13] = rl_torque if t > 1 else 0

    # torque = 200 if t > 1 else 0
    # tau[12] = torque
    # tau[13] = torque

    return tau, forces_map
