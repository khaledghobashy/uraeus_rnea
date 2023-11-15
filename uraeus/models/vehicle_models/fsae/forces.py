from typing import NamedTuple, Dict, List, Tuple

import numpy as np

from uraeus.rnea.bodies import BodyKinematics
from uraeus.models.vehicle_models.utils.forces_elements import (
    AeroForce,
    SimpleElectricMotor,
    TireMF52,
)


def construct_sping_func(stiffness: float, damping: float, preload: float):
    def spring_force(x: float, v: float):
        return (stiffness * x) + (damping * v) + preload

    return spring_force


def esitmate_stiffness_damping(mass: float, frequency: float, damping_ratio: float):
    stiffness = (2 * np.pi * frequency) ** 2 * mass
    damping = damping_ratio * (2 * np.sqrt(stiffness * mass))
    return stiffness, damping, 0 * mass * 9.81


class Forces(NamedTuple):
    aero_force = AeroForce("aero", 0.3, 0.5, 1, np.array([0, 0, 0]))

    fr_tire = TireMF52("fr_tire", "uraeus/rnea/utils/sample.tir")
    fl_tire = TireMF52("fl_tire", "uraeus/rnea/utils/sample.tir")
    rr_tire = TireMF52("rr_tire", "uraeus/rnea/utils/sample.tir")
    rl_tire = TireMF52("rl_tire", "uraeus/rnea/utils/sample.tir")

    motor = SimpleElectricMotor(
        name="rl_motor",
        min_rpm=0,
        max_rpm=10000,
        min_torque=0,
        max_torque=300,
        max_power=200e3,
        reduction_ratio=1,
    )

    fr_spring = construct_sping_func(
        *esitmate_stiffness_damping(250 * 0.45 * 0.5, 2.8, 0.8)
    )
    fl_spring = construct_sping_func(
        *esitmate_stiffness_damping(250 * 0.45 * 0.5, 2.8, 0.8)
    )
    rr_spring = construct_sping_func(
        *esitmate_stiffness_damping(250 * 0.55 * 0.5, 2.9, 0.8)
    )
    rl_spring = construct_sping_func(
        *esitmate_stiffness_damping(250 * 0.55 * 0.5, 2.9, 0.8)
    )


def evaluate_forces(
    bodies_kinematics: List[BodyKinematics],
    bodies_idx: Dict[str, int],
    forces_map: Dict[str, Dict[str, np.ndarray]],
    qdt0: np.ndarray,
    qdt1: np.ndarray,
    u: np.ndarray = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    tau = np.zeros_like(qdt0)

    chassis_kin = bodies_kinematics[bodies_idx["chassis"]]

    fr_carier_kin = bodies_kinematics[bodies_idx["fr_carier"]]
    fl_carier_kin = bodies_kinematics[bodies_idx["fl_carier"]]
    rr_carier_kin = bodies_kinematics[bodies_idx["rr_carier"]]
    rl_carier_kin = bodies_kinematics[bodies_idx["rl_carier"]]

    fr_wheel_kin = bodies_kinematics[bodies_idx["fr_wheel"]]
    fl_wheel_kin = bodies_kinematics[bodies_idx["fl_wheel"]]
    rr_wheel_kin = bodies_kinematics[bodies_idx["rr_wheel"]]
    rl_wheel_kin = bodies_kinematics[bodies_idx["rl_wheel"]]

    fr_tire_force = Forces.fr_tire(fr_wheel_kin, fr_carier_kin)
    fl_tire_force = Forces.fl_tire(fl_wheel_kin, fl_carier_kin)
    rr_tire_force = Forces.rr_tire(rr_wheel_kin, rr_carier_kin)
    rl_tire_force = Forces.rl_tire(rl_wheel_kin, rl_carier_kin)

    # fr_tire_force = Forces.fr_tire.Fz(fr_wheel_kin)
    # fl_tire_force = Forces.fl_tire.Fz(fl_wheel_kin)
    # rr_tire_force = Forces.rr_tire.Fz(rr_wheel_kin)
    # rl_tire_force = Forces.rl_tire.Fz(rl_wheel_kin)

    forces_map["chassis"]["aero"] = Forces.aero_force(chassis_kin)

    forces_map["fr_wheel"]["tire"] = fr_tire_force
    forces_map["fl_wheel"]["tire"] = fl_tire_force
    forces_map["rr_wheel"]["tire"] = rr_tire_force
    forces_map["rl_wheel"]["tire"] = rl_tire_force

    tau[6] = -Forces.fr_spring(qdt0[6], qdt1[6])
    tau[7] = -Forces.fl_spring(qdt0[7], qdt1[7])
    tau[8] = -Forces.rr_spring(qdt0[8], qdt1[8])
    tau[9] = -Forces.rl_spring(qdt0[9], qdt1[9])

    rr_throttle = rl_throttle = 1

    rr_torque = Forces.motor(rr_wheel_kin, rr_throttle)
    rr_effective_radius = 0.254

    rl_torque = Forces.motor(rl_wheel_kin, rl_throttle)
    rl_effective_radius = 0.254

    # tau[12] = rr_torque
    # tau[13] = rl_torque

    tau[12] = Forces.motor.torque_control(
        rr_torque,
        rr_tire_force[5],
        rr_effective_radius,
        Forces.rr_tire.tir_model.Fx,
    )
    tau[13] = Forces.motor.torque_control(
        rl_torque,
        rl_tire_force[5],
        rl_effective_radius,
        Forces.rl_tire.tir_model.Fx,
    )

    return tau, forces_map
