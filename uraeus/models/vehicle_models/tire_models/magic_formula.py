import logging
import re
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple

import numpy as np

from uraeus.utils.logging import construct_logger
from uraeus.rnea.quaternion.bodies import BodyKinematics
from uraeus.models.vehicle_models.tire_models.utils import (
    TireKinematics,
    evaluate_tire_kinematics,
)
from uraeus.models.vehicle_models.tire_models.contact_point_method import (
    evaluate_transient_slips,
)

_logger = construct_logger(__name__, logging.DEBUG)


@dataclass
class TireStates:
    mu_scale: float  # Scaling factor for tire patch forces
    mu_road: float  # Actual road friction coefficient
    grip_sat_x: float  # Tire grip saturation (x-direction)
    grip_sat_y: float  # Tire grip saturation (y-direction)
    kappa: float  # Slip ratio [-1:+1]
    alpha: float  # Slip angle [-PI/2:+PI/2]
    gamma: float  # Inclination angle
    vx: float  # Longitudinal speed
    vsx: float  # Longitudinal slip velocity
    vsy: float  # Lateral slip velocity (also called lateral velocity)
    omega: float  # Wheel angular velocity about its spin axis
    R_eff: float  # Effective radius
    Fz0_prime: float  # Scaled vertical load
    dfz0: float  # Normalized vertical force
    Pi0_prime: float  # Scaled inflation pressure
    dpi: float  # Normalized inflation pressure
    brx: float = 0.0  # Bristle deformation (x-direction)
    bry: float = 0.0  # Bristle deformation (y-direction)
    # disc_normal: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Temporary for debug


@dataclass
class ContactData:
    vel_x: float
    vel_y: float
    depth: float


def calculate_vertical_load_factors(Fn_mag: float, tire_par: namedtuple):
    Fz0_prime = tire_par.FNOMIN * tire_par.LFZO
    dfz0 = (Fn_mag - Fz0_prime) / Fz0_prime
    return (Fz0_prime, dfz0)


def read_tire_file(file_path: str) -> dict[str, tuple[float, str]]:
    pattern = re.compile(
        "([A-Z_]+[0-9]*)(?:\W+)( *\-?\d+[\.\d]*(e[+-]\d+)*)((?:\W+)(.*))"
    )
    # pattern = re.compile("(\w+)(?:\W+)( *\-?\d+[\.\d]*(e[+-]\d+)*)((?:\W+)(.*))")
    with open(file_path, "r") as file:
        text = file.read()

    w = re.findall(pattern, text)
    # coeff = dict(map(lambda t: (t[0], (t[1], t[-1])), w))
    coeff = dict(map(lambda t: (t[0], float(t[1])), w))

    return coeff


class MagicFormulaCoefficients(NamedTuple):
    B: float
    C: float
    D: float
    E: float


class MagicFormulaTireModel(object):

    tire_parameters: namedtuple

    def __init__(self, tire_parameters: namedtuple):
        self.tire_parameters = tire_parameters
        self._u = 0
        self._v = 0
        self._last_t = 0
        self._is_sliding = False

    def evaluate_tire_kinematics(
        self, wheel_kinematics: BodyKinematics
    ) -> TireKinematics:
        return evaluate_tire_kinematics(
            wheel_kinematics, self.tire_parameters.UNLOADED_RADIUS
        )

    def evaluate_Fx_coefficients(
        self, normal_load: float, kappa: float, gamma: float
    ) -> MagicFormulaCoefficients:
        tire_par = self.tire_parameters
        Fz = normal_load

        Fz0_prime, dfz0 = calculate_vertical_load_factors(normal_load, tire_par)
        dpi = 0.1

        Fzo_d = tire_par.LFZO * tire_par.FNOMIN
        dfz0 = (Fz - Fzo_d) / Fzo_d
        # Steady state calculation
        Cx = tire_par.PCX1 * tire_par.LCX
        Shx = (tire_par.PHX1 + tire_par.PHX2 * dfz0) * tire_par.LHX
        # Svx = Fz * (tire_par.PVX1 + tire_par.PVX2 * dfz0) * tire_par.LVX * tire_par.LMUX

        kappa_x = kappa + Shx
        gamma_x = gamma * tire_par.LGAX
        Ex = (
            (tire_par.PEX1 + tire_par.PEX2 * dfz0 + tire_par.PEX3 * (dfz0**2))
            * (1.0 - tire_par.PEX4 * np.sign(kappa_x))
            * tire_par.LEX
        )
        # if Ex > 1.0:
        #     Ex = 1.0
        mu_x = np.abs(
            tire_par.LMUX
            * (tire_par.PDX1 + tire_par.PDX2 * dfz0)
            * (1.0 + tire_par.PPX3 * dpi + tire_par.PPX4 * (dpi**2))
            * (1.0 - tire_par.PDX3 * (gamma_x**2))
            * tire_par.LMUX
        )
        Dx = mu_x * Fz
        Kx = (
            Fz
            * (tire_par.PKX1 + tire_par.PKX2 * dfz0)
            * np.exp(tire_par.PKX3 * dfz0)
            * (1.0 + tire_par.PPX1 * dpi + tire_par.PPX2 * (dpi**2))
            * tire_par.LKX
        )
        Bx = Kx / (Cx * Dx + 0.1)
        return MagicFormulaCoefficients(Bx, Cx, Dx, Ex)

    def evaluate_Fx(self, normal_load: float, kappa: float, gamma: float) -> float:
        tire_par = self.tire_parameters
        Bx, Cx, Dx, Ex = self.evaluate_Fx_coefficients(normal_load, kappa, gamma)
        Fz0_prime, dfz0 = calculate_vertical_load_factors(normal_load, tire_par)
        Shx = (tire_par.PHX1 + tire_par.PHX2 * dfz0) * tire_par.LHX
        Svx = (
            normal_load
            * (tire_par.PVX1 + tire_par.PVX2 * dfz0)
            * tire_par.LVX
            * tire_par.LMUX
        )
        kappa_x = kappa + Shx
        X1 = Bx * kappa_x
        # X1 = np.clip(X1, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        Fx0 = Dx * np.sin(Cx * np.arctan(X1 - Ex * (X1 - np.arctan(X1)))) + Svx
        return Fx0

    def evaluate_local_forces(self, normal_load, kappa, alpha):
        Fx = self.evaluate_Fx(normal_load, kappa, 0)
        Fy = 0
        self._is_sliding = abs(kappa) > 0.25
        return Fx, Fy

    def evaluate_spatial_forces(
        self, wheel_kinematics: BodyKinematics, t: float
    ) -> np.ndarray:

        # evaluate radii
        tire_kinematics = self.evaluate_tire_kinematics(wheel_kinematics)
        tire_parameters = self.tire_parameters

        normal_load = (
            tire_kinematics.vertical_deflection * tire_parameters.VERTICAL_STIFFNESS
            - wheel_kinematics.v_G[2] * tire_parameters.VERTICAL_DAMPING
        )
        normal_load = normal_load + 1

        # kappa, alpha = evaluate_tire_slips(tire_kinematics)

        (kappa, alpha), (u, v) = evaluate_transient_slips(
            tire_parameters,
            tire_kinematics,
            low_speed_threshold=2.5,
            ydt0=np.array([self._u, self._v]),
            t0=self._last_t,
            t=t,
            is_sliding=self._is_sliding,
        )
        self._u = u
        self._v = v
        self._last_t = t
        _logger.debug("kappa = %s", kappa)
        _logger.debug("alpha = %s", alpha)
        _logger.debug("u = %s", u)
        _logger.debug("v = %s", v)
        _logger.debug("is_sliding = %s", self._is_sliding)
        _logger.debug("sigma_k = %s", tire_parameters.sigma_k)
        _logger.debug("tire_kinematics = %s", tire_kinematics)

        Fx, Fy = self.evaluate_local_forces(normal_load, kappa, alpha)

        My = Fx * tire_kinematics.effect_radius

        _logger.debug("Fx_SAE = %s", Fx)
        _logger.debug("My_SAE = %s", My)

        tire_force_SAE = np.array([Fx, 0, normal_load])
        tire_torque_SAE = np.array([0, -My, 0])

        tire_force_G = tire_kinematics.sae_frame @ tire_force_SAE
        tire_torque_G = tire_kinematics.sae_frame @ tire_torque_SAE

        return np.array([*tire_force_G, *tire_torque_G])

    def __call__(self, wheel_kinematics: BodyKinematics, t: float):
        return self.evaluate_spatial_forces(wheel_kinematics, t)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    tire_params_dict = read_tire_file(
        "/workspaces/uraeus_rnea/uraeus/models/vehicle_models/tire_models/data_files/sample.tir"
    )

    tire_par = namedtuple("TirePar", tire_params_dict.keys())(**tire_params_dict)

    tire_model = MagicFormulaTireModel(tire_par)
    print(tire_model.evaluate_Fx_coefficients(2000, 0.1, 0))

    kappas = np.linspace(-1, 1, 100)
    normal_loads = np.linspace(500, 2500, 5)
    for load in normal_loads:
        Fx = [tire_model.evaluate_Fx(load, kappa, 0) for kappa in kappas]
        plt.plot(kappas, Fx, label=f"Load: {load} N")

    plt.xlabel("Kappa")
    plt.ylabel("Force (Fx)")
    plt.title("Force vs. Kappa for different Normal Loads")
    plt.grid()
    plt.legend()
    plt.show()
