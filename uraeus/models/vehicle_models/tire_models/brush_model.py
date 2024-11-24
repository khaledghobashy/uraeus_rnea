import logging
from dataclasses import dataclass

import numpy as np

from uraeus.utils.logging import construct_logger
from uraeus.rnea.bodies import BodyKinematics
from uraeus.models.vehicle_models.tire_models.utils import (
    TireKinematics,
    evaluate_tire_kinematics,
    evaluate_tire_slips,
    sigmoid,
)
from uraeus.models.vehicle_models.tire_models.contact_point_method import (
    evaluate_transient_slips,
)


def normalize(v):
    normalized = v / np.linalg.norm(v)
    return normalized


logger = construct_logger(__name__, logging.DEBUG)


@dataclass
class BrushModelParameters(object):
    mu: float
    Cfk: float
    Cfa: float
    Cfx: float
    Cfy: float
    a: float
    unloaded_radius: float
    kz: float
    cz: float
    kv_low: float  # (Ns/m) Damping coefficient at low speeds

    @property
    def sigma_k(self) -> float:
        return self.Cfk / self.Cfx

    @property
    def sigma_a(self) -> float:
        return self.Cfa / self.Cfy


class BrushTireModel(object):

    tire_parameters: BrushModelParameters

    def __init__(
        self, tire_parameters: BrushModelParameters, use_cpm_model: bool = False
    ):
        self.tire_parameters = tire_parameters
        self._u = 0
        self._v = 0
        self._last_t = 0
        self._is_sliding = False

        self._evaluate_tire_slips = (
            self._evaluate_transient_slips
            if use_cpm_model
            else self._evaluate_steady_slips
        )

    def evaluate_tire_kinematics(
        self, wheel_kinematics: BodyKinematics
    ) -> TireKinematics:
        return evaluate_tire_kinematics(
            wheel_kinematics, self.tire_parameters.unloaded_radius
        )

    def evaluate_local_forces(self, normal_load, kappa, alpha):

        tire_parameters = self.tire_parameters

        sigma_x = kappa / (1 + kappa)
        sigma_y = np.tan(alpha) / (1 + kappa)
        sigma = np.sqrt(sigma_x**2 + sigma_y**2)
        sigma_vec = np.array([sigma_x, sigma_y])

        if sigma <= 1e-5 or normal_load <= 0:
            F = np.array([0, 0])
            xt = 0
        else:
            Theta = (2 / 3) * (
                (tire_parameters.Cfx * tire_parameters.a**2)
                / (tire_parameters.mu * normal_load)
            )
            TG = Theta * sigma

            if sigma <= 1 / Theta:
                factor = 3 * (TG) - 3 * (TG) ** 2 + (TG) ** 3
                force = tire_parameters.mu * normal_load * factor
            else:
                force = tire_parameters.mu * normal_load

            # transition = sigmoid(100 * (sigma - 1 / Theta))
            # factor = (3 * TG - 3 * TG**2 + TG**3) * (1 - transition) + transition
            # force = tire_parameters.mu * normal_load * factor

            F = force * normalize(sigma_vec)
            # Pneumatic Trail
            xt = (
                (1 / 3)
                * tire_parameters.a
                * (
                    (1 - 3 * abs(TG) + 3 * TG**2 - abs(TG) ** 3)
                    / (1 - abs(TG) + (1 / 3) * TG**2)
                )
            )
            self._is_sliding = sigma > (1 / Theta)
        return F, xt

    def evaluate_spatial_forces(
        self, wheel_kinematics: BodyKinematics, t: float
    ) -> np.ndarray:

        # evaluate radii
        tire_kinematics = self.evaluate_tire_kinematics(wheel_kinematics)
        tire_parameters = self.tire_parameters

        kappa, alpha = self._evaluate_tire_slips(tire_kinematics, t)

        normal_load = (
            tire_kinematics.vertical_deflection * tire_parameters.kz
            - wheel_kinematics.v_G[2] * tire_parameters.cz
        )
        normal_load = normal_load + 1
        (Fx, Fy), xt = self.evaluate_local_forces(normal_load, kappa, alpha)

        My = Fx * tire_kinematics.effect_radius
        Mz = -xt * Fy

        logger.debug(f"Fx_SAE = {Fx}")
        logger.debug(f"Fy_SAE = {Fy}")
        logger.debug(f"Fz_SAE = {-normal_load}")
        logger.debug(f"My_SAE = {My}")
        logger.debug(f"Mz_SAE = {Mz}")

        tire_force_SAE = np.array([Fx, Fy, -normal_load])
        tire_torque_SAE = np.array([0, My, Mz])

        tire_force_G = tire_kinematics.sae_frame @ tire_force_SAE
        tire_torque_G = tire_kinematics.sae_frame @ tire_torque_SAE

        return np.array([*tire_force_G, *tire_torque_G])

    def _evaluate_transient_slips(self, tire_kinematics: TireKinematics, t: float):
        (kappa, alpha), (u, v) = evaluate_transient_slips(
            self.tire_parameters,
            tire_kinematics,
            low_speed_threshold=3,
            ydt0=np.array([self._u, self._v]),
            t0=self._last_t,
            t=t,
            is_sliding=self._is_sliding,
        )

        self._u = u
        self._v = v
        self._last_t = t
        logger.debug(f"kappa = {kappa}")
        logger.debug(f"alpha = {alpha}")
        logger.debug(f"u = {u}")
        logger.debug(f"v = {v}")
        logger.debug(f"is_sliding = {self._is_sliding}")
        logger.debug(f"tire_kinematics = {tire_kinematics}")

        return kappa, alpha

    def _evaluate_steady_slips(self, tire_kinematics: TireKinematics, t: float):
        kappa, alpha = evaluate_tire_slips(tire_kinematics)
        logger.debug(f"kappa = {kappa}")
        logger.debug(f"alpha = {alpha}")
        return kappa, alpha

    def __call__(self, wheel_kinematics: BodyKinematics, t: float):
        return self.evaluate_spatial_forces(wheel_kinematics, t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_tire_forces():
        normal_loads = np.linspace(500, 2500, 5)
        kappas = np.linspace(-1, 1, 100)

        tire_parameters = BrushModelParameters(
            mu=1,
            Cfk=150e3 * 0.2,
            Cfa=4297,
            Cfx=150e3,
            Cfy=1500,
            a=0.210,
            unloaded_radius=0.313,
            kz=650e3,
            cz=15e3,
            kv_low=100,
        )
        tire_model = BrushTireModel(tire_parameters)

        Fx_s = [
            [tire_model.evaluate_local_forces(load, kappa, 0)[0][0] for kappa in kappas]
            for load in normal_loads
        ]
        Fy_s = [
            [tire_model.evaluate_local_forces(load, 0, kappa)[0][1] for kappa in kappas]
            for load in normal_loads
        ]

        plt.figure()
        for Fx, load in zip(Fx_s, normal_loads):
            plt.plot(kappas, Fx, label=f"Load: {load} N")

        plt.xlabel("Kappa")
        plt.ylabel("Force (Fx)")
        plt.title("Force vs. Kappa for different Normal Loads")
        plt.grid()
        plt.legend()

        plt.figure()
        for Fy, load in zip(Fy_s, normal_loads):
            plt.plot(kappas, Fy, label=f"Load: {load} N")

        plt.xlabel("Kappa")
        plt.ylabel("Force (Fy)")
        plt.title("Force vs. alpha for different Normal Loads")
        plt.grid()
        plt.legend()
        plt.show()

    plot_tire_forces()
