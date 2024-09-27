from dataclasses import dataclass

import numpy as np

from uraeus.rnea.quaternion.bodies import BodyKinematics
from uraeus.models.vehicle_models.tire_models.contact_point_method import (
    evaluate_transient_slips,
)
from uraeus.models.vehicle_models.tire_models.utils import (
    TireKinematics,
    evaluate_tire_kinematics,
    evaluate_tire_slips,
    sigmoid,
)


@dataclass
class FialaTireParameters:
    mu: float
    Cfk: float
    Cfa: float
    Cfx: float
    Cfy: float
    unloaded_radius: float
    kz: float
    cz: float
    kv_low: float = 770  # (Ns/m) Damping coefficient at low speeds

    @property
    def sigma_k(self) -> float:
        return self.Cfk / self.Cfx

    @property
    def sigma_a(self) -> float:
        return self.Cfa / self.Cfy


class FialaTireModel(object):

    tire_parameters: FialaTireParameters

    def __init__(self, tire_parameters: FialaTireParameters):
        self.tire_parameters = tire_parameters
        self._u = 0
        self._v = 0
        self._last_t = 0

    def evaluate_tire_kinematics(
        self, wheel_kinematics: BodyKinematics
    ) -> TireKinematics:
        return evaluate_tire_kinematics(
            wheel_kinematics, self.tire_parameters.unloaded_radius
        )

    def evaluate_spatial_forces(
        self, wheel_kinematics: BodyKinematics, t: float
    ) -> np.ndarray:

        # evaluate radii
        tire_kinematics = self.evaluate_tire_kinematics(wheel_kinematics)
        tire_parameters = self.tire_parameters

        kappa, alpha = evaluate_tire_slips(tire_kinematics)
        # (kappa, alpha), (u, v) = evaluate_transient_slips(
        #     tire_parameters,
        #     tire_kinematics,
        #     low_speed_threshold=2,
        #     ydt0=np.array([self._u, self._v]),
        #     t0=self._last_t,
        #     t=t,
        # )
        # self._u = u
        # self._v = v
        # self._last_t = t
        # print("kappa = ", kappa)
        # print("alpha = ", alpha)
        # print("u = ", u)
        # print("v = ", v)
        # print("sigma_k = ", tire_parameters.sigma_k)

        normal_load = (
            tire_kinematics.vertical_deflection * tire_parameters.kz
            - wheel_kinematics.v_G[2] * tire_parameters.cz
        )
        normal_load = normal_load + 1
        Fx, Fy = self.evaluate_local_forces(normal_load, kappa, alpha)

        My = Fx * tire_kinematics.effect_radius

        tire_force_SAE = np.array([-Fx, 0, -normal_load])
        tire_torque_SAE = np.array([0, My, 0])

        tire_force_G = tire_kinematics.sae_frame @ tire_force_SAE
        tire_torque_G = tire_kinematics.sae_frame @ tire_torque_SAE

        print(tire_kinematics)

        return np.array([*tire_force_G, *tire_torque_G])

    def evaluate_local_forces(self, normal_load, kappa, alpha):

        tire_parameters = self.tire_parameters

        Fz = normal_load
        mu = tire_parameters.mu
        Cfa = tire_parameters.Cfa
        Cfk = tire_parameters.Cfk

        # Calculate lateral force Fy using sigmoid function for smooth transition
        alpha_peak = np.arctan(3 * mu * Fz / Cfa)
        alpha_ratio = alpha / alpha_peak
        Fy_linear = (
            -Cfa * np.tan(alpha)
            + (Cfa**2 / (3 * mu * Fz)) * abs(np.tan(alpha)) * np.tan(alpha)
            - (Cfa**3 / (27 * mu**2 * Fz**2)) * np.tan(alpha) ** 3
        )
        Fy_saturated = -mu * Fz * np.sign(alpha)
        Fy = Fy_linear * sigmoid(10 * (1 - abs(alpha_ratio))) + Fy_saturated * (
            1 - sigmoid(10 * (1 - abs(alpha_ratio)))
        )

        # Calculate longitudinal force Fx using sigmoid function for smooth transition
        kappa_peak = 3 * mu * Fz / Cfk
        kappa_ratio = kappa / kappa_peak
        Fx_linear = -(
            -Cfk * kappa
            + (Cfk**2 / (3 * mu * Fz)) * abs(kappa) * kappa
            - (Cfk**3 / (27 * mu**2 * Fz**2)) * kappa**3
        )
        Fx_saturated = mu * Fz * np.sign(kappa)
        transition = sigmoid(10 * (1 - abs(kappa_ratio)))
        Fx = (Fx_linear * transition) + (Fx_saturated * (1 - transition))

        return Fx, Fy

    def __call__(self, wheel_kinematics: BodyKinematics, t: float):
        return self.evaluate_spatial_forces(wheel_kinematics, t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_fiala_forces(Ca, Ck):
        normal_loads = np.linspace(500, 2500, 5)
        kappas = np.linspace(-1, 1, 100)

        tire_parameters = FialaTireParameters(
            mu=1,
            Cfk=1553e1,
            Cfa=4297e1,
            Cfx=1500e1,
            Cfy=1500e1,
            unloaded_radius=0.313,
            kz=150e3,
            cz=15e3,
        )

        fiala_tire_model = FialaTireModel(tire_parameters)

        plt.figure()
        for load in normal_loads:
            Fx = [
                fiala_tire_model.evaluate_local_forces(load, kappa, 0)[0]
                for kappa in kappas
            ]
            plt.plot(kappas, Fx, label=f"Load: {load} N")

        plt.xlabel("Kappa")
        plt.ylabel("Force (Fx)")
        plt.title("Force vs. Kappa for different Normal Loads")
        plt.grid()
        plt.legend()
        plt.show()

    plot_fiala_forces(21e3, 21e3)
