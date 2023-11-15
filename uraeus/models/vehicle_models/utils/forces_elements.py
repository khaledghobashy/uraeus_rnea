from typing import Callable
import numpy as np
from scipy import interpolate

from uraeus.rnea.spatial_algebra import skew_matrix
from uraeus.rnea.bodies import BodyKinematics
from uraeus.rnea.utils.tire_utils.tire_models import MF52, construct_SAE_frame


def aero_force(coeff: float, frontal_area: float, vel: float) -> float:
    return 0.5 * 1.2 * frontal_area * coeff * vel**2


class TireMF52(object):
    tir_model: MF52

    def __init__(self, name, tir_file: str):
        self.name = name
        self.tir_model = MF52(tir_file)

    def __call__(
        self, wheel_kinematics: BodyKinematics, carrier_kinematics: BodyKinematics
    ) -> np.ndarray:
        wc_vel_z = carrier_kinematics.v_B[5]
        wc_vel_x = carrier_kinematics.v_B[3]
        wc_pos_z = carrier_kinematics.p_GB[5]
        spin_axis = carrier_kinematics.R_GB[:, 1]
        omega = wheel_kinematics.v_B[1]

        # print(f"{self.name}_spin_axis = {spin_axis}")

        R_SAE_G = construct_SAE_frame(np.array([0, 0, 1]), spin_axis)
        # print(f"{self.name} R_SAE = ", R_SAE_G)

        loaded_radius, effective_radius, defflection = self.evaluate_tire_radii(
            wc_pos_z, 0
        )

        sx = self.evaluate_slip(wc_vel_x, omega, effective_radius)
        alpha = 0.0
        gamma = 0.0

        Fz = self.tir_model.Fz(defflection, wc_vel_z)
        Fx = self.tir_model.Fx(Fz, sx)
        Fy = 0  # self.tir_model.Fy(Fz, alpha, gamma)

        # print(f"{self.name}_Fy = ", Fy)

        Mx = -Fy * effective_radius
        My = Fx * effective_radius
        Mz = 0

        frc_vec_SAE = np.array([Fx, Fy, -Fz])
        trq_vec_SAE = np.array([Mx, My, Mz])

        frc_vec_G = R_SAE_G @ frc_vec_SAE
        trq_vec_G = R_SAE_G @ trq_vec_SAE

        # frc_vec_G[0] = 0
        # trq_vec_G[1] = 0
        # print(f"{self.name}_frc_G = ", frc_vec_G)
        # print(f"{self.name}_trq_G = ", trq_vec_G)

        return np.hstack([trq_vec_G, frc_vec_G])

    def evaluate_slip(self, vel: float, omega: float, effective_radius: float):
        sx = ((omega * effective_radius) - vel) / (vel + 1e-5)
        if abs(sx) > 1:
            print("SLIDING! : ", sx)
        # sx = 0
        # print(f"{self.name}_sx = {sx}")
        return sx

    def evaluate_tire_radii(self, wc_zdt0: float, ground_z: float):
        loaded_radius = wc_zdt0 - ground_z
        defflection = max(self.tir_model.coeff.UNLOADED_RADIUS - loaded_radius, 0)
        effective_radius = loaded_radius + ((2 / 3) * defflection)
        return loaded_radius, effective_radius, defflection

    def Fz(self, wheel_kinematics: BodyKinematics) -> np.ndarray:
        wc_height = wheel_kinematics.p_GB[5]
        wc_vel_z = wheel_kinematics.v_G[5]
        loaded_radius, effective_radius, defflection = self.evaluate_tire_radii(
            wc_height, 0
        )
        Fz = self.tir_model.Fz(defflection, -wc_vel_z)
        return np.array([0, 0, 0, 0, 0, Fz])


class AeroForce(object):
    name: str
    cd: float
    cl: float
    frontal_area: float
    local_pos: np.ndarray

    def __init__(
        self,
        name: str,
        cd: float,
        cl: float,
        frontal_area: float,
        local_pos: np.ndarray,
    ):
        self.name = name
        self.cd = cd
        self.cl = cl
        self.frontal_area = frontal_area
        self.local_pos = local_pos

    def __call__(self, chassis_kin: BodyKinematics) -> np.ndarray:
        vel_x = chassis_kin.v_B[3]
        aero_drag = aero_force(self.cd, self.frontal_area, vel_x)
        aero_down = aero_force(self.cl, self.frontal_area, vel_x)

        local_frc_vec = np.array([-aero_drag, 0, -aero_down])
        local_trq_vec = skew_matrix(self.local_pos) @ local_frc_vec

        global_frc_vec = chassis_kin.R_GB @ local_frc_vec
        global_trq_vec = chassis_kin.R_GB @ local_trq_vec

        spatial_force_vec = np.hstack([global_trq_vec, global_frc_vec])
        return spatial_force_vec


class SimpleElectricMotor(object):
    def __init__(
        self,
        name: str,
        min_rpm: int,
        max_rpm: int,
        min_torque: float,
        max_torque: float,
        max_power: float,
        reduction_ratio: float,
    ):
        self.name = name

        rpms = np.arange(0, max_rpm + 200, 100)
        trqs = np.array(
            [
                max_power / (rpm * (2 * np.pi / 60))
                if (max_power / (rpm * 2 * np.pi / 60) <= max_torque)
                else max_torque
                for rpm in rpms
            ]
        )

        interp_func = interpolate.interp2d(
            x=rpms,
            y=[0, 1],
            z=np.vstack([min_torque * np.ones_like(trqs), trqs]),
            # bounds_error=True,
            fill_value=0,
        )

        self._func = interp_func
        self.reduction_ratio = reduction_ratio

    def __call__(self, wheel_kinematics: BodyKinematics, throttle: float):
        omega = wheel_kinematics.v_B[1]
        rpm = (omega * self.reduction_ratio) * (30 / np.pi)
        trq = float(self._func(rpm, throttle)) * self.reduction_ratio
        return trq

    def torque_control(
        self,
        motor_torque: float,
        Fz: float,
        effective_radius: float,
        tire_fx_func: Callable[[float, float], float],
    ) -> float:
        max_grip = max([tire_fx_func(Fz, slip) for slip in np.arange(0, 1, 0.001)])
        My = 0.95 * max_grip * effective_radius
        factor = min(motor_torque, My)
        # print(f"{self.name} factor = ", factor)
        return factor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # motor = SimpleElectricMotor(
    #     name="rl_motor",
    #     min_rpm=0,
    #     max_rpm=18000,
    #     min_torque=0,
    #     max_torque=550,
    #     max_power=380e3,
    #     reduction_ratio=18000 * (np.pi / 30) / (100 / 0.3135),
    # )

    # rpms = np.arange(0, 19000, 400)
    # omegas = np.arange(0, 350, 10)
    # rpms = (omegas * motor.reduction_ratio) * (30 / np.pi)

    # plt.figure()
    # plt.plot(rpms, motor._func(rpms, 1))
    # plt.grid()

    rl_motor = SimpleElectricMotor(
        name="rl_motor",
        min_rpm=0,
        max_rpm=7000,
        min_torque=0,
        max_torque=200,
        max_power=70e3,
        reduction_ratio=1,
    )

    rpms = np.arange(0, 7000, 100)
    plt.figure()
    plt.plot(rpms, rl_motor._func(rpms, 1))
    plt.grid()

    plt.show()
