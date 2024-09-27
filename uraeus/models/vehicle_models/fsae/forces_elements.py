import numpy as np
from scipy import interpolate

from uraeus.rnea.quaternion.spatial_algebra import skew_M
from uraeus.rnea.quaternion.bodies import BodyKinematics


def aero_force(coeff: float, frontal_area: float, vel: float) -> float:
    return 0.5 * 1.2 * frontal_area * coeff * vel**2


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
        vel_x = chassis_kin.v_B[0]
        aero_drag = aero_force(self.cd, self.frontal_area, vel_x)
        aero_down = aero_force(self.cl, self.frontal_area, vel_x)

        local_frc_vec = np.array([-aero_drag, 0, -aero_down])
        local_trq_vec = skew_M @ local_frc_vec @ self.local_pos

        spatial_force_vec = np.hstack([local_frc_vec, local_trq_vec])
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

        torque_func = lambda rpm, throttle: (
            (max_power / (rpm * (2 * np.pi / 60))) * throttle
            if (max_power / (rpm * 2 * np.pi / 60) <= max_torque)
            else max_torque * throttle
        )

        # speed_grid, throttle_grid = np.meshgrid(rpms, np.array([0, 1]), indexing="ij")

        data = np.array(
            [[torque_func(rpm, throttle) for rpm in rpms] for throttle in [0, 1]]
        )

        interp_func = interpolate.RegularGridInterpolator(
            points=(rpms, np.array([0, 1])),
            values=data.T,
            bounds_error=False,
            fill_value=None,
        )
        self._func = interp_func
        self.reduction_ratio = reduction_ratio

    def __call__(self, wheel_kinematics: BodyKinematics, throttle: float):
        omega = abs(wheel_kinematics.v_B[4])
        rpm = (omega * self.reduction_ratio) * (30 / np.pi)
        trq = float(self._func((rpm, throttle))) * self.reduction_ratio
        return trq


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rl_motor = SimpleElectricMotor(
        name="rl_motor",
        min_rpm=0,
        max_rpm=7000,
        min_torque=0,
        max_torque=200,
        max_power=40e3,
        reduction_ratio=1,
    )

    rpms = np.arange(0, 7000, 100)
    plt.figure()
    plt.plot(rpms, [rl_motor._func((rpm, 1)) for rpm in rpms])
    plt.grid()

    plt.show()
