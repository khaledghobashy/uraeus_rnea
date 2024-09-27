from dataclasses import dataclass
import numpy as np

from uraeus.rnea.quaternion.bodies import BodyKinematics


@dataclass
class TireKinematics(object):

    cir_vel: float
    lon_vel: float
    slp_vel: float
    lat_vel: float
    ang_vel: float

    effect_radius: float
    loaded_radius: float
    vertical_deflection: float
    sae_frame: np.ndarray


def evaluate_tire_kinematics(
    wheel_kinematics: BodyKinematics, unloaded_radius: float
) -> TireKinematics:

    wc_pos_z = wheel_kinematics.p_GB.r[2]
    spin_axis = np.array([0, 1, 0])
    # omega = abs(wheel_kinematics.v_B[4])
    omega = wheel_kinematics.v_B[4]

    R_SAE_G = construct_SAE_frame(np.array([0, 0, 1]), spin_axis)
    loaded_radius = min(unloaded_radius, abs(wc_pos_z - 0))
    vertical_deflection = max(unloaded_radius - loaded_radius, 0)
    effect_radius = loaded_radius + ((2 / 3) * vertical_deflection)

    # Wheel Center Translational Velocity in Global Frame
    v_wc_GF = wheel_kinematics.v_G[:3]

    # Wheel Center Translational Velocity in SAE Frame
    v_wc_SAE = R_SAE_G.T @ v_wc_GF

    # Longitudinal Wheel Velocity in SAE frame
    # V_x  = abs(V_wc_SAE[0,0])
    # lon_vel = abs(v_wc_SAE[0])
    lon_vel = v_wc_SAE[0]

    # Circumferential Velocity in SAE frame
    cir_vel = omega * effect_radius

    # Longitudinal Slip Velocity in SAE frame
    slp_vel = lon_vel - cir_vel

    # Lateral Slip Velocity in SAE frame
    lat_vel = v_wc_SAE[1]

    tire_kinematics = TireKinematics(
        cir_vel,
        lon_vel,
        slp_vel,
        lat_vel,
        omega,
        effect_radius,
        loaded_radius,
        vertical_deflection,
        R_SAE_G,
    )

    return tire_kinematics


def evaluate_tire_slips(tire_kinematics: TireKinematics) -> tuple[float, float]:

    epsilon = 0.1
    kappa = np.clip(
        -tire_kinematics.slp_vel / (abs(tire_kinematics.lon_vel) + epsilon), -1.0, 1.0
    )
    alpha = np.clip(
        np.arctan2(tire_kinematics.lat_vel, abs(tire_kinematics.lon_vel) + epsilon),
        -np.pi / 2 + 0.01,
        np.pi / 2 - 0.01,
    )

    # if abs(V_x) <= self._V_low:
    #     kv_low = 0.5 * self.kv_low * (1 + np.cos(np.pi * (V_x / self._V_low)))
    #     damped = (kv_low / self.C_Fk) * V_sx
    #     #            print('damped_k = %s'%damped)
    #     k = k - damped

    #     ka_low = 0.5 * self.kv_low * (1 + np.cos(np.pi * (V_x / self._V_low)))
    #     damped = (ka_low / self.C_Fa) * V_sy
    #     #            print('damped_a = %s'%damped)
    #     a = a - damped

    return kappa, alpha


def construct_SAE_frame(terrain_normal, spin_axis):
    # Normalize the terrain normal and spin axis vectors
    spin_axis = np.array(spin_axis)
    terrain_normal = terrain_normal / np.linalg.norm(terrain_normal)
    spin_axis = spin_axis / np.linalg.norm(spin_axis)

    # Calculate the longitudinal axis (x-axis)
    x_axis = np.cross(terrain_normal, spin_axis)

    # Calculate the lateral axis (y-axis)
    y_axis = np.cross(terrain_normal, x_axis)

    # Construct the SAE tire reference frame
    sae_frame = np.column_stack((-x_axis, -y_axis, terrain_normal))

    return sae_frame


# Sigmoid function for smooth transition
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
