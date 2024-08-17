import re
from typing import Any
from dataclasses import dataclass
from collections import namedtuple

import pandas as pd

import numpy as np


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


def create_class_from_dict(param_dict):
    """
    Creates a dataclass with fields based on keys from a dictionary.

    Args:
        param_dict (dict): A dictionary containing parameter keys and values.

    Returns:
        type: A new dataclass with fields corresponding to the keys in param_dict.
    """

    @dataclass
    class DynamicClass:
        # Create fields dynamically based on keys in param_dict
        for key in param_dict.keys():
            locals()[key] = 0  # You can adjust the field type as needed

    return DynamicClass


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


class quaternion(object):

    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class SpatialPose(object):

    r: np.ndarray
    q: quaternion


class SpatialScrew(object):

    linear: np.ndarray
    angular: np.ndarray


class SpatialState(object):

    pose: SpatialPose
    v_F: SpatialPose


# def calculate_tire_states(
#     Fn_mag: float,
#     contact_data,
#     tire_par: namedtuple,
#     omega: float,
#     spin_axis: np.ndarray,
#     gamma_limit: float,
# ):
#     tire_states = TireStates()  # Initialize a TireStates object

#     # Calculate R_eff
#     tire_states.R_eff = (
#         2.0 * tire_par.UNLOADED_RADIUS + (tire_par.UNLOADED_RADIUS - contact_data.depth)
#     ) / 3.0

#     # Calculate other fields
#     tire_states.vx = np.abs(contact_data.vel_x)
#     tire_states.vsx = contact_data.vel_x - omega * tire_states.R_eff
#     tire_states.vsy = -contact_data.vel_y
#     epsilon = 0.1
#     tire_states.kappa = -tire_states.vsx / (tire_states.vx + epsilon)
#     tire_states.alpha = np.arctan2(tire_states.vsy, tire_states.vx + epsilon)
#     tire_states.omega = omega
#     tire_states.disc_normal = spin_axis
#     tire_states.Fz0_prime = tire_par.FNOMIN * tire_par.LFZO
#     tire_states.dfz0 = (Fn_mag - tire_states.Fz0_prime) / tire_states.Fz0_prime
#     tire_states.Pi0_prime = tire_par.IP_NOM * tire_par.LIP
#     tire_states.dpi = (tire_par.IP - tire_states.Pi0_prime) / tire_states.Pi0_prime

#     # Ensure kappa stays between -1 and 1
#     tire_states.kappa = np.clip(tire_states.kappa, -1.0, 1.0)

#     # Ensure alpha stays between -pi()/2 and pi()/2
#     tire_states.alpha = np.clip(tire_states.alpha, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

#     # Clamp |gamma| to specified value
#     tire_states.gamma = np.clip(tire_states.gamma, -gamma_limit, gamma_limit)

#     return tire_states


def calculate_tire_states(
    Fn_mag: float,
    contact_data,
    tire_par: namedtuple,
    omega: float,
    spin_axis: np.ndarray,
    gamma_limit: float,
) -> TireStates:
    """
    Calculates tire states based on input data.

    Args:
        Fn_mag (float): Magnitude of normal force.
        contact_data: Contact data (replace with actual data structure).
        tire_par (namedtuple): Tire parameters (replace with actual namedtuple).
        omega (float): Angular velocity of the wheel.
        spin_axis (np.ndarray): Spin axis vector.
        gamma_limit (float): Maximum allowable inclination angle.

    Returns:
        TireStates: Initialized tire states.
    """
    R_eff = (
        2.0 * tire_par.UNLOADED_RADIUS + (tire_par.UNLOADED_RADIUS - contact_data.depth)
    ) / 3.0
    vx = np.abs(contact_data.vel_x)
    vsx = contact_data.vel_x - omega * R_eff
    vsy = -contact_data.vel_y
    epsilon = 0.1
    kappa = np.clip(-vsx / (vx + epsilon), -1.0, 1.0)
    alpha = np.clip(np.arctan2(vsy, vx + epsilon), -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
    Fz0_prime = tire_par.FNOMIN * tire_par.LFZO
    dfz0 = (Fn_mag - Fz0_prime) / Fz0_prime
    # Pi0_prime = tire_par.IP_NOM * tire_par.LIP
    # dpi = (tire_par.IP - Pi0_prime) / Pi0_prime

    return TireStates(
        mu_scale=tire_par.LMUX,  # Example value (replace with actual value)
        mu_road=tire_par.LMUY,  # Example value (replace with actual value)
        grip_sat_x=0.9,  # Example value (replace with actual value)
        grip_sat_y=0.8,  # Example value (replace with actual value)
        kappa=kappa,
        alpha=alpha,
        gamma=np.clip(spin_axis[2], -gamma_limit, gamma_limit),
        vx=vx,
        vsx=vsx,
        vsy=vsy,
        omega=omega,
        R_eff=R_eff,
        Fz0_prime=Fz0_prime,
        dfz0=dfz0,
        Pi0_prime=0,
        dpi=0.1,
        brx=0.0,  # Example value (replace with actual value)
        bry=0.0,  # Example value (replace with actual value)
    )


def calculate_Fx(
    Fz: float, kappa: float, gamma: float, tire_states: TireStates, tire_par: namedtuple
):
    Fzo_d = tire_par.LFZO * tire_par.FNOMIN
    dfz0 = (Fz - Fzo_d) / Fzo_d
    tire_states.dfz0 = dfz0
    # Steady state calculation
    Cx = tire_par.PCX1 * tire_par.LCX
    Shx = (tire_par.PHX1 + tire_par.PHX2 * tire_states.dfz0) * tire_par.LHX
    Svx = (
        Fz
        * (tire_par.PVX1 + tire_par.PVX2 * tire_states.dfz0)
        * tire_par.LVX
        * tire_par.LMUX
    )
    epsilon = 0.1
    kappa = np.clip(-tire_states.vsx / (tire_states.vx + epsilon), -1.0, 1.0)

    # kappa = tire_states.kappa
    kappa_x = kappa + Shx
    gamma_x = gamma * tire_par.LGAX
    Ex = (
        (
            tire_par.PEX1
            + tire_par.PEX2 * tire_states.dfz0
            + tire_par.PEX3 * (tire_states.dfz0**2)
        )
        * (1.0 - tire_par.PEX4 * np.sign(kappa_x))
        * tire_par.LEX
    )
    # if Ex > 1.0:
    #     Ex = 1.0
    mu_x = np.abs(
        tire_states.mu_scale
        * (tire_par.PDX1 + tire_par.PDX2 * tire_states.dfz0)
        * (1.0 + tire_par.PPX3 * tire_states.dpi + tire_par.PPX4 * (tire_states.dpi**2))
        * (1.0 - tire_par.PDX3 * (gamma_x**2))
        * tire_par.LMUX
    )
    Dx = mu_x * Fz
    Kx = (
        Fz
        * (tire_par.PKX1 + tire_par.PKX2 * tire_states.dfz0)
        * np.exp(tire_par.PKX3 * tire_states.dfz0)
        * (1.0 + tire_par.PPX1 * tire_states.dpi + tire_par.PPX2 * (tire_states.dpi**2))
        * tire_par.LKX
    )
    Bx = Kx / (Cx * Dx + 0.1)
    X1 = Bx * kappa_x
    # X1 = np.clip(X1, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
    Fx0 = Dx * np.sin(Cx * np.arctan(X1 - Ex * (X1 - np.arctan(X1)))) + Svx
    return Fx0


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # with open("/workspaces/uraeus_rnea/uraeus/models/vehicle_models/utils/sample.tir", "r") as file:
    #     # text = file.readlines()
    #     # text = file.read().strip("").replace(" ", "")
    #     # print(text)
    #     text = file.read()

    # # pattern = re.compile("(\w+)+(?:\W+)+=([ \d])")
    # # pattern = re.compile("(\w+)(?:\W+)( *\-?\d+\.*\d*(e[+-])*?\d+)")
    # pattern = re.compile(
    #     "(\w+)(?:\W+)( *\-?\d+[\.\d]*(e[+-]\d+)*)((?:\W+)(.*))")

    # w = re.findall(pattern, text)
    # print(w)
    # d = dict(map(lambda t: (t[0], (t[1], t[-1])), w))
    # print(len(d))
    # print(d)

    tire_params_dict = read_tire_file(
        "/workspaces/uraeus_rnea/uraeus/models/vehicle_models/utils/sample.tir"
    )

    # tire_params_dict = read_tire_file(
    #     "/workspaces/uraeus_rnea/uraeus/models/vehicle_models/utils/Sedan_Pac02Tire.tir"
    # )
    print(tire_params_dict)
    # TireData = create_class_from_dict(tire_params_dict)
    # print(dir(TireData))
    # tire_params = TireData(**tire_params_dict)
    # print(tire_params)

    tire_par = namedtuple("TirePar", tire_params_dict.keys())(**tire_params_dict)
    print(tire_par)

    # tire_state = calculate_tire_states(
    #     250 * 9.81, contact_data, tire_par, wheel_state, np.array([0, 1, 0]), 0.1
    # )

    tire_state = TireStates(
        mu_scale=0.9,
        mu_road=0.8,
        grip_sat_x=1.0,
        grip_sat_y=0.7,
        kappa=0.1,
        alpha=np.radians(0),  # Convert degrees to radians
        gamma=np.radians(0),
        vx=20.0,
        vsx=2.0,
        vsy=0.5,
        omega=100.0,
        R_eff=0.3,
        Fz0_prime=5000.0,
        dfz0=0.1,
        Pi0_prime=2.0,
        dpi=0.02,
    )

    kappas = np.arange(-1, 1, 0.01)
    fx_s = map(
        lambda load: np.array(
            [
                calculate_Fx(load * 9.81, kappa, 0, tire_state, tire_par)
                for kappa in kappas
            ]
        ),
        (400, 450, 500, 550, 600, 650),
    )

    # fx_250 = np.array(
    #     [calculate_Fx(250 * 9.81, kappa, 0, tire_state, tire_par) for kappa in kappas]
    # )

    # plt.figure(figsize=(10, 10))
    # for fx in fx_s:
    #     plt.plot(kappas, fx)

    # plt.grid()
    # plt.show()
    # for line in text:
    #     # w = re.match(pattern, line)
    #     # print(w)
    #     # print(w.groups() if w else w)
    #     line_args = line.replace(" ", "").split("=")
    #     # print(line_args[0], " = ", line_args[1].split("$")[0])
