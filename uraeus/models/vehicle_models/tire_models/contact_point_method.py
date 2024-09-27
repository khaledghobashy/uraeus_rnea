import logging
from dataclasses import dataclass

import numpy as np
import scipy.integrate

from uraeus.utils.logging import construct_logger
from uraeus.models.vehicle_models.tire_models.utils import TireKinematics

logger = construct_logger(__name__, logging.DEBUG)


def evaluate_transient_slips(
    tire_parameters,
    tire_kinematics: TireKinematics,
    low_speed_threshold: float,
    ydt0: np.ndarray,
    t0: float,
    t: float,
    is_sliding: bool = False,
):

    if t <= t0:
        u, v = ydt0
        k = u / tire_parameters.sigma_k
        a = v / tire_parameters.sigma_a
        return (k, a), (u, v)

    slp_vel = tire_kinematics.slp_vel
    lat_vel = tire_kinematics.lat_vel
    lon_vel = tire_kinematics.lon_vel

    u = integrate_cpm(
        contact_point_method_ssode,
        np.array([ydt0[0]]),
        t0,
        t,
        (
            slp_vel,
            lon_vel,
            tire_parameters.sigma_k,
            (abs(lon_vel) < low_speed_threshold),
            is_sliding,
        ),
    )
    v = integrate_cpm(
        contact_point_method_ssode,
        np.array([ydt0[1]]),
        t0,
        t,
        (
            lat_vel,
            lon_vel,
            tire_parameters.sigma_a,
            (abs(lon_vel) < low_speed_threshold),
            is_sliding,
        ),
    )

    k = u / tire_parameters.sigma_k
    a = v / tire_parameters.sigma_a

    if abs(lon_vel) <= low_speed_threshold:
        kv_low = (
            0.5
            * tire_parameters.kv_low
            * (1 + np.cos(np.pi * (abs(lon_vel) / low_speed_threshold)))
        )
        damped = (kv_low / tire_parameters.Cfk) * slp_vel
        k = k - damped

        ka_low = (
            0.5
            * tire_parameters.kv_low
            * (1 + np.cos(np.pi * (abs(lon_vel) / low_speed_threshold)))
        )
        damped = (ka_low / tire_parameters.Cfa) * lat_vel
        a = a - damped

    return (k, a), (u, v)


def contact_point_method_ssode(
    t: float,
    ydt0: np.ndarray,
    slp_vel: float,
    lon_vel: float,
    relaxation_x: float,
    is_below_low_speed_threshold: bool,
    is_sliding: bool,
):
    cond1 = (slp_vel + (abs(lon_vel) * (ydt0 / relaxation_x))) * ydt0 < 0
    no_go = cond1 and is_below_low_speed_threshold and is_sliding
    # no_go = is_below_low_speed_threshold
    # no_go = is_sliding
    # no_go = cond1 and is_below_low_speed_threshold
    # no_go = (is_below_low_speed_threshold) and is_sliding
    # no_go = cond1 and is_sliding
    # no_go = cond1
    # no_go = False

    ydt1 = (
        np.array([0])
        if (no_go)
        else (-(1 / relaxation_x) * abs(lon_vel) * ydt0) - slp_vel
    )

    return ydt1


def integrate_cpm(ssode, ydt0, t0, t_end, args):

    max_step = np.inf
    soln = scipy.integrate.solve_ivp(
        ssode, (t0, t_end), ydt0, "BDF", t_eval=[t_end], args=args, max_step=max_step
    )
    return soln.y[-1][-1]
