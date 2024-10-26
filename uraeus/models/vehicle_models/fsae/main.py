import logging

import numpy as np
import matplotlib.pyplot as plt

from uraeus.rnea.spatial_algebra import quaternion_to_yaw
from uraeus.models.vehicle_models.fsae.topology import (
    ChassisData,
    WheelData,
    SuspentionData,
    VehicleData,
    construct_multibodytree,
)
from uraeus.rnea.multibody_models import Model, HybridModel
from uraeus.models.vehicle_models.fsae.simulations import (
    solve_for_static_equilibrium,
    acceleration_sim,
    standing_sim,
)

logging.disable(logging.DEBUG)

np.set_printoptions(precision=3)

chassis_data = ChassisData(
    mass=250,
    wheelbase=1.6,
    cg_height=0.3,
    weight_distribution_f=0.5,
    inertia_tensor=np.diag([120, 150, 150]),
)


wheel_data_front = WheelData(mass=1, inertia_tensor=np.eye(3), wc_height=0.245)
wheel_data_rear = WheelData(mass=1, inertia_tensor=np.eye(3), wc_height=0.245)

susp_data_front = SuspentionData(mass=5, trackwidth=1.2, stiffness=0, damping=0)
susp_data_rear = SuspentionData(mass=5, trackwidth=1.1, stiffness=0, damping=0)

vehicle_data = VehicleData(
    chassis=chassis_data,
    suspension_front=susp_data_front,
    suspension_rear=susp_data_rear,
    wheels_front=wheel_data_front,
    wheels_rear=wheel_data_rear,
)

topology = construct_multibodytree(vehicle_data)
# model = Model(topology)
model = HybridModel(topology, id_coordinates=[7, 9])
model.vehicle_data = vehicle_data
print(model.hybrid_dynamics_data.permutation_matrix.shape)
print(model.n)
print(model.tree_data.qdt0_names)
if __name__ == "__main__":
    system_inputs = {"throttle": 0, "steering_input": 0}
    print(solve_for_static_equilibrium(model, u0=system_inputs, vx=0 / 3.6))
    # exit()
    # res = standing_sim(model)
    res = acceleration_sim(model, u0=system_inputs, v0=30 / 3.6, tf=20)

    bodies_kinematics = [
        model.forward_kinematics_pass(
            res.qdt0_history[i, :], res.qdt1_history[i, :], res.qdt2_history[i, :]
        )[0]
        for i in range(len(res.time_history))
    ]

    chassis_kin = [
        model.get_body_kinematics("chassis", bodies) for bodies in bodies_kinematics
    ]

    chassis_p_GB = [kin.p_GB for kin in chassis_kin]
    chassis_p_BG = [kin.p_BG for kin in chassis_kin]
    chassis_v = [kin.v_B for kin in chassis_kin]
    chassis_a = [kin.a_B for kin in chassis_kin]

    plt.figure("chassis_z.png")
    plt.plot(res.time_history, [p.r[2] for p in chassis_p_GB])
    plt.grid()

    plt.figure("Vehicle yaw")
    plt.plot(res.time_history, res.qdt0_history[:, 5], label="free.yaw")
    plt.plot(
        res.time_history,
        [quaternion_to_yaw(p.q) for p in chassis_p_BG],
        label="chassis.yaw",
    )
    plt.legend()
    plt.grid()

    plt.figure("x-y pos")
    plt.title("x-y pos")
    plt.plot([p.r[0] for p in chassis_p_GB], [p.r[1] for p in chassis_p_GB], label="GB")
    plt.plot([p.r[0] for p in chassis_p_BG], [p.r[1] for p in chassis_p_BG], label="BG")
    plt.legend()
    plt.grid()

    plt.figure("chassis_x.png")
    plt.plot(res.time_history, [p.r[0] for p in chassis_p_GB])
    plt.grid()

    plt.figure("chassis_y.png")
    plt.plot(res.time_history, [p.r[1] for p in chassis_p_GB])
    plt.grid()

    plt.figure("chassis_z_vel.png")
    plt.plot(res.time_history, [p[2] for p in chassis_v])
    plt.grid()

    plt.figure("chassis_vel_x.png")
    plt.plot(res.time_history, [p[0] * 3.6 for p in chassis_v])
    plt.grid()

    plt.figure("chassis_acc_x.png")
    plt.plot(res.time_history, [p[0] / 9.81 for p in chassis_a])
    plt.grid()

    plt.figure("wheels_omega.png")
    plt.plot(
        res.time_history,
        res.qdt1_history[:, model.tree_data.qdt0_names.fr_wheel_rev.psi],
        label="fr_wheel",
    )
    plt.plot(
        res.time_history,
        res.qdt1_history[:, model.tree_data.qdt0_names.fl_wheel_rev.psi],
        label="fl_wheel",
    )
    plt.plot(
        res.time_history,
        res.qdt1_history[:, model.tree_data.qdt0_names.rr_wheel_rev.psi],
        label="rr_wheel",
    )
    plt.plot(
        res.time_history,
        res.qdt1_history[:, model.tree_data.qdt0_names.rl_wheel_rev.psi],
        label="rl_wheel",
    )
    plt.legend()
    plt.grid()

    plt.show()
