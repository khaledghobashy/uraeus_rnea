import numpy as np
import matplotlib.pyplot as plt

from uraeus.models.vehicle_models.fsae.topology import (
    ChassisData,
    WheelData,
    SuspentionData,
    VehicleData,
    construct_multibodytree,
)
from uraeus.models.vehicle_models.fsae.model import Model
from uraeus.models.vehicle_models.fsae.simulations import (
    static_equilibrium,
    acceleration_sim,
    standing_sim,
    test_eval_joints_kinematics,
)

chassis_data = ChassisData(
    mass=250,
    wheelbase=1.6,
    cg_height=0.3,
    weight_distribution_f=0.45,
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
model = Model(topology, vehicle_data)

if __name__ == "__main__":
    test_eval_joints_kinematics(model)
    print(static_equilibrium(model, 200 / 3.6))
    input()

    # res = standing_sim(model)
    res = acceleration_sim(model, 1, 10)

    bodies_kinematics = [
        model.forward_kinematics_pass(
            res.qdt0_history[i, :], res.qdt1_history[i, :], res.qdt2_history[i, :]
        )[0]
        for i in range(len(res.time_history))
    ]

    chassis_kin = [
        model.get_body_kinematics("chassis", bodies) for bodies in bodies_kinematics
    ]

    chassis_p = [kin.p_GB for kin in chassis_kin]
    chassis_v = [kin.v_B for kin in chassis_kin]
    chassis_a = [kin.a_B for kin in chassis_kin]

    plt.figure("chassis_z.png")
    plt.plot(res.time_history, [p[5] for p in chassis_p])
    plt.grid()

    plt.figure("chassis_y.png")
    plt.plot(res.time_history, [p[4] for p in chassis_p])
    plt.grid()

    plt.figure("chassis_yaw.png")
    plt.plot(res.time_history, [p[2] for p in chassis_p])
    plt.grid()

    plt.figure("chassis_z_vel.png")
    plt.plot(res.time_history, [p[5] for p in chassis_v])
    plt.grid()

    plt.figure("chassis_vel_x.png")
    plt.plot(res.time_history, [p[3] * 3.6 for p in chassis_v])
    plt.grid()

    plt.figure("chassis_acc_x.png")
    plt.plot(res.time_history, [p[3] / 9.81 for p in chassis_a])
    plt.grid()

    # plt.figure("fr_wheel_z.png")
    # plt.plot(res.time_history, y[:, 6])
    # plt.grid()

    # plt.figure("wheels_omega.png")
    # plt.plot(res.time_history, y[:, 24])
    # plt.plot(res.time_history, y[:, 25])
    # plt.plot(res.time_history, y[:, 26])
    # plt.plot(res.time_history, y[:, 27])
    # plt.grid()

    plt.show()
