import numpy as np
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# def analytic_system(qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray):

#     theta1_dt0, theta2_dt0 = qdt0
#     theta1_dt1, theta2_dt1 = qdt1
#     theta1_dt2, theta2_dt2 = qdt2

#     l1 = -5
#     l2 = -5

#     # l1_z = np.cos(theta1_dt0) * l1
#     # l1_y = np.sin(theta1_dt0) * l1

#     # l2_z = l1_z + (np.cos(theta1_dt0 + theta2_dt0) * l2)
#     # l2_y = l1_y + (np.sin(theta1_dt0 + theta2_dt0) * l2)


#     # l1_v_t = theta1_dt1 * l1
#     # l1_v_z = theta1_dt1 * (-np.sin(theta1_dt0)) * l1
#     # l1_v_y = theta1_dt1 * (np.cos(theta1_dt0)) * l1

#     # l2_v_z = l1_v_z + (
#     #     (theta1_dt1 + theta2_dt1) * -np.sin(theta1_dt0 + theta2_dt0) * l2
#     # )
#     # l2_v_y = l1_v_y + ((theta1_dt1 + theta2_dt1) * np.cos(theta1_dt0 + theta2_dt0) * l2)


#     l1_z_func = lambda theta: np.cos(theta) * l1
#     l1_y_func = lambda theta: np.sin(theta) * l1

#     l2_z_func = lambda theta1, theta2: l1_z_func(theta1) + (np.cos(theta1 + theta2) * l2)
#     l2_y_func = lambda theta1, theta2: l1_y_func(theta1) + (np.sin(theta1 + theta2) * l2)

#     l1_v_z_func = jax.jacfwd(l1_z_func)

#     l1_v_t = theta1_dt1 * l1
#     l1_v_z = theta1_dt1 * (-np.sin(theta1_dt0)) * l1
#     l1_v_y = theta1_dt1 * (np.cos(theta1_dt0)) * l1

#     l2_v_z = l1_v_z + (
#         (theta1_dt1 + theta2_dt1) * -np.sin(theta1_dt0 + theta2_dt0) * l2
#     )
#     l2_v_y = l1_v_y + ((theta1_dt1 + theta2_dt1) * np.cos(theta1_dt0 + theta2_dt0) * l2)

#     l1_a_z = (theta1_dt2 * l1 * (-np.sin(theta1_dt0))) + (
#         theta1_dt1 * l1 * (theta1_dt1 * -np.cos(theta1_dt0))
#     )

#     return ((l1_y, l1_z), (l2_y, l2_z)), ((l1_v_y, l1_v_z), (l2_v_y, l2_v_z))


def analytic_system(theta1_func, theta2_func, t):

    l1 = -5
    l2 = -5

    l1_z_func = lambda t: jnp.cos(theta1_func(t)) * l1
    l1_y_func = lambda t: jnp.sin(theta1_func(t)) * l1

    l2_z_func = lambda t: l1_z_func(t) + (jnp.cos(theta1_func(t) + theta2_func(t)) * l2)
    l2_y_func = lambda t: l1_y_func(t) + (jnp.sin(theta1_func(t) + theta2_func(t)) * l2)

    l1_v_z_func = jax.jacfwd(l1_z_func)
    l1_v_y_func = jax.jacfwd(l1_y_func)
    l2_v_z_func = jax.jacfwd(l2_z_func)
    l2_v_y_func = jax.jacfwd(l2_y_func)

    l1_a_z_func = jax.jacfwd(l1_v_z_func)
    l1_a_y_func = jax.jacfwd(l1_v_y_func)
    l2_a_z_func = jax.jacfwd(l2_v_z_func)
    l2_a_y_func = jax.jacfwd(l2_v_y_func)

    l1_y = l1_y_func(t)
    l1_z = l1_z_func(t)
    l2_y = l2_y_func(t)
    l2_z = l2_z_func(t)

    l1_v_y = l1_v_y_func(t)
    l1_v_z = l1_v_z_func(t)
    l2_v_y = l2_v_y_func(t)
    l2_v_z = l2_v_z_func(t)

    l1_a_y = l1_a_y_func(t)
    l1_a_z = l1_a_z_func(t)
    l2_a_y = l2_a_y_func(t)
    l2_a_z = l2_a_z_func(t)

    sys_kinematics = (
        ((l1_y, l1_z), (l2_y, l2_z)),
        ((l1_v_y, l1_v_z), (l2_v_y, l2_v_z)),
        ((l1_a_y, l1_a_z), (l2_a_y, l2_a_z)),
    )

    return sys_kinematics


if __name__ == "__main__":

    theta1_dt0 = lambda t: 0
    theta2_dt0 = lambda t: 2 * t

    theta1_dt1 = jax.jacfwd(theta1_dt0)
    theta2_dt1 = jax.jacfwd(theta2_dt0)
    time_array = np.linspace(0, 2 * np.pi, 100)

    # analytic_kinematics = [
    #     analytic_system(
    #         np.array([theta1_dt0(t), theta2_dt0(t)]),
    #         np.array([theta1_dt1(t), theta2_dt1(t)]),
    #         np.array([0, 0]),
    #     )
    #     for t in time_array
    #     # for t in time_array[0:1]
    # ]
    analytic_kinematics = [
        analytic_system(theta1_dt0, theta2_dt0, t) for t in time_array
    ]
    sys_pose_G, sys_vel_G, sys_a_G = zip(*analytic_kinematics)

    l1_pose_G, l2_pose_G = zip(*sys_pose_G)
    l1_pose_G_y, l1_pose_G_z = zip(*l1_pose_G)
    l2_pose_G_y, l2_pose_G_z = zip(*l2_pose_G)

    l1_vel_G, l2_vel_G = zip(*sys_vel_G)
    l1_vel_G_y, l1_vel_G_z = zip(*l1_vel_G)
    l2_vel_G_y, l2_vel_G_z = zip(*l2_vel_G)

    l1_acc_G, l2_acc_G = zip(*sys_a_G)
    l1_acc_G_y, l1_acc_G_z = zip(*l1_acc_G)
    l2_acc_G_y, l2_acc_G_z = zip(*l2_acc_G)

    sys_pose_y = list(zip(np.zeros(len(time_array)), l1_pose_G_y, l2_pose_G_y))
    sys_pose_z = list(zip(np.zeros(len(time_array)), l1_pose_G_z, l2_pose_G_z))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7.5), height_ratios=(2, 1, 1))
    fig.tight_layout()

    def animate(i):
        print(time_array[i])
        ax1.cla()
        ax2.cla()
        ax3.cla()

        ax1.grid()
        ax2.grid()
        ax3.grid()

        ax1.set_xlim([-10, 10])
        ax1.set_ylim([-10, 10])

        ax2.set_xlim([0, max(time_array)])
        ax2.set_ylim([min(l2_vel_G_y), max(l2_vel_G_y)])

        ax3.set_xlim([0, max(time_array)])
        ax3.set_ylim([min(l2_acc_G_y), max(l2_acc_G_y)])

        ax1.plot(sys_pose_y[i], sys_pose_z[i])
        ax1.plot(sys_pose_y[i], sys_pose_z[i], "o")

        ax2.plot(time_array[:i], l1_vel_G_y[:i], label="l1.vel.y")
        ax2.plot(time_array[:i], l1_vel_G_z[:i], label="l1.vel.z")
        ax2.plot(time_array[:i], l2_vel_G_y[:i], label="l2.vel.y")
        ax2.plot(time_array[:i], l2_vel_G_z[:i], label="l2.vel.z")
        ax2.legend()

        ax3.plot(time_array[:i], l1_acc_G_y[:i], label="l1.acc.y")
        ax3.plot(time_array[:i], l1_acc_G_z[:i], label="l1.acc.z")
        ax3.plot(time_array[:i], l2_acc_G_y[:i], label="l2.acc.y")
        ax3.plot(time_array[:i], l2_acc_G_z[:i], label="l2.acc.z")
        ax3.legend()
        return

    plt.grid()
    ani = animation.FuncAnimation(fig, animate, frames=99, interval=50)
    plt.show()
