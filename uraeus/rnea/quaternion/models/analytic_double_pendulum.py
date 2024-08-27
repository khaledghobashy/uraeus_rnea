import jax
import jax.numpy as jnp
import numpy as np


def analytic_system(l1, l2, theta1_func, theta2_func, t):

    l1_z_func = lambda t: jnp.cos(-theta1_func(t)) * -l1
    l1_y_func = lambda t: jnp.sin(-theta1_func(t)) * -l1

    l2_z_func = lambda t: l1_z_func(t) + (
        jnp.cos(-theta1_func(t) + -theta2_func(t)) * -l2
    )
    l2_y_func = lambda t: l1_y_func(t) + (
        jnp.sin(-theta1_func(t) + -theta2_func(t)) * -l2
    )

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


def inverse_dynamics(l1, l2, m1, m2, theta1_func, theta2_func, t):

    g = 9.81

    sys_kinematics = analytic_system(l1, l2, theta1_func, theta2_func, t)
    l1_kin, l2_kin = zip(*sys_kinematics)
    l1_pose_G, l1_vel_G, l1_acc_G = l1_kin
    l2_pose_G, l2_vel_G, l2_acc_G = l2_kin
    # l1_acc_G, l2_acc_G = zip(*acc_G)

    # f2_z = (m2 * l2_acc_G[1]) - (m2 * g)
    f2_z = (m2 * g) - (m2 * l2_acc_G[1])
    f2_y = -m2 * l2_acc_G[0]
    torque2 = (
        f2_z * -np.sin(-theta1_func(t) + -theta2_func(t))
        + f2_y * np.cos(-theta1_func(t) + -theta2_func(t))
    ) * l2
    # print("torque2 = ", torque2)

    f1_z = (m1 * g) + f2_z - (m1 * l1_acc_G[1])
    f1_y = f2_y - (m1 * l1_acc_G[0])
    torque1 = (
        f1_z * -np.sin(-theta1_func(t)) + f1_y * np.cos(-theta1_func(t))
    ) * l1 + torque2
    # print("torque1 = ", torque1)

    # print("f0_z = ", f0_z)

    return (f1_y, f1_z, torque1), (f2_y, f2_z, torque2)


class AnalyticDoublePendulum(object):

    def __init__(self, l1, l2, m1, m2):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

    def evaluate_kinematics(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray):

        theta1_dt0, theta2_dt0 = qdt0
        theta1_dt1, theta2_dt1 = qdt1
        theta1_dt2, theta2_dt2 = qdt2

        c1 = np.cos(theta1_dt0)
        s1 = np.sin(theta1_dt0)
        c2 = np.cos(theta1_dt0 + theta2_dt0)
        s2 = np.sin(theta1_dt0 + theta2_dt0)

        d1 = np.array([s1, -c1])
        n1 = np.array([c1, s1])
        d2 = np.array([s2, -c2])
        n2 = np.array([c2, s2])

        r1dt0 = self.l1 * d1
        r2dt0 = r1dt0 + (self.l2 * d2)

        r1dt1 = self.l1 * theta1_dt1 * n1
        r2dt1 = (self.l2 * (theta1_dt1 + theta2_dt1) * n2) + r1dt1

        r1dt2 = (self.l1 * theta1_dt2 * n1) + (theta1_dt1**2 * self.l1 * -d1)
        r2dt2 = (
            (self.l2 * (theta1_dt2 + theta2_dt2) * n2)
            + ((theta1_dt1 + theta2_dt1) ** 2 * self.l2 * -d2)
            + r1dt2
        )

        kinematics = ((r1dt0, r2dt0), (r1dt1, r2dt1), (r1dt2, r2dt2))

        return kinematics

    # def evaluate_inverse_dynamics(
    #     self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    # ):

    #     (r1dt0, r2dt0), _, (r1dt2, r2dt2) = self.evaluate_kinematics(qdt0, qdt1, qdt2)

    #     g = 9.81
    #     # FBD for l2 -> reactions at joint 2
    #     F2g = np.array([0, -self.m2 * g])
    #     F2i = self.m2 * r2dt2
    #     Fj2 = -F2g - F2i

    #     Tj2 = -(np.cross((r2dt0 - r1dt0), F2g)) - (np.cross((r2dt0 - r1dt0), F2i))

    #     # FBD for l1 -> reactions at joint 1
    #     F1g = np.array([0, -self.m1 * g])
    #     F1i = self.m1 * r1dt2

    #     Fj1 = Fj2 - F1g - F1i
    #     Tj1 = Tj2 - np.cross(r1dt0, F1g) - np.cross(r1dt0, F1i) - np.cross(r1dt0, -Fj2)

    #     forces = (np.array([*Fj1, Tj1]), np.array([*Fj2, Tj2]))
    #     return forces

    def evaluate_inverse_dynamics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ):

        (r1dt0, r2dt0), _, (r1dt2, r2dt2) = self.evaluate_kinematics(qdt0, qdt1, qdt2)

        g = 9.81
        # FBD for l2 -> reactions at joint 2
        F2g = np.array([0, -self.m2 * g])
        F2i = self.m2 * r2dt2
        Fj2 = -F2g - F2i

        Tj2 = -(np.cross((r2dt0 - r1dt0), F2g)) - (np.cross((r2dt0 - r1dt0), F2i))

        # FBD for l1 -> reactions at joint 1
        F1g = np.array([0, -self.m1 * g])
        F1i = self.m1 * r1dt2

        Fj1 = Fj2 - F1g - F1i
        Tj1 = Tj2 - np.cross(r1dt0, F1g) - np.cross(r1dt0, F1i) + np.cross(r1dt0, Fj2)

        forces = (np.array([*Fj1, Tj1]), np.array([*Fj2, Tj2]))
        return forces


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    from uraeus.rnea.quaternion.utils import PlotData, plot_animated

    theta1_dt0 = lambda t: np.radians(45)
    theta2_dt0 = lambda t: np.radians(0)

    time_array = np.linspace(0, 2 * np.pi, 100)

    inverse_dynamics_func = lambda t: inverse_dynamics(
        5, 5, 1, 1, theta1_dt0, theta2_dt0, t
    )
    reactions = [inverse_dynamics_func(t) for t in time_array]
    j1F, j2F = zip(*reactions)
    j1F_y, j1F_z, j1F_tau = zip(*j1F)
    j2F_y, j2F_z, j2F_tau = zip(*j2F)

    plt.figure(figsize=(10, 10))
    plt.plot(time_array, j1F_y, label="j1.y")
    plt.plot(time_array, j1F_z, label="j1.z")
    plt.plot(time_array, j2F_y, label="j2.y")
    plt.plot(time_array, j2F_z, label="j2.z")
    plt.grid()
    plt.legend()

    plt.figure(figsize=(10, 10))
    plt.plot(time_array, j1F_tau, label="j1.tau")
    plt.plot(time_array, j2F_tau, label="j2.tau")
    plt.grid()
    plt.legend()
    plt.show()
    # analytical_system = lambda t: analytic_system(5, 5, theta1_dt0, theta2_dt0, t)

    # analytic_kinematics = [analytical_system(t) for t in time_array]
    # sys_pose_G, sys_vel_G, sys_a_G = zip(*analytic_kinematics)

    # l1_pose_G, l2_pose_G = zip(*sys_pose_G)
    # l1_pose_G_y, l1_pose_G_z = zip(*l1_pose_G)
    # l2_pose_G_y, l2_pose_G_z = zip(*l2_pose_G)

    # l1_vel_G, l2_vel_G = zip(*sys_vel_G)
    # l1_vel_G_y, l1_vel_G_z = zip(*l1_vel_G)
    # l2_vel_G_y, l2_vel_G_z = zip(*l2_vel_G)

    # l1_acc_G, l2_acc_G = zip(*sys_a_G)
    # l1_acc_G_y, l1_acc_G_z = zip(*l1_acc_G)
    # l2_acc_G_y, l2_acc_G_z = zip(*l2_acc_G)

    # sys_pose_y = list(zip(np.zeros(len(time_array)), l1_pose_G_y, l2_pose_G_y))
    # sys_pose_z = list(zip(np.zeros(len(time_array)), l1_pose_G_z, l2_pose_G_z))

    # plot1 = PlotData(
    #     title="System Animation",
    #     x_axis=sys_pose_y,
    #     y_axes=[sys_pose_z],
    #     x_label="y",
    #     y_label=["z"],
    #     x_limit=(-10, 10),
    #     y_limit=(-10, 10),
    #     animated=True,
    #     show_accumulated=False,
    # )

    # plot2 = PlotData(
    #     title="System Velocities",
    #     x_axis=time_array,
    #     y_axes=[l1_vel_G_y, l1_vel_G_z, l2_vel_G_y, l2_vel_G_z],
    #     x_label="time",
    #     y_label=["l1.y", "l1.z", "l2.y", "l2.z"],
    #     x_limit=(0, max(time_array)),
    #     y_limit=(min(l2_vel_G_y), max(l2_vel_G_y)),
    #     animated=True,
    #     show_accumulated=True,
    # )

    # fig, animator = plot_animated((2, 1), [plot1, plot2])
    # ani = animation.FuncAnimation(fig, animator, frames=99, interval=50)
    # plt.show()
