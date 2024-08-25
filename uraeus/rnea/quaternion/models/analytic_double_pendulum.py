import jax
import jax.numpy as jnp


def analytic_system(l1, l2, theta1_func, theta2_func, t):

    # l1 = -5
    # l2 = -5

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


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    from uraeus.rnea.quaternion.utils import PlotData, plot_animated

    theta1_dt0 = lambda t: 0
    theta2_dt0 = lambda t: 2 * t

    time_array = np.linspace(0, 2 * np.pi, 100)

    analytical_system = lambda t: analytic_system(5, 5, theta1_dt0, theta2_dt0, t)

    analytic_kinematics = [analytical_system(t) for t in time_array]
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

    plot1 = PlotData(
        title="System Animation",
        x_axis=sys_pose_y,
        y_axes=[sys_pose_z],
        x_label="y",
        y_label=["z"],
        x_limit=(-10, 10),
        y_limit=(-10, 10),
        animated=True,
        show_accumulated=False,
    )

    plot2 = PlotData(
        title="System Velocities",
        x_axis=time_array,
        y_axes=[l1_vel_G_y, l1_vel_G_z, l2_vel_G_y, l2_vel_G_z],
        x_label="time",
        y_label=["l1.y", "l1.z", "l2.y", "l2.z"],
        x_limit=(0, max(time_array)),
        y_limit=(min(l2_vel_G_y), max(l2_vel_G_y)),
        animated=True,
        show_accumulated=True,
    )

    fig, animator = plot_animated((2, 1), [plot1, plot2])
    ani = animation.FuncAnimation(fig, animator, frames=99, interval=50)
    plt.show()
