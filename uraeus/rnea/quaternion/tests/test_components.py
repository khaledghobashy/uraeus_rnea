import numpy as np

from uraeus.rnea.quaternion.spatial_algebra import (
    SpatialPose,
    transform_vector,
    quaternion_from_axis_angle,
)
from uraeus.rnea.quaternion.joints import RevoluteJoint, JointConfigInputs
from uraeus.rnea.quaternion.mobilizers import RevoluteMobilizer
from uraeus.rnea.quaternion.bodies import RigidBody, RigidBodyData


def double_pendulum(psi1, psi2):
    l0 = RigidBody("l0", RigidBodyData())
    l1 = RigidBody("l1", RigidBodyData(np.array([0, 0, -5])))
    l2 = RigidBody("l2", RigidBodyData(np.array([0, 0, -5])))

    j_axis = np.array([1, 0, 0])
    z_axis_G = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis_G, j_axis)
    angle = np.arccos(z_axis_G @ j_axis)

    p_j1l1 = SpatialPose(
        np.array([0, 0, 0]), quaternion_from_axis_angle(angle, rot_axis)
    )

    j1 = RevoluteMobilizer()
    j2 = RevoluteMobilizer()
    j1_kin = j1.evaluate_kinematics(np.array([psi1]), np.array([0]), np.array([0]))
    j2_kin = j2.evaluate_kinematics(np.array([psi2]), np.array([0]), np.array([0]))

    # print(l1.kinematics.p_BG)
    # print(l2.kinematics.p_BG)

    # print(j1_kin.p_FM @ p_j1l1 @ l1.kinematics.p_BG @ j2_kin.p_FM @ l2.kinematics.p_BG)

    # l1_kin = l1.kinematics.p_BG @ (j1_kin.p_FM @ p_j1l1).inv()
    # l2_kin = (l2.kinematics.p_BG.inv() @ (j2_kin.p_FM @ p_j1l1).inv()) @ l1_kin.inv()

    p_l0l1 = j1_kin.p_FM.inv() @ p_j1l1
    p_l1l2 = j2_kin.p_FM.inv() @ p_j1l1

    l1_kin = p_l0l1
    l2_kin = p_l1l2 @ p_l0l1

    # print(l1_kin)
    # print(l2_kin)

    return l1_kin, l2_kin


def test1():

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    angles = np.linspace(0, 2 * np.pi, 100)

    l1_r_y = []
    l1_r_z = []
    for i in angles:
        l1_p_G, l2_p_G = double_pendulum(0, i)
        l1_r_G = l1_p_G.r
        l2_r_G = l2_p_G.r
        # l1_r_G = transform_vector(l1_p_G.q, l1_p_G.r)
        # l2_r_G = transform_vector(l2_p_G.q, l2_p_G.r)

        print("l1_r_G.norm = ", np.linalg.norm(l1_r_G))
        print("l2_r_G.norm = ", np.linalg.norm(l2_r_G))
        print("l1_r_G.x = ", float(l1_r_G[0]))
        print("l2_r_G.x = ", float(l2_r_G[0]))

        l1_r_y.append((0, float(l1_r_G[1]), float(l2_r_G[1])))
        l1_r_z.append((0, float(l1_r_G[2]), float(l2_r_G[2])))

    def animate(i):
        print(angles[i])
        plt.cla()
        plt.grid()
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.plot(l1_r_y[i], l1_r_z[i])
        plt.plot(l1_r_y[i], l1_r_z[i], "o")
        return

    fig = plt.figure(figsize=(10, 10))
    plt.grid()
    ani = animation.FuncAnimation(fig, animate, frames=99, interval=50)
    plt.show()


# def test_transformation_order()

test1()
