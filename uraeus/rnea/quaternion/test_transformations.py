import numpy as np
import jax.numpy as jnp

from uraeus.rnea.spatial_algebra import (
    get_orientation_matrix_from_transformation,
    get_pose_from_transformation,
    spatial_motion_transformation,
    spatial_transform_transpose,
    hsplit,
    vsplit,
    vector_from_skew,
)

from uraeus.rnea.quaternion.spatial_algebra import (
    rot_x,
    rot_y,
    rot_z,
    SpatialPose,
    transform_vector,
    dcm_to_quaternion,
    quaternion_to_dcm,
    quaternion_multiply,
    quaternion_from_axis_angle,
    quaternion_inverse,
)


def get_pos_from_transformation(X_PS: jnp.ndarray) -> jnp.ndarray:
    left_half, _ = hsplit(X_PS)
    b00, b10 = vsplit(left_half)

    skewed_matrix = b00.T @ b10
    p_PS = vector_from_skew(-skewed_matrix)
    r_PS = p_PS

    return r_PS


def test_quaternion_transformation():
    q_AB = np.random.rand(4)
    q_AB = q_AB / np.linalg.norm(q_AB)
    q_BC = np.random.rand(4)
    q_BC = q_BC / np.linalg.norm(q_BC)

    print(q_AB, q_BC)

    # R_AB = rot_z(np.radians(35)) @ rot_y(10) @ rot_x(30)
    # R_BC = rot_z(np.radians(50)) @ rot_y(137) @ rot_x(20)

    R_AB = quaternion_to_dcm(q_AB)
    R_BC = quaternion_to_dcm(q_BC)
    R_AC = R_BC @ R_AB

    # q_AB = np.array(dcm_to_quaternion(R_AB), dtype=float)
    # q_BC = np.array(dcm_to_quaternion(R_BC), dtype=float)
    q_AC = quaternion_multiply(q_AB, q_BC)
    print(np.linalg.norm(q_AB), np.linalg.norm(q_BC), np.linalg.norm(q_AC))

    r_AB = np.array([1, 5, 0])
    r_BC = np.array([0, 1, 13])

    p_AB = SpatialPose(r_AB, q_AB)
    p_BC = SpatialPose(r_BC, q_BC)
    # p_AC = p_AB @ p_BC
    p_AC = p_BC @ p_AB
    # p_CA = p_AC.inv()

    X_AB = spatial_motion_transformation(jnp.array(R_AB, dtype=float), r_AB)
    X_BC = spatial_motion_transformation(jnp.array(R_BC, dtype=float), r_BC)

    X_AC = X_BC @ X_AB
    X_CA = spatial_transform_transpose(X_AC)

    # print(get_pos_from_transformation(X_AB))

    true_pos1 = get_pose_from_transformation(X_AC)[3:]
    true_pos2 = get_pos_from_transformation(X_AC)

    print(f"True pose = {true_pos1}, {np.linalg.norm(true_pos1)}")
    print(f"True pose = {-transform_vector(q_AC, true_pos2)}")
    print(f"True pose = {true_pos2}, {np.linalg.norm(true_pos2)}")
    print(f"Test pose = {p_AC.r}, {np.linalg.norm(p_AC.r)}")
    # print(f"True quat = {get_orientation_matrix_from_transformation(X_AC)}")
    print(f"True orientation = \n{R_AC}")
    print(f"Test orientation = \n{quaternion_to_dcm(p_AC.q)}")
    print(f"Test orientation = \n{quaternion_to_dcm(q_AC)}")
    np.testing.assert_almost_equal(quaternion_to_dcm(p_AC.q), np.array(R_AC))
    np.testing.assert_almost_equal(quaternion_to_dcm(q_AC), np.array(R_AC))
    # np.testing.assert_almost_equal(p_AC.q, np.array(dcm_to_quaternion(get_orientation_matrix_from_transformation(X_AC))))
    np.testing.assert_almost_equal(p_AC.r, np.array(true_pos2))


def double_pendulum_pose(psi_1, psi_2):
    z_axis_G = np.array([0, 0, 1])
    rot_axis = np.array([1, 0, 0])
    j1_angle = psi_1
    j2_angle = psi_2
    j1_quat = quaternion_from_axis_angle(j1_angle, rot_axis)
    j2_quat = quaternion_from_axis_angle(j2_angle, rot_axis)

    j1_pose = SpatialPose(np.array([0, 0, 0]), j1_quat)
    l1_pose = SpatialPose(np.array([0, 0, -5]), quaternion_from_axis_angle(0, z_axis_G))
    j2_pose = SpatialPose(np.array([0, 0, 0]), j2_quat)
    l2_pose = SpatialPose(np.array([0, 0, -5]), quaternion_from_axis_angle(0, z_axis_G))

    l1_pose_new = j1_pose @ l1_pose
    l2_pose_new = l1_pose_new @ j2_pose @ l2_pose

    # print(f"l1_pose_new = {l1_pose_new}")
    # print(f"l1.r_G = {transform_vector(l1_pose_new.q, l1_pose_new.r)}")
    # print(f"l2_pose_new = {l2_pose_new}")
    # print(f"l2.r_G = {transform_vector(l2_pose_new.q, l2_pose_new.r)}")

    print(f"j2.r_G = {transform_vector(j2_pose.q, j2_pose.r)}")
    print(f"l1.r_G = {transform_vector(l1_pose_new.q, l1_pose_new.r)}")
    print(f"l2.r_G = {transform_vector(l2_pose_new.q, l2_pose_new.r)}")
    print("")

    return l1_pose_new, l2_pose_new


def double_pendulum_pose(psi_1, psi_2):
    z_axis_G = np.array([0, 0, 1])
    rot_axis = np.array([1, 0, 0])
    j1_angle = psi_1
    j2_angle = psi_2

    j_axis = np.array([1, 0, 0])
    rot_axis1 = np.cross(z_axis_G, j_axis)
    angle = np.arccos(z_axis_G @ j_axis)

    j1_transform = quaternion_from_axis_angle(angle, rot_axis1)
    j2_transform = quaternion_from_axis_angle(angle, rot_axis1)

    j1_quat = quaternion_multiply(
        quaternion_from_axis_angle(j1_angle, z_axis_G), quaternion_inverse(j1_transform)
    )
    j2_quat = quaternion_multiply(
        quaternion_from_axis_angle(j2_angle, z_axis_G), quaternion_inverse(j2_transform)
    )
    # j2_quat = quaternion_from_axis_angle(j2_angle, z_axis_G)

    j1_pose = SpatialPose(np.array([0, 0, 0]), j1_quat)
    l1_pose = SpatialPose(np.array([0, 0, -5]), quaternion_from_axis_angle(0, z_axis_G))
    j2_pose = SpatialPose(np.array([0, 0, 0]), j2_quat)
    l2_pose = SpatialPose(np.array([0, 0, -5]), quaternion_from_axis_angle(0, z_axis_G))

    l1_pose_new = j1_pose @ l1_pose
    l2_pose_new = l1_pose_new @ j2_pose @ l2_pose

    # print(f"l1_pose_new = {l1_pose_new}")
    # print(f"l1.r_G = {transform_vector(l1_pose_new.q, l1_pose_new.r)}")
    # print(f"l2_pose_new = {l2_pose_new}")
    # print(f"l2.r_G = {transform_vector(l2_pose_new.q, l2_pose_new.r)}")

    print(f"j2.r_G = {transform_vector(j2_pose.q, j2_pose.r)}")
    print(f"l1.r_G = {transform_vector(l1_pose_new.q, l1_pose_new.r)}")
    print(f"l2.r_G = {transform_vector(l2_pose_new.q, l2_pose_new.r)}")
    print("")

    return l1_pose_new, l2_pose_new


def test_successive_transform():
    z_axis_G = np.array([0, 0, 1])
    rot_axis = np.array([1, 0, 0])
    j1_in_angle = 0
    j2_in_angle = 0
    j1_quat = quaternion_from_axis_angle(j1_in_angle, rot_axis)
    j2_quat = quaternion_from_axis_angle(j2_in_angle, rot_axis)

    rot_axis1 = np.cross(z_axis_G, rot_axis)
    orient_angle = np.arccos(z_axis_G @ rot_axis)

    j1_transform = quaternion_from_axis_angle(orient_angle, rot_axis1)
    j2_transform = quaternion_from_axis_angle(orient_angle, rot_axis1)

    j1_trn_base = quaternion_from_axis_angle(j1_in_angle, z_axis_G)
    j2_trn_base = quaternion_from_axis_angle(j2_in_angle, z_axis_G)

    print(j1_quat)
    print(quaternion_multiply(j1_transform, j1_trn_base))

    print(j2_quat)
    print(quaternion_multiply(j2_transform, j2_trn_base))


def test_2():

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    angles = np.linspace(0, 2 * np.pi, 100)

    l1_r_y = []
    l1_r_z = []
    for i in angles:
        l1_p_G, l2_p_G = double_pendulum_pose(i, 0)
        l1_r_G = transform_vector(l1_p_G.q, l1_p_G.r)
        l2_r_G = transform_vector(l2_p_G.q, l2_p_G.r)

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


# test_2()

if __name__ == "__main__":

    # z_axis_G = np.array([0, 0, 1])
    # rot_axis = np.array([1, 0, 0])
    # j1_angle = np.radians(45)
    # j2_angle = 0

    # j_axis = np.array([1, 0, 0])
    # rot_axis1 = np.cross(z_axis_G, j_axis)
    # angle = np.arccos(z_axis_G @ j_axis)

    # j1_transform = quaternion_from_axis_angle(angle, rot_axis1)

    # j1_quat = quaternion_multiply(
    #     quaternion_from_axis_angle(j1_angle, z_axis_G), j1_transform
    # )

    # print(quaternion_to_dcm(j1_transform))
    # print("")
    # print(quaternion_to_dcm(j1_quat))

    # test_2()

    # test_successive_transform()

    test_quaternion_transformation()
