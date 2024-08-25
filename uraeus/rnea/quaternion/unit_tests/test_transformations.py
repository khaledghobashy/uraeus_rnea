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
    transform_screw,
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


def normalize(v):
    return v / np.sqrt(v @ v)


def test_quaternion_operations():
    axis1 = np.random.rand(3)
    axis2 = np.random.rand(3)
    q1 = quaternion_from_axis_angle(np.radians(45), axis1)
    q2 = quaternion_from_axis_angle(np.radians(30), axis2)
    print(f"q1.norm = {np.linalg.norm(q1)}")
    print(f"q2.norm = {np.linalg.norm(q2)}")

    q12 = quaternion_multiply(q1, q2)
    q21 = quaternion_multiply(q2, q1)
    q12_i = quaternion_inverse(q12)
    q1i2i = quaternion_multiply(quaternion_inverse(q2), quaternion_inverse(q1))

    # print(f"q12 = {q12}")
    print(f"q12_i = {q12_i}")
    print(f"q12_i.norm = {np.linalg.norm(q12_i)}")
    print(f"q1i2i = {q1i2i}")
    print(f"q1i2i.norm = {np.linalg.norm(q1i2i)}")
    # print(f"q21 = {q21}")

    p1 = SpatialPose(axis1, q1)
    p2 = SpatialPose(axis2, q2)

    p12 = p1 @ p2
    p12_i = p12.inv()
    p1i2i = p2.inv() @ p1.inv()
    print(f"p12_i = {p12_i}")
    print(f"p1i2i = {p1i2i}")

    print(f"p1 @ p1.inv() = {p1 @ p1.inv()}")
    print(f"p12 @ p12.inv() = {p12 @ p12.inv()}")

    np.testing.assert_almost_equal(
        np.array((p12 @ p12.inv()).r),
        np.zeros(
            3,
        ),
    )
    # np.testing.assert_almost_equal(np.array(p12_i.r), np.array(p1i2i.r))
    np.testing.assert_almost_equal(np.array(p12_i.q), np.array(p1i2i.q))


def test_pose_transformation():
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

    r_AB = R_AB.T @ -np.array([1, 5, 0])
    r_BC = R_BC.T @ -np.array([0, 1, 13])

    p_AB = SpatialPose(r_AB, q_AB)
    p_BC = SpatialPose(r_BC, q_BC)
    p_AC = p_BC @ p_AB
    p_CA1 = (p_BC @ p_AB).inv()
    p_CA2 = p_AB.inv() @ p_BC.inv()
    print(f"p_CA1.r = {p_CA1.r}, {np.linalg.norm(p_CA1.r)}")
    print(f"p_CA2.r = {p_CA2.r}, {np.linalg.norm(p_CA2.r)}")

    X_AB = spatial_motion_transformation(jnp.array(R_AB, dtype=float), r_AB)
    X_BC = spatial_motion_transformation(jnp.array(R_BC, dtype=float), r_BC)

    X_AC = X_BC @ X_AB
    X_CA1 = spatial_transform_transpose(X_AC)
    X_CA2 = spatial_transform_transpose(X_AB) @ spatial_transform_transpose(X_BC)

    X_AC_ID = get_pose_from_transformation(X_AC @ X_CA1)[3:]
    p_AC_ID = (p_CA1 @ p_AC).r
    # p_AC_ID = (p_AC @ p_CA1).r
    print(f"X_AC_ID = {X_AC_ID}, {np.linalg.norm(X_AC_ID)}")
    print(f"p_AC_ID = {p_AC_ID}, {np.linalg.norm(p_AC_ID)}")
    np.testing.assert_almost_equal(np.array(X_CA1), np.array(X_CA2))

    # print(get_pos_from_transformation(X_AB))

    true_pos1 = get_pose_from_transformation(X_AC)[3:]
    true_pos1T = get_pose_from_transformation(X_CA1)[3:]
    true_pos2 = get_pos_from_transformation(X_AC)

    print(f"True pose1 = {true_pos1}, {np.linalg.norm(true_pos1)}")
    print(f"True pose1.T = {true_pos1T}, {np.linalg.norm(true_pos1T)}")
    print(f"Test pose1 = {-transform_vector(q_AC, true_pos2)}")
    print(f"True pose2 = {true_pos2}, {np.linalg.norm(true_pos2)}")
    # print(f"True pose2 = {true_pos2}, {np.linalg.norm(true_pos2)}")
    print(f"Test pose2 = {p_AC.r}, {np.linalg.norm(p_AC.r)}")
    # print(f"True quat = {get_orientation_matrix_from_transformation(X_AC)}")
    print(f"True orientation = \n{R_AC}")
    print(f"Test orientation = \n{quaternion_to_dcm(p_AC.q)}")
    print(f"Test orientation = \n{quaternion_to_dcm(q_AC)}")
    np.testing.assert_almost_equal(quaternion_to_dcm(p_AC.q), np.array(R_AC))
    # np.testing.assert_almost_equal(quaternion_to_dcm(q_AC), np.array(R_AC))
    # np.testing.assert_almost_equal(p_AC.q, np.array(dcm_to_quaternion(get_orientation_matrix_from_transformation(X_AC))))
    np.testing.assert_almost_equal(p_AC.r, np.array(true_pos2))


def test_pose_inverse_operations():
    print("Testing SpatialPose Inverse Operations!")
    q_AB = normalize(np.random.rand(4))
    q_BC = normalize(np.random.rand(4))

    r_AB = np.random.rand(3)
    r_BC = np.random.rand(3)
    # r_AB = np.array([1, 2, 3])
    # r_BC = np.array([5, 6, 7])
    r_AC = r_AB + r_BC

    # print(f"r_AB_G = {r_AB}")
    # print(f"r_BC_G = {r_BC}")
    # print(f"r_AC_G = {r_AC}, norm = {np.linalg.norm(r_AC)}")

    p_AB = SpatialPose(r_AB, q_AB)
    p_BC = SpatialPose(transform_vector(q_AB, r_BC), q_BC)

    p_AC = p_BC @ p_AB
    p_AC_inv1 = (p_BC @ p_AB).inv()
    p_AC_inv2 = p_AB.inv() @ p_BC.inv()

    p_I1 = p_AC @ p_AC_inv1
    p_I2 = p_AC @ p_AC_inv2

    p_I0 = SpatialPose.Identity()

    # print(f"r_AC_G = {transform_vector(p_AC.inv().q, p_AC.)}")

    # print(f"p_AC = {p_AC}")
    # print(f"r_AC_G = {p_AC_inv1.inv().r}, norm = {np.linalg.norm(p_AC_inv1.inv().r)}")
    # print(
    #     f"r_AC_G = {transform_vector(p_AC_inv1.q, p_AC_inv1.r)}, norm = {np.linalg.norm(p_AC_inv1.inv().r)}"
    # )
    # print(f"p_AC_inv1 = {p_AC_inv1}, norm = {np.linalg.norm(p_AC_inv1.r)}")
    # print(f"p_AC_inv2 = {p_AC_inv2}, norm = {np.linalg.norm(p_AC_inv2.r)}")
    # print(f"p_I1 = {p_I1}")
    # print(f"p_I2 = {p_I2}")

    np.testing.assert_almost_equal(np.array(p_AC_inv1.r), np.array(p_AC_inv2.r))
    np.testing.assert_almost_equal(np.array(p_AC_inv1.q), np.array(p_AC_inv2.q))
    np.testing.assert_almost_equal(np.array(p_I1.r), np.array(p_I0.r))
    np.testing.assert_almost_equal(np.array(p_I1.q), np.array(p_I0.q))
    np.testing.assert_almost_equal(np.array(p_I2.r), np.array(p_I0.r))
    np.testing.assert_almost_equal(np.array(p_I2.q), np.array(p_I0.q))


def test_screw_operations():

    v_A = np.array([0, 0, 0, 1, 0, 0])
    p_AB = SpatialPose(np.array([5, 0, 0]), np.array([0.70710678, 0, -0.70710678, 0]))
    v_B = transform_screw(p_AB, v_A)
    print(v_B)


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

    # test_quaternion_transformation()

    # test_quaternion_operations()
    # test_pose_transformation()
    # test_pose_operations()
    # test_pose_inverse_operations()
    test_screw_operations()
