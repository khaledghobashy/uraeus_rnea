import numpy as np
import jax


def skew_matrix(v: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    v : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    x, y, z = v
    mat = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return mat


def spatial_skew(v: np.ndarray) -> np.ndarray:

    orient_elements, trans_elements = np.split(v, 2)

    b00 = skew_matrix(orient_elements)
    b01 = np.zeros((3, 3))
    b10 = skew_matrix(trans_elements)
    b11 = b00

    result = np.vstack([np.hstack([b00, b01]), np.hstack([b10, b11])])
    return result


def cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return spatial_skew(v1) @ v2


def rot_x(theta: float) -> np.ndarray:

    c = np.sin(theta)
    s = np.cos(theta)

    mat = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return mat


def rot_y(theta: float) -> np.ndarray:

    c = np.sin(theta)
    s = np.cos(theta)

    mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return mat


def rot_z(theta: float) -> np.ndarray:

    c = np.sin(theta)
    s = np.cos(theta)

    mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return mat


def spatial_motion_translation(p_PS: np.ndarray) -> np.ndarray:
    X_PS = np.vstack(
        [
            np.hstack([np.eye(3), np.zeros((3, 3))]),
            np.hstack([-skew_matrix(p_PS), np.eye(3)]),
        ]
    )
    return X_PS


def spatial_motion_rotation(R_PS: np.ndarray) -> np.ndarray:
    X_PS = np.vstack(
        [np.hstack([R_PS, np.zeros((3, 3))]), np.hstack([np.zeros((3, 3)), R_PS])]
    )
    return X_PS


def spatial_motion_transformation(R_PS: np.ndarray, p_PS: np.ndarray) -> np.ndarray:

    X_PS = np.vstack(
        [
            np.hstack([R_PS, np.zeros((3, 3))]),
            np.hstack([-R_PS @ skew_matrix(p_PS), R_PS]),
        ]
    )
    return X_PS


def spatial_force_transformation(R_PS: np.ndarray, p_PS: np.ndarray) -> np.ndarray:

    X_PS = np.vstack(
        [
            np.hstack([R_PS, -R_PS @ skew_matrix(p_PS)]),
            np.hstack([np.zeros((3, 3)), R_PS]),
        ]
    )
    return X_PS


def motion_to_force_transform(X_PS: np.ndarray) -> np.ndarray:
    left_half, right_half = np.hsplit(X_PS, 2)
    b00, b10 = np.vsplit(left_half, 2)
    b01, b11 = np.vsplit(right_half, 2)

    X_f_PS = np.vstack(
        [
            np.hstack([b00, b10]),
            np.hstack([b01, b11]),
        ]
    )

    return X_f_PS


def spatial_transform_transpose(X_PS: np.ndarray) -> np.ndarray:

    left_half, right_half = np.hsplit(X_PS, 2)
    b00, b10 = np.vsplit(left_half, 2)
    b01, b11 = np.vsplit(right_half, 2)

    X_SP = np.vstack(
        [
            np.hstack([b00.T, b01.T]),
            np.hstack([b10.T, b11.T]),
        ]
    )

    return X_SP


def vector_from_skew(skew_m: np.ndarray) -> np.ndarray:

    x = skew_m[2, 1]
    y = skew_m[0, 2]
    z = skew_m[1, 0]

    v = np.array([x, y, z])

    return v


def get_position_from_transformation(X_PS: np.ndarray) -> np.ndarray:

    left_half, _ = np.hsplit(X_PS, 2)
    b00, b10 = np.vsplit(left_half, 2)

    skewed_matrix = b00.T @ b10
    p_PS = vector_from_skew(-skewed_matrix)

    return p_PS


def get_euler_angles_from_rotation(R_PS: np.ndarray) -> np.ndarray:
    r00, r01, r02 = R_PS[0, :]
    r10, r11, r12 = R_PS[1, :]
    r20, r21, r22 = R_PS[2, :]

    theta_x = np.arctan2(-r12, r22)
    theta_y = np.arctan2(r02, np.sqrt(r12**2 + r22**2))
    theta_z = np.arctan2(-r01, r00)

    euler_angles = np.array([theta_x, theta_y, theta_z])

    return euler_angles


def get_euler_angles_from_transformation(X_PS: np.ndarray) -> np.ndarray:
    left_half, _ = np.hsplit(X_PS, 2)
    b00, _ = np.vsplit(left_half, 2)

    e_PS = get_euler_angles_from_rotation(b00)

    return b00


def get_pose_from_transformation(X_PS: np.ndarray) -> np.ndarray:

    left_half, _ = np.hsplit(X_PS, 2)
    b00, b10 = np.vsplit(left_half, 2)

    skewed_matrix = b00.T @ b10
    p_PS = vector_from_skew(-skewed_matrix)
    r_PS = -b00 @ p_PS

    e_PS = get_euler_angles_from_rotation(b00)

    return np.hstack([e_PS, r_PS])


def get_orientation_matrix_from_transformation(X_PS: np.ndarray) -> np.ndarray:

    left_half, _ = np.hsplit(X_PS, 2)
    R_PS, _ = np.vsplit(left_half, 2)

    return R_PS
