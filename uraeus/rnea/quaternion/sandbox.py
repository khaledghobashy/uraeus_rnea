import numpy as np

from uraeus.rnea.quaternion.spatial_algebra import (
    skew_M,
    levi_cevita_tensor,
    SpatialPose,
)
from uraeus.rnea.quaternion.algorithms_operations import evaluate_joint_inertia_force
from uraeus.rnea.quaternion.bodies import BodyKinematics

from uraeus.rnea.spatial_algebra import cross
import uraeus.rnea.algorithms_operations as true_operations
import uraeus.rnea.bodies as true_bodies


def spatial_cross(v1, v2):

    v1_v, v1_w = np.split(v1, 2)
    v2_v, v2_w = np.split(v2, 2)

    v3_v = (skew_M @ v1_v @ v2_w) + (skew_M @ v1_w @ v2_v)
    v3_w = (skew_M @ v1_w) @ v2_w

    return -np.array([*v3_v, *v3_w])


def test_skew_matmul_order():

    v1 = np.random.rand(3)
    v2 = np.random.rand(3)

    v3_1 = (skew_M @ v1) @ v2
    v3_2 = (-skew_M @ v2) @ v1

    print(v3_1)
    print(v3_2)

    np.testing.assert_almost_equal(v3_1, v3_2)


def test_spatial_cross():

    v1_v = np.random.rand(3)
    v1_w = np.random.rand(3)
    v1_q = np.array([*v1_v, *v1_w])
    v1_s = np.array([*v1_w, *v1_v])

    v2_v = np.random.rand(3)
    v2_w = np.random.rand(3)
    v2_q = np.array([*v2_v, *v2_w])
    v2_s = np.array([*v2_w, *v2_v])

    v3s = cross(v1_s, v2_s)
    v3q = spatial_cross(v1_q, v2_q)

    v3s_ = np.array([*np.split(v3s, 2)[1], *np.split(v3s, 2)[0]])

    print(v3s)
    print(v3q)
    print(v3s_)

    np.testing.assert_almost_equal(v3q, v3s_)


def test_joint_inertia_force():

    v1_v = np.random.rand(3)
    v1_w = np.random.rand(3)
    v1_q = np.array([*v1_v, *v1_w])
    v1_s = np.array([*v1_w, *v1_v])

    v2_v = np.random.rand(3)
    v2_w = np.random.rand(3)
    v2_q = np.array([*v2_v, *v2_w])
    v2_s = np.array([*v2_w, *v2_v])

    a1_v = np.random.rand(3)
    a1_w = np.random.rand(3)
    a1_q = np.array([*a1_v, *a1_w])
    a1_s = np.array([*a1_w, *a1_v])

    a2_v = np.random.rand(3)
    a2_w = np.random.rand(3)
    a2_q = np.array([*a2_v, *a2_w])
    a2_s = np.array([*a2_w, *a2_v])

    bq = BodyKinematics(
        SpatialPose.Identity(),
        SpatialPose.Identity(),
        v1_q,
        a1_q,
        np.zeros((6,)),
        np.zeros((6,)),
    )
    bs = true_bodies.get_initialized_body_kinematics(
        np.zeros(
            3,
        ),
        np.eye(3),
    )
    bs = true_bodies.BodyKinematics(
        bs.X_BG, bs.X_GB, bs.p_GB, bs.R_GB, v1_s, a1_s, bs.v_G, bs.a_G
    )
    # bs.v_B = v1_s
    # bs.a_B = a1_s

    lin_inertia, ang_inertia = np.split(np.random.rand(6), 2)
    inertia_tensor_q = np.diag([*lin_inertia, *ang_inertia])
    inertia_tensor_s = np.diag([*ang_inertia, *lin_inertia])

    jq_inr = evaluate_joint_inertia_force(bq, inertia_tensor_q, [])
    js_inr = true_operations.evaluate_joint_inertia_force(bs, inertia_tensor_s, [])
    js_inr_ = np.array([*np.split(js_inr, 2)[1], *np.split(js_inr, 2)[0]])

    print("jq_inr = ", jq_inr)
    print("js_inr_ = ", js_inr_)

    np.testing.assert_almost_equal(jq_inr, js_inr_)


if __name__ == "__main__":

    # test_skew_matmul_order()
    # test_spatial_cross()
    test_joint_inertia_force()
