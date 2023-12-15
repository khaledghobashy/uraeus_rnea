import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


from uraeus.rnea.cmbs.rev2.joints import (
    RevoluteJoint,
    SphericalJoint,
    UniversalJoint,
    RotationActuator,
    JointConfigInputs,
    CylindricalJoint,
    TranslationActuator,
)
from uraeus.rnea.cmbs.rev2.bodies import RigidBodyData
from uraeus.rnea.cmbs.rev2.topologies import (
    MultiBodyGraph,
    construct_multibody_graph_data,
    construct_qdt0,
    kinematic_sim,
)


def centered(*args):
    return np.sum(args, 0) / len(args)


def oriented(*args):
    if len(args) == 2:
        v = args[1] - args[0]

    elif len(args) == 3:
        a1 = args[1] - args[0]
        a2 = args[2] - args[0]
        v = np.cross(a2, a1, axisa=0, axisb=0)

    v = v / np.linalg.norm(v)
    return v


# Tire radius
tire_radius = 254

# Upper Control Arm hardpoints
# ============================
ucaf = np.array([-235, 213, 89 + tire_radius])
ucar = np.array([170, 262, 61 + tire_radius])
ucao = np.array([7, 466, 80 + tire_radius])

# Lower Control Arm hardpoints
# ============================
lcaf = np.array([-235, 213, -90 + tire_radius])
lcar = np.array([170, 262, -62 + tire_radius])
lcao = np.array([-7, 483, -80 + tire_radius])

# Tie-Rod hardpoints
# ==================
tri = np.array([-122, 227, -122 + tire_radius])
tro = np.array([-122, 456, -132 + tire_radius])

# Push-Rod hardpoints
# ===================
pushrod_rocker = np.array([6.5, 347, 341 + tire_radius])
pushrod_uca = np.array([6.5, 412, 106 + tire_radius])

# Shock absorber hardpoints
# =========================
shock_chassis = np.array([6.5, 22.5, 377 + tire_radius])
shock_rocker = np.array([6.5, 263, 399 + tire_radius])
shock_mid = centered(shock_chassis, shock_rocker)

rocker_chassis = np.array([6.5, 280, 320 + tire_radius])

# Wheel center
# ============
# wc = np.array([0, 525, 0 + tire_radius])

# rack_center
# ===========
# rack_center = np.array([0, 0, 54 + tire_radius])

# Global axes
# ===========
x_axis = np.array([1.0, 0.0, 0.0])
y_axis = np.array([0.0, 1.0, 0.0])
z_axis = np.array([0.0, 0.0, 1.0])

# Helper Axes
# ===========

# Axis for revolute joint connecting uca to chassis
uca_axis = ucaf - ucar

# Axis for revolute joint connecting uca to chassis
lca_axis = lcaf - lcar

# Axis along tie-rod endpoints
tie_axis = tri - tro

# Axis along push-rod endpoints
push_axis = pushrod_uca - pushrod_rocker

# Axis along shock-absorber endpoints
shock_axis = shock_chassis - shock_rocker


model = MultiBodyGraph("double_wishbone")


# The chassis is treated as the ground part for the system
model.add_body("chassis", RigidBodyData())

# upper control arm
model.add_body("uca", RigidBodyData(centered(ucaf, ucar, ucao)))

# lower control arm
model.add_body("lca", RigidBodyData(centered(lcaf, lcar, lcao)))

# upright
model.add_body("upright", RigidBodyData(centered(ucao, lcao, tro)))

# tie-rod
model.add_body("tie_rod", RigidBodyData(centered(tri, tro)))
model.add_body("push_rod", RigidBodyData(centered(pushrod_rocker, pushrod_uca)))

# rocker
model.add_body(
    "rocker", RigidBodyData(centered(shock_rocker, rocker_chassis, pushrod_rocker))
)

# shock absorber parts
model.add_body("shock_upper", RigidBodyData(centered(shock_chassis, shock_mid)))
model.add_body("shock_lower", RigidBodyData(centered(shock_rocker, shock_mid)))

# wheel-hub
# model.add_body("wheel", RigidBodyData(wc))

# steering rack
# model.add_body("rack", RigidBodyData(rack_center))

rev_uca_config = JointConfigInputs(pos=centered(ucaf, ucar), z_axis_1_G=uca_axis)
model.add_joint("rev_uca", "chassis", "uca", RevoluteJoint, rev_uca_config)

rev_lca_config = JointConfigInputs(pos=centered(lcaf, lcar), z_axis_1_G=lca_axis)
model.add_joint("rev_lca", "chassis", "lca", RevoluteJoint, rev_lca_config)

sph_uca_upright_cfg = JointConfigInputs(pos=ucao, z_axis_1_G=z_axis)
model.add_joint(
    "sph_uca_upright", "uca", "upright", SphericalJoint, sph_uca_upright_cfg
)

sph_lca_upright_cfg = JointConfigInputs(pos=lcao, z_axis_1_G=z_axis)
model.add_joint(
    "sph_lca_upright", "lca", "upright", SphericalJoint, sph_lca_upright_cfg
)

sph_tie_upright_cfg = JointConfigInputs(pos=tro, z_axis_1_G=z_axis)
model.add_joint(
    "sph_tie_upright", "tie_rod", "upright", SphericalJoint, sph_tie_upright_cfg
)

uni_tie_chassis_cfg = JointConfigInputs(
    pos=tri, z_axis_1_G=tie_axis, z_axis_2_G=-tie_axis
)
model.add_joint(
    "uni_tie_chassis", "chassis", "tie_rod", UniversalJoint, uni_tie_chassis_cfg
)

sph_push_uca_cfg = JointConfigInputs(pos=pushrod_uca, z_axis_1_G=z_axis)
model.add_joint("sph_push_uca", "push_rod", "uca", SphericalJoint, sph_push_uca_cfg)

uni_push_rocker_cfg = JointConfigInputs(
    pos=pushrod_rocker, z_axis_1_G=push_axis, z_axis_2_G=-push_axis
)
model.add_joint(
    "uni_push_rocker", "push_rod", "rocker", UniversalJoint, uni_push_rocker_cfg
)

uni_shock_chassis_cfg = JointConfigInputs(
    pos=shock_chassis, z_axis_1_G=shock_axis, z_axis_2_G=-shock_axis
)
model.add_joint(
    "uni_shock_chassis",
    "shock_upper",
    "chassis",
    UniversalJoint,
    uni_shock_chassis_cfg,
)

uni_shock_rocker_cfg = JointConfigInputs(
    pos=shock_rocker, z_axis_1_G=shock_axis, z_axis_2_G=-shock_axis
)
model.add_joint(
    "uni_shock_rocker",
    "shock_lower",
    "rocker",
    UniversalJoint,
    uni_shock_rocker_cfg,
)

cyl_shock_cyl_cfg = JointConfigInputs(pos=shock_mid, z_axis_1_G=shock_axis)
model.add_joint(
    "cyl_shock_cyl",
    "shock_upper",
    "shock_lower",
    CylindricalJoint,
    cyl_shock_cyl_cfg,
)


rev_rocker_chassis_cfg = JointConfigInputs(
    pos=rocker_chassis,
    z_axis_1_G=oriented(rocker_chassis, shock_rocker, pushrod_rocker),
)
model.add_joint(
    "rev_rocker_chassis",
    "rocker",
    "chassis",
    RevoluteJoint,
    rev_rocker_chassis_cfg,
)

dst_shock_cyl_cfg = JointConfigInputs(pos=shock_mid, z_axis_1_G=shock_axis)
model.add_actuator(
    "dst_shock_cyl",
    "cyl_shock_cyl",
    TranslationActuator,
    lambda t: 15 * jnp.sin(t),
)

# model.add_joint("act_rev", "uca", "chassis", RotationActuator, rev_uca_config)

# rev_hub_upright_cfg = JointConfigInputs(pos=wc, z_axis_1_G=y_axis)
# model.add_joint(
#     "rev_hub_upright", "wheel", "upright", RevoluteJoint, rev_hub_upright_cfg
# )

numerical_model = construct_multibody_graph_data(model)


if __name__ == "__main__":
    from uraeus.rnea.cmbs.rev2.utils import timer

    print(model.graph.nodes)
    print(model.graph.edges)

    # graph = nx.Graph(model.graph.edges)
    # nx.draw(graph, with_labels=True, font_weight="bold")
    # plt.show()

    t_array = np.arange(0, 2 * np.pi, 1e-2)
    qdt0 = construct_qdt0(model)

    kinematic_sim = timer(kinematic_sim)

    print("Compiling")
    qdt0s, qdt1s, qdt2s = kinematic_sim(numerical_model, qdt0, t_array[0:2])
    print("")
    qdt0s, qdt1s, qdt2s = kinematic_sim(numerical_model, qdt0, t_array)

    plt.figure("upright_x")
    plt.plot(t_array, [q[3 * 7 + 0] for q in qdt0s])
    plt.grid()

    plt.figure("upright_y")
    plt.plot(t_array, [q[3 * 7 + 1] for q in qdt0s])
    plt.grid()

    plt.figure("upright_z")
    plt.plot(t_array, [q[3 * 7 + 2] for q in qdt0s])
    plt.grid()

    plt.show()
