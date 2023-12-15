from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

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

model = MultiBodyGraph("double_wishbone")


# The chassis is treated as the ground part for the system
model.add_body("chassis", RigidBodyData())

# upper control arm
model.add_body("uca", RigidBodyData(centered(ucaf, ucar, ucao)))

# lower control arm
model.add_body("lca", RigidBodyData(centered(lcaf, lcar, lcao)))

# upright
model.add_body("upright", RigidBodyData(centered(ucao, lcao)))

# tie-rod
model.add_body("tie_rod", RigidBodyData(centered(tri, tro)))

# uca joint
rev_uca_config = JointConfigInputs(pos=centered(ucaf, ucar), z_axis_1_G=uca_axis)
model.add_joint("rev_uca", "chassis", "uca", RevoluteJoint, rev_uca_config)

# lca joint
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

model.add_actuator("act_rev", "rev_uca", RotationActuator, lambda t: 0.7 * jnp.sin(t))

numerical_model = construct_multibody_graph_data(model)

if __name__ == "__main__":
    print(model.graph.nodes)
    print(model.graph.edges)
    # print(model.dof)

    # graph = nx.Graph(model.graph.edges)
    # nx.draw(graph, with_labels=True, font_weight="bold")
    # plt.show()

    # from uraeus.rnea.cmbs.models.topologies import minimizer

    # qdt0 = construct_qdt0(model)
    # q = minimizer(numerical_model, qdt0, 0.2)
    # print(q)
    # quit()

    t_array = np.arange(0, 2 * np.pi, 1e-2)
    qdt0 = construct_qdt0(model)

    qdt0s, qdt1s, qdt2s = kinematic_sim(numerical_model, qdt0, t_array)

    plt.figure("upright")
    plt.plot(t_array, [q[3 * 7 + 0] for q in qdt0s], label="x")
    plt.plot(t_array, [q[3 * 7 + 1] for q in qdt0s], label="y")
    plt.plot(t_array, [q[3 * 7 + 2] for q in qdt0s], label="z")
    plt.legend()
    plt.grid()

    plt.show()
