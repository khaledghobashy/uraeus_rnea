from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

from uraeus.rnea.cmbs.rev2.joints import (
    RevoluteJoint,
    SphericalJoint,
    UniversalJoint,
    RotationActuator,
    JointConfigInputs,
)
from uraeus.rnea.cmbs.rev2.bodies import RigidBodyData
from uraeus.rnea.cmbs.rev2.topologies import (
    MultiBodyGraph,
    construct_multibody_graph_data,
    construct_qdt0,
    kinematic_sim,
)

# print(jax.make_jaxpr(RevoluteJoint.pos_constraint)())
# quit()


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


point_A = np.array([0, 0, 0])
point_B = np.array([0, 0, 2.00])
point_C = np.array([-7.50, -8.50, 6.50])
point_D = np.array([-4.00, -8.50, 0])

x_axis = np.array([1.0, 0.0, 0.0])
y_axis = np.array([0.0, 1.0, 0.0])
z_axis = np.array([0.0, 0.0, 1.0])

axis_bc = point_C - point_B
axis_cd = point_D - point_C

graph = MultiBodyGraph("fourbar")

graph.add_body("l0", RigidBodyData())
graph.add_body("l1", RigidBodyData(location=centered(point_A, point_B)))
graph.add_body("l2", RigidBodyData(location=centered(point_B, point_C)))
graph.add_body("l3", RigidBodyData(location=centered(point_C, point_D)))


j1_config = JointConfigInputs(pos=point_A, z_axis_1_G=x_axis)
j2_config = JointConfigInputs(pos=point_B, z_axis_1_G=z_axis)
j3_config = JointConfigInputs(pos=point_C, z_axis_1_G=axis_bc, z_axis_2_G=axis_cd)
j4_config = JointConfigInputs(pos=point_D, z_axis_1_G=y_axis)

print(
    RevoluteJoint(
        name="rev",
        body_i=graph.bodies["l0"],
        body_j=graph.bodies["l1"],
        joint_config=j1_config,
    )
)

graph.add_joint("j1", "l0", "l1", RevoluteJoint, j1_config)
graph.add_joint("j2", "l1", "l2", SphericalJoint, j2_config)
graph.add_joint("j3", "l2", "l3", UniversalJoint, j3_config)
graph.add_joint("j4", "l3", "l0", RevoluteJoint, j4_config)
graph.add_actuator("act", "j1", RotationActuator, lambda t: 2 * np.pi * t)

multibody_data = construct_multibody_graph_data(graph)

print(graph.graph.nodes)
print(graph.graph.edges)

qdt0 = construct_qdt0(graph)


print(multibody_data.pos_constraints(qdt0, np.pi / 2).shape)
# print(multibody_data.vel_constraints(qdt0, np.pi / 2))
# print(mbs_eq.acc_eqn(qdt0, qdt0, np.pi / 2))
# print(mbs_eq.constraint_jacobian(qdt0, np.pi / 2))
# print(static_equilibrium(qdt0, 0, multibody_data))


def static_equilibrium(qdt0, t: float) -> np.ndarray:
    x0 = qdt0
    x, d, success, msg = fsolve(
        multibody_data.pos_constraints,
        x0,
        args=(t,),
        fprime=multibody_data.jacobian,
        full_output=True,
    )
    return x, d["fjac"]


# print(static_equilibrium(qdt0, 1))

# quit()


t_array = np.arange(0, 1, 1e-2)
from uraeus.rnea.cmbs.rev2.utils import timer

kinematic_sim = timer(kinematic_sim)

print("Compiling")
qdt0s, qdt1s, qdt2s = kinematic_sim(multibody_data, qdt0, t_array[0:2])
print("")
qdt0s, qdt1s, qdt2s = kinematic_sim(multibody_data, qdt0, t_array)

qdt1s_d = [(qdt0s[i] - qdt0s[i - 1]) / 1e-2 for i in np.arange(1, len(qdt0s))]
qdt1s_d = np.diff(qdt0s, axis=0) / 1e-2
# qdt2s_d = np.gradient(qdt1s_d, axis=1) / np.gradient(t_array)

plt.figure("l1_pos")
plt.plot(t_array, [q[8] for q in qdt0s])
plt.plot(t_array, [q[9] for q in qdt0s])
plt.grid()

plt.figure("l2_pos")
plt.plot(t_array, [q[7 + 7] for q in qdt0s])
plt.plot(t_array, [q[8 + 7] for q in qdt0s])
plt.plot(t_array, [q[9 + 7] for q in qdt0s])
plt.grid()

plt.figure("l3_pos")
plt.plot(t_array, [q[7 + 7 * 2] for q in qdt0s])
plt.plot(t_array, [q[8 + 7 * 2] for q in qdt0s])
plt.plot(t_array, [q[9 + 7 * 2] for q in qdt0s])
plt.grid()

plt.figure("l2_vel")
plt.plot(t_array, [q[7 + 7] for q in qdt1s])
plt.plot(t_array, [q[8 + 7] for q in qdt1s])
plt.plot(t_array, [q[9 + 7] for q in qdt1s])
plt.grid()

plt.figure("l2_vel_d")
plt.plot(t_array[1:], [q[7 + 7] for q in qdt1s_d])
plt.plot(t_array[1:], [q[8 + 7] for q in qdt1s_d])
plt.plot(t_array[1:], [q[9 + 7] for q in qdt1s_d])
plt.grid()


plt.figure("l3_vel")
plt.plot(t_array, [q[7 + 7 * 2] for q in qdt1s])
plt.plot(t_array, [q[8 + 7 * 2] for q in qdt1s])
plt.plot(t_array, [q[9 + 7 * 2] for q in qdt1s])
plt.grid()

# plt.figure("l2_acc")
# plt.plot(t_array, [q[7 + 7] for q in qdt2s])
# plt.plot(t_array, [q[8 + 7] for q in qdt2s])
# plt.plot(t_array, [q[9 + 7] for q in qdt2s])
# plt.grid()

# plt.figure("l3_acc")
# plt.plot(t_array, [q[7 + 7 * 2] for q in qdt2s])
# plt.plot(t_array, [q[8 + 7 * 2] for q in qdt2s])
# plt.plot(t_array, [q[9 + 7 * 2] for q in qdt2s])
# plt.grid()


plt.show()
