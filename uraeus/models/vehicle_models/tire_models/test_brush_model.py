import logging
from dataclasses import dataclass

import numpy as np

from uraeus.utils.logging import construct_logger
from uraeus.rnea.bodies import RigidBodyData
from uraeus.rnea.joints import (
    JointConfigInputs,
    TranslationalJoint,
    RevoluteJoint,
    CylindricalJoint,
)
from uraeus.rnea.multibody_models import (
    MultiBodyTree,
    Model,
    HybridModel,
    reconstruct_system_coordinates,
)
from uraeus.models.vehicle_models.tire_models.brush_model import (
    BrushModelParameters,
    BrushTireModel,
)
from uraeus.models.vehicle_models.fsae.forces_elements import (
    SimpleElectricMotor,
)

logger = construct_logger(__name__, logging.DEBUG)


@dataclass
class Forces(object):
    tire_parameters = BrushModelParameters(
        mu=1.3,
        Cfk=50e3,
        Cfa=50e3,
        Cfx=150e3,
        Cfy=150e3,
        a=0.10,
        unloaded_radius=0.313,
        kz=150e3,
        cz=15e3,
        kv_low=0,
    )

    tire = BrushTireModel(tire_parameters)

    motor = SimpleElectricMotor(
        name="rl_motor",
        min_rpm=0,
        max_rpm=10000,
        min_torque=0,
        max_torque=300,
        max_power=200e3,
        reduction_ratio=1,
    )


class QuarterCarModel(object):

    def __init__(self):
        radius = 0.313
        tree = MultiBodyTree("tire_test_tree")

        slider_body = RigidBodyData(
            np.array([0, 0, radius]), np.array([1, 0, 0, 0]), 1, np.eye(3)
        )

        carier_body = RigidBodyData(
            np.array([0, 0, radius]), np.array([1, 0, 0, 0]), 10, np.eye(3)
        )
        wheel_body = RigidBodyData(
            np.array([0, 0, radius]), np.array([1, 0, 0, 0]), 10, np.eye(3)
        )

        j1_data = JointConfigInputs(np.array([0, 0, radius]), np.array([1, 0, 0]), None)
        j2_data = JointConfigInputs(np.array([0, 0, radius]), np.array([0, 0, 1]), None)
        j3_data = JointConfigInputs(np.array([0, 0, radius]), np.array([0, 1, 0]), None)

        tree.add_joint(
            "j1", "ground", "slider", slider_body, TranslationalJoint, j1_data
        )
        tree.add_joint("j2", "slider", "carier", carier_body, CylindricalJoint, j2_data)
        tree.add_joint("j3", "carier", "wheel", wheel_body, RevoluteJoint, j3_data)

        # self.model = Model(tree)
        self.model = HybridModel(tree, id_coordinates=[2])
        logger.debug(self.model.hybrid_dynamics_data.permutation_matrix)

    def evaluate_force_inputs(
        self, model, t: float, ydt0: np.ndarray, u: dict[str, float]
    ):
        qdt0, qdt1 = ydt0.reshape(2, -1)
        qdt2 = np.zeros((model.dof,))

        model = self.model
        bodies_kinematics, _ = model.forward_kinematics_pass(qdt0, qdt1, qdt2)
        wheel_kin = model.get_body_kinematics("wheel", bodies_kinematics)

        tire_force = Forces.tire(wheel_kin, t)

        wn = 2 * (2 * np.pi)
        damping_ratio = 0.8
        stiffness = wn**2 * 65
        damping = damping_ratio * (2 * np.sqrt(stiffness * 65))

        model.forces_map["wheel"]["global"]["tire"] = tire_force
        model.forces_map["wheel"]["global"]["load"] = np.array(
            [0, 0, -65 * 9.81, 0, 0, 0]
        )
        torque = (100) if t > 1 else 0
        torque = 0 if t > 5 else torque
        steering_torque = 0
        tau = np.array(
            [0, (-stiffness * qdt0[1] - damping * qdt1[1]), steering_torque, torque]
        )
        logger.debug("Input torque = %s", torque)
        logger.debug("Tire Fz = %s", tire_force[2])
        logger.debug("Tire Fx = %s", tire_force[0])
        logger.debug("Tire Fy = %s", tire_force[1])
        logger.debug("Tire My = %s", tire_force[4])
        return tau, model.forces_map

    def evaluate_motion_inputs(
        self, model, t: float, ydt0: np.ndarray, u: dict[str, float]
    ):
        return steering_function(t)


if __name__ == "__main__":

    import scipy.integrate as integrate
    import matplotlib.pyplot as plt

    def steering_function(t):
        angle = 0 * min(np.radians(10) * (t - 1), np.radians(10)) if t > 1 else 0
        return np.array([angle]), np.array([0]), np.array([0])

    quarter_car_model = QuarterCarModel()

    def simulate(ssode, ydt0, t_end):

        time_history = []
        qdt0_history = []
        qdt1_history = []
        qdt2_history = []

        stepper = integrate.BDF(ssode, 0, ydt0, t_end)
        while stepper.status == "running":
            y = stepper.y
            ydt1 = ssode(stepper.t, y)
            qdt0_fd, qdt1_fd = y.reshape(2, -1)
            _, qdt2_fd = ydt1.reshape(2, -1)

            print("sim-time = ", stepper.t)
            print(qdt0_fd)

            qdt0_id, qdt1_id, qdt2_id = steering_function(stepper.t)

            qdt0, qdt1, qdt2 = reconstruct_system_coordinates(
                quarter_car_model.model,
                (qdt0_fd, qdt1_fd, qdt2_fd),
                (qdt0_id, qdt1_id, qdt2_id),
            )
            print(qdt0)

            time_history.append(stepper.t)
            qdt0_history.append(qdt0)
            qdt1_history.append(qdt1)
            qdt2_history.append(qdt2)
            stepper.step()

        return (
            np.array(time_history),
            np.array(qdt0_history),
            np.array(qdt1_history),
            np.array(qdt2_history),
        )

    ssode = lambda t, ydt0: quarter_car_model.model.ssode(
        t,
        ydt0,
        {},
        quarter_car_model.evaluate_motion_inputs,
        quarter_car_model.evaluate_force_inputs,
    )

    v = 0 / 3.6
    r = 0.31
    ydt0 = np.array([0, -5.19672528e-03, 0, v, 0, v / r])
    true_t, true_qdt0, true_qdt1, true_qdt2 = simulate(ssode, ydt0, 10)

    bodies_kin, joints_kin = zip(
        *map(
            quarter_car_model.model.forward_kinematics_pass,
            true_qdt0,
            true_qdt1,
            true_qdt2,
        )
    )

    wheel_kin = [
        quarter_car_model.model.get_body_kinematics("wheel", kin) for kin in bodies_kin
    ]
    wheel_pGB = [b.p_GB for b in wheel_kin]
    wheel_vG = [b.v_G for b in wheel_kin]
    wheel_vB = [b.v_B for b in wheel_kin]

    plt.figure("wheel.vel.x")
    plt.title("wheel.vel.x")
    plt.plot(true_t, [v[0] * 3.6 for v in wheel_vG])
    plt.grid()

    plt.figure("wheel.ang.vel_G")
    plt.plot(true_t, [v[3] for v in wheel_vG], label="ang.v.x")
    plt.plot(true_t, [v[4] for v in wheel_vG], label="ang.v.y")
    plt.plot(true_t, [v[5] for v in wheel_vG], label="ang.v.z")
    plt.legend()
    plt.grid()

    plt.figure("wheel.ang.vel_B")
    plt.plot(true_t, [v[3] for v in wheel_vB], label="ang.v.x")
    plt.plot(true_t, [v[4] for v in wheel_vB], label="ang.v.y")
    plt.plot(true_t, [v[5] for v in wheel_vB], label="ang.v.z")
    plt.legend()
    plt.grid()

    plt.figure("wheel.pos.z")
    plt.title("wheel.pos.z")
    plt.plot(true_t, [p.r[2] for p in wheel_pGB])
    plt.grid()

    plt.figure("wheel.pos.x")
    plt.title("wheel.pos.x")
    plt.plot(true_t, [p.r[0] for p in wheel_pGB])
    plt.grid()

    plt.figure("Steering angle")
    plt.title("Steering angle")
    plt.plot(true_t, true_qdt0[:, 2], label="qdt0[2]")
    plt.plot(true_t, [steering_function(t)[0] for t in true_t], label="steering_input")
    plt.grid()

    plt.figure("x-y pos")
    plt.title("x-y pos")
    plt.plot([p.r[0] for p in wheel_pGB], [p.r[1] for p in wheel_pGB])
    plt.grid()

    plt.show()
