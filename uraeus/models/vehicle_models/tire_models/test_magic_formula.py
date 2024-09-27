import logging
from dataclasses import dataclass
from collections import namedtuple

import numpy as np

from uraeus.utils.logging import construct_logger
from uraeus.rnea.bodies import RigidBodyData
from uraeus.rnea.joints import (
    JointConfigInputs,
    TranslationalJoint,
    RevoluteJoint,
)
from uraeus.rnea.topologies import MultiBodyTree, Model
from uraeus.models.vehicle_models.tire_models.magic_formula import (
    read_tire_file,
    MagicFormulaTireModel,
)
from uraeus.models.vehicle_models.fsae.forces_elements import (
    SimpleElectricMotor,
)


logger = construct_logger(__name__, logging.DEBUG)


@dataclass
class Forces(object):

    tir_file = "/workspaces/uraeus_rnea/uraeus/models/vehicle_models/tire_models/data_files/sample.tir"
    tire_params_dict = read_tire_file(tir_file)
    tire_params_dict["sigma_k"] = 0.1
    tire_params_dict["sigma_a"] = 0.02
    tire_params_dict["kv_low"] = -700
    tire_params_dict["Cfk"] = 24e3
    tire_params_dict["Cfa"] = 24e3
    tire_parameters = namedtuple("TirePar", tire_params_dict.keys())(**tire_params_dict)

    tire = MagicFormulaTireModel(tire_parameters)

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
        radius = 0.3135
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
        tree.add_joint(
            "j2", "slider", "carier", carier_body, TranslationalJoint, j2_data
        )
        tree.add_joint("j3", "carier", "wheel", wheel_body, RevoluteJoint, j3_data)

        self.model = Model(tree)

    def calculate_external_forces(
        self, model, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray, t
    ):
        model = self.model
        bodies_kinematics, _ = model.forward_kinematics_pass(qdt0, qdt1, qdt2)
        wheel_kin = model.get_body_kinematics("wheel", bodies_kinematics)
        tire_force = Forces.tire(wheel_kin, t)

        corner_mass = 200
        wn = 2 * (2 * np.pi)
        damping_ratio = 0.8
        stiffness = wn**2 * corner_mass
        damping = damping_ratio * (2 * np.sqrt(stiffness * corner_mass))

        model.forces_map["wheel"]["global"]["tire"] = tire_force
        model.forces_map["wheel"]["global"]["load"] = np.array(
            [0, 0, -corner_mass * 9.81, 0, 0, 0]
        )
        # torque = (10 * t) if t > 1 else 0
        torque = (100) if t > 1 else 0
        torque = 0 if t > 2 else torque
        tau = np.array([0, (-stiffness * qdt0[1] - damping * qdt1[1]), torque])
        logger.debug("Input torque = %s", torque)
        logger.debug("Tire Fz = %s", tire_force[2])
        logger.debug("Tire Fx = %s", tire_force[0])
        logger.debug("Tire Fy = %s", tire_force[1])
        logger.debug("Tire My = %s", tire_force[4])
        return tau, model.forces_map


if __name__ == "__main__":

    import scipy.integrate as integrate
    import matplotlib.pyplot as plt

    def simulate(ssode, ydt0, t_end):

        time_history = []
        qdt0_history = []
        qdt1_history = []
        qdt2_history = []

        stepper = integrate.BDF(ssode, 0, ydt0, t_end)
        while stepper.status == "running":
            y = stepper.y
            ydt1 = ssode(stepper.t, y)
            qdt0, qdt1 = y.reshape(2, -1)
            _, qdt2 = ydt1.reshape(2, -1)

            print("sim-time = ", stepper.t)
            print(qdt0)

            time_history.append(stepper.t)
            qdt0_history.append(qdt0)
            qdt1_history.append(qdt1)
            qdt2_history.append(qdt2)
            stepper.step()

        return time_history, qdt0_history, qdt1_history, qdt2_history

    quarter_car_model = QuarterCarModel()

    v = 0 / 3.6
    r = 0.31
    true_t, true_qdt0, true_qdt1, true_qdt2 = simulate(
        lambda ydt0, t: quarter_car_model.model.ssode(
            ydt0, t, quarter_car_model.calculate_external_forces
        ),
        np.array([0, -5.19672528e-03, 0, v, 0, v / r]),
        # np.array([0, 0.0, 0, v, 0, v / r]),
        3,
    )

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

    plt.figure("wheel.vel.x")
    plt.title("wheel.vel.x")
    plt.plot(true_t, [v[0] for v in wheel_vG])
    plt.grid()

    plt.figure("wheel.ang.vel")
    plt.plot(true_t, [v[3] for v in wheel_vG], label="ang.v.x")
    plt.plot(true_t, [v[4] for v in wheel_vG], label="ang.v.y")
    plt.plot(true_t, [v[5] for v in wheel_vG], label="ang.v.z")
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

    plt.show()
