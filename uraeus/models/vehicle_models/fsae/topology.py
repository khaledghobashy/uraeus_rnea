from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
from uraeus.rnea.bodies import RigidBody, RigidBodyData
from uraeus.rnea.joints import (
    FreeJoint,
    JointData,
    JointConfigInputs,
    RevoluteJoint,
    TranslationalJoint,
)

from uraeus.rnea.topologies import MultiBodyTree

from .utils import RightSuspensionJoint, LeftSuspensionJoint


@dataclass
class SuspentionData(object):
    mass: float
    trackwidth: float
    stiffness: float
    damping: float


@dataclass
class ChassisData(object):
    mass: float
    wheelbase: float
    cg_height: float
    weight_distribution_f: float
    inertia_tensor: np.ndarray


@dataclass
class WheelData(object):
    mass: float
    inertia_tensor: np.ndarray
    wc_height: float


@dataclass
class VehicleData(object):
    chassis: ChassisData
    suspension_front: SuspentionData
    suspension_rear: SuspentionData
    wheels_front: WheelData
    wheels_rear: WheelData


class BodiesData(NamedTuple):
    chassis: RigidBodyData
    fr_carier: RigidBodyData
    fl_carier: RigidBodyData
    rr_carier: RigidBodyData
    rl_carier: RigidBodyData
    fr_wheel: RigidBodyData
    fl_wheel: RigidBodyData
    rr_wheel: RigidBodyData
    rl_wheel: RigidBodyData


class JointsData(NamedTuple):
    free: JointData
    fr_susp: JointData
    fl_susp: JointData
    rr_susp: JointData
    rl_susp: JointData
    fr_wheel_rev: JointData
    fl_wheel_rev: JointData
    rr_wheel_rev: JointData
    rl_wheel_rev: JointData


def construct_bodies_data(vehicle_data: VehicleData) -> BodiesData:
    chassis = vehicle_data.chassis
    susp_front = vehicle_data.suspension_front
    susp_rear = vehicle_data.suspension_rear
    wheels_front = vehicle_data.wheels_front
    wheels_rear = vehicle_data.wheels_rear

    chassis_data = RigidBodyData(
        location=np.array(
            [
                -(1 - chassis.weight_distribution_f) * chassis.wheelbase,
                0,
                chassis.cg_height,
            ]
        ),
        orientation=np.eye(3),
        mass=chassis.mass,
        inertia_tensor=chassis.inertia_tensor,
    )

    fr_carrier_data = RigidBodyData(
        location=np.array(
            [
                0,
                -susp_front.trackwidth / 2,
                wheels_front.wc_height,
            ]
        ),
        orientation=np.eye(3),
        mass=susp_front.mass,
        inertia_tensor=np.eye(3),
    )

    fl_carrier_data = RigidBodyData(
        location=np.array(
            [
                0,
                susp_front.trackwidth / 2,
                wheels_front.wc_height,
            ]
        ),
        orientation=np.eye(3),
        mass=susp_front.mass,
        inertia_tensor=np.eye(3),
    )

    rr_carrier_data = RigidBodyData(
        location=np.array(
            [
                -chassis.wheelbase,
                -susp_rear.trackwidth / 2,
                wheels_rear.wc_height,
            ]
        ),
        orientation=np.eye(3),
        mass=susp_rear.mass,
        inertia_tensor=np.eye(3),
    )

    rl_carrier_data = RigidBodyData(
        location=np.array(
            [
                -chassis.wheelbase,
                susp_rear.trackwidth / 2,
                wheels_rear.wc_height,
            ]
        ),
        orientation=np.eye(3),
        mass=susp_rear.mass,
        inertia_tensor=np.eye(3),
    )

    fr_wheel_data = RigidBodyData(
        location=np.array(
            [
                0,
                -susp_front.trackwidth / 2,
                wheels_front.wc_height,
            ]
        ),
        orientation=np.eye(3),
        mass=wheels_front.mass,
        inertia_tensor=wheels_front.inertia_tensor,
    )

    fl_wheel_data = RigidBodyData(
        location=np.array(
            [
                0,
                susp_front.trackwidth / 2,
                wheels_front.wc_height,
            ]
        ),
        orientation=np.eye(3),
        mass=wheels_front.mass,
        inertia_tensor=wheels_front.inertia_tensor,
    )

    rr_wheel_data = RigidBodyData(
        location=np.array(
            [
                -chassis.wheelbase,
                -susp_rear.trackwidth / 2,
                wheels_rear.wc_height,
            ]
        ),
        orientation=np.eye(3),
        mass=wheels_rear.mass,
        inertia_tensor=wheels_rear.inertia_tensor,
    )

    rl_wheel_data = RigidBodyData(
        location=np.array(
            [
                -chassis.wheelbase,
                susp_rear.trackwidth / 2,
                wheels_rear.wc_height,
            ]
        ),
        orientation=np.eye(3),
        mass=wheels_rear.mass,
        inertia_tensor=wheels_rear.inertia_tensor,
    )

    bodies_data = BodiesData(
        chassis=chassis_data,
        fr_carier=fr_carrier_data,
        fl_carier=fl_carrier_data,
        rr_carier=rr_carrier_data,
        rl_carier=rl_carrier_data,
        fr_wheel=fr_wheel_data,
        fl_wheel=fl_wheel_data,
        rr_wheel=rr_wheel_data,
        rl_wheel=rl_wheel_data,
    )

    return bodies_data


def construct_joints_data(vehicle_data: VehicleData) -> JointsData:
    chassis = vehicle_data.chassis
    susp_front = vehicle_data.suspension_front
    wheel_front = vehicle_data.wheels_front
    susp_rear = vehicle_data.suspension_rear
    wheel_rear = vehicle_data.wheels_rear

    free_joint = JointConfigInputs(
        pos=np.array(
            [
                -(1 - chassis.weight_distribution_f) * chassis.wheelbase,
                0,
                chassis.cg_height,
            ]
        ),
        z_axis=np.array([0, 0, 1]),
        x_axis=np.array([1, 0, 0]),
    )

    fr_susp = JointConfigInputs(
        pos=np.array([0, -susp_front.trackwidth / 2, wheel_front.wc_height]),
        z_axis=np.array([0, 0, 1]),
        x_axis=None,  # np.array([1, 0, 0]),
    )

    fr_wheel = JointConfigInputs(
        pos=np.array([0, -susp_front.trackwidth / 2, wheel_front.wc_height]),
        z_axis=np.array([0, 1, 0]),
        x_axis=None,  # np.array([0, 0, 1]),
    )

    fl_susp = JointConfigInputs(
        pos=np.array([0, susp_front.trackwidth / 2, wheel_front.wc_height]),
        z_axis=np.array([0, 0, 1]),
        x_axis=None,  # np.array([1, 0, 0]),
    )

    fl_wheel = JointConfigInputs(
        pos=np.array([0, susp_front.trackwidth / 2, wheel_front.wc_height]),
        z_axis=np.array([0, 1, 0]),
        x_axis=None,  # np.array([0, 0, 1]),
    )

    rr_susp = JointConfigInputs(
        pos=np.array(
            [-chassis.wheelbase, -susp_rear.trackwidth / 2, wheel_rear.wc_height]
        ),
        z_axis=np.array([0, 0, 1]),
        x_axis=None,  # np.array([1, 0, 0]),
    )

    rr_wheel = JointConfigInputs(
        pos=np.array(
            [-chassis.wheelbase, -susp_rear.trackwidth / 2, wheel_rear.wc_height]
        ),
        z_axis=np.array([0, 1, 0]),
        x_axis=None,  # np.array([0, 0, 1]),
    )

    rl_susp = JointConfigInputs(
        pos=np.array(
            [-chassis.wheelbase, susp_rear.trackwidth / 2, wheel_rear.wc_height]
        ),
        z_axis=np.array([0, 0, 1]),
        x_axis=None,  # np.array([1, 0, 0]),
    )

    rl_wheel = JointConfigInputs(
        pos=np.array(
            [-chassis.wheelbase, susp_rear.trackwidth / 2, wheel_rear.wc_height]
        ),
        z_axis=np.array([0, 1, 0]),
        x_axis=None,  # np.array([0, 0, 1]),
    )

    configs = JointsData(
        free=free_joint,
        fr_susp=fr_susp,
        fl_susp=fl_susp,
        rr_susp=rr_susp,
        rl_susp=rl_susp,
        fr_wheel_rev=fr_wheel,
        fl_wheel_rev=fl_wheel,
        rr_wheel_rev=rr_wheel,
        rl_wheel_rev=rl_wheel,
    )

    return configs


def construct_multibodytree(vehicle_data: VehicleData) -> MultiBodyTree:
    bodies_data = construct_bodies_data(vehicle_data)
    joints_data = construct_joints_data(vehicle_data)

    tree = MultiBodyTree("vehicle")

    tree.add_joint(
        joint_name="free_joint",
        predecessor="ground",
        successor="chassis",
        succ_data=bodies_data.chassis,
        joint_type=FreeJoint,
        joint_data=joints_data.free,
    )

    # tree = emulate_free_joint(tree, "ground", "chassis", bodies_data.chassis)

    tree.add_joint(
        joint_name="fr_susp",
        predecessor="chassis",
        successor="fr_carier",
        succ_data=bodies_data.fr_carier,
        joint_type=RightSuspensionJoint,
        joint_data=joints_data.fr_susp,
    )

    tree.add_joint(
        joint_name="fl_susp",
        predecessor="chassis",
        successor="fl_carier",
        succ_data=bodies_data.fl_carier,
        joint_type=LeftSuspensionJoint,
        joint_data=joints_data.fl_susp,
    )

    tree.add_joint(
        joint_name="rr_susp",
        predecessor="chassis",
        successor="rr_carier",
        succ_data=bodies_data.rr_carier,
        joint_type=RightSuspensionJoint,
        joint_data=joints_data.rr_susp,
    )

    tree.add_joint(
        joint_name="rl_susp",
        predecessor="chassis",
        successor="rl_carier",
        succ_data=bodies_data.rl_carier,
        joint_type=LeftSuspensionJoint,
        joint_data=joints_data.rl_susp,
    )

    tree.add_joint(
        joint_name="fr_wheel_rev",
        predecessor="fr_carier",
        successor="fr_wheel",
        succ_data=bodies_data.fr_wheel,
        joint_type=RevoluteJoint,
        joint_data=joints_data.fr_wheel_rev,
    )

    tree.add_joint(
        joint_name="fl_wheel_rev",
        predecessor="fl_carier",
        successor="fl_wheel",
        succ_data=bodies_data.fl_wheel,
        joint_type=RevoluteJoint,
        joint_data=joints_data.fl_wheel_rev,
    )

    tree.add_joint(
        joint_name="rr_wheel_rev",
        predecessor="rr_carier",
        successor="rr_wheel",
        succ_data=bodies_data.rr_wheel,
        joint_type=RevoluteJoint,
        joint_data=joints_data.rr_wheel_rev,
    )

    tree.add_joint(
        joint_name="rl_wheel_rev",
        predecessor="rl_carier",
        successor="rl_wheel",
        succ_data=bodies_data.rl_wheel,
        joint_type=RevoluteJoint,
        joint_data=joints_data.rl_wheel_rev,
    )

    return tree


def emulate_free_joint(
    model: MultiBodyTree, predecessor: str, successor: str, succ_data: RigidBodyData
) -> MultiBodyTree:
    dummy_body_data = RigidBodyData()

    x_axis_config = JointConfigInputs(
        pos=np.zeros((3,)), z_axis=np.array([1, 0, 0]), x_axis=None
    )
    y_axis_config = JointConfigInputs(
        pos=np.zeros((3,)), z_axis=np.array([0, 1, 0]), x_axis=None
    )
    z_axis_config = JointConfigInputs(
        pos=np.zeros((3,)), z_axis=np.array([0, 0, 1]), x_axis=None
    )

    model.add_joint(
        joint_name="x_rotation",
        predecessor=predecessor,
        successor="d1",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=x_axis_config,
    )

    model.add_joint(
        joint_name="y_rotation",
        predecessor="d1",
        successor="d2",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=y_axis_config,
    )

    model.add_joint(
        joint_name="z_rotation",
        predecessor="d2",
        successor="d3",
        succ_data=dummy_body_data,
        joint_type=RevoluteJoint,
        joint_data=z_axis_config,
    )

    model.add_joint(
        joint_name="x_trans",
        predecessor="d3",
        successor="d4",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=x_axis_config,
    )

    model.add_joint(
        joint_name="y_trans",
        predecessor="d4",
        successor="d5",
        succ_data=dummy_body_data,
        joint_type=TranslationalJoint,
        joint_data=y_axis_config,
    )

    model.add_joint(
        joint_name="z_trans",
        predecessor="d5",
        successor=successor,
        succ_data=succ_data,
        joint_type=TranslationalJoint,
        joint_data=z_axis_config,
    )

    return model
