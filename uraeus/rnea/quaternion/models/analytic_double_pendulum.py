import jax
import jax.numpy as jnp
import numpy as np


class AnalyticDoublePendulum(object):

    def __init__(self, l1, l2, m1, m2):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

    def evaluate_kinematics(self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray):

        theta1_dt0, theta2_dt0 = qdt0
        theta1_dt1, theta2_dt1 = qdt1
        theta1_dt2, theta2_dt2 = qdt2

        c1 = np.cos(theta1_dt0)
        s1 = np.sin(theta1_dt0)
        c2 = np.cos(theta1_dt0 + theta2_dt0)
        s2 = np.sin(theta1_dt0 + theta2_dt0)

        d1 = np.array([s1, -c1])
        n1 = np.array([c1, s1])
        d2 = np.array([s2, -c2])
        n2 = np.array([c2, s2])

        r1dt0 = self.l1 * d1
        r2dt0 = r1dt0 + (self.l2 * d2)

        r1dt1 = self.l1 * theta1_dt1 * n1
        r2dt1 = (self.l2 * (theta1_dt1 + theta2_dt1) * n2) + r1dt1

        r1dt2 = (self.l1 * theta1_dt2 * n1) + (theta1_dt1**2 * self.l1 * -d1)
        r2dt2 = (
            (self.l2 * (theta1_dt2 + theta2_dt2) * n2)
            + ((theta1_dt1 + theta2_dt1) ** 2 * self.l2 * -d2)
            + r1dt2
        )

        kinematics = ((r1dt0, r2dt0), (r1dt1, r2dt1), (r1dt2, r2dt2))

        return kinematics

    def evaluate_inverse_dynamics(
        self, qdt0: np.ndarray, qdt1: np.ndarray, qdt2: np.ndarray
    ):

        (r1dt0, r2dt0), _, (r1dt2, r2dt2) = self.evaluate_kinematics(qdt0, qdt1, qdt2)

        g = 9.81
        # FBD for l2 -> reactions at joint 2
        F2g = np.array([0, -self.m2 * g])
        F2i = self.m2 * r2dt2
        Fj2 = F2g - F2i

        Tj2 = (np.cross((r2dt0 - r1dt0), F2g)) - (np.cross((r2dt0 - r1dt0), F2i))

        # FBD for l1 -> reactions at joint 1
        F1g = np.array([0, -self.m1 * g])
        F1i = self.m1 * r1dt2

        Fj1 = Fj2 + F1g - F1i
        Tj1 = Tj2 + np.cross(r1dt0, F1g) - np.cross(r1dt0, F1i) + np.cross(r1dt0, Fj2)

        forces = (np.array([*Fj1, Tj1]), np.array([*Fj2, Tj2]))
        return forces
