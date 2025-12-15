# robot_kinematics.py
import numpy as np


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ])


def transl(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])


class RobotArmRRRR:
    """
    4DOF R-R-R-R (все суставы вращательные), фиксированные длины звеньев.

    q1: база (Z)
    q2: плечо (Y)
    q3: локоть (Y)
    q4: "3-й локоть"/запястье (Y) - держит красную точку (кончик)

    l1: плечо
    l2: предплечье
    l3: запястье (короткое звено до наконечника)
    """
    def __init__(self, l1=0.4, l2=0.35, l3=0.20):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def forward(self, q1, q2, q3, q4):
        T0 = np.eye(4)

        # база/плечо в начале координат
        T1 = rot_z(q1)

        # локоть
        T2 = T1 @ rot_y(q2) @ transl(self.l1, 0, 0)

        # запястье (вторая "кость")
        T3 = T2 @ rot_y(q3) @ transl(self.l2, 0, 0)

        # третий локоть / кисть
        T4 = T3 @ rot_y(q4) @ transl(self.l3, 0, 0)

        p0 = T0 @ np.array([0, 0, 0, 1])  # база
        p1 = T1 @ np.array([0, 0, 0, 1])  # сустав 1
        p2 = T2 @ np.array([0, 0, 0, 1])  # сустав 2
        p3 = T3 @ np.array([0, 0, 0, 1])  # сустав 3
        p4 = T4 @ np.array([0, 0, 0, 1])  # красная точка (кончик)

        points = np.stack([p0[:3], p1[:3], p2[:3], p3[:3], p4[:3]], axis=0)
        return points
