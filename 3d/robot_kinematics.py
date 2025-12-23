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
    def __init__(self, l1=0.4, l2=0.35, l3=0.20):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def forward(self, q1, q2, q3, q4):
        # База (0,0,0)
        T0 = np.eye(4)
        
        # Поворот базы q1 (вокруг Z)
        T1 = T0 @ rot_z(q1)
        
        # Звено 1: поворот q2, длина l1
        T2 = T1 @ rot_y(q2) @ transl(self.l1, 0, 0)
        
        # Звено 2: поворот q3, длина l2
        T3 = T2 @ rot_y(q3) @ transl(self.l2, 0, 0)
        
        # Звено 3: поворот q4, длина l3
        T4 = T3 @ rot_y(q4) @ transl(self.l3, 0, 0)
        
        p0 = T0 @ np.array([0, 0, 0, 1])
        p1 = T1 @ np.array([0, 0, 0, 1])
        p2 = T2 @ np.array([0, 0, 0, 1])
        p3 = T3 @ np.array([0, 0, 0, 1])
        p4 = T4 @ np.array([0, 0, 0, 1])
        
        return np.stack([p0[:3], p1[:3], p2[:3], p3[:3], p4[:3]], axis=0)
