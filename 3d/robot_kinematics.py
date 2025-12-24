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
        # Начало координат в базе
        T = np.eye(4)
        
        # 1. Поворот базы вокруг Z
        T = T @ rot_z(q1)
        p0 = (T @ np.array([0, 0, 0, 1]))[:3]
        
        # 2. Первое звено (l1) - поворот q2 вокруг Y, затем смещение на l1 по X
        T = T @ rot_y(q2) @ transl(self.l1, 0, 0)
        p1 = (T @ np.array([0, 0, 0, 1]))[:3]
        
        # 3. Второе звено (l2) - поворот q3, затем смещение на l2 по X
        T = T @ rot_y(q3) @ transl(self.l2, 0, 0)
        p2 = (T @ np.array([0, 0, 0, 1]))[:3]
        
        # 4. Третье звено (l3) - поворот q4, затем смещение на l3 по X
        T = T @ rot_y(q4) @ transl(self.l3, 0, 0)
        p3 = (T @ np.array([0, 0, 0, 1]))[:3]
        
        return np.array([p0, p1, p2, p3, p3])  # 5 точек для отрисовки
