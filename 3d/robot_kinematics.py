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
        # Базовая матрица
        T_base = np.eye(4)
        
        # Звено 1: Поворот базы q1
        T1 = T_base @ rot_z(q1)
        p0 = (T1 @ np.array([0, 0, 0, 1]))[:3] # Точка базы
        
        # Звено 2: Поворот плеча q2 + смещение l1
        T2 = T1 @ rot_y(q2) @ transl(self.l1, 0, 0)
        p1 = (T2 @ np.array([0, 0, 0, 1]))[:3]
        
        # Звено 3: Поворот локтя q3 + смещение l2
        T3 = T2 @ rot_y(q3) @ transl(self.l2, 0, 0)
        p2 = (T3 @ np.array([0, 0, 0, 1]))[:3]
        
        # Звено 4: Поворот кисти q4 + смещение l3
        T4 = T3 @ rot_y(q4) @ transl(self.l3, 0, 0)
        p3 = (T4 @ np.array([0, 0, 0, 1]))[:3]
        
        # Возвращаем точки для графики и матрицы для отчета
        points = np.array([p0, p1, p2, p3, p3])
        matrices = [T1, T2, T3, T4]
        
        return points, matrices
