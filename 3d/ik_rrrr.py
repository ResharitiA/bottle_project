import numpy as np

def inverse_kinematics_rrrr(target, l1=0.4, l2=0.35, l3=0.20):
    x, y, z = target

    # 1. Поворот базы
    q1 = np.arctan2(y, x)

    # 2. Плоская задача (r, z)
    r_end = np.hypot(x, y)
    z_end = z

    # Целимся запястьем, предполагая, что кисть горизонтальна
    r_wrist = r_end - l3
    z_wrist = z_end

    # Решаем треугольник для l1, l2
    d_sq = r_wrist**2 + z_wrist**2
    d = np.sqrt(d_sq)

    if d > (l1 + l2):
        d = l1 + l2
        r_wrist = d * r_wrist / np.sqrt(d_sq)
        z_wrist = d * z_wrist / np.sqrt(d_sq)
        d_sq = d**2

    # Угол локтя (внутренний угол gamma)
    cos_gamma = (l1**2 + l2**2 - d_sq) / (2 * l1 * l2)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    
    # Наш q3 = -(pi - gamma), т.к. в FK это поворот относительно предыдущего звена
    q3 = -(np.pi - gamma)

    # Угол плеча q2
    alpha = np.arctan2(z_wrist, r_wrist)
    cos_beta = (l1**2 + d_sq - l2**2) / (2 * l1 * d)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    
    q2 = alpha + beta

    # Угол кисти (чтобы горизонтально)
    q4 = -(q2 + q3)

    return q1, q2, q3, q4
