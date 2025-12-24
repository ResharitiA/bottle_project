import numpy as np

def inverse_kinematics_rrrr(target, l1=0.4, l2=0.35, l3=0.20):
    x = float(target[0])
    y = float(target[1])
    z = float(target[2])
    
    q1 = np.arctan2(y, x)
    r = np.sqrt(x*x + y*y)
    r_wrist = r - l3
    z_wrist = z
    
    d = np.sqrt(r_wrist*r_wrist + z_wrist*z_wrist)
    
    reach_max = l1 + l2
    reach_min = abs(l1 - l2)
    
    if d > reach_max:
        d = reach_max
    if d < reach_min:
        d = reach_min
    
    cos_q3 = (l1*l1 + l2*l2 - d*d) / (2.0 * l1 * l2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3 = -(np.pi - np.arccos(cos_q3))
    
    alpha = np.arctan2(z_wrist, r_wrist)
    cos_beta = (l1*l1 + d*d - l2*l2) / (2.0 * l1 * d)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    q2 = alpha + np.arccos(cos_beta)
    
    q4 = -(q2 + q3)
    
    return q1, q2, q3, q4
