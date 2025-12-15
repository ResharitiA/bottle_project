# demo_ik.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from robot_kinematics import RobotArm3DOF
from visualizer import plot_robot


def demo_inverse_kinematics():
    robot = RobotArm3DOF(l1=0.3, l2=0.3)

    # Желаемая точка в пространстве (можешь менять)
    x_des, y_des, z_des = 0.3, 0.2, 0.4

    q = robot.inverse_kinematics(x_des, y_des, z_des)
    if q is None:
        print("Точка недостижима для выбранной геометрии.")
        return

    print("Найденные обобщённые координаты q:", q)

    T0e, points = robot.forward_kinematics(q)
    print("Конечное положение схвата (из FK):")
    print(T0e)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_robot(ax, points)
    ax.scatter([x_des], [y_des], [z_des], color='red', s=60, label='цель')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    demo_inverse_kinematics()
