# visualizer.py
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

from robot_kinematics import RobotArm3DOF


def plot_robot(ax, points):
    """
    points: np.array shape (3, 3) – точки звеньев.
    """
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    ax.plot(xs, ys, zs, '-o', linewidth=3, markersize=6, color='tab:blue')

    # Настройка осей
    max_range = 0.8
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max_range * 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3DOF манипулятор (скелет)')


def demo_static():
    robot = RobotArm3DOF(l1=0.3, l2=0.3)
    q = np.array([np.deg2rad(30), np.deg2rad(30), 0.0])

    T0e, points = robot.forward_kinematics(q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_robot(ax, points)
    plt.show()


if __name__ == "__main__":
    demo_static()
