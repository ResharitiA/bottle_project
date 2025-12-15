# demo_sliders.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider

from pathlib import Path
import sys

# Добавляем путь к папке 3d в sys.path
BASE_DIR = Path(__file__).resolve().parents[1]  # это папка robot
THREED_DIR = BASE_DIR / "3d"
sys.path.append(str(THREED_DIR))

from robot_kinematics import RobotArm3DOF


def create_plot():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    max_range = 0.8
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max_range * 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3DOF манипулятор (управление ползунками)')

    return fig, ax


def main():
    robot = RobotArm3DOF(l1=0.3, l2=0.3)

    # Начальные значения
    q1_0 = np.deg2rad(30)
    q2_0 = np.deg2rad(30)
    d3_0 = 0.0

    fig, ax = create_plot()

    # Вычисляем начальное положение
    T0e, points = robot.forward_kinematics([q1_0, q2_0, d3_0])
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    # Линия манипулятора
    (line,) = ax.plot(xs, ys, zs, "-o", linewidth=3, markersize=6, color="tab:blue")
    # Точка схвата (клешня)
    (ee_point,) = ax.plot([xs[-1]], [ys[-1]], [zs[-1]], "ro", markersize=8)

    # Оставляем место снизу под слайдеры
    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Оси для слайдеров
    ax_q1 = plt.axes([0.1, 0.20, 0.8, 0.03])
    ax_q2 = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_d3 = plt.axes([0.1, 0.10, 0.8, 0.03])

    # Слайдеры:
    slider_q1 = Slider(
        ax=ax_q1,
        label="q1 (deg)",
        valmin=-180,
        valmax=180,
        valinit=np.rad2deg(q1_0),
    )

    slider_q2 = Slider(
        ax=ax_q2,
        label="q2 (deg)",
        valmin=-90,
        valmax=90,
        valinit=np.rad2deg(q2_0),
    )

    slider_d3 = Slider(
        ax=ax_d3,
        label="d3 (m)",
        valmin=-0.2,
        valmax=0.2,
        valinit=d3_0,
    )

    def update(val):
        q1 = np.deg2rad(slider_q1.val)
        q2 = np.deg2rad(slider_q2.val)
        d3 = slider_d3.val

        _, pts = robot.forward_kinematics([q1, q2, d3])

        xs_new = pts[:, 0]
        ys_new = pts[:, 1]
        zs_new = pts[:, 2]

        line.set_xdata(xs_new)
        line.set_ydata(ys_new)
        line.set_3d_properties(zs_new)

        ee_point.set_xdata([xs_new[-1]])
        ee_point.set_ydata([ys_new[-1]])
        ee_point.set_3d_properties([zs_new[-1]])

        fig.canvas.draw_idle()

    slider_q1.on_changed(update)
    slider_q2.on_changed(update)
    slider_d3.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
