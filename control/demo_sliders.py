# demo_sliders.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
THREED_DIR = BASE_DIR / "3d"
sys.path.append(str(THREED_DIR))

from robot_kinematics import RobotArmRRRR


def create_plot():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    max_range = 1.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-0.1, max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("RRRR манипулятор (добавлен 3-й локоть q4)")

    return fig, ax


def main():
    robot = RobotArmRRRR(l1=0.4, l2=0.35, l3=0.20)

    q1_0 = np.deg2rad(0)
    q2_0 = np.deg2rad(30)
    q3_0 = np.deg2rad(-40)
    q4_0 = np.deg2rad(20)

    fig, ax = create_plot()

    pts = robot.forward(q1_0, q2_0, q3_0, q4_0)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

    (line,) = ax.plot(xs, ys, zs, "-o", linewidth=3, markersize=6, color="tab:blue")
    (ee_point,) = ax.plot([xs[-1]], [ys[-1]], [zs[-1]], "ro", markersize=8)

    plt.subplots_adjust(left=0.15, bottom=0.32)

    ax_q1 = plt.axes([0.15, 0.24, 0.7, 0.03])
    ax_q2 = plt.axes([0.15, 0.19, 0.7, 0.03])
    ax_q3 = plt.axes([0.15, 0.14, 0.7, 0.03])
    ax_q4 = plt.axes([0.15, 0.09, 0.7, 0.03])

    s_q1 = Slider(ax=ax_q1, label="База q1 (deg)", valmin=-180, valmax=180, valinit=np.rad2deg(q1_0))
    s_q2 = Slider(ax=ax_q2, label="Плечо q2 (deg)", valmin=-90, valmax=90, valinit=np.rad2deg(q2_0))
    s_q3 = Slider(ax=ax_q3, label="Локоть q3 (deg)", valmin=-150, valmax=150, valinit=np.rad2deg(q3_0))
    s_q4 = Slider(ax=ax_q4, label="3-й локоть q4 (deg)", valmin=-180, valmax=180, valinit=np.rad2deg(q4_0))

    def update(val):
        q1 = np.deg2rad(s_q1.val)
        q2 = np.deg2rad(s_q2.val)
        q3 = np.deg2rad(s_q3.val)
        q4 = np.deg2rad(s_q4.val)

        pts_new = robot.forward(q1, q2, q3, q4)
        xs_new, ys_new, zs_new = pts_new[:, 0], pts_new[:, 1], pts_new[:, 2]

        line.set_xdata(xs_new)
        line.set_ydata(ys_new)
        line.set_3d_properties(zs_new)

        ee_point.set_xdata([xs_new[-1]])
        ee_point.set_ydata([ys_new[-1]])
        ee_point.set_3d_properties([zs_new[-1]])

        fig.canvas.draw_idle()

    s_q1.on_changed(update)
    s_q2.on_changed(update)
    s_q3.on_changed(update)
    s_q4.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
