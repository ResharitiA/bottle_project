import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
import sys
import random

# ---- пути и импорт кинематики ----
BASE_DIR = Path(__file__).resolve().parents[1]
THREED_DIR = BASE_DIR / "3d"
sys.path.append(str(THREED_DIR))

from robot_kinematics import RobotArmRRRR

# ---------- настройки робота / единиц ----------
MM = 1000.0
N_JOINTS = 4  # 4DOF RRRR
LINK_LENGTHS_MM = [400, 350, 200, 0]
R_WORK_MM = sum(abs(l) for l in LINK_LENGTHS_MM)

robot = RobotArmRRRR(l1=0.4, l2=0.35, l3=0.20)


# ---------- вспомогательные функции ----------
def fk_mm(q):
    """Конечный эффектор в мм (берём последнюю точку)."""
    q1, q2, q3, q4 = q[:4]
    pts = robot.forward(q1, q2, q3, q4)  # в метрах
    p_end = pts[-1]
    return p_end * MM


def inside_workspace(pt_mm):
    return np.linalg.norm(pt_mm) <= R_WORK_MM + 1e-6


def generate_random_points(n=10):
    pts = []
    while len(pts) < n:
        r = R_WORK_MM * (random.random() ** (1.0 / 3.0))
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        pts.append([x, y, z])
    return np.array(pts)


# ---------- основная функция ----------
def main():
    # начальные углы (домашняя поза)
    q = np.zeros(N_JOINTS)

    # фигура
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.30, right=0.98, bottom=0.05, top=0.95)

    # оси в мм
    lim = R_WORK_MM * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, lim)
    ax.set_xlabel("X (мм)")
    ax.set_ylabel("Y (мм)")
    ax.set_zlabel("Z (мм)")

    # сфера рабочей области
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = R_WORK_MM * np.outer(np.cos(u), np.sin(v))
    ys = R_WORK_MM * np.outer(np.sin(u), np.sin(v))
    zs = R_WORK_MM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="purple", alpha=0.1, linewidth=0)

    # линия манипулятора
    def all_points_mm(q_local):
        q1, q2, q3, q4 = q_local[:4]
        pts = robot.forward(q1, q2, q3, q4)  # метры
        return np.asarray(pts) * MM

    pts = all_points_mm(q)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, colors=["b"] * (len(pts) - 1), linewidths=3)
    ax.add_collection3d(lc)

    # суставы: база, колено, кисть, конец (красный)
    joint_scatter = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=["k", "k", "k", "k", "r"],
        s=[30, 30, 30, 30, 40]
    )

    # случайные точки
    scatter = None
    labels = []

    def draw_random_points():
        nonlocal scatter, labels
        if scatter is not None:
            scatter.remove()
        for t in labels:
            t.remove()
        labels.clear()

        pts_rand = generate_random_points(10)
        scatter = ax.scatter(pts_rand[:, 0], pts_rand[:, 1], pts_rand[:, 2],
                             c="g", s=30)
        for i, p in enumerate(pts_rand):
            txt = ax.text(p[0], p[1], p[2], f"луч {i+1}", fontsize=8)
            labels.append(txt)
        fig.canvas.draw_idle()
        return pts_rand

    rand_pts = draw_random_points()

    # ----- GUI слева -----
    axbox_x1 = plt.axes([0.03, 0.78, 0.20, 0.04])
    axbox_y1 = plt.axes([0.03, 0.72, 0.20, 0.04])
    axbox_z1 = plt.axes([0.03, 0.66, 0.20, 0.04])

    x1_box = TextBox(axbox_x1, "X1:", initial="0")
    y1_box = TextBox(axbox_y1, "Y1:", initial="0")
    z1_box = TextBox(axbox_z1, "Z1:", initial="0")

    axbox_x2 = plt.axes([0.03, 0.58, 0.20, 0.04])
    axbox_y2 = plt.axes([0.03, 0.52, 0.20, 0.04])
    axbox_z2 = plt.axes([0.03, 0.46, 0.20, 0.04])

    x2_box = TextBox(axbox_x2, "X2:", initial="0")
    y2_box = TextBox(axbox_y2, "Y2:", initial="0")
    z2_box = TextBox(axbox_z2, "Z2:", initial="0")

    # кнопки
    ax_btn_calc = plt.axes([0.03, 0.38, 0.20, 0.05])
    btn_calc = Button(ax_btn_calc, "Рассчитать углы", color="lightgreen")

    ax_btn_home = plt.axes([0.03, 0.30, 0.20, 0.05])
    btn_home = Button(ax_btn_home, "Исходное положение", color="lightblue")

    ax_btn_stop = plt.axes([0.03, 0.22, 0.20, 0.05])
    btn_stop = Button(ax_btn_stop, "СТОП", color="lightcoral")

    ax_btn_rand = plt.axes([0.03, 0.14, 0.20, 0.05])
    btn_rand = Button(ax_btn_rand, "Обновить случайные точки", color="0.9")

    # текст результатов
    ax_result = plt.axes([0.03, 0.02, 0.20, 0.10])
    ax_result.axis("off")
    result_text = ax_result.text(0.0, 1.0, "", va="top", fontsize=9)

    moving = {"active": False}

    # ----- перерисовка робота -----
    def redraw(q_local):
        pts_local = all_points_mm(q_local)
        segs_local = np.stack([pts_local[:-1], pts_local[1:]], axis=1)
        lc.set_segments(segs_local)
        joint_scatter._offsets3d = (
            pts_local[:, 0], pts_local[:, 1], pts_local[:, 2]
        )
        fig.canvas.draw_idle()

    def set_result_from_q(q_local):
        lines = []
        for i, angle in enumerate(q_local, start=1):
            lines.append(f"Угол {i}: {np.rad2deg(angle):6.2f}°")
        result_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    # ----- обработчики -----
    def on_calc(event):
        result_text.set_text("ИК ещё не реализована")
        fig.canvas.draw_idle()

    def on_home(event):
        nonlocal q
        q = np.zeros(N_JOINTS)
        redraw(q)
        set_result_from_q(q)

    def on_stop(event):
        moving["active"] = False

    def on_rand(event):
        nonlocal rand_pts
        rand_pts = draw_random_points()

    btn_calc.on_clicked(on_calc)
    btn_home.on_clicked(on_home)
    btn_stop.on_clicked(on_stop)
    btn_rand.on_clicked(on_rand)

    def on_click(event):
        result_text.set_text("Клик по точке: ИК ещё не реализована")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    set_result_from_q(q)
    plt.show()
    fig.canvas.mpl_connect("button_press_event", on_click)

    set_result_from_q(q)
    plt.show()


if __name__ == "__main__":
    main()
