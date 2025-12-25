import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path
import sys
import random

BASE_DIR = Path(__file__).resolve().parents[1]
THREED_DIR = BASE_DIR / "3d"
sys.path.append(str(THREED_DIR))

from robot_kinematics import RobotArmRRRR
from ik_rrrr import inverse_kinematics_rrrr

MM = 1000.0
N_JOINTS = 4
LINK_LENGTHS_MM = [400, 350, 200, 0]
R_WORK_MM = sum(abs(l) for l in LINK_LENGTHS_MM)

robot = RobotArmRRRR(l1=0.4, l2=0.35, l3=0.20)

# Глобальная переменная для хранения последних матриц
current_matrices_data = []

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

def ik_mm(target_mm, q0=None):
    target_m = np.asarray(target_mm) / MM
    q1, q2, q3, q4 = inverse_kinematics_rrrr(target_m, l1=robot.l1, l2=robot.l2, l3=robot.l3)
    
    # ФИКС ЗЕРКАЛЬНОСТИ
    q2 = -q2
    q3 = -q3
    q4 = -(q2 + q3)
    
    return np.array([q1, q2, q3, q4])

def main():
    global current_matrices_data
    q = np.zeros(N_JOINTS)

    fig = plt.figure(figsize=(14, 8)) 
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.35, right=0.98, bottom=0.05, top=0.95)

    lim = R_WORK_MM * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, lim)
    ax.set_xlabel("X (мм)")
    ax.set_ylabel("Y (мм)")
    ax.set_zlabel("Z (мм)")

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = R_WORK_MM * np.outer(np.cos(u), np.sin(v))
    ys = R_WORK_MM * np.outer(np.sin(u), np.sin(v))
    zs = R_WORK_MM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="purple", alpha=0.1, linewidth=0)

    # --- Функция отрисовки ---
    def get_robot_state(q_local):
        q1, q2, q3, q4 = q_local[:4]
        pts, matrices = robot.forward(q1, q2, q3, q4)
        return np.asarray(pts) * MM, matrices

    pts, matrices = get_robot_state(q)
    current_matrices_data = matrices # Сохраняем сразу

    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, colors=["b"] * (len(pts) - 1), linewidths=3)
    ax.add_collection3d(lc)

    joint_scatter = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=["k", "k", "k", "k", "r"],
        s=[30, 30, 30, 30, 40]
    )

    rand_pts = generate_random_points(10)
    scatter = ax.scatter(rand_pts[:, 0], rand_pts[:, 1], rand_pts[:, 2], c="g", s=30)
    labels = []
    for i, p in enumerate(rand_pts):
        labels.append(ax.text(p[0], p[1], p[2], f"L{i+1}", fontsize=8))

    # ================= GUI ЭЛЕМЕНТЫ =================
    
    # 1. Ввод координат (IK)
    axbox_x = plt.axes([0.05, 0.85, 0.10, 0.04])
    axbox_y = plt.axes([0.05, 0.80, 0.10, 0.04])
    axbox_z = plt.axes([0.05, 0.75, 0.10, 0.04])
    
    box_x = TextBox(axbox_x, "X:", initial="0")
    box_y = TextBox(axbox_y, "Y:", initial="0")
    box_z = TextBox(axbox_z, "Z:", initial="0")
    
    btn_calc_ax = plt.axes([0.05, 0.69, 0.20, 0.05])
    btn_calc = Button(btn_calc_ax, "Рассчитать (XYZ)", color="lightgreen")

    # 2. Ввод углов (FK)
    plt.text(0.05, 0.65, "Ввод углов (град):", transform=fig.transFigure, fontsize=10, fontweight='bold')
    
    axbox_q1 = plt.axes([0.05, 0.60, 0.10, 0.04])
    axbox_q2 = plt.axes([0.05, 0.55, 0.10, 0.04])
    axbox_q3 = plt.axes([0.05, 0.50, 0.10, 0.04])
    axbox_q4 = plt.axes([0.05, 0.45, 0.10, 0.04])

    box_q1 = TextBox(axbox_q1, "Q1:", initial="0")
    box_q2 = TextBox(axbox_q2, "Q2:", initial="0")
    box_q3 = TextBox(axbox_q3, "Q3:", initial="0")
    box_q4 = TextBox(axbox_q4, "Q4:", initial="0")

    btn_fk_ax = plt.axes([0.05, 0.39, 0.20, 0.05])
    btn_fk = Button(btn_fk_ax, "Установить углы", color="orange")

    # 3. Управление
    btn_home_ax = plt.axes([0.05, 0.30, 0.20, 0.05])
    btn_home = Button(btn_home_ax, "Домой", color="lightblue")

    btn_rand_ax = plt.axes([0.05, 0.24, 0.20, 0.05])
    btn_rand = Button(btn_rand_ax, "Новые точки", color="0.9")

    # 4. КНОПКА ПОКАЗА МАТРИЦ
    btn_mat_ax = plt.axes([0.05, 0.15, 0.20, 0.05])
    btn_mat = Button(btn_mat_ax, "Показать матрицы", color="violet")

    # Текст статуса
    ax_result = plt.axes([0.05, 0.02, 0.20, 0.10])
    ax_result.axis("off")
    result_text = ax_result.text(0.0, 1.0, "Готов", va="top", fontsize=9)

    def redraw(q_local):
        global current_matrices_data
        pts_local, mats = get_robot_state(q_local)
        current_matrices_data = mats # Обновляем глобальные данные
        
        segs_local = np.stack([pts_local[:-1], pts_local[1:]], axis=1)
        lc.set_segments(segs_local)
        joint_scatter._offsets3d = (pts_local[:, 0], pts_local[:, 1], pts_local[:, 2])
        fig.canvas.draw_idle()
        
        # Обновляем поля углов
        box_q1.set_val(f"{np.rad2deg(q_local[0]):.1f}")
        box_q2.set_val(f"{np.rad2deg(q_local[1]):.1f}")
        box_q3.set_val(f"{np.rad2deg(q_local[2]):.1f}")
        box_q4.set_val(f"{np.rad2deg(q_local[3]):.1f}")

    def on_calc(event):
        nonlocal q
        try:
            x, y, z = float(box_x.text), float(box_y.text), float(box_z.text)
            target = np.array([x, y, z])
            if not inside_workspace(target):
                result_text.set_text("Вне рабочей зоны!")
                return
            q_new = ik_mm(target, q)
            q = np.asarray(q_new, dtype=float)
            redraw(q)
            result_text.set_text(f"IK OK.")
        except Exception as e:
            result_text.set_text(f"Ошибка: {e}")

    def on_set_angles(event):
        nonlocal q
        try:
            d1 = float(box_q1.text)
            d2 = float(box_q2.text)
            d3 = float(box_q3.text)
            d4 = float(box_q4.text)
            q = np.deg2rad([d1, d2, d3, d4])
            redraw(q)
            result_text.set_text(f"Углы установлены.")
        except ValueError:
            result_text.set_text("Ошибка ввода углов")

    def on_show_matrices(event):
        # Создаем новое окно для текста
        fig_m = plt.figure(figsize=(6, 8))
        fig_m.canvas.manager.set_window_title("Матрицы трансформации")
        
        ax_m = fig_m.add_subplot(111)
        ax_m.axis("off") # Скрываем оси
        
        # Формируем текст
        text_str = "ТЕКУЩИЕ МАТРИЦЫ:\n\n"
        for i, M in enumerate(current_matrices_data):
            text_str += f"Звено {i+1} (T{i+1}):\n"
            # Форматируем красиво
            mat_str = np.array2string(M, precision=2, suppress_small=True, separator=', ')
            text_str += mat_str + "\n\n"
            
        ax_m.text(0.05, 0.95, text_str,  family='monospace', va='top', fontsize=10)
        plt.show()

    def on_home(event):
        nonlocal q
        q = np.zeros(N_JOINTS)
        redraw(q)
        result_text.set_text("Домой")

    def on_rand(event):
        nonlocal rand_pts, labels
        for t in labels: t.remove()
        labels.clear()
        rand_pts = generate_random_points(10)
        scatter._offsets3d = (rand_pts[:, 0], rand_pts[:, 1], rand_pts[:, 2])
        for i, p in enumerate(rand_pts):
            labels.append(ax.text(p[0], p[1], p[2], f"L{i+1}", fontsize=8))
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal q
        if event.inaxes is not ax: return
        if event.button != 1: return 
        
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None: return

        min_dist = float('inf')
        closest_idx = -1
        
        for i, pt in enumerate(rand_pts):
            x2d, y2d, _ = proj3d.proj_transform(pt[0], pt[1], pt[2], ax.get_proj())
            dist = np.sqrt((x2d - click_x)**2 + (y2d - click_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if closest_idx >= 0 and min_dist < 0.05:
            target = rand_pts[closest_idx]
            box_x.set_val(f"{target[0]:.0f}")
            box_y.set_val(f"{target[1]:.0f}")
            box_z.set_val(f"{target[2]:.0f}")
            try:
                q_new = ik_mm(target, q)
                q = np.asarray(q_new, dtype=float)
                redraw(q)
                result_text.set_text(f"Цель: Луч {closest_idx+1}")
            except Exception as e:
                result_text.set_text("Ошибка IK")

    # Подключение кнопок
    btn_calc.on_clicked(on_calc)
    btn_fk.on_clicked(on_set_angles)
    btn_home.on_clicked(on_home)
    btn_rand.on_clicked(on_rand)
    btn_mat.on_clicked(on_show_matrices) # Новая кнопка
    
    fig.canvas.mpl_connect("button_press_event", on_click)

    redraw(q)
    plt.show()

if __name__ == "__main__":
    main()
