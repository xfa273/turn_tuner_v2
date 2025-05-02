#!/usr/bin/env python3
import math
import sys
import numpy as np
import tkinter
import tkinter.messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import partial
import multiprocessing as mp
import config  # config.py を参照

# グローバル変数（描画スケール）
SCALE_FACTOR = 1  # 1: ハーフサイズ, 2: クラシックサイズ（後でメイン処理で設定）

# ==========================
# ＜パラメータ探索用シミュレーション関数群＞
# ==========================

def integrate_phase(phi_func, t_phase, v, dt):
    x = 0.0
    y = 0.0
    t = 0.0
    n_steps = int(t_phase / dt)
    remainder = t_phase - n_steps * dt

    for i in range(n_steps):
        t_mid = t + dt/2
        phi = phi_func(t_mid)
        x += v * math.cos(phi) * dt
        y += v * math.sin(phi) * dt
        t += dt

    if remainder > 1e-12:
        t_mid = t + remainder/2
        phi = phi_func(t_mid)
        x += v * math.cos(phi) * remainder
        y += v * math.sin(phi) * remainder
        t += remainder

    phi_final = phi_func(t_phase)
    return x, y, phi_final

def simulate_turn(turning_angle_deg, translational_velocity, angular_acceleration_deg, rear_offset_distance, dt):
    theta_tot = turning_angle_deg * math.pi / 180.0
    alpha_mag = angular_acceleration_deg * math.pi / 180.0
    phi0 = math.pi/2  # 初期進行角（北向き）

    # 旋回角を３等分（「入り」設定）
    theta1 = theta_tot / 3.0
    theta2 = theta_tot / 3.0

    # フェーズ2：クロソイド入り（加速）
    t1 = math.sqrt(2 * theta1 / alpha_mag)
    def phi_phase2(t):
        return phi0 - 0.5 * alpha_mag * t * t
    x2, y2, _ = integrate_phase(phi_phase2, t1, translational_velocity, dt)
    phi1 = phi0 - theta1

    # フェーズ3：円弧区間（定角速度）
    w1 = -alpha_mag * t1  # フェーズ2終了時の角速度
    t2 = theta2 / (alpha_mag * t1)
    def phi_phase3(t):
        return phi1 + w1 * t
    x3, y3, _ = integrate_phase(phi_phase3, t2, translational_velocity, dt)
    phi2 = phi1 + w1 * t2

    # フェーズ4：クロソイド出口（減速）
    t3 = t1
    def phi_phase4(t):
        return phi2 + w1 * t + 0.5 * alpha_mag * t * t
    x4, y4, phi_phase4_end = integrate_phase(phi_phase4, t3, translational_velocity, dt)
    phi_final = phi_phase4_end

    # フェーズ5：後オフセット直進
    x5 = rear_offset_distance * math.cos(phi_final)
    y5 = rear_offset_distance * math.sin(phi_final)

    x_turn = x2 + x3 + x4 + x5
    y_turn = y2 + y3 + y4 + y5

    return x_turn, y_turn, phi_final

def simulate_full_trajectory(turning_angle_deg, translational_velocity, angular_acceleration_deg,
                             front_offset_distance, rear_offset_distance, dt):
    x_turn, y_turn, phi_final = simulate_turn(turning_angle_deg, translational_velocity,
                                              angular_acceleration_deg, rear_offset_distance, dt)
    phi0 = math.pi/2
    x_pre = front_offset_distance * math.cos(phi0)
    y_pre = front_offset_distance * math.sin(phi0)
    x_total = x_pre + x_turn
    y_total = y_pre + y_turn
    return x_total, y_total, phi_final

def compute_max_angular_velocity(turning_angle_deg, angular_acceleration_deg):
    theta_tot = turning_angle_deg * math.pi / 180.0
    theta1 = theta_tot / 3.0
    alpha_mag = angular_acceleration_deg * math.pi / 180.0
    max_w_rad = math.sqrt(2 * theta1 * alpha_mag)
    max_w_deg = max_w_rad * 180 / math.pi
    return max_w_rad, max_w_deg

def compute_error(angular_acceleration_deg, rear_offset_distance, target_x, target_y,
                  turning_angle_deg, translational_velocity, front_offset_distance, dt):
    x_end, y_end, phi_final = simulate_full_trajectory(turning_angle_deg, translational_velocity,
                                                       angular_acceleration_deg, front_offset_distance,
                                                       rear_offset_distance, dt)
    error = math.sqrt((x_end - target_x)**2 + (y_end - target_y)**2)
    return error, x_end, y_end, phi_final

def evaluate_candidate(args):
    (acc_deg, rear_offset, target_x, target_y,
     turning_angle_deg, translational_velocity, front_offset_distance, dt) = args
    err, x_end, y_end, phi_final = compute_error(acc_deg, rear_offset, target_x, target_y,
                                                 turning_angle_deg, translational_velocity, front_offset_distance, dt)
    return (acc_deg, rear_offset, err, x_end, y_end, phi_final)

def search_parameters_parallel(target_x, target_y, turning_angle_deg, translational_velocity, front_offset_distance, dt,
                               min_acc_deg, max_acc_deg, acc_step,
                               min_rear_offset, max_rear_offset, rear_offset_step):
    # 後ろオフセット距離は前オフセット距離と同じ値を使用
    rear_offset = front_offset_distance
    
    candidates = []
    acc = min_acc_deg
    while acc <= max_acc_deg:
        candidates.append((acc, rear_offset, target_x, target_y,
                           turning_angle_deg, translational_velocity, front_offset_distance, dt))
        acc += acc_step
    total_candidates = len(candidates)
    best_error = float('inf')
    best_result = None
    pool = mp.Pool()
    results = []
    for i, res in enumerate(pool.imap_unordered(evaluate_candidate, candidates), 1):
        results.append(res)
        progress = i / total_candidates * 100
        sys.stdout.write(f"\rProgress: {i}/{total_candidates} ({progress:5.1f}%)")
        sys.stdout.flush()
    pool.close()
    pool.join()
    print("")
    for acc_deg, rear_offset, err, x_end, y_end, phi_final in results:
        if err < best_error:
            best_error = err
            best_result = (acc_deg, rear_offset, x_end, y_end, phi_final)
    return best_result, best_error

# ==========================
# ＜描画ツール（GUI）用関数群＞
# ==========================

# グローバル変数（入口・出口の角度設定用）
ini_x = 0.0
ini_y = 45.0  # ハーフサイズの場合は45, クラシックの場合は90に変更
ini_angle = 0.0
fin_angle = None  # 後で SetAngleFromConfig() により設定

# グローバル変数 for 角度ボタン管理
angle_buttons = {}
selected_angle = None

def SetAngleFromConfig():
    """
    config.py の TURNING_ANGLE_DEG の値を使い、
    45, 90, 135, 180 の場合は「入り」設定によりグローバル変数を設定する。
    ハーフサイズの場合は ini_y = 45、クラシックの場合は ini_y = 90 とする。
    """
    global ini_x, ini_y, ini_angle, fin_angle, selected_angle
    default_y = 45.0 if SCALE_FACTOR == 1 else 90.0
    ta = config.TURNING_ANGLE_DEG
    if ta == 45:
        ini_x = 0.0; ini_y = default_y; ini_angle = 0.0; fin_angle = 45.0
    elif ta == 90:
        ini_x = 0.0; ini_y = default_y; ini_angle = 0.0; fin_angle = 90.0
    elif ta == 135:
        ini_x = 0.0; ini_y = default_y; ini_angle = 0.0; fin_angle = 135.0
    elif ta == 180:
        ini_x = 0.0; ini_y = default_y; ini_angle = 0.0; fin_angle = 180.0
    else:
        tkinter.messagebox.showerror("Error", "旋回角度の設定値は45, 90, 135, 180のいずれかでなければなりません。")
        sys.exit(1)
    selected_angle = ta
    update_angle_button_colors()

def update_angle_button_colors():
    for ang, btn in angle_buttons.items():
        if ang == selected_angle:
            btn.config(bg="lightblue")
        else:
            btn.config(bg="SystemButtonFace")

def SetAngle(foge):
    """
    GUIの角度選択ボタン用。 foge の値によりグローバル変数を更新する。
    SCALE_FACTOR に応じ、ハーフサイズなら ini_y=45、クラシックなら ini_y=90 に設定する。
    """
    global ini_x, ini_y, ini_angle, fin_angle, selected_angle
    default_y = 45.0 if SCALE_FACTOR == 1 else 90.0
    if foge == 90:
        ini_x = 0.0; ini_y = default_y; ini_angle = 0.0; fin_angle = 90.0
    elif foge == 180:
        ini_x = 0.0; ini_y = default_y; ini_angle = 0.0; fin_angle = 180.0
    elif foge == 45:
        ini_x = 0.0; ini_y = default_y; ini_angle = 0.0; fin_angle = 45.0
    elif foge == 135:
        ini_x = 0.0; ini_y = default_y; ini_angle = 0.0; fin_angle = 135.0
    elif foge == 91:
        ini_x = 0.0; ini_y = 90.0; ini_angle = 45.0; fin_angle = 90.0
    elif foge == 46:
        ini_x = 0.0; ini_y = 90.0; ini_angle = 45.0; fin_angle = 45.0
    elif foge == 136:
        ini_x = 0.0; ini_y = 90.0; ini_angle = 45.0; fin_angle = 135.0
    selected_angle = foge
    update_angle_button_colors()

def DrawCanvas(canvas, ax):
    s = SCALE_FACTOR
    ax.plot(np.array([-45.0, -45.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([0.0, 0.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([45.0, 45.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([90.0, 90.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([135.0, 135.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([-45.0, 135.0]) * s, np.array([0.0, 0.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([-45.0, 135.0]) * s, np.array([45.0, 45.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([-45.0, 135.0]) * s, np.array([90.0, 90.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([-45.0, 135.0]) * s, np.array([135.0, 135.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([-45.0, 135.0]) * s, np.array([180.0, 180.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([0.0, -45.0]) * s, np.array([180.0, 135.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([90.0, -45.0]) * s, np.array([180.0, 45.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([135.0, 0.0]) * s, np.array([135.0, 0.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([135.0, 90.0]) * s, np.array([45.0, 0.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([90.0, 135.0]) * s, np.array([180.0, 135.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([0.0, 135.0]) * s, np.array([180.0, 45.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([-45.0, 90.0]) * s, np.array([135.0, 0.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([-45.0, 0.0]) * s, np.array([45.0, 0.0]) * s, color="gray", linestyle="dashed")
    ax.plot(np.array([-48, -48, -42, -42, -48]) * s, np.array([-3, 3, 3, -3, -3]) * s, color="red", linestyle="solid")
    ax.plot(np.array([-48, -48, -42, -42, -48]) * s, np.array([87, 93, 93, 87, 87]) * s, color="red", linestyle="solid")
    ax.plot(np.array([-48, -48, -42, -42, -48]) * s, np.array([177, 183, 183, 177, 177]) * s, color="red", linestyle="solid")
    ax.plot(np.array([42, 42, 48, 48, 42]) * s, np.array([-3, 3, 3, -3, -3]) * s, color="red", linestyle="solid")
    ax.plot(np.array([42, 42, 48, 48, 42]) * s, np.array([87, 93, 93, 87, 87]) * s, color="red", linestyle="solid")
    ax.plot(np.array([42, 42, 48, 48, 42]) * s, np.array([177, 183, 183, 177, 177]) * s, color="red", linestyle="solid")
    ax.plot(np.array([132, 132, 138, 138, 132]) * s, np.array([-3, 3, 3, -3, -3]) * s, color="red", linestyle="solid")
    ax.plot(np.array([132, 132, 138, 138, 132]) * s, np.array([87, 93, 93, 87, 87]) * s, color="red", linestyle="solid")
    ax.plot(np.array([132, 132, 138, 138, 132]) * s, np.array([177, 183, 183, 177, 177]) * s, color="red", linestyle="solid")
    canvas.draw()

def DrawTrace(canvas, ax):
    speed = float(Set_Speed.get())
    low_AngVel = float(Set_low_AngVel.get())
    Low_AngAcl = float(Set_Low_AngAcl.get())
    pri_offset = float(Set_pri_offset.get())
    post_offset = float(Set_post_offset.get())
    K_slip_angle = float(Set_K_SP.get())
    Width = float(Set_Width.get())
    speed_ms = speed / 1000.0
    ax.cla()
    DrawCanvas(canvas, ax)
    fin_x = ini_x + pri_offset * np.sin(np.deg2rad(ini_angle))
    fin_y = ini_y + pri_offset * np.cos(np.deg2rad(ini_angle))
    ax.plot(np.array([ini_x, fin_x]), np.array([ini_y, fin_y]), color="blue", linestyle="solid")
    ax.plot(np.array([ini_x + (Width/2) * np.cos(np.deg2rad(ini_angle)), fin_x + (Width/2) * np.cos(np.deg2rad(ini_angle))]),
            np.array([ini_y - (Width/2) * np.sin(np.deg2rad(ini_angle)), fin_y - (Width/2) * np.sin(np.deg2rad(ini_angle))]),
            color="#888888", linestyle="solid")
    ax.plot(np.array([ini_x - (Width/2) * np.cos(np.deg2rad(ini_angle)), fin_x - (Width/2) * np.cos(np.deg2rad(ini_angle))]),
            np.array([ini_y + (Width/2) * np.sin(np.deg2rad(ini_angle)), fin_y + (Width/2) * np.sin(np.deg2rad(ini_angle))]),
            color="#888888", linestyle="solid")
    now_Angle = ini_angle
    s_now_Angle = ini_angle
    now_AngVel = 0
    while now_AngVel < low_AngVel:
        befor_x = fin_x; befor_y = fin_y
        now_AngVel += Low_AngAcl / 1000.0
        now_Angle += now_AngVel / 1000.0
        s_now_Angle = max(now_Angle - (now_AngVel * speed_ms * K_slip_angle), 0)
        fin_x = befor_x + speed_ms * np.sin(np.deg2rad(s_now_Angle))
        fin_y = befor_y + speed_ms * np.cos(np.deg2rad(s_now_Angle))
        ax.plot(np.array([befor_x, fin_x]), np.array([befor_y, fin_y]), color="green", linestyle="solid")
        ax.plot(np.array([befor_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                np.array([befor_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                color="#888888", linestyle="solid")
        ax.plot(np.array([befor_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                np.array([befor_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                color="#888888", linestyle="solid")
    now_AngVel = low_AngVel
    use_angle = now_Angle - ini_angle
    while now_Angle < ((ini_angle + fin_angle) - use_angle):
        befor_x = fin_x; befor_y = fin_y
        now_Angle += low_AngVel / 1000.0
        s_now_Angle = now_Angle - (low_AngVel * speed_ms * K_slip_angle)
        fin_x = befor_x + speed_ms * np.sin(np.deg2rad(s_now_Angle))
        fin_y = befor_y + speed_ms * np.cos(np.deg2rad(s_now_Angle))
        ax.plot(np.array([befor_x, fin_x]), np.array([befor_y, fin_y]), color="red", linestyle="solid")
        ax.plot(np.array([befor_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                np.array([befor_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                color="#888888", linestyle="solid")
        ax.plot(np.array([befor_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                np.array([befor_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                color="#888888", linestyle="solid")
    while now_AngVel > 0:
        befor_x = fin_x; befor_y = fin_y
        now_AngVel -= Low_AngAcl / 1000.0
        now_Angle += now_AngVel / 1000.0
        s_now_Angle = now_Angle - (now_AngVel * speed_ms * K_slip_angle)
        fin_x = befor_x + speed_ms * np.sin(np.deg2rad(s_now_Angle))
        fin_y = befor_y + speed_ms * np.cos(np.deg2rad(s_now_Angle))
        ax.plot(np.array([befor_x, fin_x]), np.array([befor_y, fin_y]), color="green", linestyle="solid")
        ax.plot(np.array([befor_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                np.array([befor_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                color="#888888", linestyle="solid")
        ax.plot(np.array([befor_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                np.array([befor_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                color="#888888", linestyle="solid")
    befor_x = fin_x; befor_y = fin_y
    fin_x = befor_x + post_offset * np.sin(np.deg2rad(ini_angle + fin_angle))
    fin_y = befor_y + post_offset * np.cos(np.deg2rad(ini_angle + fin_angle))
    ax.plot(np.array([befor_x, fin_x]), np.array([befor_y, fin_y]), color="blue", linestyle="solid")
    ax.plot(np.array([befor_x + (Width/2)*np.cos(np.deg2rad(ini_angle + fin_angle)),
                      fin_x + (Width/2)*np.cos(np.deg2rad(ini_angle + fin_angle))]),
            np.array([befor_y - (Width/2)*np.sin(np.deg2rad(ini_angle + fin_angle)),
                      fin_y - (Width/2)*np.sin(np.deg2rad(ini_angle + fin_angle))]),
            color="#888888", linestyle="solid")
    ax.plot(np.array([befor_x - (Width/2)*np.cos(np.deg2rad(ini_angle + fin_angle)),
                      fin_x - (Width/2)*np.cos(np.deg2rad(ini_angle + fin_angle))]),
            np.array([befor_y + (Width/2)*np.sin(np.deg2rad(ini_angle + fin_angle)),
                      fin_y + (Width/2)*np.sin(np.deg2rad(ini_angle + fin_angle))]),
            color="#888888", linestyle="solid")
    canvas.draw()

# ==========================
# ＜メイン処理＞
# ==========================
if __name__ == "__main__":
    try:
        TRANSLATIONAL_VELOCITY = config.TRANSLATIONAL_VELOCITY
        TURNING_ANGLE_DEG = config.TURNING_ANGLE_DEG
        TARGET_X = config.TARGET_X
        TARGET_Y = config.TARGET_Y
        FRONT_OFFSET_DISTANCE = config.FRONT_OFFSET_DISTANCE
        MIN_ACC_DEG = config.MIN_ACC_DEG
        MAX_ACC_DEG = config.MAX_ACC_DEG
        ACC_STEP = config.ACC_STEP
        MIN_REAR_OFFSET = config.MIN_REAR_OFFSET
        MAX_REAR_OFFSET = config.MAX_REAR_OFFSET
        REAR_OFFSET_STEP = config.REAR_OFFSET_STEP
        DT = config.DT

        if TURNING_ANGLE_DEG not in (45, 90, 135, 180):
            tkinter.messagebox.showerror("Error", "旋回角度の設定値は45, 90, 135, 180のいずれかでなければなりません。")
            sys.exit(1)

        if TARGET_X >= 180 or TARGET_Y >= 180:
            SCALE_FACTOR = 2
        else:
            SCALE_FACTOR = 1

        SetAngleFromConfig()

        print("----- Parameter Search Start -----")
        best_result, best_error = search_parameters_parallel(
            TARGET_X, TARGET_Y, TURNING_ANGLE_DEG, TRANSLATIONAL_VELOCITY, FRONT_OFFSET_DISTANCE, DT,
            MIN_ACC_DEG, MAX_ACC_DEG, ACC_STEP,
            MIN_REAR_OFFSET, MAX_REAR_OFFSET, REAR_OFFSET_STEP
        )
        if best_result is None:
            print("探索結果が見つかりませんでした。")
            sys.exit(1)
        best_acc_deg, best_rear_offset, x_end, y_end, phi_final = best_result
        _, max_w_deg = compute_max_angular_velocity(TURNING_ANGLE_DEG, best_acc_deg)
        print(f"探索完了：最適角加速度 = {best_acc_deg:.2f} deg/s², 最適後オフセット = {best_rear_offset:.2f} mm, 最大角速度 = {max_w_deg:.2f} deg/s")
        print("----- Parameter Search End -----\n")

        root = tkinter.Tk()
        root.title("たーんしみゅれーた")
        fig, ax1 = plt.subplots(figsize=(7.0, 7.0))
        fig.gca().set_aspect("equal", adjustable="box")
        Canvas = FigureCanvasTkAgg(fig, master=root)
        Canvas.get_tk_widget().grid(row=0, rowspan=10, column=1)

        lbl_speed = tkinter.Label(text="速度 (mm/s)")
        lbl_speed.grid(row=0, column=2)
        lbl_AngVel = tkinter.Label(text="最大角速度 (deg/s)")
        lbl_AngVel.grid(row=1, column=2)
        lbl_AngAcl = tkinter.Label(text="角加速度 (deg/s²)")
        lbl_AngAcl.grid(row=2, column=2)
        lbl_pri_offset = tkinter.Label(text="入口オフセット (mm)")
        lbl_pri_offset.grid(row=3, column=2)
        lbl_post_offset = tkinter.Label(text="出口オフセット (mm)")
        lbl_post_offset.grid(row=4, column=2)
        lbl_K_SP = tkinter.Label(text="スリップアングル係数")
        lbl_K_SP.grid(row=5, column=2)
        lbl_Width = tkinter.Label(text="機体の横幅 (mm)")
        lbl_Width.grid(row=6, column=2)

        Set_Speed = tkinter.Entry(width=10)
        Set_Speed.grid(row=0, column=3)
        Set_low_AngVel = tkinter.Entry(width=10)
        Set_low_AngVel.grid(row=1, column=3)
        Set_Low_AngAcl = tkinter.Entry(width=10)
        Set_Low_AngAcl.grid(row=2, column=3)
        Set_pri_offset = tkinter.Entry(width=10)
        Set_pri_offset.grid(row=3, column=3)
        Set_post_offset = tkinter.Entry(width=10)
        Set_post_offset.grid(row=4, column=3)
        Set_K_SP = tkinter.Entry(width=10)
        Set_K_SP.grid(row=5, column=3)
        Set_Width = tkinter.Entry(width=10)
        Set_Width.grid(row=6, column=3)

        Set_Speed.insert(0, f"{TRANSLATIONAL_VELOCITY:.2f}")
        Set_low_AngVel.insert(0, f"{max_w_deg:.2f}")
        Set_Low_AngAcl.insert(0, f"{best_acc_deg:.2f}")
        Set_pri_offset.insert(0, f"{FRONT_OFFSET_DISTANCE:.2f}")
        Set_post_offset.insert(0, f"{best_rear_offset:.2f}")
        Set_K_SP.insert(0, f"{0.00:.2f}")
        # 機体幅は、SCALE_FACTOR に応じて使い分ける
        if SCALE_FACTOR == 1:
            Set_Width.insert(0, f"{config.ROBOT_WIDTH_HALF:.2f}")
        else:
            Set_Width.insert(0, f"{config.ROBOT_WIDTH_CLASSIC:.2f}")

        DrawCanvas(Canvas, ax1)

        angle_buttons = {}
        selected_angle = config.TURNING_ANGLE_DEG

        btn_90 = tkinter.Button(text="90°", width=10, command=partial(SetAngle, 90))
        btn_90.grid(row=0, column=0)
        angle_buttons[90] = btn_90

        btn_180 = tkinter.Button(text="180°", width=10, command=partial(SetAngle, 180))
        btn_180.grid(row=1, column=0)
        angle_buttons[180] = btn_180

        btn_in45 = tkinter.Button(text="入45°", width=10, command=partial(SetAngle, 45))
        btn_in45.grid(row=2, column=0)
        angle_buttons[45] = btn_in45

        btn_out45 = tkinter.Button(text="出45°", width=10, command=partial(SetAngle, 46))
        btn_out45.grid(row=3, column=0)
        angle_buttons[46] = btn_out45

        btn_V90 = tkinter.Button(text="V90°", width=10, command=partial(SetAngle, 91))
        btn_V90.grid(row=4, column=0)
        angle_buttons[91] = btn_V90

        btn_in135 = tkinter.Button(text="入135°", width=10, command=partial(SetAngle, 135))
        btn_in135.grid(row=5, column=0)
        angle_buttons[135] = btn_in135

        btn_out135 = tkinter.Button(text="出135°", width=10, command=partial(SetAngle, 136))
        btn_out135.grid(row=6, column=0)
        angle_buttons[136] = btn_out135

        update_angle_button_colors()

        btn_generate = tkinter.Button(text="軌道生成", width=10, command=partial(DrawTrace, Canvas, ax1))
        btn_generate.grid(row=8, column=2)

        # GUI表示直後の1回だけ自動で軌道生成
        root.after(100, lambda: DrawTrace(Canvas, ax1))

        root.mainloop()

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        input(">>")
