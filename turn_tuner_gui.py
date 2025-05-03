#!/usr/bin/env python3
import math
import sys
import os
import json
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'MS Gothic'  # Windows用日本語フォント
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
from functools import partial
import multiprocessing as mp
import config  # config.py を参照
import turn_params  # ターン用パラメータ

# グローバル変数（描画スケール）
SCALE_FACTOR = 1  # 1: ハーフサイズ, 2: クラシックサイズ（後でメイン処理で設定）

# ==========================
# ＜パラメータ探索用シミュレーション関数群＞
# ==========================

def integrate_phase(phi_func, t_phase, v, dt):
    """位相ごとの数値積分"""
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

def simulate_turn(turning_angle_deg, translational_velocity, angular_acceleration_deg, rear_offset_distance, dt, slip_coefficient=1.0):
    """ターン動作のシミュレーション
    Parameters:
        turning_angle_deg: 旋回角度 [deg]
        translational_velocity: 並進速度 [mm/s]
        angular_acceleration_deg: 角加速度 [deg/s^2]
        rear_offset_distance: 後オフセット距離 [mm]
        dt: 時間ステップ [s]
        slip_coefficient: スリップアングル係数 (1.0=スリップなし)
    """
    theta_tot = turning_angle_deg * math.pi / 180.0
    # スリップアングル係数を角加速度に適用
    alpha_mag = angular_acceleration_deg * math.pi / 180.0 * slip_coefficient
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
    t2 = theta2 / abs(w1)
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
                             front_offset_distance, rear_offset_distance, dt, slip_coefficient=1.0):
    """全体の軌道シミュレーション（前オフセット＋ターン＋後オフセット）
    Parameters:
        turning_angle_deg: 旋回角度 [deg]
        translational_velocity: 並進速度 [mm/s]
        angular_acceleration_deg: 角加速度 [deg/s^2]
        front_offset_distance: 前オフセット距離 [mm]
        rear_offset_distance: 後オフセット距離 [mm]
        dt: 時間ステップ [s]
        slip_coefficient: スリップアングル係数 (1.0=スリップなし)
    """
    x_turn, y_turn, phi_final = simulate_turn(turning_angle_deg, translational_velocity,
                                              angular_acceleration_deg, rear_offset_distance, dt, slip_coefficient)
    phi0 = math.pi/2
    x_pre = front_offset_distance * math.cos(phi0)
    y_pre = front_offset_distance * math.sin(phi0)
    x_total = x_pre + x_turn
    y_total = y_pre + y_turn
    return x_total, y_total, phi_final

def compute_max_angular_velocity(turning_angle_deg, angular_acceleration_deg):
    """最大角速度の計算"""
    theta_tot = turning_angle_deg * math.pi / 180.0
    theta1 = theta_tot / 3.0
    alpha_mag = angular_acceleration_deg * math.pi / 180.0
    max_w_rad = math.sqrt(2 * theta1 * alpha_mag)
    max_w_deg = max_w_rad * 180 / math.pi
    return max_w_rad, max_w_deg

def is_diagonal_turn(turn_type):
    """斜め入りのターンかどうかを判定する関数"""
    return "45deg入り" in turn_type or "135deg入り" in turn_type

def compute_error(angular_acceleration_deg, rear_offset_distance, target_x, target_y,
                  turning_angle_deg, translational_velocity, front_offset_distance, dt, slip_coefficient=1.0):
    """目標位置との誤差計算"""
    x_end, y_end, phi_final = simulate_full_trajectory(turning_angle_deg, translational_velocity,
                                                       angular_acceleration_deg, front_offset_distance,
                                                       rear_offset_distance, dt, slip_coefficient)
    # 目標位置との二乗誤差
    error = (x_end - target_x)**2 + (y_end - target_y)**2
    return error, x_end, y_end, phi_final

def evaluate_candidate(args):
    """探索候補の評価"""
    angular_acceleration_deg, rear_offset, target_x, target_y, turning_angle_deg, translational_velocity, front_offset_distance, dt = args
    error, x_end, y_end, phi_final = compute_error(angular_acceleration_deg, rear_offset, target_x, target_y,
                                                   turning_angle_deg, translational_velocity, front_offset_distance, dt)
    return error, angular_acceleration_deg, rear_offset, x_end, y_end, phi_final

def search_parameters_parallel(target_x, target_y, turning_angle_deg, translational_velocity, front_offset_distance, dt,
                               min_acc_deg, max_acc_deg, acc_step,
                               min_rear_offset, max_rear_offset, rear_offset_step):
    """パラメータの並列探索"""
    candidates = []
    
    # 探索対象のパラメータ組み合わせを作成
    for angular_acceleration_deg in np.arange(min_acc_deg, max_acc_deg + acc_step, acc_step):
        for rear_offset in np.arange(min_rear_offset, max_rear_offset + rear_offset_step, rear_offset_step):
            candidates.append((angular_acceleration_deg, rear_offset, target_x, target_y, turning_angle_deg, translational_velocity, front_offset_distance, dt))

    # マルチプロセスによる並列計算
    with mp.Pool() as pool:
        results = pool.map(evaluate_candidate, candidates)

    # 誤差最小のパラメータを選択
    best_result = None
    min_error = float('inf')

    for result in results:
        error, angular_acceleration_deg, rear_offset, x_end, y_end, phi_final = result
        if error < min_error:
            min_error = error
            best_result = (angular_acceleration_deg, rear_offset, x_end, y_end, phi_final)

    return best_result

# ==========================
# ＜GUIアプリケーションクラス＞
# ==========================

class TurnTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ターンシミュレーター v2")
        
        # 閉じるボタンが押されたときの処理を設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 設定ファイルのパス
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        
        # 設定ファイルから設定を読み込む
        self.settings = self.load_settings()
        
        # 最適化結果保存用の変数
        self.optimized_values = None
        
        # ロボットサイズ設定を取得
        self.robot_settings = self.settings["robot_settings"]
        
        # 前回の選択状態を取得
        self.robot_size = self.settings["current_selection"]["robot_size"]
        self.turn_type = self.settings["current_selection"]["turn_type"]
        
        # turn_paramsモジュールを正しく参照
        import turn_params as tp  # エラー回避のために再度インポート
        self.scale_factor = tp.ROBOT_SIZES[self.robot_size]["scale"]
        
        # ロボットの幅を設定から読み込み
        if self.robot_size in self.robot_settings:
            self.robot_width = self.robot_settings[self.robot_size]["width"]
        else:
            # 設定になければデフォルト値を使用
            self.robot_width = tp.ROBOT_SIZES[self.robot_size]["width"]
            # 設定に追加
            self.robot_settings[self.robot_size] = {"width": self.robot_width}
            self.save_settings()
        
        # 元のグローバル変数をクラスのメンバ変数として再定義
        self.ini_x = 0.0
        self.ini_y = 45.0 * self.scale_factor  # ハーフサイズの場合は45, クラシックの場合は90
        self.ini_angle = 0.0
        self.fin_angle = 90.0  # 初期値、load_turn_paramsで上書きされる
        
        # 動作パラメータの初期化（デフォルト設定）
        self.load_turn_params()
        
        # GUI作成
        self.create_gui()
        
        # 初期表示のみ行い、自動軸道計算は行わない
        self.root.after(100, self.initialize_plot)
        
    def load_settings(self):
        """設定ファイルから設定を読み込む"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                    
                    # 当初フォーマットの場合は新フォーマットに変換
                    if isinstance(settings_data, dict) and "robot_settings" not in settings_data:
                        # 旧形式（ロボットサイズの設定のみ）
                        return {
                            "robot_settings": settings_data,
                            "current_selection": {
                                "robot_size": "ハーフ",
                                "turn_type": "小回り90deg"
                            },
                            "parameters": {
                                "velocity": 250,
                                "front_offset": 10
                            }
                        }
                    return settings_data
            else:
                # 新規作成時のデフォルト設定
                return {
                    "robot_settings": {
                        "ハーフ": {"width": 36.0},
                        "クラシック": {"width": 70.0}
                    },
                    "current_selection": {
                        "robot_size": "ハーフ",
                        "turn_type": "小回り90deg"
                    },
                    "parameters": {
                        "velocity": 250,
                        "front_offset": 10,
                        "slip_coefficient": 1.0,
                        "target_error": 2.0,
                        "acc_step": 100.0,
                        "precision": 10
                    }
                }
        except Exception as e:
            print(f"設定読み込みエラー: {e}")
            # エラー時のデフォルト設定
            return {
                "robot_settings": {
                    "ハーフ": {"width": 36.0},
                    "クラシック": {"width": 70.0}
                },
                "current_selection": {
                    "robot_size": "ハーフ",
                    "turn_type": "小回り90deg"
                },
                "parameters": {
                    "velocity": 250,
                    "front_offset": 10,
                    "slip_coefficient": 1.0,
                    "target_error": 2.0,
                    "acc_step": 100.0,
                    "precision": 10
                }
            }
    
    def save_settings(self):
        """設定をファイルに保存する"""
        try:
            # 全体設定データの作成
            settings_data = {
                # ロボットサイズに関する設定
                "robot_settings": self.robot_settings,
                
                # 現在の選択状態
                "current_selection": {
                    "robot_size": self.robot_size,
                    "turn_type": self.turn_type
                },
                
                # 現在の入力値
                "parameters": {
                    "velocity": self.translational_velocity,
                    "front_offset": self.display_front_offset,
                    "slip_coefficient": self.slip_coefficient,
                    "target_error": float(self.target_y_error_entry.get()),
                    "acc_step": float(self.acc_step_entry.get()),
                    "precision": self.precision_slider.get()
                }
            }
            
            # 設定ファイルに保存
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"設定保存エラー: {e}")
            tkinter.messagebox.showerror("設定保存エラー", f"設定の保存中にエラーが発生しました: {e}")
    
    def update_robot_width(self):
        """機体の幅を更新して保存する"""
        try:
            new_width = float(self.width_entry.get())
            if new_width <= 0:
                tkinter.messagebox.showerror("エラー", "機体の幅は正の値でなければなりません")
                # 元の値に戻す
                self.width_entry.delete(0, tk.END)
                self.width_entry.insert(0, str(self.robot_width))
                return
                
            # 幅を更新
            self.robot_width = new_width
            
            # 設定に保存
            if self.robot_size not in self.robot_settings:
                self.robot_settings[self.robot_size] = {}
            self.robot_settings[self.robot_size]["width"] = new_width
            self.save_settings()
            
            # 軸道再計算
            self.generate_trajectory()
            
            tkinter.messagebox.showinfo("設定更新", "機体の幅を更新しました")
            
        except ValueError:
            tkinter.messagebox.showerror("エラー", "機体の幅には数値を入力してください")
            # 元の値に戻す
            self.width_entry.delete(0, tk.END)
            self.width_entry.insert(0, str(self.robot_width))
        except Exception as e:
            tkinter.messagebox.showerror("エラー", f"予期せぬエラーが発生しました: {e}")
            # 元の値に戻す
            self.width_entry.delete(0, tk.END)
            self.width_entry.insert(0, str(self.robot_width))
    
    def load_turn_params(self):
        """現在の設定に基づいてターンパラメータを読み込む"""
        # turn_paramsモジュールへの参照を一貫させる
        import turn_params as tp
        
        turns_dict = tp.HALF_TURNS if self.robot_size == "ハーフ" else tp.CLASSIC_TURNS
        turn_params_dict = turns_dict[self.turn_type]
        
        # パラメータ設定
        self.turning_angle_deg = turn_params_dict["angle"]
        self.target_x = turn_params_dict["target_x"]
        self.target_y = turn_params_dict["target_y"]
        
        # 速度は保存された値から読み込み（初回起動時はデフォルト値）
        if "parameters" in self.settings and "velocity" in self.settings["parameters"]:
            self.translational_velocity = self.settings["parameters"]["velocity"]
        else:
            self.translational_velocity = turn_params_dict["default_velocity"]

        # 前オフセットは保存された値から読み込み（初回起動時はデフォルト値）
        if "parameters" in self.settings and "front_offset" in self.settings["parameters"]:
            # 保存された値は常に表示用の値（小回り90degの場合はすでに半区画分引かれている）
            display_offset = self.settings["parameters"]["front_offset"]
        else:
            # 初回起動時はデフォルト値を使用
            if self.turn_type == "小回り90deg":
                half_cell = 45.0 if self.robot_size == "ハーフ" else 90.0
                display_offset = turn_params_dict["default_front_offset"] - half_cell
            else:
                display_offset = turn_params_dict["default_front_offset"]
        
        # 表示用値を設定
        self.display_front_offset = display_offset
        
        # 小回り90degの場合、計算用には半区画分を加算
        if self.turn_type == "小回り90deg":
            half_cell = 45.0 if self.robot_size == "ハーフ" else 90.0
            self.front_offset_distance = display_offset + half_cell
        else:
            # それ以外のターンは表示と計算で同じ値を使用
            self.front_offset_distance = display_offset
        
        # 計算パラメータ
        self.dt = tp.CALC_PARAMS["dt"]
        self.min_acc_deg = tp.CALC_PARAMS["min_acc_deg"]
        self.max_acc_deg = tp.CALC_PARAMS["max_acc_deg"]
        self.acc_step = tp.CALC_PARAMS["acc_step"]
        self.min_rear_offset = tp.CALC_PARAMS["min_rear_offset"]
        self.max_rear_offset = tp.CALC_PARAMS["max_rear_offset"]
        self.rear_offset_step = tp.CALC_PARAMS["rear_offset_step"]
        
        # スリップアングル係数の初期化
        if "parameters" in self.settings and "slip_coefficient" in self.settings["parameters"]:
            self.slip_coefficient = self.settings["parameters"]["slip_coefficient"]
        else:
            self.slip_coefficient = tp.CALC_PARAMS["default_slip_coefficient"]
        
    def create_gui(self):
        """GUIコンポーネントを作成する"""
        # メインフレームの作成
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左側フレーム（設定部分）
        left_frame = ttk.Frame(main_frame, padding=5)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 右側フレーム（グラフ表示部分）
        right_frame = ttk.Frame(main_frame, padding=5)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ===== 左側フレームの内容 =====
        # turn_paramsモジュールへの参照を一貫させる
        import turn_params as tp
        
        # 機体サイズ選択
        ttk.Label(left_frame, text="機体サイズ:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.size_combo = ttk.Combobox(left_frame, values=list(tp.ROBOT_SIZES.keys()), width=15, state="readonly")
        self.size_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.size_combo.set(self.robot_size)
        self.size_combo.bind("<<ComboboxSelected>>", self.on_size_changed)
        
        # 動作タイプ選択
        ttk.Label(left_frame, text="動作タイプ:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.turn_combo = ttk.Combobox(left_frame, values=list(tp.HALF_TURNS.keys()), width=15, state="readonly")
        self.turn_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.turn_combo.set(self.turn_type)
        self.turn_combo.bind("<<ComboboxSelected>>", self.on_turn_changed)
        
        # 並進速度設定
        ttk.Label(left_frame, text="並進速度 [mm/s]:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.velocity_entry = ttk.Entry(left_frame, width=15)
        self.velocity_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        self.velocity_entry.insert(0, str(self.translational_velocity))
        
        # 前オフセット距離設定
        ttk.Label(left_frame, text="前オフセット [mm]:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.front_offset_entry = ttk.Entry(left_frame, width=15)
        self.front_offset_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
        # 表示用の値を使用して表示フィールドを初期化
        self.front_offset_entry.insert(0, str(self.display_front_offset))
        
        # 機体幅設定
        ttk.Label(left_frame, text="機体の幅 [mm]:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.width_entry = ttk.Entry(left_frame, width=15)
        self.width_entry.grid(row=4, column=1, sticky=tk.W, pady=5)
        self.width_entry.insert(0, str(self.robot_width))
        
        # スリップアングル係数設定
        ttk.Label(left_frame, text="スリップ係数:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.slip_entry = ttk.Entry(left_frame, width=15)
        self.slip_entry.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # 設定ファイルからスリップ係数を読み込む
        slip_coefficient = 1.0  # デフォルト値
        if "slip_coefficient" in self.settings["parameters"]:
            slip_coefficient = self.settings["parameters"]["slip_coefficient"]
            self.slip_coefficient = slip_coefficient  # 内部変数も更新
        self.slip_entry.insert(0, str(slip_coefficient))
        
        # 機体幅更新ボタン
        self.width_update_btn = ttk.Button(left_frame, text="幅を更新", command=self.update_robot_width)
        self.width_update_btn.grid(row=6, column=0, columnspan=2, pady=5)
        
        # 結果表示ラベル
        ttk.Label(left_frame, text="--- 計算結果 ---").grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=10)
        
        # 列ヘダー
        ttk.Label(left_frame, text="初期探索値", font=(None, 8)).grid(row=7, column=1, sticky=tk.W, pady=1)
        ttk.Label(left_frame, text="最適化後", font=(None, 8)).grid(row=7, column=2, sticky=tk.W, pady=1)
        
        ttk.Label(left_frame, text="角加速度 [deg/s²]:").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.acc_label = ttk.Label(left_frame, text="---")
        self.acc_label.grid(row=8, column=1, sticky=tk.W, pady=2)
        self.opt_acc_label = ttk.Label(left_frame, text="---", foreground="blue")
        self.opt_acc_label.grid(row=8, column=2, sticky=tk.W, pady=2)
        
        ttk.Label(left_frame, text="最大角速度 [deg/s]:").grid(row=9, column=0, sticky=tk.W, pady=2)
        self.ang_vel_label = ttk.Label(left_frame, text="---")
        self.ang_vel_label.grid(row=9, column=1, sticky=tk.W, pady=2)
        self.opt_ang_vel_label = ttk.Label(left_frame, text="---", foreground="blue")
        self.opt_ang_vel_label.grid(row=9, column=2, sticky=tk.W, pady=2)
        
        ttk.Label(left_frame, text="後オフセット [mm]:").grid(row=10, column=0, sticky=tk.W, pady=2)
        self.rear_offset_label = ttk.Label(left_frame, text="---")
        self.rear_offset_label.grid(row=10, column=1, sticky=tk.W, pady=2)
        self.opt_rear_offset_label = ttk.Label(left_frame, text="---", foreground="blue")
        self.opt_rear_offset_label.grid(row=10, column=2, sticky=tk.W, pady=2)
        
        # Y差分情報
        ttk.Label(left_frame, text="Y誤差 [mm]:").grid(row=11, column=0, sticky=tk.W, pady=2)
        self.y_error_label = ttk.Label(left_frame, text="---")
        self.y_error_label.grid(row=11, column=1, sticky=tk.W, pady=2)
        self.opt_y_error_label = ttk.Label(left_frame, text="---", foreground="blue")
        self.opt_y_error_label.grid(row=11, column=2, sticky=tk.W, pady=2)
        
        ttk.Label(left_frame, text="所要時間 [ms]:").grid(row=12, column=0, sticky=tk.W, pady=2)
        self.time_label = ttk.Label(left_frame, text="---")
        self.time_label.grid(row=12, column=1, sticky=tk.W, pady=2)
        
        # 図の説明
        ttk.Label(left_frame, text="--- シミュレーション表示 ---").grid(row=13, column=0, columnspan=3, sticky=tk.W, pady=10)
        ttk.Label(left_frame, text="●: スタート地点").grid(row=14, column=0, sticky=tk.W)
        ttk.Label(left_frame, text="■: ターゲット地点").grid(row=14, column=1, sticky=tk.W)
        ttk.Label(left_frame, text="赤線: 軸道").grid(row=15, column=0, sticky=tk.W)
        ttk.Label(left_frame, text="青線: クロソイド入り").grid(row=15, column=1, sticky=tk.W)
        ttk.Label(left_frame, text="緑線: 定速円弧").grid(row=16, column=0, sticky=tk.W)
        ttk.Label(left_frame, text="黄線: クロソイド抜け").grid(row=16, column=1, sticky=tk.W)
        
        # 計算精度の設定
        ttk.Label(left_frame, text="計算精度:").grid(row=17, column=0, sticky=tk.W, pady=5)
        self.precision_frame = ttk.Frame(left_frame)
        self.precision_frame.grid(row=17, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(self.precision_frame, text="高速").pack(side=tk.LEFT, padx=2)
        self.precision_slider = ttk.Scale(self.precision_frame, from_=0, to=10, orient=tk.HORIZONTAL, length=100)
        self.precision_slider.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.precision_frame, text="高精度").pack(side=tk.LEFT, padx=2)
        
        # デフォルト値を設定
        if "precision" in self.settings["parameters"]:
            self.precision_slider.set(self.settings["parameters"]["precision"])
        else:
            # 新規の場合は最高精度をデフォルトに
            self.precision_slider.set(10)
        
        # プログレスバー
        ttk.Label(left_frame, text="計算進捗:").grid(row=18, column=0, sticky=tk.W, pady=5)
        self.progress = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=150, mode='determinate')
        self.progress.grid(row=18, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        # 計算実行ボタン
        self.calc_button = ttk.Button(left_frame, text="軌道計算", command=self.generate_trajectory)
        self.calc_button.grid(row=19, column=0, columnspan=3, pady=10)
        
        # 角加速度調整セクション
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=20, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(left_frame, text="角加速度調整", font=("Helvetica", 10, "bold")).grid(row=21, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # 角加速度入力フィールド
        ttk.Label(left_frame, text="角加速度 [円/秒²]:").grid(row=22, column=0, sticky=tk.W, pady=5)
        self.acc_entry = ttk.Entry(left_frame, width=10)
        self.acc_entry.grid(row=22, column=1, sticky=tk.W, pady=5)
        self.acc_entry.insert(0, "0.0")
        
        # 軸道再計算ボタン
        self.recalc_button = ttk.Button(left_frame, text="軸道再計算", command=self.recalculate_trajectory)
        self.recalc_button.grid(row=23, column=0, columnspan=2, pady=5)
        # 初期状態では無効化
        self.recalc_button.config(state=tk.DISABLED)
        
        # 自動調整セクション
        ttk.Label(left_frame, text="自動角加速度最適化").grid(row=24, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # 目標誤差とステップ幅
        ttk.Label(left_frame, text="目標誤差 [mm]:").grid(row=25, column=0, sticky=tk.W, pady=2)
        self.target_y_error_entry = ttk.Entry(left_frame, width=6)  # 変数名は互換性のために維持
        self.target_y_error_entry.grid(row=25, column=1, sticky=tk.W, pady=2)
        
        # 設定ファイルから目標誤差を読み込む
        target_error = 2.0  # デフォルト値
        if "target_error" in self.settings["parameters"]:
            target_error = self.settings["parameters"]["target_error"]
        elif "target_y_error" in self.settings["parameters"]:
            # 互換性のための移行処理
            target_error = self.settings["parameters"]["target_y_error"]
        self.target_y_error_entry.insert(0, str(target_error))
        
        ttk.Label(left_frame, text="ステップ幅 [円/秒²]:").grid(row=26, column=0, sticky=tk.W, pady=2)
        self.acc_step_entry = ttk.Entry(left_frame, width=6)
        self.acc_step_entry.grid(row=26, column=1, sticky=tk.W, pady=2)
        
        # 設定ファイルからステップ幅を読み込む
        acc_step = 100.0  # デフォルト値
        if "acc_step" in self.settings["parameters"]:
            acc_step = self.settings["parameters"]["acc_step"]
        self.acc_step_entry.insert(0, str(acc_step))
        
        # 自動調整ボタン
        self.auto_adjust_button = ttk.Button(left_frame, text="角加速度自動最適化", command=self.auto_adjust_acceleration)
        self.auto_adjust_button.grid(row=27, column=0, columnspan=2, pady=5)
        # 初期状態では無効化
        self.auto_adjust_button.config(state=tk.DISABLED)
        
        # ===== 右側フレームの内容 =====
        # Matplotlib図の作成
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        
    def on_size_changed(self, event):
        """機体サイズが変更されたときの処理"""
        # turn_paramsモジュールへの参照を一貫させる
        import turn_params as tp
        
        self.robot_size = self.size_combo.get()
        self.scale_factor = tp.ROBOT_SIZES[self.robot_size]["scale"]
        
        # 設定から機体幅を読み込み
        if self.robot_size in self.robot_settings:
            self.robot_width = self.robot_settings[self.robot_size]["width"]
        else:
            # 設定になければデフォルト値を使用
            self.robot_width = tp.ROBOT_SIZES[self.robot_size]["width"]
            # 設定に追加
            self.robot_settings[self.robot_size] = {"width": self.robot_width}
            self.save_settings()
        
        # 機体幅の表示を更新
        self.width_entry.delete(0, tk.END)
        self.width_entry.insert(0, str(self.robot_width))
        
        # サイズ変更に応じてグローバル変数相当の値を更新
        # ハーフサイズのy初期位置は45mm、クラシックサイズは90mm
        default_y = 45.0 if self.robot_size == "ハーフ" else 90.0
        self.ini_y = default_y
        
        # 動作タイプのコンボボックス更新
        turns_dict = tp.HALF_TURNS if self.robot_size == "ハーフ" else tp.CLASSIC_TURNS
        self.turn_combo['values'] = list(turns_dict.keys())
        
        # 前の値が利用可能か確認し、なければ最初の値を設定
        if self.turn_type not in turns_dict:
            self.turn_type = list(turns_dict.keys())[0]
            self.turn_combo.set(self.turn_type)
        
        # パラメータ再読み込み
        self.load_turn_params()
        
        # 入力欄の更新
        self.velocity_entry.delete(0, tk.END)
        self.velocity_entry.insert(0, str(self.translational_velocity))
        
        self.front_offset_entry.delete(0, tk.END)
        # 小回り90degの場合は表示用の値を使用
        self.front_offset_entry.insert(0, str(self.display_front_offset))
    
    def on_turn_changed(self, event):
        """動作タイプが変更されたときの処理"""
        self.turn_type = self.turn_combo.get()
        
        # パラメータ再読み込み
        self.load_turn_params()
        
        # 入力欄の更新
        self.velocity_entry.delete(0, tk.END)
        self.velocity_entry.insert(0, str(self.translational_velocity))
        
        self.front_offset_entry.delete(0, tk.END)
        # 小回り90degの場合は表示用の値を使用
        self.front_offset_entry.insert(0, str(self.display_front_offset))
    
    def generate_trajectory(self):
        """軸道計算と描画を実行"""
        try:
            # 入力値の取得
            self.translational_velocity = float(self.velocity_entry.get())
            
            # 前オフセット距離の処理
            # 入力された値は常に表示用の値
            self.display_front_offset = float(self.front_offset_entry.get())
            
            # スリップ係数を入力欄から取得
            self.slip_coefficient = float(self.slip_entry.get())
            
            # 小回り90degの場合、計算用には半区画分を加算
            if self.turn_type == "小回り90deg":
                half_cell = 45.0 if self.robot_size == "ハーフ" else 90.0
                self.front_offset_distance = self.display_front_offset + half_cell
            else:
                # それ以外のターンは表示用値と計算用値は同じ
                self.front_offset_distance = self.display_front_offset
            
            # 計算精度の取得
            precision_value = self.precision_slider.get()
            
            # 探索の細かさをスライダーの値から計算
            # 0 (高速・低精度) から 10 (高精度・低速) のスケール
            # 角加速度の探索刷み幅を調整 (10がデフォルトの最高精度、低精度では大きく)
            adjusted_acc_step = self.acc_step * (1 + (10 - precision_value) * 9.9)  # 10～1000 の範囲 (10がデフォルト)
            
            # 後オフセット距離の探索刷み幅を調整 (0.1がデフォルトの最高精度、低精度では大きく)
            adjusted_rear_offset_step = self.rear_offset_step * (1 + (10 - precision_value) * 29.9)  # 0.1～3.0 の範囲
            
            # 現在の設定を保存
            self.settings["current_selection"]["robot_size"] = self.robot_size
            self.settings["current_selection"]["turn_type"] = self.turn_type
            self.settings["parameters"]["velocity"] = self.translational_velocity
            self.settings["parameters"]["front_offset"] = self.display_front_offset
            self.settings["parameters"]["slip_coefficient"] = self.slip_coefficient
            self.settings["parameters"]["precision"] = precision_value
            self.save_settings()
            
            # 斜め入りのターンかどうかを判定
            is_diagonal = is_diagonal_turn(self.turn_type)
            
            # 計算ボタンを無効化
            self.calc_button.config(state=tk.DISABLED)
            
            # プログレスバーをリセット
            self.progress['value'] = 0
            self.root.update()
            
            if is_diagonal:
                # 斜め入りターンの場合は角加速度と後オフセットの両方を探索
                best_acc_deg = None
                best_rear_offset = None
                best_error = float('inf')
                
                # 角加速度と後オフセット距離の候補値を生成
                acc_candidates = np.arange(self.min_acc_deg, self.max_acc_deg + 1e-10, adjusted_acc_step)
                rear_offset_candidates = np.arange(self.min_rear_offset, self.max_rear_offset + 1e-10, adjusted_rear_offset_step)
                
                # 全探索数を計算
                total_iterations = len(acc_candidates) * len(rear_offset_candidates)
                current_iteration = 0
                
                # プログレスバーの最大値を設定
                self.progress['maximum'] = total_iterations
                
                for acc_deg in acc_candidates:
                    for rear_offset in rear_offset_candidates:
                        # 軸道をシミュレーション
                        x_end, y_end, phi_final = simulate_full_trajectory(
                            self.turning_angle_deg, self.translational_velocity,
                            acc_deg, self.front_offset_distance, rear_offset, self.dt,
                            self.slip_coefficient
                        )
                        
                        # 目標地点との誤差計算
                        error = math.sqrt((x_end - self.target_x)**2 + (y_end - self.target_y)**2)
                        
                        if error < best_error:
                            best_error = error
                            best_acc_deg = acc_deg
                            best_rear_offset = rear_offset
                        
                        # 進捗を更新
                        current_iteration += 1
                        self.progress['value'] = current_iteration
                        
                        # 10回に1回表示更新
                        if current_iteration % 10 == 0:
                            self.root.update()
            else:
                # 通常のターンの場合は後オフセットを前オフセットと同じにして角加速度のみ探索
                best_rear_offset = self.front_offset_distance
                best_acc_deg = None
                best_error = float('inf')
                
                # ターン名から小回り90度ターンかどうかを判定
                is_small_90deg = "小回り90deg" in self.turn_type
                
                # スリップ係数が1.0を超える場合の補正処理（小回り90度ターンのみ対象）
                slip_correction_enabled = is_small_90deg and self.slip_coefficient > 1.0
                
                # 角加速度の候補値を生成
                acc_candidates = np.arange(self.min_acc_deg, self.max_acc_deg + 1e-10, adjusted_acc_step)
                
                # 全探索数を計算（スリップ補正を行う場合は追加の探索が必要）
                additional_iterations = 0
                if slip_correction_enabled:
                    additional_iterations = int(len(acc_candidates) * 0.5)  # 追加探索の回数（候補の半分程度）
                
                total_iterations = len(acc_candidates) + additional_iterations
                current_iteration = 0
                
                # プログレスバーの最大値を設定
                self.progress['maximum'] = total_iterations
                
                # 第1段階: 通常の探索で最適角加速度を見つける
                for acc_deg in acc_candidates:
                    # 軸道をシミュレーション
                    x_end, y_end, phi_final = simulate_full_trajectory(
                        self.turning_angle_deg, self.translational_velocity,
                        acc_deg, self.front_offset_distance, best_rear_offset, self.dt,
                        self.slip_coefficient
                    )
                    
                    # 目標地点との誤差計算
                    error = math.sqrt((x_end - self.target_x)**2 + (y_end - self.target_y)**2)
                    
                    if error < best_error:
                        best_error = error
                        best_acc_deg = acc_deg
                        best_y_end = y_end  # Y座標を保存
                    
                    # 進捗を更新
                    current_iteration += 1
                    self.progress['value'] = current_iteration
                    
                    # 5回に1回表示更新
                    if current_iteration % 5 == 0:
                        self.root.update()
                
                # 第2段階: 小回り90度ターンかつスリップ係数>1.0の場合、Y座標に基づく補正を適用
                if slip_correction_enabled and best_acc_deg is not None:
                    # Y座標の目標値と現在の差
                    y_error = best_y_end - self.target_y
                    
                    # Y座標が目標より大きい場合（外側に膨らんでいる場合）
                    if y_error > 2.0:  # 2mm以上のずれがある場合のみ補正
                        print(f"スリップ補正開始: 現在のY誤差 = {y_error:.2f}mm")
                        
                        # 現在の最適角加速度から段階的に角加速度を上げていく
                        corrected_acc_deg = best_acc_deg
                        correction_step = adjusted_acc_step * 0.5  # より細かいステップで補正
                        best_corrected_error = best_error
                        correction_iterations = 0
                        
                        while correction_iterations < additional_iterations:
                            # 角加速度を少し上げる
                            corrected_acc_deg += correction_step
                            
                            # 上限を超えないようにする
                            if corrected_acc_deg > self.max_acc_deg:
                                break
                            
                            # 補正した角加速度でシミュレーション
                            x_end, y_end, phi_final = simulate_full_trajectory(
                                self.turning_angle_deg, self.translational_velocity,
                                corrected_acc_deg, self.front_offset_distance, best_rear_offset, self.dt,
                                self.slip_coefficient
                            )
                            
                            # 目標地点との誤差計算
                            error = math.sqrt((x_end - self.target_x)**2 + (y_end - self.target_y)**2)
                            
                            # Y座標の誤差が小さくなり、かつ全体の誤差も改善される場合に採用
                            new_y_error = abs(y_end - self.target_y)
                            if new_y_error < abs(y_error) and error <= best_corrected_error * 1.2:  # 多少の誤差増加は許容
                                best_acc_deg = corrected_acc_deg
                                best_corrected_error = error
                                y_error = y_end - self.target_y
                                print(f"スリップ補正: 角加速度={best_acc_deg:.2f}, Y誤差={y_error:.2f}mm")
                                
                                # 十分に改善された場合は終了
                                if abs(y_error) < 1.0:  # 1mm以下になったら十分
                                    break
                            
                            # 進捗を更新
                            current_iteration += 1
                            self.progress['value'] = current_iteration
                            correction_iterations += 1
                            
                            # 表示更新
                            if correction_iterations % 2 == 0:
                                self.root.update()
                        
                        print(f"スリップ補正完了: 最終角加速度 = {best_acc_deg:.2f}, Y誤差 = {y_error:.2f}mm")
            
            # 計算ボタンを再有効化
            self.calc_button.config(state=tk.NORMAL)
            
            if best_acc_deg is None:
                tkinter.messagebox.showerror("エラー", "探索結果が見つかりませんでした。")
                return
            
            # 最適パラメータで最終シミュレーション
            x_end, y_end, phi_final = simulate_full_trajectory(
                self.turning_angle_deg, self.translational_velocity,
                best_acc_deg, self.front_offset_distance, best_rear_offset, self.dt,
                self.slip_coefficient
            )
            max_w_rad, max_w_deg = compute_max_angular_velocity(self.turning_angle_deg, best_acc_deg)
            
            # 結果の表示
            self.acc_label.config(text=f"{best_acc_deg:.2f}")
            self.ang_vel_label.config(text=f"{max_w_deg:.2f}")
            self.rear_offset_label.config(text=f"{best_rear_offset:.2f}")
            
            # 所要時間の計算
            theta1 = self.turning_angle_deg/3 * math.pi/180
            alpha_mag = best_acc_deg * math.pi/180
            t1 = math.sqrt(2 * theta1 / alpha_mag)
            w1 = -alpha_mag * t1
            t2 = theta1 / abs(w1)
            t3 = t1
            t_straight1 = self.front_offset_distance / self.translational_velocity
            t_straight2 = best_rear_offset / self.translational_velocity
            total_time = (t1 + t2 + t3 + t_straight1 + t_straight2) * 1000  # ms単位で表示
            
            # プログレスバーを最大値にセットして計算完了を示す
            self.progress['value'] = self.progress['maximum']
            self.time_label.config(text=f"{total_time:.2f}")
            
            # 軸道の描画（実際の終点座標を取得）
            plot_x_end, plot_y_end = self.plot_trajectory(best_acc_deg, best_rear_offset)
            
            # 計算終点とグラフ描画の終点の差を確認
            calc_diff = math.sqrt((x_end - plot_x_end)**2 + (y_end - plot_y_end)**2)
            if calc_diff > 1.0:  # 1mm以上の差があればログ表示
                print(f"\n警告: 計算終点とグラフ描画終点に不一致があります: {calc_diff:.2f}mm")
                print(f"  計算終点: ({x_end:.2f}, {y_end:.2f}), グラフ描画終点: ({plot_x_end:.2f}, {plot_y_end:.2f})")
            
            # 小回り90degターンの場合、目標座標を調整
            adjusted_target_x = self.target_x
            adjusted_target_y = self.target_y
            
            # 小回り90degターンの場合、目標座標にY方向に半区画分の調整が必要
            if "90deg" in self.turn_type:
                half_cell = 45.0 if self.robot_size == "ハーフ" else 90.0
                adjusted_target_y += half_cell
            
            # 調整後の目標座標とグラフ描画の終点座標から差分を計算
            x_diff = plot_x_end - adjusted_target_x
            y_diff = plot_y_end - adjusted_target_y
            error = math.sqrt(x_diff**2 + y_diff**2)
            
            # 座標差分情報をグラフに表示
            position_info = f"目標差: X={x_diff:.2f}mm, Y={y_diff:.2f}mm, 距離={error:.2f}mm"
            self.ax.text(0.05, 0.15, position_info, transform=self.ax.transAxes, fontsize=10, color='purple')
            
            # スリップ係数情報を追加
            slip_info = f"スリップ係数: {self.slip_coefficient:.2f}"
            self.ax.text(0.05, 0.10, slip_info, transform=self.ax.transAxes, fontsize=10, color='blue')
            
            # 計算結果のコンソール表示
            print("\n===== 計算結果 =====")
            print(f"ターンタイプ: {self.robot_size} {self.turn_type}")
            print(f"スリップ係数: {self.slip_coefficient:.2f}")
            print(f"計算検討数: {current_iteration}")
            print(f"角加速度: {best_acc_deg:.2f}円/秒²")
            print(f"最高角速度: {max_w_deg:.2f}円/秒")
            print(f"後オフセット: {best_rear_offset:.2f}mm")
            # 目標座標の表示を調整前と調整後の両方を表示
            if adjusted_target_y != self.target_y:
                print(f"元の目標座標: X={self.target_x:.2f}mm, Y={self.target_y:.2f}mm")
                print(f"調整後目標座標: X={adjusted_target_x:.2f}mm, Y={adjusted_target_y:.2f}mm (半区画調整済)")
            else:
                print(f"目標座標: X={adjusted_target_x:.2f}mm, Y={adjusted_target_y:.2f}mm")
                
            print(f"到達座標: X={plot_x_end:.2f}mm, Y={plot_y_end:.2f}mm")
            print(f"終了位置差分: X={x_diff:.2f}mm, Y={y_diff:.2f}mm, 距離={error:.2f}mm")
            
            # 角加速度調整用の入力欄を更新
            self.acc_entry.delete(0, tk.END)
            self.acc_entry.insert(0, f"{best_acc_deg:.2f}")
            
            # 再計算ボタンと自動調整ボタンを有効化
            self.recalc_button.config(state=tk.NORMAL)
            self.auto_adjust_button.config(state=tk.NORMAL)
            
            # 計算結果を保存して手動調整時に使用する
            self.current_best_acc_deg = best_acc_deg
            self.current_best_rear_offset = best_rear_offset
            
            # 実際の終点座標と調整済み目標座標を保存
            self.current_plot_end = (plot_x_end, plot_y_end)
            self.current_adjusted_target = (adjusted_target_x, adjusted_target_y)
            
        except ValueError as e:
            tkinter.messagebox.showerror("入力エラー", f"数値の入力に問題があります: {e}")
        except Exception as e:
            tkinter.messagebox.showerror("エラー", f"計算中にエラーが発生しました: {e}")
    
    def recalculate_trajectory(self):
        """角加速度を手動で調整して軸道を再計算する"""
        try:
            # 再計算ボタンを無効化
            self.recalc_button.config(state=tk.DISABLED)
            
            # 入力値の取得
            manual_acc_deg = float(self.acc_entry.get())
            
            # 始時刻を記録
            start_time = time.time()
            
            # 計算に使用するパラメータを手動設定値と前回の値を使用
            best_acc_deg = manual_acc_deg
            best_rear_offset = self.current_best_rear_offset
            
            # 最大角速度を計算
            _, max_w_deg = compute_max_angular_velocity(self.turning_angle_deg, best_acc_deg)
            
            # 軸道をシミュレーション
            x_end, y_end, phi_final = simulate_full_trajectory(
                self.turning_angle_deg, self.translational_velocity,
                best_acc_deg, self.front_offset_distance, best_rear_offset, self.dt,
                self.slip_coefficient
            )
            
            # 軸道の描画（実際の終点座標を取得）
            plot_x_end, plot_y_end = self.plot_trajectory(best_acc_deg, best_rear_offset)
            
            # 計算終点とグラフ描画の終点の差を確認
            calc_diff = math.sqrt((x_end - plot_x_end)**2 + (y_end - plot_y_end)**2)
            if calc_diff > 1.0:  # 1mm以上の差があればログ表示
                print(f"\n警告: 計算終点とグラフ描画終点に不一致があります: {calc_diff:.2f}mm")
                print(f"  計算終点: ({x_end:.2f}, {y_end:.2f}), グラフ描画終点: ({plot_x_end:.2f}, {plot_y_end:.2f})")
            
            # 小回り90degターンの場合、目標座標を調整
            adjusted_target_x = self.target_x
            adjusted_target_y = self.target_y
            
            # 小回り90degターンの場合、目標座標にY方向に半区画分の調整が必要
            if "90deg" in self.turn_type:
                half_cell = 45.0 if self.robot_size == "ハーフ" else 90.0
                adjusted_target_y += half_cell
            
            # 調整後の目標座標とグラフ描画の終点座標から差分を計算
            x_diff = plot_x_end - adjusted_target_x
            y_diff = plot_y_end - adjusted_target_y
            error = math.sqrt(x_diff**2 + y_diff**2)
            
            # 座標差分情報をグラフに表示
            position_info = f"目標差: X={x_diff:.2f}mm, Y={y_diff:.2f}mm, 距離={error:.2f}mm"
            self.ax.text(0.05, 0.15, position_info, transform=self.ax.transAxes, fontsize=10, color='purple')
            
            # スリップ係数情報を追加
            slip_info = f"スリップ係数: {self.slip_coefficient:.2f}"
            self.ax.text(0.05, 0.10, slip_info, transform=self.ax.transAxes, fontsize=10, color='blue')
            
            # 計算終了時間を記録
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # ミリ秒に変換
            
            # 再計算結果のコンソール表示
            print("\n===== 手動調整再計算結果 =====")
            print(f"ターンタイプ: {self.robot_size} {self.turn_type}")
            print(f"スリップ係数: {self.slip_coefficient:.2f}")
            print(f"角加速度(手動設定): {best_acc_deg:.2f}円/秒²")
            print(f"最高角速度: {max_w_deg:.2f}円/秒")
            print(f"後オフセット: {best_rear_offset:.2f}mm")
            
            # 目標座標の表示を調整前と調整後の両方を表示
            if adjusted_target_y != self.target_y:
                print(f"元の目標座標: X={self.target_x:.2f}mm, Y={self.target_y:.2f}mm")
                print(f"調整後目標座標: X={adjusted_target_x:.2f}mm, Y={adjusted_target_y:.2f}mm (半区画調整済)")
            else:
                print(f"目標座標: X={adjusted_target_x:.2f}mm, Y={adjusted_target_y:.2f}mm")
                
            print(f"到達座標: X={plot_x_end:.2f}mm, Y={plot_y_end:.2f}mm")
            print(f"終了位置差分: X={x_diff:.2f}mm, Y={y_diff:.2f}mm, 距離={error:.2f}mm")
            print(f"計算時間: {total_time:.2f}ms")
            
            # 45度ターンの場合、対角誤差も計算して表示
            if "45deg" in self.turn_type:
                diagonal_offset = 90.0 if self.robot_size == "ハーフ" else 180.0
                diagonal_error = (plot_x_end - plot_y_end + diagonal_offset) / math.sqrt(2)
                print(f"対角誤差: {diagonal_error:.2f}mm (y=x+{diagonal_offset:.1f}直線より)")
                
                # グラフにも対角誤差を表示
                diagonal_info = f"対角誤差 (y=x+{diagonal_offset:.1f}): {diagonal_error:.2f}mm"
                self.ax.text(0.05, 0.05, diagonal_info, transform=self.ax.transAxes, fontsize=10, color='red')
            
            # 再計算ボタンを有効化
            self.recalc_button.config(state=tk.NORMAL)
            self.auto_adjust_button.config(state=tk.NORMAL)
            
        except ValueError as e:
            tkinter.messagebox.showerror("入力エラー", f"角加速度値の入力に問題があります: {e}")
            # 元の値に戻す
            self.acc_entry.delete(0, tk.END)
            self.acc_entry.insert(0, str(self.current_best_acc_deg))
            # 再計算ボタンを有効化
            self.recalc_button.config(state=tk.NORMAL)
            self.auto_adjust_button.config(state=tk.NORMAL)
        except Exception as e:
            tkinter.messagebox.showerror("エラー", f"再計算中にエラーが発生しました: {e}")
            # 再計算ボタンを有効化
            self.recalc_button.config(state=tk.NORMAL)
            self.auto_adjust_button.config(state=tk.NORMAL)
            
    def auto_adjust_acceleration(self):
        """角加速度を自動調整してターンタイプに応じた誤差を目標値以下にする
        90degターンではY方向の誤差を、180degターンではX方向の誤差を最適化する"""
        try:
            # 自動調整ボタンを無効化
            self.auto_adjust_button.config(state=tk.DISABLED)
            self.recalc_button.config(state=tk.DISABLED)
            
            # パラメータ取得
            target_error = float(self.target_y_error_entry.get())  # 目標誤差
            acc_step = float(self.acc_step_entry.get())  # 角加速度のステップ幅
            
            # 現在の角加速度を取得
            current_acc_deg = self.current_best_acc_deg
            
            # ターンタイプに応じた最適化対象軸を決定
            is_180deg_turn = "180deg" in self.turn_type
            is_45deg_turn = "45deg" in self.turn_type
            
            if is_180deg_turn:
                target_axis = "X"
            elif is_45deg_turn:
                target_axis = "Diagonal"  # 毎度軸を指定
            else:
                target_axis = "Y"  # 90度ターンなどの場合
            
            # 最適化開始メッセージ
            print("\n===== 角加速度自動最適化開始 =====")
            print(f"ターンタイプ: {self.robot_size} {self.turn_type}")
            print(f"スリップ係数: {self.slip_coefficient:.2f}")
            print(f"最適化対象軸: {target_axis} 方向")
            print(f"目標誤差: {target_error:.2f}mm")
            print(f"初期角加速度: {current_acc_deg:.2f}円/秒²")
            print(f"後オフセット: {self.current_best_rear_offset:.2f}mm")
            
            # プログレスバーをリセット
            self.progress['value'] = 0
            self.root.update()
            
            # 最初の軸道をシミュレーション
            x_end, y_end, phi_final = simulate_full_trajectory(
                self.turning_angle_deg, self.translational_velocity,
                current_acc_deg, self.front_offset_distance, self.current_best_rear_offset, self.dt,
                self.slip_coefficient
            )
            
            # 軌道の描画（実際の終点座標を取得）
            plot_x_end, plot_y_end = self.plot_trajectory(current_acc_deg, self.current_best_rear_offset)
            
            # 調整済み目標座標との差分計算
            adjusted_target_x, adjusted_target_y = self.current_adjusted_target
            current_x_diff = plot_x_end - adjusted_target_x
            current_y_diff = plot_y_end - adjusted_target_y
            
            # ターンタイプに応じて最適化する誤差を選択
            if is_180deg_turn:
                # 180degターンの場合はX方向の誤差を最適化
                current_error = current_x_diff
                print(f"\n初期状態: X誤差 = {current_x_diff:.2f}mm")
            elif is_45deg_turn:
                # 45degターンの場合は終点からy=x+offset直線までの距離を誤差として使用
                # ロボットサイズに応じてオフセットを設定
                if self.robot_size == "ハーフ":
                    diagonal_offset = 90.0  # ハーフサイズの場合はy=x+90直線
                else:  # クラシックサイズ
                    diagonal_offset = 180.0  # クラシックサイズの場合はy=x+180直線
                
                # y=x+offset直線からの距離を計算
                # 式: (x - (y-offset))/sqrt(2) = (x-y+offset)/sqrt(2)
                current_error = (plot_x_end - plot_y_end + diagonal_offset) / math.sqrt(2)
                print(f"\n初期状態: 対角誤差 = {current_error:.2f}mm (終点座標: X={plot_x_end:.2f}, Y={plot_y_end:.2f}, 対角オフセット: {diagonal_offset:.1f}mm)")
            else:
                # 90degターンの場合はY方向の誤差を最適化
                current_error = current_y_diff
                print(f"\n初期状態: Y誤差 = {current_y_diff:.2f}mm")
            
            # 誤差が目標以下ならそのまま終了
            if abs(current_error) <= target_error:
                print(f"現在の{target_axis}誤差がすでに目標以下です。最適化は不要です。")
                # ボタンを再有効化
                self.auto_adjust_button.config(state=tk.NORMAL)
                self.recalc_button.config(state=tk.NORMAL)
                return
            
            # 誤差の符号を確認
            # 90degターン: Y誤差が正なら外側に膨らんでいるので角加速度を上げる
            # 180degターン: X誤差が正なら外側に膨らんでいるので角加速度を上げる
            # 45degターン: 対角誤差が正なら角加速度を下げ、負なら角加速度を上げる
            if is_45deg_turn:
                # 45度ターンの場合は符号が逆
                increase_acc = current_error < 0
            else:
                # 90度と180度ターンの場合
                increase_acc = current_error > 0
            
            if increase_acc:
                print("外側に膨らんでいるため、角加速度を上げて調整します")
            else:
                print("内側に膨らんでいるため、角加速度を下げて調整します")
                # ステップ幅を負にする
                acc_step = -acc_step
            
            # プログレスバーの最大値を設定（最大100ステップまで）
            max_iterations = 100
            self.progress['maximum'] = max_iterations
            
            # 実際の調整ループ
            best_acc_deg = current_acc_deg
            best_error = abs(current_error)  # 選択した軸の誤差を追跡
            iteration = 0
            improved = False
            
            # ステップ幅の調整回数をカウントする変数
            step_adjustment_count = 0
            max_step_adjustments = 3  # 最大調整回数
            
            while iteration < max_iterations:
                # 角加速度を調整
                new_acc_deg = current_acc_deg + acc_step * (iteration + 1)
                
                # 角加速度の範囲を超えたら終了
                if new_acc_deg < self.min_acc_deg or new_acc_deg > self.max_acc_deg:
                    print(f"\n角加速度が範囲外になりました: {new_acc_deg:.2f}円/秒²")
                    break
                
                # 軸道をシミュレーション
                x_end, y_end, phi_final = simulate_full_trajectory(
                    self.turning_angle_deg, self.translational_velocity,
                    new_acc_deg, self.front_offset_distance, self.current_best_rear_offset, self.dt,
                    self.slip_coefficient
                )
                
                # 軌道の描画（実際の終点座標を取得）
                plot_x_end, plot_y_end = self.plot_trajectory(new_acc_deg, self.current_best_rear_offset)
                
                # 調整済み目標座標との差分計算
                current_x_diff = plot_x_end - adjusted_target_x
                current_y_diff = plot_y_end - adjusted_target_y
                
                # ターンタイプに応じて最適化する誤差を選択
                if is_180deg_turn:
                    # 180degターンの場合はX方向の誤差を最適化
                    current_error = current_x_diff
                    current_error_abs = abs(current_error)
                elif is_45deg_turn:
                    # 45degターンの場合は終点からy=x+offset直線までの距離を誤差として使用
                    # ロボットサイズに応じてオフセットを設定
                    if self.robot_size == "ハーフ":
                        diagonal_offset = 90.0  # ハーフサイズの場合はy=x+90直線
                    else:  # クラシックサイズ
                        diagonal_offset = 180.0  # クラシックサイズの場合はy=x+180直線
                    
                    # y=x+offset直線からの距離を計算
                    # 式: (x - (y-offset))/sqrt(2) = (x-y+offset)/sqrt(2)
                    current_error = (plot_x_end - plot_y_end + diagonal_offset) / math.sqrt(2)
                    current_error_abs = abs(current_error)
                else:
                    # 90degターンの場合はY方向の誤差を最適化
                    current_error = current_y_diff
                    current_error_abs = abs(current_error)
                
                # 結果を表示
                print(f"ステップ {iteration+1}: 角加速度={new_acc_deg:.2f}, {target_axis}誤差={current_error:.2f}mm")
                
                # 誤差が目標以下になったか確認
                if current_error_abs <= target_error:
                    print(f"\n目標誤差に到達しました: {current_error_abs:.2f}mm <= {target_error:.2f}mm")
                    best_acc_deg = new_acc_deg
                    improved = True
                    break
                
                # 前回より改善したか
                if current_error_abs < best_error:
                    best_error = current_error_abs
                    best_acc_deg = new_acc_deg
                    improved = True
                    
                # 逆に差分が大きくなったか（負と正の間を通過した可能性）
                # 45度ターンでは誤差の符号と角加速度の関係が逆
                if is_45deg_turn:
                    # 45度ターンは誤差の符号が逆になるので条件も逆にする
                    sign_reversal = (current_error < 0 and not increase_acc) or (current_error > 0 and increase_acc)
                else:
                    # 90度または180度ターン
                    sign_reversal = (current_error > 0 and not increase_acc) or (current_error < 0 and increase_acc)
                
                if sign_reversal:
                    # ステップ幅の調整回数をカウントアップ
                    step_adjustment_count += 1
                    
                    # 調整回数が上限を超えたか確認
                    if step_adjustment_count >= max_step_adjustments:
                        print(f"\n{target_axis}誤差の符号が{step_adjustment_count}回変化しました。")
                        print(f"目標誤差 ({target_error:.2f}mm) に対してステップ幅 ({acc_step:.2f}) が大きすぎるか、")
                        print(f"条件を満たすパラメータが見つかりません。最も近い値を採用します。")
                        break  # 探索終了
                    
                    print(f"\nY誤差の符号が変化しました。最適値を通過した可能性があります。")
                    # ステップ幅を半分にして逆方向に探索
                    acc_step = -acc_step / 2
                    current_acc_deg = best_acc_deg  # 現在までの最適値から再探索
                    iteration = 0  # カウンターをリセット
                    continue
                
                # 進捗を更新
                iteration += 1
                self.progress['value'] = iteration
                self.root.update()
            
            # 最終結果を表示
            if improved:
                print(f"\n===== 角加速度最適化結果 =====")
                print(f"最適角加速度: {best_acc_deg:.2f}円/秒² (初期値: {self.current_best_acc_deg:.2f}円/秒²)")
                print(f"最小{target_axis}誤差: {best_error:.2f}mm (目標: {target_error:.2f}mm)")
                
                # ターン角度を取得
                # ターンタイプの文字列から角度を抽出
                if "45deg" in self.turn_type:
                    turning_angle_deg = 45.0
                elif "90deg" in self.turn_type:
                    turning_angle_deg = 90.0
                elif "135deg" in self.turn_type:
                    turning_angle_deg = 135.0
                elif "180deg" in self.turn_type:
                    turning_angle_deg = 180.0
                else:
                    # デフォルト値
                    turning_angle_deg = 90.0
                
                # 最高角速度と後オフセットを計算
                # 関数はタプル(max_w_rad, max_w_deg)を返すので、度数法の値（第２要素）だけを取得
                _, max_w_deg = compute_max_angular_velocity(turning_angle_deg, best_acc_deg)
                
                # 最適化された軌道の終点座標を取得
                # plot_trajectoryメソッドを利用して現在の最適化された軌道を描画し、終点座標を取得
                best_plot_x_end, best_plot_y_end = self.plot_trajectory(best_acc_deg, self.current_best_rear_offset)
                
                # X方向の誤差を計算
                best_x_diff = best_plot_x_end - adjusted_target_x
                print(f"X方向の誤差: {best_x_diff:.2f}mm")
                
                # ターンタイプに応じて後ろオフセット距離を調整
                if "90deg" in self.turn_type:
                    # 後オフセット値を現在の計算で使用している値から取得
                    original_rear_offset = self.current_best_rear_offset
                    # X方向の誤差を相殺するように後ろオフセットを調整
                    # 誤差とは逆方向に調整する必要があるので減算を使用
                    adjusted_rear_offset = original_rear_offset - best_x_diff
                    best_rear_offset = adjusted_rear_offset
                    print(f"後ろオフセット距離を調整: {original_rear_offset:.2f}mm → {best_rear_offset:.2f}mm (X誤差: {best_x_diff:.2f}mm)")
                    
                    # 調整後の後ろオフセットで軌道を再計算して表示
                    final_plot_x_end, final_plot_y_end = self.plot_trajectory(best_acc_deg, best_rear_offset)
                    print(f"調整後の終点座標: X={final_plot_x_end:.2f}mm, Y={final_plot_y_end:.2f}mm")
                    print(f"調整後のX誤差: {final_plot_x_end - adjusted_target_x:.2f}mm")
                    print(f"調整後のY誤差: {final_plot_y_end - adjusted_target_y:.2f}mm")
                    
                    # 内部変数を更新して次回の計算でも使用されるようにする
                    self.current_best_rear_offset = best_rear_offset
                elif "45deg" in self.turn_type:
                    # 後オフセット値を現在の計算で使用している値から取得
                    original_rear_offset = self.current_best_rear_offset
                    
                    # ロボットサイズに応じて角加速度最適化用の対角オフセットを設定
                    if self.robot_size == "ハーフ":
                        forward_diagonal_offset = 90.0  # y=x+90直線
                        rear_diagonal_offset = 180.0    # y=-x+180直線
                    else:  # クラシックサイズ
                        forward_diagonal_offset = 180.0  # y=x+180直線
                        rear_diagonal_offset = 360.0     # y=-x+360直線
                    
                    # 角加速度最適化後の位置から、y=x+offset直線からの対角誤差を計算
                    # 対角誤差は (x-y+offset)/sqrt(2) で計算される
                    forward_diagonal_error = (best_plot_x_end - best_plot_y_end + forward_diagonal_offset) / math.sqrt(2)
                    print(f"対角誤差 (y=x+{forward_diagonal_offset}): {forward_diagonal_error:.2f}mm")
                    
                    # 理想的な終点位置は前方対角直線と後方対角直線の交点
                    # y=x+forward_offset と y=-x+rear_offset の交点は (rear_offset-forward_offset)/2, (rear_offset+forward_offset)/2)
                    ideal_x = (rear_diagonal_offset - forward_diagonal_offset) / 2
                    ideal_y = (rear_diagonal_offset + forward_diagonal_offset) / 2
                    print(f"理想的な終点位置: X={ideal_x:.2f}mm, Y={ideal_y:.2f}mm (対角直線の交点)")
                    
                    # 現在の終点から理想位置までの距離を計算
                    distance_to_ideal = math.sqrt((best_plot_x_end - ideal_x)**2 + (best_plot_y_end - ideal_y)**2)
                    print(f"理想点までの距離: {distance_to_ideal:.2f}mm")
                    
                    # 後方対角直線 (y=-x+rear_offset) からの対角誤差を計算
                    # 式: (x+y-rear_offset)/sqrt(2)
                    rear_diagonal_error = (best_plot_x_end + best_plot_y_end - rear_diagonal_offset) / math.sqrt(2)
                    print(f"後方対角誤差 (y=-x+{rear_diagonal_offset}): {rear_diagonal_error:.2f}mm")
                    
                    # 後方対角誤差を相殖するように後オフセットを調整
                    # 誤差とは逆方向に調整する必要があるので減算を使用
                    adjusted_rear_offset = original_rear_offset - rear_diagonal_error
                    best_rear_offset = adjusted_rear_offset
                    print(f"後ろオフセット距離を調整: {original_rear_offset:.2f}mm → {best_rear_offset:.2f}mm (後方対角誤差: {rear_diagonal_error:.2f}mm)")
                    
                    # 調整後の後ろオフセットで軸道を再計算して表示
                    final_plot_x_end, final_plot_y_end = self.plot_trajectory(best_acc_deg, best_rear_offset)
                    
                    # 調整後の対角誤差を計算 (前方と後方の両方)
                    final_forward_error = (final_plot_x_end - final_plot_y_end + forward_diagonal_offset) / math.sqrt(2)
                    final_rear_error = (final_plot_x_end + final_plot_y_end - rear_diagonal_offset) / math.sqrt(2)
                    
                    print(f"調整後の終点座標: X={final_plot_x_end:.2f}mm, Y={final_plot_y_end:.2f}mm")
                    print(f"調整後の前方対角誤差: {final_forward_error:.2f}mm (y=x+{forward_diagonal_offset:.1f})")
                    print(f"調整後の後方対角誤差: {final_rear_error:.2f}mm (y=-x+{rear_diagonal_offset:.1f})")
                    
                    # 内部変数を更新して次回の計算でも使用されるようにする
                    self.current_best_rear_offset = best_rear_offset
                else:
                    # 対応していないターンタイプは調整しない
                    best_rear_offset = self.current_best_rear_offset
                
                # GUIに最適化後の値を安全に表示
                try:
                    self.opt_acc_label.config(text=f"{best_acc_deg:.2f}")
                    self.opt_ang_vel_label.config(text=f"{max_w_deg:.2f}")
                    self.opt_rear_offset_label.config(text=f"{best_rear_offset:.2f}")
                    self.opt_y_error_label.config(text=f"{best_error:.2f}")
                    print(f"\n結果表示成功: 角加速度={best_acc_deg:.2f}, 最大角速度={max_w_deg:.2f}, 後オフセット={best_rear_offset:.2f}, {target_axis}誤差={best_error:.2f}")
                except Exception as format_err:
                    print(f"\n値の表示中にエラーが発生しました: {format_err}")
                    # エラー発生時も安全に表示
                    self.opt_acc_label.config(text=str(best_acc_deg))
                    self.opt_ang_vel_label.config(text=str(max_w_deg))
                    self.opt_rear_offset_label.config(text=str(best_rear_offset))
                    self.opt_y_error_label.config(text=str(best_error))
                
                # 再計算で使用する最適化後の値を保存
                self.optimized_values = {
                    "acc_deg": best_acc_deg,
                    "max_w_deg": max_w_deg,
                    "rear_offset": best_rear_offset,
                    "error": best_error,
                    "target_error": target_error,
                    "initial_acc_deg": self.current_best_acc_deg
                }
                
                # 最適角加速度を入力欄に設定
                self.acc_entry.delete(0, tk.END)
                self.acc_entry.insert(0, f"{best_acc_deg:.2f}")
                
                # 最終的なシミュレーションを実行
                self.recalculate_trajectory()
                
                # 自動調整ボタンを再有効化
                self.auto_adjust_button.config(state=tk.NORMAL)
            else:
                print(f"\n最適化に失敗しました。最良の角加速度: {best_acc_deg:.2f}円/秒², {target_axis}誤差: {best_error:.2f}mm")
                # ボタンを再有効化
                self.auto_adjust_button.config(state=tk.NORMAL)
                self.recalc_button.config(state=tk.NORMAL)
            
        except ValueError as e:
            tkinter.messagebox.showerror("入力エラー", f"自動調整用パラメータの入力に問題があります: {e}")
            # ボタンを再有効化
            self.auto_adjust_button.config(state=tk.NORMAL)
            self.recalc_button.config(state=tk.NORMAL)
        except Exception as e:
            tkinter.messagebox.showerror("エラー", f"自動調整中にエラーが発生しました: {e}")
            # ボタンを再有効化
            self.auto_adjust_button.config(state=tk.NORMAL)
            self.recalc_button.config(state=tk.NORMAL)
    
    def plot_trajectory(self, best_acc_deg, best_rear_offset):
        """軸道の描画（元のDrawTrace関数のロジックを使用）
        実際の終点座標を返すように改善"""
        # max_w_degの計算
        _, max_w_deg = compute_max_angular_velocity(self.turning_angle_deg, best_acc_deg)

        # 必要なパラメータを設定
        self.fin_angle = self.turning_angle_deg  # ターン角度を設定
        
        # 元のグローバル変数をローカル変数として定義
        pri_offset = self.front_offset_distance
        post_offset = best_rear_offset
        _, max_w_deg = compute_max_angular_velocity(self.turning_angle_deg, best_acc_deg)
        low_AngVel = max_w_deg
        Low_AngAcl = best_acc_deg
        Width = self.robot_width
        speed = self.translational_velocity
        speed_ms = speed / 1000.0
        
        # スリップアングル係数をUIから設定した値に基づいて計算
        # スリップ係数が1を超える場合は、その超過分をスリップアングル係数として使用
        K_slip_angle = max(0.0, (self.slip_coefficient - 1.0) * 0.05)  # スリップを設定値に合わせて調整
        
        # 初期化
        self.ax.clear()
        
        # 格子の描画
        self.draw_grid()
        
        # 軌道計算と描画
        # フェーズの時間計算
        alpha_mag = best_acc_deg * math.pi / 180.0
        theta1 = self.turning_angle_deg * math.pi / 180.0 / 3.0
        t1 = math.sqrt(2 * theta1 / alpha_mag)
        w1 = -alpha_mag * t1
        t2 = theta1 / abs(w1)
        t3 = t1
        t_pre = self.front_offset_distance / self.translational_velocity
        t_post = best_rear_offset / self.translational_velocity
        t_total = t_pre + t1 + t2 + t3 + t_post
        
        # 時間刻みによる軌道生成
        t_list = np.linspace(0, t_total, 200)
        x_list = []
        y_list = []
        phi_list = []
        
        for t in t_list:
            # フェーズ1: 前オフセット直進
            if t <= t_pre:
                x = t * self.translational_velocity
                y = 0
                phi = math.pi/2
            # フェーズ2: クロソイド入り
            elif t <= t_pre + t1:
                t_rel = t - t_pre
                x_pre = self.front_offset_distance
                y_pre = 0
                phi0 = math.pi/2
                
                phi = phi0 - 0.5 * alpha_mag * t_rel * t_rel
                v_x = self.translational_velocity * math.cos(phi)
                v_y = self.translational_velocity * math.sin(phi)
                
                # シンプルな積分（精度は低いが十分）
                dt = t1 / 100
                x_rel = 0
                y_rel = 0
                for i in range(int(t_rel / dt)):
                    t_i = i * dt
                    phi_i = phi0 - 0.5 * alpha_mag * t_i * t_i
                    x_rel += self.translational_velocity * math.cos(phi_i) * dt
                    y_rel += self.translational_velocity * math.sin(phi_i) * dt
                
                x = x_pre + x_rel
                y = y_pre + y_rel
            # フェーズ3: 円弧
            elif t <= t_pre + t1 + t2:
                t_rel = t - (t_pre + t1)
                
                # フェーズ2までの計算
                x_phase2, y_phase2, phi1 = self.simulate_phase(self.translational_velocity, alpha_mag, t1)
        
        # --- 元のDrawTrace関数のロジックを再現 ---
        # 前オフセット区間
        fin_x = self.ini_x + pri_offset * np.sin(np.deg2rad(self.ini_angle))
        fin_y = self.ini_y + pri_offset * np.cos(np.deg2rad(self.ini_angle))
        self.ax.plot(np.array([self.ini_x, fin_x]), np.array([self.ini_y, fin_y]), color="blue", linestyle="solid")
        self.ax.plot(np.array([self.ini_x + (Width/2) * np.cos(np.deg2rad(self.ini_angle)), fin_x + (Width/2) * np.cos(np.deg2rad(self.ini_angle))]),
                np.array([self.ini_y - (Width/2) * np.sin(np.deg2rad(self.ini_angle)), fin_y - (Width/2) * np.sin(np.deg2rad(self.ini_angle))]),
                color="#888888", linestyle="solid")
        self.ax.plot(np.array([self.ini_x - (Width/2) * np.cos(np.deg2rad(self.ini_angle)), fin_x - (Width/2) * np.cos(np.deg2rad(self.ini_angle))]),
                np.array([self.ini_y + (Width/2) * np.sin(np.deg2rad(self.ini_angle)), fin_y + (Width/2) * np.sin(np.deg2rad(self.ini_angle))]),
                color="#888888", linestyle="solid")
        
        # 角速度加速区間
        now_Angle = self.ini_angle
        s_now_Angle = self.ini_angle
        now_AngVel = 0
        while now_AngVel < low_AngVel:
            befor_x = fin_x; befor_y = fin_y
            now_AngVel += Low_AngAcl / 1000.0
            now_Angle += now_AngVel / 1000.0
            s_now_Angle = max(now_Angle - (now_AngVel * speed_ms * K_slip_angle), 0)
            fin_x = befor_x + speed_ms * np.sin(np.deg2rad(s_now_Angle))
            fin_y = befor_y + speed_ms * np.cos(np.deg2rad(s_now_Angle))
            self.ax.plot(np.array([befor_x, fin_x]), np.array([befor_y, fin_y]), color="green", linestyle="solid")
            self.ax.plot(np.array([befor_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                    np.array([befor_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                    color="#888888", linestyle="solid")
            self.ax.plot(np.array([befor_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                    np.array([befor_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                    color="#888888", linestyle="solid")
        
        # 定速区間
        now_AngVel = low_AngVel
        use_angle = now_Angle - self.ini_angle
        while now_Angle < ((self.ini_angle + self.fin_angle) - use_angle):
            befor_x = fin_x; befor_y = fin_y
            now_Angle += low_AngVel / 1000.0
            s_now_Angle = now_Angle - (low_AngVel * speed_ms * K_slip_angle)
            fin_x = befor_x + speed_ms * np.sin(np.deg2rad(s_now_Angle))
            fin_y = befor_y + speed_ms * np.cos(np.deg2rad(s_now_Angle))
            self.ax.plot(np.array([befor_x, fin_x]), np.array([befor_y, fin_y]), color="red", linestyle="solid")
            self.ax.plot(np.array([befor_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                    np.array([befor_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                    color="#888888", linestyle="solid")
            self.ax.plot(np.array([befor_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                    np.array([befor_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                    color="#888888", linestyle="solid")
        
        # 角速度減速区間
        while now_AngVel > 0:
            befor_x = fin_x; befor_y = fin_y
            now_AngVel -= Low_AngAcl / 1000.0
            now_Angle += now_AngVel / 1000.0
            s_now_Angle = now_Angle - (now_AngVel * speed_ms * K_slip_angle)
            fin_x = befor_x + speed_ms * np.sin(np.deg2rad(s_now_Angle))
            fin_y = befor_y + speed_ms * np.cos(np.deg2rad(s_now_Angle))
            self.ax.plot(np.array([befor_x, fin_x]), np.array([befor_y, fin_y]), color="green", linestyle="solid")
            self.ax.plot(np.array([befor_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x + (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                    np.array([befor_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y - (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                    color="#888888", linestyle="solid")
            self.ax.plot(np.array([befor_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle)), fin_x - (Width/2)*np.cos(np.deg2rad(s_now_Angle))]),
                    np.array([befor_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle)), fin_y + (Width/2)*np.sin(np.deg2rad(s_now_Angle))]),
                    color="#888888", linestyle="solid")
        
        # 後オフセット区間
        befor_x = fin_x; befor_y = fin_y
        fin_x = befor_x + post_offset * np.sin(np.deg2rad(self.ini_angle + self.fin_angle))
        fin_y = befor_y + post_offset * np.cos(np.deg2rad(self.ini_angle + self.fin_angle))
        self.ax.plot(np.array([befor_x, fin_x]), np.array([befor_y, fin_y]), color="blue", linestyle="solid")
        self.ax.plot(np.array([befor_x + (Width/2)*np.cos(np.deg2rad(self.ini_angle + self.fin_angle)),
                        fin_x + (Width/2)*np.cos(np.deg2rad(self.ini_angle + self.fin_angle))]),
                np.array([befor_y - (Width/2)*np.sin(np.deg2rad(self.ini_angle + self.fin_angle)),
                        fin_y - (Width/2)*np.sin(np.deg2rad(self.ini_angle + self.fin_angle))]),
                color="#888888", linestyle="solid")
        self.ax.plot(np.array([befor_x - (Width/2)*np.cos(np.deg2rad(self.ini_angle + self.fin_angle)),
                        fin_x - (Width/2)*np.cos(np.deg2rad(self.ini_angle + self.fin_angle))]),
                np.array([befor_y + (Width/2)*np.sin(np.deg2rad(self.ini_angle + self.fin_angle)),
                        fin_y + (Width/2)*np.sin(np.deg2rad(self.ini_angle + self.fin_angle))]),
                color="#888888", linestyle="solid")
        
        # キャンバスの更新
        self.canvas.draw()
        
        # 実際の終点座標を返す（後オフセット区間の終点）
        return fin_x, fin_y
    
    def simulate_phase(self, velocity, alpha_mag, t_phase):
        """単一フェーズのシミュレーション（簡易版）"""
        phi0 = math.pi/2
        dt = t_phase / 100
        x = 0
        y = 0
        
        for i in range(100):
            t = i * dt
            phi = phi0 - 0.5 * alpha_mag * t * t
            x += velocity * math.cos(phi) * dt
            y += velocity * math.sin(phi) * dt
        
        phi_final = phi0 - 0.5 * alpha_mag * t_phase * t_phase
        return x, y, phi_final
    
    def initialize_plot(self):
        """初期プロットを表示（軸道計算なし）"""
        # プロットを初期化
        self.ax.clear()
        
        # 軸の設定
        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Y [mm]")
        self.ax.set_title(f"{self.robot_size} - {self.turn_type}")
        
        # 格子と柱を描画
        self.draw_grid()
        
        # スタート地点と目標地点を描画
        s = self.scale_factor
        self.ax.plot(0 * s, self.ini_y, 'o', color="blue", markersize=8)  # スタート地点
        self.ax.plot(self.target_x, self.target_y, 's', color="green", markersize=8)  # ターゲット地点
        
        # 軸の範囲を設定
        if self.robot_size == "ハーフ":
            self.ax.set_xlim(-50, 150)
            self.ax.set_ylim(-10, 190)
        else:  # クラシック
            self.ax.set_xlim(-100, 300)
            self.ax.set_ylim(-20, 380)
        
        # キャンバスを更新
        self.canvas.draw()
        
        # ユーザーにメッセージを表示
        self.acc_label.config(text="---")
        self.ang_vel_label.config(text="---")
        self.rear_offset_label.config(text="---")
        self.time_label.config(text="---")
    
    def draw_grid(self):
        """格子線と迷路の柱の描画（元のDrawCanvas関数を移植）"""
        s = self.scale_factor
        
        # 迷路の壁と格子線
        self.ax.plot(np.array([-45.0, -45.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([0.0, 0.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([45.0, 45.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([90.0, 90.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([135.0, 135.0]) * s, np.array([0.0, 180.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([-45.0, 135.0]) * s, np.array([0.0, 0.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([-45.0, 135.0]) * s, np.array([45.0, 45.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([-45.0, 135.0]) * s, np.array([90.0, 90.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([-45.0, 135.0]) * s, np.array([135.0, 135.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([-45.0, 135.0]) * s, np.array([180.0, 180.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([0.0, -45.0]) * s, np.array([180.0, 135.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([90.0, -45.0]) * s, np.array([180.0, 45.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([135.0, 0.0]) * s, np.array([135.0, 0.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([135.0, 90.0]) * s, np.array([45.0, 0.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([90.0, 135.0]) * s, np.array([180.0, 135.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([0.0, 135.0]) * s, np.array([180.0, 45.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([-45.0, 90.0]) * s, np.array([135.0, 0.0]) * s, color="gray", linestyle="dashed")
        self.ax.plot(np.array([-45.0, 0.0]) * s, np.array([45.0, 0.0]) * s, color="gray", linestyle="dashed")
        
        # 迷路の柱（赤い正方形）
        self.ax.plot(np.array([-48, -48, -42, -42, -48]) * s, np.array([-3, 3, 3, -3, -3]) * s, color="red", linestyle="solid")
        self.ax.plot(np.array([-48, -48, -42, -42, -48]) * s, np.array([87, 93, 93, 87, 87]) * s, color="red", linestyle="solid")
        self.ax.plot(np.array([-48, -48, -42, -42, -48]) * s, np.array([177, 183, 183, 177, 177]) * s, color="red", linestyle="solid")
        self.ax.plot(np.array([42, 42, 48, 48, 42]) * s, np.array([-3, 3, 3, -3, -3]) * s, color="red", linestyle="solid")
        self.ax.plot(np.array([42, 42, 48, 48, 42]) * s, np.array([87, 93, 93, 87, 87]) * s, color="red", linestyle="solid")
        self.ax.plot(np.array([42, 42, 48, 48, 42]) * s, np.array([177, 183, 183, 177, 177]) * s, color="red", linestyle="solid")
        self.ax.plot(np.array([132, 132, 138, 138, 132]) * s, np.array([-3, 3, 3, -3, -3]) * s, color="red", linestyle="solid")
        self.ax.plot(np.array([132, 132, 138, 138, 132]) * s, np.array([87, 93, 93, 87, 87]) * s, color="red", linestyle="solid")
        self.ax.plot(np.array([132, 132, 138, 138, 132]) * s, np.array([177, 183, 183, 177, 177]) * s, color="red", linestyle="solid")
        
        # 軸の設定
        self.ax.set_xlim(-50 * s, 150 * s)
        self.ax.set_ylim(-10 * s, 200 * s)
        self.ax.set_xlabel('X [mm]')
        self.ax.set_ylabel('Y [mm]')
        self.ax.set_title(f'{self.robot_size} {self.turn_type} シミュレーション')
        self.ax.grid(False)  # グリッドは不要（迷路の格子線があるため）
    
    def on_closing(self):
        """Windowが閉じられるときの処理"""
        print("\nプログラムを終了します...")
        
        # タプルが関係する変数フォーマットの記録をクリアする
        self.current_plot_end = None
        self.current_adjusted_target = None
        self.optimized_values = None
        
        # 設定を保存
        self.save_settings()
        
        # Windowを破棄
        self.root.destroy()

    def draw_robot(self, x, y, phi, color):
        """ロボットを描画"""
        width = self.robot_width
        length = width  # 正方形と仮定
        
        # ロボットの四隅の座標（ローカル座標系）
        corners_local = [
            [-width/2, length/2],  # 左前
            [width/2, length/2],   # 右前
            [width/2, -length/2],  # 右後
            [-width/2, -length/2], # 左後
            [-width/2, length/2]   # 閉じるために最初の点を繰り返す
        ]
        
        # グローバル座標系に変換
        corners_global_x = []
        corners_global_y = []
        
        for corner in corners_local:
            # 回転行列による変換
            rot_x = corner[0] * math.cos(phi) - corner[1] * math.sin(phi)
            rot_y = corner[0] * math.sin(phi) + corner[1] * math.cos(phi)
            
            # 平行移動
            global_x = x + rot_x
            global_y = y + rot_y
            
            corners_global_x.append(global_x)
            corners_global_y.append(global_y)
        
        # ロボットの輪郭を描画
        self.ax.plot(corners_global_x, corners_global_y, color, linewidth=2)
        
        # 進行方向を示す矢印
        arrow_length = width * 0.8
        arrow_x = x + arrow_length * math.cos(phi)
        arrow_y = y + arrow_length * math.sin(phi)
        self.ax.arrow(x, y, arrow_length * math.cos(phi), arrow_length * math.sin(phi), 
                    head_width=width/5, head_length=width/3, fc=color, ec=color)

# ==========================
# メイン関数
# ==========================
def main():
    try:
        # マルチプロセッシングの設定
        mp.freeze_support()
        
        # GUIアプリケーションの起動
        root = tk.Tk()
        app = TurnTunerApp(root)
        root.mainloop()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"エラーが発生しました: {e}")
    finally:
        print("プログラムを終了します...")
        input("何かキーを押して終了してください...")

if __name__ == "__main__":
    main()
