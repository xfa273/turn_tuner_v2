#!/usr/bin/env python3
# turn_params.py
# マイクロマウスのターン動作パラメータ定義

# 機体サイズの定義
ROBOT_SIZES = {
    "ハーフ": {"width": 36.00, "scale": 1},
    "クラシック": {"width": 64.00, "scale": 2}
}

# 動作タイプの定義 (ハーフサイズ)
HALF_TURNS = {
    "小回り90deg": {
        "angle": 90,
        "target_x": 90,
        "target_y": 90,
        "default_velocity": 250,
        "default_front_offset": 55  # 10+45
    },
    "大回り90deg": {
        "angle": 90,
        "target_x": 90,
        "target_y": 90,
        "default_velocity": 450,
        "default_front_offset": 0
    },
    "大回り180deg": {
        "angle": 180,
        "target_x": 90,
        "target_y": 0,
        "default_velocity": 450,
        "default_front_offset": 0
    },
    "45deg入り": {
        "angle": 45,
        "target_x": 45,
        "target_y": 90,
        "default_velocity": 450,
        "default_front_offset": 0
    },
    "135deg入り": {
        "angle": 135,
        "target_x": 90,
        "target_y": 45,
        "default_velocity": 450,
        "default_front_offset": 15
    }
}

# 動作タイプの定義 (クラシックサイズ)
CLASSIC_TURNS = {
    "小回り90deg": {
        "angle": 90,
        "target_x": 180,
        "target_y": 180,
        "default_velocity": 600,
        "default_front_offset": 100  # 10+90
    },
    "大回り90deg": {
        "angle": 90,
        "target_x": 180,
        "target_y": 180,
        "default_velocity": 900,
        "default_front_offset": 0
    },
    "大回り180deg": {
        "angle": 180,
        "target_x": 180,
        "target_y": 0,
        "default_velocity": 900,
        "default_front_offset": 0
    },
    "45deg入り": {
        "angle": 45,
        "target_x": 90,
        "target_y": 180,
        "default_velocity": 900,
        "default_front_offset": 0
    },
    "135deg入り": {
        "angle": 135,
        "target_x": 180,
        "target_y": 90,
        "default_velocity": 900,
        "default_front_offset": 30
    }
}

# 計算パラメータ（探索範囲の設定）
CALC_PARAMS = {
    "dt": 1e-3,                    # 数値積分の刷み時間 [s]
    "min_acc_deg": 100,            # 最小角加速度 [deg/s²]
    "max_acc_deg": 30000,          # 最大角加速度 [deg/s²]
    "acc_step": 10,                # 角加速度の刷み幅（精度向上のため10に変更）
    "min_rear_offset": 0,          # 最小後オフセット距離 [mm]
    "max_rear_offset": 180,        # 最大後オフセット距離 [mm]
    "rear_offset_step": 0.1,        # 後オフセット距離の刷み幅 [mm]
    "default_slip_coefficient": 1.0 # スリップアングル係数のデフォルト値（1.0でスリップなし）
}
