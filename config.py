# config.py
# -----------------------------------------
# 走行に関するパラメータ（ユーザが編集する部分）
# -----------------------------------------

# ハーフ小回り90deg
""" TRANSLATIONAL_VELOCITY = 250    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 90     # 旋回角度 [deg]
TARGET_X               = 90   # 目標最終位置 x [mm]
TARGET_Y               = 90      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 10+45    # 前オフセット距離 [mm] """

# クラシック小回り90deg
""" TRANSLATIONAL_VELOCITY = 600    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 90     # 旋回角度 [deg]
TARGET_X               = 180   # 目標最終位置 x [mm]
TARGET_Y               = 180      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 10+90    # 前オフセット距離 [mm] """

# ハーフ大回り90deg
""" TRANSLATIONAL_VELOCITY = 450    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 90     # 旋回角度 [deg]
TARGET_X               = 90   # 目標最終位置 x [mm]
TARGET_Y               = 90      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 0    # 前オフセット距離 [mm] """

# クラシック大回り90deg
""" TRANSLATIONAL_VELOCITY = 900    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 90     # 旋回角度 [deg]
TARGET_X               = 180   # 目標最終位置 x [mm]
TARGET_Y               = 180      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 0    # 前オフセット距離 [mm] """

# ハーフ大回り180deg
""" TRANSLATIONAL_VELOCITY = 450    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 180     # 旋回角度 [deg]
TARGET_X               = 90   # 目標最終位置 x [mm]
TARGET_Y               = 0      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 0    # 前オフセット距離 [mm] """

# クラシック大回り180deg
TRANSLATIONAL_VELOCITY = 900    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 180     # 旋回角度 [deg]
TARGET_X               = 180   # 目標最終位置 x [mm]
TARGET_Y               = 0      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 0    # 前オフセット距離 [mm]

# ハーフ45deg
""" TRANSLATIONAL_VELOCITY = 450    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 45     # 旋回角度 [deg]
TARGET_X               = 45   # 目標最終位置 x [mm]
TARGET_Y               = 90      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 0    # 前オフセット距離 [mm] """

# クラシック45deg
""" TRANSLATIONAL_VELOCITY = 900    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 45     # 旋回角度 [deg]
TARGET_X               = 90   # 目標最終位置 x [mm]
TARGET_Y               = 180      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 0    # 前オフセット距離 [mm] """

# ハーフ135deg
""" TRANSLATIONAL_VELOCITY = 450    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 135     # 旋回角度 [deg]
TARGET_X               = 90   # 目標最終位置 x [mm]
TARGET_Y               = 45      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 15    # 前オフセット距離 [mm] """

# クラシック135deg
""" TRANSLATIONAL_VELOCITY = 900    # 並進速度 [mm/s]
TURNING_ANGLE_DEG      = 135     # 旋回角度 [deg]
TARGET_X               = 180   # 目標最終位置 x [mm]
TARGET_Y               = 90      # 目標最終位置 y [mm]
FRONT_OFFSET_DISTANCE  = 30    # 前オフセット距離 [mm] """

# 機体の横幅（mm）
ROBOT_WIDTH_HALF = 36.00        # ハーフサイズ用
ROBOT_WIDTH_CLASSIC = 64.00     # クラシック用


# -----------------------------------------
# 計算に関するパラメータ（探索の範囲・刻みなど）
# -----------------------------------------
DT = 1e-3                     # 数値積分の刻み時間 [s]

# 角加速度の探索範囲 [deg/s²]
MIN_ACC_DEG  = 100           # 最小角加速度
MAX_ACC_DEG  = 30000          # 最大角加速度
ACC_STEP     = 100            # 角加速度の刻み幅

# 後オフセット距離（旋回後に直進する距離）の探索範囲 [mm]
MIN_REAR_OFFSET = 0           # 最小後オフセット距離
MAX_REAR_OFFSET = 180        # 最大後オフセット距離
REAR_OFFSET_STEP = 0.1         # 後オフセット距離の刻み幅
