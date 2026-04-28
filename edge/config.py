"""
Edge Anomaly Detection Configuration
端侧异常检测配置 - 机器狗端
"""

import os

# ========== 中间机器配置（转发节点）==========
RELAY_SERVER = {
    "host": os.environ.get("RELAY_HOST", "192.168.1.100"),  # 中间机器局域网IP
    "port": int(os.environ.get("RELAY_PORT", 9000)),
    "frame_endpoint": "/api/edge/frame",
    "status_endpoint": "/api/edge/status",
}

# ========== 本地Jigsaw服务配置 ==========
JIGSAW_SERVER = {
    "host": "localhost",
    "port": 8000,
    "detect_endpoint": "/detect",
    "batch_detect_endpoint": "/batch_detect",
    "health_endpoint": "/health",
}

# ========== RealSense相机配置 ==========
REALSENSE_CONFIG = {
    "resolution": (640, 480),
    "fps": 30,
    "enable_color": True,
    "enable_depth": True,
    "color_format": "bgr8",
    "depth_format": "z16",
}

# ========== 检测参数配置 ==========
DETECTION_CONFIG = {
    "anomaly_threshold": 0.5,  # 分数>=此值判定为异常
    "fps_sample": 5,  # 每秒采样帧数（实际处理频率）
    "frame_skip": 6,  # 每30fps跳过6帧，实现5fps采样
}

# ========== 网络重连配置 ==========
NETWORK_CONFIG = {
    "max_retries": 5,  # 最大重试次数
    "retry_interval_base": 2.0,  # 重试间隔基数（秒）
    "retry_interval_max": 10.0,  # 最大重试间隔（秒）
    "request_timeout": 5.0,  # 单次请求超时（秒）
    "cache_max_frames": 100,  # 本地缓存最大帧数
}

# ========== Jigsaw模型配置 ==========
JIGSAW_MODEL_CONFIG = {
    "checkpoint_path": "pre_trained/stc_78.76_sample7.pth",  # 模型权重路径
    "sample_num": 7,  # 帧数
    "gpu_id": 0,  # GPU ID，空字符串表示CPU
}

# ========== MAE服务配置 ==========
MAE_SERVER = {
    "host": "localhost",
    "port": 8001,  # 与Jigsaw端口区分
    "detect_endpoint": "/detect",
    "health_endpoint": "/health",
    "reset_endpoint": "/reset",
}

# ========== MAE模型配置 ==========
MAE_MODEL_CONFIG = {
    "checkpoint_dir": "./edge/detection/pre_trained/aed-mae/stc",  # ShanghaiTech checkpoint目录
    "teacher_ckpt": "checkpoint-best.pth",
    "student_ckpt": "checkpoint-best-student.pth",
    "input_size": (160, 320),  # ShanghaiTech配置 (H, W)
    "patch_size": 8,
    "buffer_size": 7,  # 帧缓冲大小
    "mask_ratio": 0.5,  # MAE mask比例
    "gpu_id": 0,  # GPU ID，-1表示使用CPU
}

# ========== 日志配置 ==========
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}