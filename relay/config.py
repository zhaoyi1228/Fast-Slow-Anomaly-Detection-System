"""
Relay Anomaly Detection Configuration
中间节点配置 - 转发机器端
"""

import os

# ========== 端侧接收服务配置 ==========
EDGE_RECEIVER = {
    "host": "0.0.0.0",  # 监听所有接口
    "port": int(os.environ.get("EDGE_RECEIVER_PORT", 9000)),
    "frame_endpoint": "/api/edge/frame",
    "status_endpoint": "/api/edge/status",
}

# ========== 云侧服务配置（通过VPN访问）==========
CLOUD_SERVER = {
    "host": os.environ.get("CLOUD_HOST", "10.8.0.1"),  # 云侧VPN IP
    "port": int(os.environ.get("CLOUD_PORT", 8001)),
    "detect_endpoint": "/api/v1/detect",
    "health_endpoint": "/api/v1/health",
}

# ========== Gradio可视化配置 ==========
GRADIO_SERVER = {
    "host": "127.0.0.1",  # 仅本地访问
    "port": 7860,
    "share": False,  # 不创建公网链接
}

# ========== 融合检测参数配置 ==========
FUSION_CONFIG = {
    "jigsaw_threshold": 0.4,  # Jigsaw分数阈值
    "window_size_seconds": 5.0,  # 滑动窗口大小（秒）
    "window_threshold_percent": 0.3,  # 窗口内异常帧比例阈值
    "deep_analysis_batch_size": 5,  # 发送给云侧的帧数
    "deep_analysis_min_interval": 3.0,  # 深度分析最小间隔（秒）
}

# ========== 帧缓冲配置 ==========
FRAME_BUFFER_CONFIG = {
    "max_frames": 150,  # 最大缓存帧数（约30秒@5fps）
    "cleanup_interval": 10.0,  # 清理间隔（秒）
    "frame_ttl": 60.0,  # 帧过期时间（秒）
}

# ========== 网络重连配置 ==========
NETWORK_CONFIG = {
    "max_retries": 3,
    "retry_interval_base": 2.0,
    "retry_interval_max": 10.0,
    "request_timeout": 60.0,  # 云侧请求超时（秒）- Agent分析需要时间
    "health_check_interval": 30.0,  # 健康检查间隔（秒）
}

# ========== 劝阻文本模板 ==========
DETERRENCE_TEMPLATES = {
    "fighting": "警告：检测到打架斗殴行为！请立即停止违法行为。",
    "running": "注意：检测到奔跑行为！请在步行区域慢速行走。",
    "falling": "警告：检测到有人摔倒！请注意安全。",
    "crowd_gathering": "注意：检测到人群聚集！请保持秩序。",
    "vehicle_intrusion": "警告：检测到车辆闯入！行人请注意安全。",
    "bicycle": "注意：检测到自行车！请注意行人安全。",
    "default": "警告：检测到异常行为！请注意安全。",
}

# ========== 异常类型关键词映射 ==========
ANOMALY_KEYWORDS = {
    "fighting": ["fight", "打架", "斗殴", "punch", "kick", "violence"],
    "running": ["run", "奔跑", "跑步", "sprint", "chase"],
    "falling": ["fall", "摔倒", "跌倒", "collapse"],
    "crowd_gathering": ["crowd", "聚集", "gathering"],
    "vehicle_intrusion": ["vehicle", "车辆", "car", "truck"],
    "bicycle": ["bicycle", "自行车", "bike"],
}