"""
Cloud Anomaly Detection Configuration
云侧配置 - 服务器端Agent服务
"""

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ========== API服务配置 ==========
API_SERVER = {
    "host": os.environ.get("API_HOST", "0.0.0.0"),
    "port": int(os.environ.get("API_PORT", 8001)),
    "workers": 1,  # 推荐单worker（Agent全局单例）
}

# ========== Agent配置 ==========
AGENT_CONFIG = {
    "config_path": os.environ.get("ANOMALY_AGENT_CONFIG_PATH", os.path.join("AnomalyAgent", "config", "api_config.yaml")) if os.environ.get("ANOMALY_AGENT_CONFIG_PATH", os.path.join("AnomalyAgent", "config", "api_config.yaml")) else "/home/zhaoyi/media/projects_zy/anomaly_agent",
    "temp_frame_dir": os.environ.get("CLOUD_TEMP_FRAME_DIR", "/home/zhaoyi/media/temp_frames"),  # VLM可访问路径
    "memory_checkpoint_dir": os.environ.get("CLOUD_MEMORY_CHECKPOINT_DIR", "/home/zhaoyi/media/memory_checkpoints"),  # VLM可访问路径
    "cleanup_temp_frames": _env_bool("CLOUD_CLEANUP_TEMP_FRAMES", False),  # 默认不清理，方便回溯
    "save_results": _env_bool("CLOUD_SAVE_RESULTS", True),  # 保存检测结果到result子文件夹
}

# ========== 检测参数 ==========
DETECTION_CONFIG = {
    "max_frames_per_request": 100,
}

# ========== 服务超时配置 ==========
TIMEOUT_CONFIG = {
    "request_timeout": 300,  # 请求超时（秒）
    "max_retry": 3,
}

# ========== 外部路径与服务语义 ==========
RESOURCE_PATHS = {
    "anomaly_agent_project_path": os.environ.get("ANOMALY_AGENT_PROJECT_PATH", os.path.join(os.getcwd(), "AnomalyAgent")) if os.environ.get("ANOMALY_AGENT_PROJECT_PATH", os.path.join(os.getcwd(), "AnomalyAgent")) else "/home/zhaoyi/media/projects_zy/anomaly_agent" ,
    "semantic_similarity_model_path": os.environ.get("SEMANTIC_SIMILARITY_MODEL_PATH", ""),
}