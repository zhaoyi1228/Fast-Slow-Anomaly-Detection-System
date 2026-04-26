"""FastAPI service that wraps MemoryEnhancedVADAgent via AnomalyAgent.api."""

import time
import uuid
import os
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import API_SERVER, AGENT_CONFIG, DETECTION_CONFIG, RESOURCE_PATHS


def _resolve_anomaly_agent_path(raw_path: str) -> str:
    """Resolve a usable AnomalyAgent import root.

    Accept either:
    - the repository root that contains `main_memory_vad.py`
    - a parent directory that contains an `AnomalyAgent/` subdirectory
    """
    candidates = []
    if raw_path:
        base = Path(raw_path).expanduser().resolve()
        candidates.extend([base, base / "AnomalyAgent"])

    cwd_base = Path(os.getcwd()).resolve()
    candidates.extend([cwd_base / "AnomalyAgent", cwd_base])

    for candidate in candidates:
        if (candidate / "main_memory_vad.py").exists():
            return str(candidate)

    return str(Path(raw_path).expanduser()) if raw_path else str(cwd_base / "AnomalyAgent")


ANOMALY_AGENT_PATH = _resolve_anomaly_agent_path(RESOURCE_PATHS["anomaly_agent_project_path"])
if ANOMALY_AGENT_PATH not in sys.path:
    sys.path.insert(0, ANOMALY_AGENT_PATH)

AGENT_AVAILABLE = True
try:
    from main_memory_vad import (
        Config,
        load_config_from_yaml,
        create_brain_llm_model,
        create_memory_tool_llm_model,
        create_vlm_model,
        create_tools,
        create_memory_agent,
        configure_semantic_similarity,
        has_memory_tool_llm_override,
    )
    from api.handlers.frame_processor import FrameProcessor
    from api.handlers.detection_handler import DetectionHandler
except Exception as e:
    AGENT_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"[cloud.api_server] Failed to import AnomalyAgent stack: {IMPORT_ERROR}")


# Pydantic模型定义
class FrameInput(BaseModel):
    """单帧输入"""
    image_base64: str
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None
    jigsaw_score: Optional[float] = None


class DetectRequest(BaseModel):
    """检测请求"""
    frames: List[FrameInput]
    video_id: Optional[str] = None
    scene_type: Optional[str] = "general"
    dataset: Optional[str] = "ped2"
    batch_size: Optional[int] = None
    anomaly_rules: Optional[str] = None


class DetectResponse(BaseModel):
    """检测响应"""
    success: bool
    request_id: str
    video_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    processing_time_ms: int
    error: Optional[str] = None


class MemoryStatsResponse(BaseModel):
    """Memory统计响应"""
    working_memory: Dict[str, Any]
    episodic_memory: Dict[str, Any]
    semantic_memory: Dict[str, Any]


# FastAPI应用
app = FastAPI(
    title="Cloud Anomaly Detection API",
    description="云侧Agent深度异常检测服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
agent_instance = None
detection_handler = None
frame_processor = None
agent_config = None
request_counter = 0


def get_agent():
    """获取或创建Agent实例"""
    global agent_instance, detection_handler, frame_processor, agent_config

    if agent_instance is None and AGENT_AVAILABLE:
        try:
            config_path = AGENT_CONFIG["config_path"]
            if os.path.exists(config_path):
                agent_config = load_config_from_yaml(config_path)
            else:
                agent_config = Config()

            if RESOURCE_PATHS["semantic_similarity_model_path"]:
                agent_config.semantic_similarity_model_path = RESOURCE_PATHS["semantic_similarity_model_path"]

            configure_semantic_similarity(agent_config)
            brain_model = create_brain_llm_model(agent_config)
            memory_tool_model = create_memory_tool_llm_model(agent_config) if has_memory_tool_llm_override(agent_config) else brain_model
            vlm_model_dict = create_vlm_model(agent_config)
            tools = create_tools(agent_config, vlm_model_dict, memory_tool_model)

            checkpoint_dir = AGENT_CONFIG["memory_checkpoint_dir"]
            os.makedirs(checkpoint_dir, exist_ok=True)
            agent_instance = create_memory_agent(agent_config, brain_model, memory_tool_model, tools, checkpoint_dir)

            frame_processor = FrameProcessor(
                temp_dir=AGENT_CONFIG["temp_frame_dir"],
                cleanup=AGENT_CONFIG["cleanup_temp_frames"],
            )
            detection_handler = DetectionHandler(
                agent=agent_instance,
                frame_processor=frame_processor,
                config=agent_config,
            )
        except Exception as e:
            print(f"Agent创建失败: {e}")
            agent_instance = None
    return agent_instance


def _validate_frames(frames: List[FrameInput]):
    for idx, frame in enumerate(frames):
        if not frame.image_base64 or not frame.image_base64.strip():
            raise HTTPException(status_code=400, detail=f"frame[{idx}].image_base64 is required")


async def run_detection(request: DetectRequest) -> Dict[str, Any]:
    """执行检测"""
    agent = get_agent()

    if agent is None:
        detail = IMPORT_ERROR if not AGENT_AVAILABLE else "Agent服务未就绪"
        raise HTTPException(status_code=503, detail=detail)

    try:
        assert detection_handler is not None
        result = await detection_handler.detect(
            frames_input=[frame.model_dump() for frame in request.frames],
            video_id=request.video_id,
            scene_type=request.scene_type or "general",
            dataset=request.dataset or "ped2",
            anomaly_rules=request.anomaly_rules,
            batch_size=request.batch_size,
            input_type="base64",
        )
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Detection failed"))
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        print(f"Agent检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    print(f"API服务启动: {API_SERVER['host']}:{API_SERVER['port']}")
    print(f"AnomalyAgent路径: {RESOURCE_PATHS['anomaly_agent_project_path']}")

    # 预加载Agent（可选）
    if AGENT_AVAILABLE:
        get_agent()


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    global request_counter

    payload = {
        "status": "ok",
        "agent_available": AGENT_AVAILABLE,
        "agent_loaded": agent_instance is not None,
        "request_count": request_counter,
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }
    if not AGENT_AVAILABLE:
        payload["import_error"] = IMPORT_ERROR
    return payload


@app.post("/api/v1/detect")
async def detect(request: DetectRequest):
    """
    执行异常检测

    Args:
        request: 检测请求，包含帧列表

    Returns:
        DetectResponse: 检测结果
    """
    global request_counter

    request_counter += 1
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # 验证请求
    if not request.frames:
        raise HTTPException(status_code=400, detail="帧列表不能为空")

    if len(request.frames) > DETECTION_CONFIG["max_frames_per_request"]:
        raise HTTPException(
            status_code=400,
            detail=f"帧数超过限制: {DETECTION_CONFIG['max_frames_per_request']}"
        )

    try:
        _validate_frames(request.frames)
        result = await run_detection(request)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return DetectResponse(
            success=result.get("success", True),
            request_id=result.get("request_id", request_id),
            video_id=result.get("video_id", request.video_id),
            result=result.get("result"),
            processing_time_ms=processing_time_ms
        )
    except HTTPException:
        raise


@app.get("/api/v1/memory/stats")
async def get_memory_stats():
    """获取Memory系统统计"""
    agent = get_agent()
    if agent is None or not getattr(agent, "enable_memory", False):
        raise HTTPException(status_code=404, detail="Memory系统未初始化")

    stats = agent.get_memory_stats()
    stats["total_requests"] = request_counter
    return stats


@app.post("/api/v1/memory/reset")
async def reset_memory(memory_type: str = "all"):
    """重置Memory"""
    agent = get_agent()
    if agent is None or not getattr(agent, "enable_memory", False):
        raise HTTPException(status_code=404, detail="Memory系统未初始化")

    memory = getattr(agent, "memory", None)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory系统未初始化")

    if memory_type in ("working", "all") and hasattr(memory.working, "reset"):
        memory.working.reset()
    if memory_type in ("episodic", "all") and hasattr(memory.episodic, "reset"):
        memory.episodic.reset()
    if memory_type in ("semantic", "all") and hasattr(memory.semantic, "reset"):
        memory.semantic.reset()

    return {"status": "ok", "reset_type": memory_type}


@app.post("/api/v1/memory/save")
async def save_memory():
    """保存Memory状态"""
    agent = get_agent()
    if agent is None or not getattr(agent, "enable_memory", False):
        raise HTTPException(status_code=404, detail="Memory系统未初始化")

    try:
        checkpoint_dir = AGENT_CONFIG["memory_checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = agent.save_memory(checkpoint_dir)
        return {"status": "ok", "checkpoint_path": checkpoint_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/config")
async def get_config():
    """获取服务配置（隐藏敏感信息）"""
    return {
        "agent_config": AGENT_CONFIG,
        "detection_config": DETECTION_CONFIG,
        "api_server": API_SERVER,
        "anomaly_agent_project_path": RESOURCE_PATHS["anomaly_agent_project_path"],
        "agent_yaml_config_path": AGENT_CONFIG["config_path"],
    }


def main():
    """启动API服务"""
    import uvicorn

    app.state.start_time = time.time()

    uvicorn.run(
        app,
        host=API_SERVER["host"],
        port=API_SERVER["port"],
        workers=API_SERVER["workers"]
    )


if __name__ == "__main__":
    main()