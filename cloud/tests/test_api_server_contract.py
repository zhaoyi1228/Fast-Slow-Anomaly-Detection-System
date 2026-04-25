import importlib
import sys
import types

import pytest
from fastapi import HTTPException


def _load_api_server_with_stubs(monkeypatch):
    fake_main = types.ModuleType("main_memory_vad")

    class FakeConfig:
        enable_memory = True
        batch_size = 5
        semantic_similarity_model_path = ""

    fake_main.Config = FakeConfig
    fake_main.load_config_from_yaml = lambda path: FakeConfig()
    fake_main.create_brain_llm_model = lambda config: object()
    fake_main.create_memory_tool_llm_model = lambda config: object()
    fake_main.create_vlm_model = lambda config: {"main_model": object()}
    fake_main.create_tools = lambda config, vlm_model_dict, memory_tool_model: [object()]
    fake_main.create_memory_agent = lambda config, brain_model, memory_tool_model, tools, save_dir: object()
    fake_main.configure_semantic_similarity = lambda config: None
    fake_main.has_memory_tool_llm_override = lambda config: False

    fake_api_handlers = types.ModuleType("api.handlers")
    fake_frame_processor_module = types.ModuleType("api.handlers.frame_processor")
    fake_detection_handler_module = types.ModuleType("api.handlers.detection_handler")

    class FakeFrameProcessor:
        def __init__(self, temp_dir, cleanup):
            self.temp_dir = temp_dir
            self.cleanup = cleanup

    class FakeDetectionHandler:
        def __init__(self, agent, frame_processor, config):
            self.agent = agent
            self.frame_processor = frame_processor
            self.config = config

        async def detect(self, **kwargs):
            return {
                "success": True,
                "request_id": "req-1",
                "video_id": kwargs.get("video_id") or "generated-video",
                "result": {
                    "anomaly_score": 0.2,
                    "is_anomaly": False,
                    "anomaly_type": "none",
                    "description": "normal",
                    "explanation": "no anomaly",
                    "frame_scores": [0.2],
                    "batch_results": [],
                },
            }

    fake_frame_processor_module.FrameProcessor = FakeFrameProcessor
    fake_detection_handler_module.DetectionHandler = FakeDetectionHandler

    monkeypatch.setitem(sys.modules, "main_memory_vad", fake_main)
    monkeypatch.setitem(sys.modules, "api.handlers", fake_api_handlers)
    monkeypatch.setitem(sys.modules, "api.handlers.frame_processor", fake_frame_processor_module)
    monkeypatch.setitem(sys.modules, "api.handlers.detection_handler", fake_detection_handler_module)

    sys.modules.pop("anomaly_detection_system.cloud.service.api_server", None)
    return importlib.import_module("anomaly_detection_system.cloud.service.api_server")


def test_validate_frames_rejects_empty_base64(monkeypatch):
    module = _load_api_server_with_stubs(monkeypatch)

    with pytest.raises(HTTPException) as exc:
        module._validate_frames([module.FrameInput(image_base64="")])

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_run_detection_returns_503_when_agent_unavailable(monkeypatch):
    module = _load_api_server_with_stubs(monkeypatch)
    monkeypatch.setattr(module, "get_agent", lambda: None)
    monkeypatch.setattr(module, "AGENT_AVAILABLE", False)
    monkeypatch.setattr(module, "IMPORT_ERROR", "import failed", raising=False)

    request = module.DetectRequest(frames=[module.FrameInput(image_base64="abc")])

    with pytest.raises(HTTPException) as exc:
        await module.run_detection(request)

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_run_detection_uses_detection_handler_contract(monkeypatch):
    module = _load_api_server_with_stubs(monkeypatch)
    module.get_agent()

    captured = {}

    async def fake_detect(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "request_id": "req-2",
            "video_id": "video-2",
            "result": {
                "anomaly_score": 0.1,
                "is_anomaly": False,
                "anomaly_type": "none",
                "description": "ok",
                "explanation": "ok",
                "frame_scores": [0.1],
                "batch_results": [],
            },
        }

    module.detection_handler.detect = fake_detect
    request = module.DetectRequest(
        frames=[module.FrameInput(image_base64="abc", frame_id=1, timestamp=1.2, jigsaw_score=0.3)],
        video_id="video-2",
        scene_type="general",
        dataset="ped2",
    )

    result = await module.run_detection(request)

    assert captured["input_type"] == "base64"
    assert captured["video_id"] == "video-2"
    assert captured["frames_input"][0]["frame_id"] == 1
    assert result["success"] is True
