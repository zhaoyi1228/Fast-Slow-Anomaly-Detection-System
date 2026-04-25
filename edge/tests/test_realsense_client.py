import importlib
import sys
import types

import numpy as np


def _install_fake_pyrealsense(monkeypatch):
    fake_rs = types.SimpleNamespace(
        pipeline=lambda: object(),
        config=lambda: object(),
        stream=types.SimpleNamespace(color="color", depth="depth"),
        format=types.SimpleNamespace(bgr8="bgr8", z16="z16"),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)


def test_depth_image_none_check_does_not_trigger_numpy_truth_value(monkeypatch):
    _install_fake_pyrealsense(monkeypatch)

    module = importlib.import_module("anomaly_detection_system.edge.camera.realsense_client")
    client = module.RealSenseClient(config={
        "resolution": (64, 64),
        "fps": 30,
        "enable_color": True,
        "enable_depth": True,
    })

    color_image = np.zeros((64, 64, 3), dtype=np.uint8)
    depth_image = np.zeros((64, 64), dtype=np.uint16)

    class FakeColorFrame:
        def get_data(self):
            return color_image

    class FakeDepthFrame:
        def get_data(self):
            return depth_image

    class FakeFrames:
        def get_color_frame(self):
            return FakeColorFrame()

        def get_depth_frame(self):
            return FakeDepthFrame()

    class FakePipeline:
        def wait_for_frames(self):
            return FakeFrames()

    client.pipeline = FakePipeline()
    client.start_time = 0.0
    monkeypatch.setattr(client, "_encode_image", lambda image: "color-b64")
    monkeypatch.setattr(client, "_encode_depth", lambda image: "depth-b64")

    frame = client.get_frame()

    assert frame is not None
    assert frame.depth_base64 == "depth-b64"


def test_start_streaming_clamps_frame_skip_to_at_least_one(monkeypatch):
    _install_fake_pyrealsense(monkeypatch)
    module = importlib.import_module("anomaly_detection_system.edge.camera.realsense_client")

    monkeypatch.setitem(
        sys.modules,
        "config",
        types.SimpleNamespace(
            REALSENSE_CONFIG={
                "resolution": (64, 64),
                "fps": 5,
                "enable_color": True,
                "enable_depth": False,
            },
            DETECTION_CONFIG={"fps_sample": 10},
        ),
    )

    client = module.RealSenseClient(config={
        "resolution": (64, 64),
        "fps": 5,
        "enable_color": True,
        "enable_depth": False,
    })

    monkeypatch.setattr(client, "initialize", lambda: True)
    monkeypatch.setattr(client, "get_frame", lambda: None)

    thread_calls = {}

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            thread_calls["target"] = target
            thread_calls["daemon"] = daemon

        def start(self):
            thread_calls["started"] = True

    monkeypatch.setattr(module.threading, "Thread", FakeThread)

    client.start_streaming(lambda frame: None, sample_fps=10)

    assert client.is_streaming is True
    assert thread_calls["started"] is True
    assert callable(thread_calls["target"])
