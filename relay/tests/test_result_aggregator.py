import time

from anomaly_detection_system.relay.aggregator.result_aggregator import ResultAggregator
from anomaly_detection_system.relay.receiver.edge_receiver import FrameResult
from anomaly_detection_system.relay.sender.cloud_client import DeepAnalysisResult


class FakeCloudClient:
    def __init__(self):
        self.calls = []

    def detect(self, frames):
        self.calls.append(frames)
        return DeepAnalysisResult(
            anomaly_score=0.9,
            is_anomaly=True,
            anomaly_type="running",
            description="detected anomaly",
            explanation="window-triggered cloud analysis",
            frame_scores=[0.9 for _ in frames],
            processing_time_ms=12,
            raw_response={"success": True},
        )


def _make_frame(frame_id: int, score: float, now: float) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        timestamp=frame_id * 0.2,
        color_base64=f"frame-{frame_id}",
        depth_base64=None,
        jigsaw_score=score,
        spatial_score=score,
        temporal_score=score,
        is_anomalous=score < 0.4,
        received_time=now,
    )


def test_frame_callback_is_called_after_add_frame_result():
    callback_calls = []
    aggregator = ResultAggregator(
        cloud_client=None,
        window_size_seconds=5.0,
        window_threshold=0.3,
        jigsaw_threshold=0.4,
        frame_callback=lambda frame, fusion: callback_calls.append((frame.frame_id, fusion.final_decision)),
    )

    aggregator.add_frame_result(_make_frame(1, 0.8, time.time()))

    assert callback_calls == [(1, "normal")]


def test_deep_analysis_runs_without_holding_lock_and_updates_result():
    cloud_client = FakeCloudClient()
    aggregator = ResultAggregator(
        cloud_client=cloud_client,
        window_size_seconds=5.0,
        window_threshold=0.3,
        jigsaw_threshold=0.4,
    )

    now = time.time()
    results = []
    for idx, score in enumerate([0.2, 0.3, 0.2, 0.8], start=1):
        results.append(aggregator.add_frame_result(_make_frame(idx, score, now + idx * 0.01)))

    assert len(cloud_client.calls) >= 1
    assert any(result.deep_analysis is not None for result in results)
    assert any(result.final_decision == "confirmed_anomaly" for result in results)
    assert aggregator.get_status()["deep_analysis_in_progress"] is False
