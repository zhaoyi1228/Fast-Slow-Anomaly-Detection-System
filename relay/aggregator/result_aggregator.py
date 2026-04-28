"""
Result Aggregator
结果聚合器 - 聚合Jigsaw快速检测和云侧Agent深度分析结果
"""

import time
import threading
from collections import deque
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from receiver.edge_receiver import FrameResult
from sender.cloud_client import DeepAnalysisResult


@dataclass
class WindowStatus:
    """滑动窗口状态"""
    window_start_time: float
    total_frames: int
    anomalous_frames: int
    anomaly_ratio: float
    should_trigger_deep_analysis: bool


@dataclass
class FusionResult:
    """融合检测结果"""
    frame_id: int
    timestamp: float
    anomaly_score: float  # 异常分数
    spatial_score: float = 0.0
    temporal_score: float = 0.0
    window_status: Optional[WindowStatus] = None
    deep_analysis: Optional[DeepAnalysisResult] = None
    deterrence_text: Optional[str] = None
    final_decision: str = "normal"  # normal, suspicious, confirmed_anomaly


class DeterrenceGenerator:
    """劝阻文本生成器"""

    def __init__(self):
        from config import DETERRENCE_TEMPLATES, ANOMALY_KEYWORDS
        self.templates = DETERRENCE_TEMPLATES
        self.keywords = ANOMALY_KEYWORDS

    def generate(self, anomaly_type: str, description: str, explanation: str) -> str:
        """根据异常类型生成劝阻文本"""
        anomaly_type_lower = anomaly_type.lower()

        # 精确匹配
        if anomaly_type_lower in self.templates:
            return self.templates[anomaly_type_lower]

        # 关键词匹配
        combined_text = f"{anomaly_type} {description} {explanation}".lower()
        for category, keyword_list in self.keywords.items():
            for keyword in keyword_list:
                if keyword.lower() in combined_text:
                    return self.templates.get(category, self.templates["default"])

        return self.templates["default"]


class ResultAggregator:
    """结果聚合器主类"""

    def __init__(self, cloud_client=None, window_size_seconds: float = None,
                 window_threshold: float = None, jigsaw_threshold: float = None,
                 frame_callback=None, deep_analysis_callback=None):
        """
        初始化聚合器

        Args:
            cloud_client: 云侧客户端
            window_size_seconds: 滑动窗口大小
            window_threshold: 窗口内异常帧比例阈值
            jigsaw_threshold: Jigsaw分数阈值
            frame_callback: 帧处理完成回调
            deep_analysis_callback: 深度分析完成回调，传递(frames_to_analyze, fusion_result)
        """
        from config import FUSION_CONFIG

        self.cloud_client = cloud_client
        self.window_size = window_size_seconds or FUSION_CONFIG["window_size_seconds"]
        self.window_threshold = window_threshold or FUSION_CONFIG["window_threshold_percent"]
        self.jigsaw_threshold = jigsaw_threshold or FUSION_CONFIG["jigsaw_threshold"]
        self.deep_analysis_interval = FUSION_CONFIG["deep_analysis_min_interval"]

        # 滑动窗口
        self._window_buffer: deque = deque()

        # 劝阻文本生成器
        self.deterrence_gen = DeterrenceGenerator()

        # 状态
        self._lock = threading.Lock()
        self._deep_analysis_in_progress = False
        self._last_deep_analysis_time = 0
        self._frame_callback = frame_callback
        self._deep_analysis_callback = deep_analysis_callback

        # 统计
        self.stats = {
            "total_frames": 0,
            "deep_analyses_triggered": 0,
            "confirmed_anomalies": 0,
        }

        # 结果历史
        self._result_history: List[FusionResult] = []

    def add_frame_result(self, frame_result: FrameResult) -> FusionResult:
        """
        添加帧结果并处理

        Args:
            frame_result: 端侧发送的帧检测结果

        Returns:
            FusionResult: 融合检测结果
        """
        frames_to_analyze = None
        with self._lock:
            self.stats["total_frames"] += 1

            # 添加到滑动窗口
            self._window_buffer.append(frame_result)
            self._evict_old_frames()

            # 获取窗口状态
            window_status = self._get_window_status()

            # 创建融合结果
            fusion_result = FusionResult(
                frame_id=frame_result.frame_id,
                timestamp=frame_result.timestamp,
                anomaly_score=frame_result.anomaly_score,
                spatial_score=frame_result.spatial_score,
                temporal_score=frame_result.temporal_score,
                window_status=window_status
            )

            # 判断是否触发深度分析
            current_time = time.time()
            should_analyze = (
                window_status.should_trigger_deep_analysis and
                not self._deep_analysis_in_progress and
                (current_time - self._last_deep_analysis_time) > self.deep_analysis_interval
            )

            if should_analyze and self.cloud_client:
                self._deep_analysis_in_progress = True
                frames_to_analyze = self._select_frames_for_deep_analysis()

            # 确定最终决策
            fusion_result.final_decision = self._determine_final_decision(fusion_result)

            # 保存到历史
            self._result_history.append(fusion_result)

        if frames_to_analyze:
            fusion_result = self._perform_deep_analysis(fusion_result, frames_to_analyze)
            with self._lock:
                fusion_result.final_decision = self._determine_final_decision(fusion_result)
                if self._result_history and self._result_history[-1].frame_id == fusion_result.frame_id:
                    self._result_history[-1] = fusion_result

        if self._frame_callback:
            try:
                self._frame_callback(frame_result, fusion_result)
            except Exception as e:
                print(f"frame callback failed: {e}")

        return fusion_result

    def _evict_old_frames(self):
        """移除超出时间窗口的帧"""
        current_time = time.time()
        cutoff_time = current_time - self.window_size

        while self._window_buffer and self._window_buffer[0].received_time < cutoff_time:
            self._window_buffer.popleft()

    def _get_window_status(self) -> WindowStatus:
        """获取窗口状态"""
        if not self._window_buffer:
            return WindowStatus(
                window_start_time=time.time(),
                total_frames=0,
                anomalous_frames=0,
                anomaly_ratio=0.0,
                should_trigger_deep_analysis=False
            )

        total = len(self._window_buffer)
        anomalous = sum(1 for f in self._window_buffer if f.anomaly_score < self.jigsaw_threshold)
        ratio = anomalous / total if total > 0 else 0.0

        return WindowStatus(
            window_start_time=self._window_buffer[0].received_time,
            total_frames=total,
            anomalous_frames=anomalous,
            anomaly_ratio=ratio,
            should_trigger_deep_analysis=ratio >= self.window_threshold
        )

    def _select_frames_for_deep_analysis(self) -> List[FrameResult]:
        from config import FUSION_CONFIG

        anomalous_frames = [f for f in self._window_buffer if f.anomaly_score < self.jigsaw_threshold]
        batch_size = FUSION_CONFIG["deep_analysis_batch_size"]
        if len(anomalous_frames) >= batch_size:
            return anomalous_frames[-batch_size:]
        window_frames = list(self._window_buffer)
        return window_frames[-batch_size:]

    def _perform_deep_analysis(self, fusion_result: FusionResult, frames_to_analyze: List[FrameResult]) -> FusionResult:
        """执行深度分析"""

        try:
            # 转换为云侧客户端需要的格式
            frames_data = [
                {
                    "frame_id": f.frame_id,
                    "image_base64": f.color_base64,
                    "timestamp": f.timestamp,
                    "anomaly_score": f.anomaly_score,
                }
                for f in frames_to_analyze
            ]

            # 调用云侧分析
            deep_result = self.cloud_client.detect(frames_data)
            fusion_result.deep_analysis = deep_result

            # 生成劝阻文本
            if deep_result.is_anomaly:
                fusion_result.deterrence_text = self.deterrence_gen.generate(
                    deep_result.anomaly_type,
                    deep_result.description,
                    deep_result.explanation
                )
                self.stats["confirmed_anomalies"] += 1

            # 调用深度分析回调，传递帧列表和结果
            if self._deep_analysis_callback and frames_to_analyze:
                try:
                    self._deep_analysis_callback(frames_to_analyze, fusion_result)
                except Exception as e:
                    print(f"深度分析回调失败: {e}")

        except Exception as e:
            print(f"深度分析失败: {e}")

        finally:
            with self._lock:
                self.stats["deep_analyses_triggered"] += 1
                self._deep_analysis_in_progress = False
                self._last_deep_analysis_time = time.time()

        return fusion_result

    def _determine_final_decision(self, fusion_result: FusionResult) -> str:
        """确定最终决策"""
        # 有深度分析结果
        if fusion_result.deep_analysis:
            if fusion_result.deep_analysis.is_anomaly:
                return "confirmed_anomaly"
            else:
                return "normal"

        # 基于窗口状态
        if fusion_result.window_status:
            if fusion_result.window_status.should_trigger_deep_analysis:
                return "suspicious"
            elif fusion_result.anomaly_score < self.jigsaw_threshold:
                return "suspicious"

        return "normal"

    def get_recent_frames(self, count: int = 5) -> List[FrameResult]:
        """获取最近N帧"""
        with self._lock:
            if len(self._window_buffer) <= count:
                return list(self._window_buffer)
            return list(self._window_buffer)[-count:]

    def get_result_history(self, count: int = None) -> List[FusionResult]:
        """获取结果历史"""
        with self._lock:
            if count:
                return self._result_history[-count:]
            return list(self._result_history)

    def reset(self):
        """重置聚合器"""
        with self._lock:
            self._window_buffer.clear()
            self._result_history.clear()
            self._deep_analysis_in_progress = False
            self._last_deep_analysis_time = 0
            self.stats = {
                "total_frames": 0,
                "deep_analyses_triggered": 0,
                "confirmed_anomalies": 0,
            }

    def get_status(self) -> Dict[str, Any]:
        """获取聚合器状态"""
        with self._lock:
            window_status = self._get_window_status()
            return {
                "window_size_seconds": self.window_size,
                "window_total_frames": window_status.total_frames,
                "window_anomalous_frames": window_status.anomalous_frames,
                "window_anomaly_ratio": window_status.anomaly_ratio,
                "jigsaw_threshold": self.jigsaw_threshold,
                "window_threshold_percent": self.window_threshold,
                "deep_analysis_in_progress": self._deep_analysis_in_progress,
                "stats": self.stats,
            }


def main():
    """测试聚合器"""
    from sender.cloud_client import CloudClient

    cloud_client = CloudClient()
    aggregator = ResultAggregator(cloud_client)

    # 添加测试帧
    for i in range(10):
        frame = FrameResult(
            frame_id=i,
            timestamp=i * 0.2,
            color_base64="test_base64",
            depth_base64=None,
            anomaly_score=0.3 if i < 5 else 0.7,
            spatial_score=0.5,
            temporal_score=0.5,
            is_anomalous=i < 5,
            received_time=time.time()
        )
        result = aggregator.add_frame_result(frame)
        print(f"帧{i}: 分数={result.anomaly_score:.2f}, 决策={result.final_decision}")

    print(f"\n状态: {aggregator.get_status()}")
    cloud_client.stop()


if __name__ == "__main__":
    main()