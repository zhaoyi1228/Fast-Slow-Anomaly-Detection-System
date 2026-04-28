"""
Frame Buffer
帧缓冲管理 - 管理从端侧接收的帧数据，支持滑动窗口和帧检索
"""

import time
import threading
from collections import deque
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .edge_receiver import FrameResult


class FrameBuffer:
    """帧缓冲管理器"""

    def __init__(self, max_frames: int = None, frame_ttl: float = None):
        """
        初始化帧缓冲

        Args:
            max_frames: 最大缓存帧数
            frame_ttl: 帧过期时间（秒）
        """
        from config import FRAME_BUFFER_CONFIG

        self.max_frames = max_frames or FRAME_BUFFER_CONFIG["max_frames"]
        self.frame_ttl = frame_ttl or FRAME_BUFFER_CONFIG["frame_ttl"]

        self._buffer: deque = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

        # 清理线程
        self._cleanup_thread = None
        self._cleanup_stop = threading.Event()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """启动清理线程"""
        from config import FRAME_BUFFER_CONFIG

        cleanup_interval = FRAME_BUFFER_CONFIG["cleanup_interval"]

        def cleanup_loop():
            while not self._cleanup_stop.is_set():
                time.sleep(cleanup_interval)
                self._cleanup_expired()

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def add_frame(self, frame_result: FrameResult):
        """添加帧到缓冲区"""
        with self._lock:
            self._buffer.append(frame_result)

    def get_frames(self, seconds: float = None) -> List[FrameResult]:
        """
        获取最近N秒内的帧

        Args:
            seconds: 时间范围（秒），默认返回全部

        Returns:
            List[FrameResult]: 帧列表
        """
        with self._lock:
            if seconds is None:
                return list(self._buffer)

            current_time = time.time()
            cutoff_time = current_time - seconds

            return [f for f in self._buffer if f.received_time >= cutoff_time]

    def get_anomalous_frames(self, threshold: float = None, count: int = None) -> List[FrameResult]:
        """
        获取异常帧

        Args:
            threshold: Jigsaw分数阈值
            count: 返回帧数

        Returns:
            List[FrameResult]: 异常帧列表
        """
        from config import FUSION_CONFIG

        threshold = threshold or FUSION_CONFIG["jigsaw_threshold"]

        with self._lock:
            anomalous = [f for f in self._buffer if f.anomaly_score < threshold]

            if count:
                return anomalous[-count:]
            return anomalous

    def get_recent_frames(self, count: int = 5) -> List[FrameResult]:
        """获取最近N帧"""
        with self._lock:
            if len(self._buffer) <= count:
                return list(self._buffer)
            return list(self._buffer)[-count:]

    def _cleanup_expired(self):
        """清理过期帧"""
        current_time = time.time()
        cutoff_time = current_time - self.frame_ttl

        with self._lock:
            while self._buffer and self._buffer[0].received_time < cutoff_time:
                self._buffer.popleft()

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()

    def size(self) -> int:
        """获取缓冲区大小"""
        with self._lock:
            return len(self._buffer)

    def get_status(self) -> Dict[str, Any]:
        """获取缓冲区状态"""
        with self._lock:
            if not self._buffer:
                return {
                    "size": 0,
                    "oldest_frame_time": None,
                    "newest_frame_time": None,
                }

            return {
                "size": len(self._buffer),
                "max_size": self.max_frames,
                "oldest_frame_time": self._buffer[0].received_time,
                "newest_frame_time": self._buffer[-1].received_time,
                "frame_ttl": self.frame_ttl,
            }

    def stop(self):
        """停止缓冲管理器"""
        self._cleanup_stop.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)


def main():
    """测试帧缓冲"""
    buffer = FrameBuffer(max_frames=10)

    # 添加测试帧
    for i in range(5):
        frame = FrameResult(
            frame_id=i,
            timestamp=i * 0.2,
            color_base64="test",
            depth_base64=None,
            anomaly_score=0.3 if i % 2 == 0 else 0.7,
            spatial_score=0.5,
            temporal_score=0.5,
            is_anomalous=i % 2 == 0,
            received_time=time.time()
        )
        buffer.add_frame(frame)

    print(f"缓冲区大小: {buffer.size()}")
    print(f"异常帧数: {len(buffer.get_anomalous_frames())}")
    print(f"最近2秒帧数: {len(buffer.get_frames(seconds=2))}")

    buffer.stop()


if __name__ == "__main__":
    main()