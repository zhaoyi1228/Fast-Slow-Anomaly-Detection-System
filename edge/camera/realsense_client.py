"""
RealSense Camera Client
RealSense深度相机客户端 - 用于机器狗端侧视频采集
"""

import time
import base64
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class FrameData:
    """帧数据结构"""
    frame_id: int
    timestamp: float
    color_image: np.ndarray
    depth_image: np.ndarray
    color_base64: str
    depth_base64: Optional[str] = None


class RealSenseClient:
    """RealSense相机客户端"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化RealSense相机

        Args:
            config: 相机配置字典，默认使用config.py中的REALSENSE_CONFIG
        """
        from config import REALSENSE_CONFIG
        self.config = config or REALSENSE_CONFIG

        self.pipeline = None
        self.config_rs = None
        self.is_streaming = False
        self.frame_counter = 0
        self.start_time = None

        self._stream_thread = None
        self._callback = None
        self._stop_event = threading.Event()

    def initialize(self) -> bool:
        """
        初始化相机管道

        Returns:
            bool: 是否成功初始化
        """
        try:
            self.pipeline = rs.pipeline()
            self.config_rs = rs.config()

            # 配置彩色流
            if self.config["enable_color"]:
                self.config_rs.enable_stream(
                    rs.stream.color,
                    self.config["resolution"][0],
                    self.config["resolution"][1],
                    rs.format.bgr8,
                    self.config["fps"]
                )

            # 配置深度流
            if self.config["enable_depth"]:
                self.config_rs.enable_stream(
                    rs.stream.depth,
                    self.config["resolution"][0],
                    self.config["resolution"][1],
                    rs.format.z16,
                    self.config["fps"]
                )

            # 启动管道
            self.pipeline.start(self.config_rs)
            self.start_time = time.time()
            return True

        except Exception as e:
            print(f"RealSense初始化失败: {e}")
            return False

    def get_frame(self) -> Optional[FrameData]:
        """
        获取单帧图像

        Returns:
            FrameData: 包含彩色图像、深度图像和base64编码的帧数据
        """
        if not self.pipeline:
            return None

        try:
            frames = self.pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                return None

            # 获取图像数据
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            # 转换为base64
            color_base64 = self._encode_image(color_image)
            depth_base64 = self._encode_depth(depth_image) if depth_image is not None else None

            self.frame_counter += 1
            timestamp = time.time() - self.start_time

            return FrameData(
                frame_id=self.frame_counter,
                timestamp=timestamp,
                color_image=color_image,
                depth_image=depth_image,
                color_base64=color_base64,
                depth_base64=depth_base64
            )

        except Exception as e:
            print(f"获取帧失败: {e}")
            return None

    def _encode_image(self, image: np.ndarray) -> str:
        """将图像编码为base64字符串"""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')

    def _encode_depth(self, depth: np.ndarray) -> str:
        """将深度图编码为base64字符串（可视化后编码）"""
        # 将深度图可视化为彩色图
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        return self._encode_image(depth_colormap)

    def start_streaming(self, callback: Callable[[FrameData], None],
                        sample_fps: int = None):
        """
        启动流式采集，通过回调函数处理每帧

        Args:
            callback: 帧处理回调函数
            sample_fps: 采样帧率，默认使用DETECTION_CONFIG中的fps_sample
        """
        from config import DETECTION_CONFIG

        if not self.initialize():
            raise RuntimeError("相机初始化失败")

        self._callback = callback
        self._stop_event.clear()
        self.is_streaming = True

        sample_fps = sample_fps or DETECTION_CONFIG["fps_sample"]
        actual_fps = self.config["fps"]
        frame_skip = max(1, int(actual_fps // sample_fps))

        def _stream_loop():
            skip_counter = 0
            while not self._stop_event.is_set():
                try:
                    frame_data = self.get_frame()
                    if frame_data:
                        skip_counter += 1
                        if skip_counter >= frame_skip:
                            skip_counter = 0
                            if self._callback:
                                self._callback(frame_data)
                except Exception as e:
                    print(f"流处理错误: {e}")
                    time.sleep(0.1)

        self._stream_thread = threading.Thread(target=_stream_loop, daemon=True)
        self._stream_thread.start()

    def stop_streaming(self):
        """停止流式采集"""
        self._stop_event.set()
        self.is_streaming = False

        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

    def get_status(self) -> Dict[str, Any]:
        """获取相机状态"""
        return {
            "is_streaming": self.is_streaming,
            "frame_count": self.frame_counter,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            "actual_fps": self.frame_counter / (time.time() - self.start_time) if self.start_time and self.frame_counter > 0 else 0,
        }

    def test_camera(self, duration: int = 5) -> bool:
        """
        测试相机是否正常工作

        Args:
            duration: 测试持续时间（秒）

        Returns:
            bool: 测试是否通过
        """
        print("测试RealSense相机...")
        if not self.initialize():
            print("相机初始化失败")
            return False

        frames_received = 0
        test_start = time.time()

        while time.time() - test_start < duration:
            frame = self.get_frame()
            if frame:
                frames_received += 1

        self.pipeline.stop()

        expected_frames = duration * self.config["fps"]
        print(f"测试完成: 接收 {frames_received} 帧，预期约 {expected_frames} 帧")

        return frames_received > expected_frames * 0.8


def main():
    """测试RealSense相机"""
    client = RealSenseClient()
    if client.test_camera(duration=3):
        print("相机测试通过!")
    else:
        print("相机测试失败!")


if __name__ == "__main__":
    main()