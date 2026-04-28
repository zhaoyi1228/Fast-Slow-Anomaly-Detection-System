"""
Run Edge Detection
端侧主运行脚本 - 机器狗端异常检测主入口

功能：
1. 采集RealSense相机视频流
2. 本地Jigsaw快速检测
3. 发送检测结果到中间机器
"""

import os
import sys
import time
import signal
import argparse
import threading
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RELAY_SERVER, JIGSAW_SERVER, DETECTION_CONFIG, LOG_CONFIG, MAE_SERVER
from camera.realsense_client import RealSenseClient, FrameData
from communication.relay_client import RelayClient
from detection.jigsaw_service import create_jigsaw_client
from detection.mae_service import create_mae_client


class EdgeDetector:
    """端侧检测器主类"""

    def __init__(self, relay_host: str = None, relay_port: int = None,
                 detector_type: str = 'jigsaw',
                 detector_host: str = None, detector_port: int = None):
        """
        初始化端侧检测器

        Args:
            relay_host: 中间机器IP
            relay_port: 中间机器端口
            detector_type: 检测器类型 ('jigsaw' 或 'mae')
            detector_host: 检测服务主机（默认本地）
            detector_port: 检测服务端口（默认根据类型选择）
        """
        # 检测器类型
        self.detector_type = detector_type

        # 通信客户端
        self.relay_client = RelayClient(relay_host, relay_port)

        # 根据检测器类型确定默认host和port，创建客户端
        if detector_type == 'mae':
            host = detector_host or MAE_SERVER["host"]
            port = detector_port or MAE_SERVER["port"]
            self.detector_client = create_mae_client(host, port)
            self.detector_name = "MAE"
        else:
            host = detector_host or JIGSAW_SERVER["host"]
            port = detector_port or JIGSAW_SERVER["port"]
            self.detector_client = create_jigsaw_client(host, port)
            self.detector_name = "Jigsaw"

        self.detector_host = host
        self.detector_port = port

        # RealSense相机
        self.camera = RealSenseClient()

        # 状态
        self.is_running = False
        self.frame_counter = 0
        self.start_time = None
        self._stop_event = threading.Event()

        # 统计
        self.stats = {
            "total_frames": 0,
            "detector_success": 0,
            "detector_failed": 0,
            "relay_success": 0,
            "relay_cached": 0,
            "relay_failed": 0,
        }

    def start(self):
        """启动检测"""
        print("=" * 50)
        print("端侧异常检测系统启动")
        print(f"检测器类型: {self.detector_name}")
        print("=" * 50)

        # 检查连接
        print(f"\n检查{self.detector_name}服务 ({self.detector_host}:{self.detector_port})...")
        detector_ok = self.detector_client.health_check()
        print(f"{self.detector_name}服务状态: {'正常' if detector_ok else '异常'}")

        print(f"\n检查中间机器 ({self.relay_client.host}:{self.relay_client.port})...")
        relay_ok = self.relay_client.health_check()
        print(f"中间机器状态: {'正常' if relay_ok else '异常'}")

        if not detector_ok:
            print(f"\n警告: {self.detector_name}服务未就绪，请先启动服务")
            print("启动命令: python start_edge.py --start-service --detector-type {self.detector_type}")

        print("\n启动相机采集...")
        self.is_running = True
        self.start_time = time.time()

        # 启动流式采集
        self.camera.start_streaming(self._process_frame, DETECTION_CONFIG["fps_sample"])

        print(f"检测已启动，采样频率: {DETECTION_CONFIG['fps_sample']} fps")
        print("按 Ctrl+C 停止检测\n")

    def _process_frame(self, frame_data: FrameData):
        """
        处理单帧数据

        Args:
            frame_data: 帧数据
        """
        if not self.is_running:
            return

        self.stats["total_frames"] += 1
        self.frame_counter += 1

        # 步骤1: 本地检测
        detector_result = self.detector_client.detect_single(frame_data.color_base64)

        if detector_result.get("error"):
            self.stats["detector_failed"] += 1
            print(f"[{self.frame_counter}] {self.detector_name}检测失败: {detector_result['error']}")
            return

        self.stats["detector_success"] += 1

        # 根据检测器类型处理结果
        if self.detector_type == 'mae':
            anomaly_score = detector_result.get("anomaly_score", 0.0)
            teacher_score = detector_result.get("teacher_score", 0.0)
            st_score = detector_result.get("student_teacher_score", 0.0)
            buffer_ready = detector_result.get("buffer_ready", False)
            # MAE分数越高越异常，阈值判断由下游决定
            is_anomalous = anomaly_score > DETECTION_CONFIG["anomaly_threshold"]  # 临时阈值，实际由下游判断
            status = "异常?" if is_anomalous else "正常"
            if not buffer_ready:
                status = "缓冲未就绪"
        else:
            anomaly_score = detector_result.get("anomaly_score", 1.0)
            spatial_score = detector_result.get("spatial_score", 1.0)
            temporal_score = detector_result.get("temporal_score", 1.0)
            # 分数越低越异常
            is_anomalous = anomaly_score < DETECTION_CONFIG["anomaly_threshold"]
            status = "异常?" if is_anomalous else "正常"

        # 步骤2: 发送到中间机器
        send_data = {
            "frame_id": frame_data.frame_id,
            "timestamp": frame_data.timestamp,
            "color_base64": frame_data.color_base64,
            "depth_base64": frame_data.depth_base64,
            "detector_type": self.detector_type,
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
        }

        # 根据检测器类型添加额外信息
        if self.detector_type == 'mae':
            send_data["teacher_score"] = teacher_score
            send_data["student_teacher_score"] = st_score
            send_data["buffer_ready"] = buffer_ready
        else:
            send_data["spatial_score"] = spatial_score
            send_data["temporal_score"] = temporal_score

        send_result = self.relay_client.send_frame(send_data)

        if send_result.get("status") == "success":
            self.stats["relay_success"] += 1
        elif send_result.get("status") == "cached":
            self.stats["relay_cached"] += 1
        else:
            self.stats["relay_failed"] += 1

        # 打印状态
        elapsed = time.time() - self.start_time
        fps = self.stats["total_frames"] / elapsed if elapsed > 0 else 0
        relay_status_info = self.relay_client.get_status()
        relay_status = relay_status_info.get("is_connected", False)
        cached_frames = relay_status_info.get("cached_frames", 0)

        print(f"[{self.frame_counter}] {self.detector_name}={anomaly_score:.3f} ({status}) "
              f"| Relay={'连接' if relay_status else '断开'} "
              f"| FPS={fps:.1f} | 缓存={cached_frames}")

        # 如果检测到异常，打印警告
        if is_anomalous:
            print(f"  >>> 潜在异常帧! 分数: {anomaly_score:.3f}")

    def stop(self):
        """停止检测"""
        print("\n停止检测...")
        self.is_running = False
        self._stop_event.set()

        # 停止相机
        self.camera.stop_streaming()

        # 尝试发送缓存的帧
        cached_sent = self.relay_client.flush_cache()
        print(f"发送缓存帧: {cached_sent}")

        # 停止通信客户端
        self.relay_client.stop()

        # 打印统计
        print("\n" + "=" * 50)
        print("检测统计:")
        print(f"  检测器类型: {self.detector_name}")
        print(f"  总帧数: {self.stats['total_frames']}")
        print(f"  检测器成功: {self.stats['detector_success']}")
        print(f"  检测器失败: {self.stats['detector_failed']}")
        print(f"  Relay成功: {self.stats['relay_success']}")
        print(f"  Relay缓存: {self.stats['relay_cached']}")
        print(f"  Relay失败: {self.stats['relay_failed']}")
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"  运行时间: {elapsed:.1f}秒")
        print("=" * 50)

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "is_running": self.is_running,
            "detector_type": self.detector_type,
            "detector_name": self.detector_name,
            "detector_host": self.detector_host,
            "detector_port": self.detector_port,
            "frame_counter": self.frame_counter,
            "camera": self.camera.get_status(),
            "relay": self.relay_client.get_status(),
            "detector_connected": self.detector_client.health_check(),
            "stats": self.stats,
        }


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='端侧异常检测')
    parser.add_argument('--relay-host', type=str, default=RELAY_SERVER["host"],
                        help='中间机器IP地址')
    parser.add_argument('--relay-port', type=int, default=RELAY_SERVER["port"],
                        help='中间机器端口')
    parser.add_argument('--detector-type', type=str, default='jigsaw',
                        choices=['jigsaw', 'mae'],
                        help='检测器类型 (默认jigsaw)')
    parser.add_argument('--host', type=str, default=None,
                        help='检测服务主机（默认localhost）')
    parser.add_argument('--port', type=int, default=None,
                        help='检测服务端口（默认根据detector-type: jigsaw=8000, mae=8001）')

    args = parser.parse_args()

    # 根据detector-type确定默认端口
    if args.host is None:
        args.host = "localhost"
    if args.port is None:
        if args.detector_type == 'mae':
            args.port = MAE_SERVER["port"]
        else:
            args.port = JIGSAW_SERVER["port"]

    # 创建检测器
    detector = EdgeDetector(
        relay_host=args.relay_host,
        relay_port=args.relay_port,
        detector_type=args.detector_type,
        detector_host=args.host,
        detector_port=args.port
    )

    # 注册信号处理
    def signal_handler(sig, frame):
        detector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动检测
    detector.start()

    # 保持运行
    try:
        while detector.is_running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        detector.stop()


if __name__ == "__main__":
    main()