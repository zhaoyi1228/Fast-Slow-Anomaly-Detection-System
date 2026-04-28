"""
Edge Receiver
端侧数据接收服务 - Flask服务，接收机器狗发送的帧数据和检测结果
"""

import time
import threading
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FrameResult:
    """帧检测结果"""
    frame_id: int
    timestamp: float
    color_base64: str
    depth_base64: Optional[str]
    anomaly_score: float  # 异常分数（支持 jigsaw_score 或 anomaly_score）
    spatial_score: float
    temporal_score: float
    is_anomalous: bool
    received_time: float
    detector_type: Optional[str] = None  # 检测器类型


class EdgeReceiver:
    """端侧数据接收服务"""

    def __init__(self, port: int = None, frame_buffer=None, aggregator=None):
        """
        初始化接收服务

        Args:
            port: 服务端口
            frame_buffer: 帧缓冲管理器
            aggregator: 结果聚合器
        """
        from config import EDGE_RECEIVER

        self.port = port or EDGE_RECEIVER["port"]
        self.app = Flask(__name__)
        self.frame_buffer = frame_buffer
        self.aggregator = aggregator

        # 统计
        self.stats = {
            "total_frames_received": 0,
            "anomalous_frames_received": 0,
            "start_time": None,
        }

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/api/edge/frame', methods=['POST'])
        def receive_frame():
            """接收单帧数据"""
            try:
                data = request.get_json()

                # 验证必要字段
                required_fields = ['frame_id', 'timestamp', 'color_base64']
                for field in required_fields:
                    if field not in data:
                        return jsonify({"error": f"缺少字段: {field}"}), 400

                # 获取异常分数（兼容 anomaly_score 和 jigsaw_score）
                anomaly_score = data.get('anomaly_score', data.get('jigsaw_score', 0.0))

                # 创建帧结果
                frame_result = FrameResult(
                    frame_id=data['frame_id'],
                    timestamp=data['timestamp'],
                    color_base64=data['color_base64'],
                    depth_base64=data.get('depth_base64'),
                    anomaly_score=anomaly_score,
                    spatial_score=data.get('spatial_score', 1.0),
                    temporal_score=data.get('temporal_score', 1.0),
                    is_anomalous=data.get('is_anomalous', False),
                    received_time=time.time(),
                    detector_type=data.get('detector_type', 'jigsaw')
                )

                # 添加到帧缓冲
                if self.frame_buffer:
                    self.frame_buffer.add_frame(frame_result)

                # 添加到聚合器
                if self.aggregator:
                    self.aggregator.add_frame_result(frame_result)

                # 更新统计
                self.stats["total_frames_received"] += 1
                if frame_result.is_anomalous:
                    self.stats["anomalous_frames_received"] += 1

                return jsonify({
                    "status": "success",
                    "frame_id": frame_result.frame_id,
                    "anomaly_score": frame_result.anomaly_score,
                    "detector_type": frame_result.detector_type,
                    "buffer_size": self.frame_buffer.size() if self.frame_buffer else 0
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/edge/status', methods=['GET'])
        def get_status():
            """获取接收服务状态"""
            buffer_status = {}
            if self.frame_buffer:
                buffer_status = self.frame_buffer.get_status()

            aggregator_status = {}
            if self.aggregator:
                aggregator_status = self.aggregator.get_status()

            elapsed = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0

            return jsonify({
                "status": "running",
                "port": self.port,
                "uptime_seconds": elapsed,
                "frames_received": self.stats["total_frames_received"],
                "anomalous_frames": self.stats["anomalous_frames_received"],
                "buffer": buffer_status,
                "aggregator": aggregator_status,
            })

        @self.app.route('/api/edge/frames', methods=['GET'])
        def get_frames():
            """获取缓存的帧列表"""
            if not self.frame_buffer:
                return jsonify({"error": "帧缓冲未启用"}), 400

            seconds = request.args.get('seconds', default=5.0, type=float)
            frames = self.frame_buffer.get_frames(seconds)

            return jsonify({
                "count": len(frames),
                "frames": [{
                    "frame_id": f.frame_id,
                    "timestamp": f.timestamp,
                    "anomaly_score": f.anomaly_score,
                    "detector_type": f.detector_type,
                    "is_anomalous": f.is_anomalous,
                } for f in frames]
            })

        @self.app.route('/api/edge/reset', methods=['POST'])
        def reset():
            """重置接收服务"""
            if self.frame_buffer:
                self.frame_buffer.clear()
            if self.aggregator:
                self.aggregator.reset()

            self.stats["total_frames_received"] = 0
            self.stats["anomalous_frames_received"] = 0

            return jsonify({"status": "ok"})

    def run(self):
        """启动服务"""
        self.stats["start_time"] = time.time()
        print(f"端侧接收服务启动，端口: {self.port}")
        self.app.run(host="0.0.0.0", port=self.port, threaded=True)


def main():
    """测试接收服务"""
    receiver = EdgeReceiver()
    receiver.run()


if __name__ == "__main__":
    main()