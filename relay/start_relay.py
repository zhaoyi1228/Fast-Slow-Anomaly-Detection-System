"""
Start Relay Services
中间节点启动脚本 - 同时启动接收服务、云侧客户端和Gradio界面
"""

import os
import sys
import time
import signal
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EDGE_RECEIVER, CLOUD_SERVER, GRADIO_SERVER
from typing import Dict, Any
from receiver.edge_receiver import EdgeReceiver
from receiver.frame_buffer import FrameBuffer
from sender.cloud_client import CloudClient
from aggregator.result_aggregator import ResultAggregator
from visualization.gradio_app import GradioApp, VisualizationState


class RelayNode:
    """中间节点主类"""

    def __init__(self, edge_port: int = None, cloud_host: str = None,
                 cloud_port: int = None, gradio_port: int = None):
        """
        初始化中间节点

        Args:
            edge_port: 端侧接收端口
            cloud_host: 云侧服务器IP
            cloud_port: 云侧服务端口
            gradio_port: Gradio界面端口
        """
        self.edge_port = edge_port or EDGE_RECEIVER["port"]
        self.cloud_host = cloud_host or CLOUD_SERVER["host"]
        self.cloud_port = cloud_port or CLOUD_SERVER["port"]
        self.gradio_port = gradio_port or GRADIO_SERVER["port"]

        # 初始化组件
        self.cloud_client = CloudClient(self.cloud_host, self.cloud_port)
        self.frame_buffer = FrameBuffer()
        self.state = VisualizationState()
        self.aggregator = ResultAggregator(
            cloud_client=self.cloud_client,
            frame_callback=lambda frame, result: self.state.update_frame(frame, result),
            deep_analysis_callback=lambda frames, result: self.state.update_deep_analysis(frames, result),
        )

        self.receiver = EdgeReceiver(
            port=self.edge_port,
            frame_buffer=self.frame_buffer,
            aggregator=self.aggregator
        )

        self.gradio_app = GradioApp(
            state=self.state,
            port=self.gradio_port
        )

        # 状态
        self.is_running = False
        self._threads = []

    def _keep_alive_without_gradio(self):
        """当 Gradio 启动失败时，保持接收/转发链路继续运行。"""
        print("Relay 将继续以无界面模式运行。")
        print(f"端侧仍可发送到: http://127.0.0.1:{self.edge_port}/api/edge/status")
        print("按 Ctrl+C 停止中间节点。")
        while self.is_running:
            time.sleep(1.0)

    def start(self):
        """启动所有服务"""
        print("=" * 50)
        print("中间节点启动")
        print("=" * 50)
        print(f"端侧接收端口: {self.edge_port}")
        print(f"云侧服务器: {self.cloud_host}:{self.cloud_port}")
        print(f"Gradio界面: http://127.0.0.1:{self.gradio_port}")
        print("=" * 50)

        self.is_running = True

        # 启动接收服务（后台线程）
        def run_receiver():
            self.receiver.run()

        receiver_thread = threading.Thread(target=run_receiver, daemon=True)
        receiver_thread.start()
        self._threads.append(receiver_thread)

        # 等待接收服务启动
        time.sleep(1.0)

        # 检查云侧连接
        cloud_ok = self.cloud_client.health_check()
        print(f"\n云侧连接状态: {'正常' if cloud_ok else '异常（请检查VPN连接）'}")

        # 启动Gradio（主线程）
        print("\n启动Gradio界面...")
        try:
            self.gradio_app.run()
        except Exception as exc:
            print("\n警告: Gradio界面启动失败，已切换为无界面模式。")
            print(f"Gradio错误: {exc}")
            self._keep_alive_without_gradio()

    def stop(self):
        """停止所有服务"""
        print("\n停止中间节点...")
        self.is_running = False

        # 停止各组件
        self.frame_buffer.stop()
        self.cloud_client.stop()
        self.aggregator.reset()

        # 打印统计
        print(f"\n接收帧数: {self.receiver.stats['total_frames_received']}")
        print(f"异常帧数: {self.receiver.stats['anomalous_frames_received']}")
        print(f"深度分析触发次数: {self.aggregator.stats['deep_analyses_triggered']}")
        print(f"确认异常次数: {self.aggregator.stats['confirmed_anomalies']}")

    def get_status(self) -> Dict[str, Any]:
        """获取节点状态"""
        return {
            "is_running": self.is_running,
            "edge_port": self.edge_port,
            "cloud_connected": self.cloud_client.get_status().get("is_connected", False),
            "buffer_size": self.frame_buffer.size(),
            "aggregator": self.aggregator.get_status(),
            "receiver_stats": self.receiver.stats,
        }


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='中间节点启动')
    parser.add_argument('--edge-port', type=int, default=EDGE_RECEIVER["port"],
                        help='端侧接收端口')
    parser.add_argument('--cloud-host', type=str, default=CLOUD_SERVER["host"],
                        help='云侧服务器IP')
    parser.add_argument('--cloud-port', type=int, default=CLOUD_SERVER["port"],
                        help='云侧服务端口')
    parser.add_argument('--gradio-port', type=int, default=GRADIO_SERVER["port"],
                        help='Gradio界面端口')

    args = parser.parse_args()

    # 创建中间节点
    node = RelayNode(
        edge_port=args.edge_port,
        cloud_host=args.cloud_host,
        cloud_port=args.cloud_port,
        gradio_port=args.gradio_port
    )

    # 注册信号处理
    def signal_handler(sig, frame):
        node.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动
    node.start()


if __name__ == "__main__":
    main()