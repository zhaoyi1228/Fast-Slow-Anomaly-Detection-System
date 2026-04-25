"""
Gradio App
Gradio可视化界面 - 实时显示检测结果
"""

import time
import base64
import threading
import numpy as np
import cv2
import gradio as gr
from typing import Dict, Any, List, Optional
from collections import deque


class VisualizationState:
    """可视化状态管理"""

    def __init__(self):
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_jigsaw_score = 1.0
        self.latest_decision = "normal"
        self.latest_deterrence = ""
        self.score_history: deque = deque(maxlen=100)
        self.anomaly_events: List[Dict[str, Any]] = []
        self.deep_analysis_result = None
        self._lock = threading.Lock()

    def update_frame(self, frame_result, fusion_result):
        """更新帧显示"""
        with self._lock:
            # 解码图像
            if frame_result.color_base64:
                self.latest_frame = self._decode_base64(frame_result.color_base64)
            self.latest_frame_id = frame_result.frame_id
            self.latest_jigsaw_score = frame_result.jigsaw_score

            # 更新决策
            self.latest_decision = fusion_result.final_decision
            self.latest_deterrence = fusion_result.deterrence_text or ""

            # 更新分数历史
            self.score_history.append({
                "frame_id": frame_result.frame_id,
                "score": frame_result.jigsaw_score,
                "time": time.time()
            })

            # 更新深度分析结果
            self.deep_analysis_result = fusion_result.deep_analysis

            # 记录异常事件
            if fusion_result.final_decision == "confirmed_anomaly":
                self.anomaly_events.append({
                    "frame_id": frame_result.frame_id,
                    "time": time.strftime("%H:%M:%S"),
                    "type": fusion_result.deep_analysis.anomaly_type if fusion_result.deep_analysis else "unknown",
                    "text": fusion_result.deterrence_text or "异常检测"
                })

    def _decode_base64(self, image_base64: str) -> np.ndarray:
        """解码base64图像"""
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return image

    def get_display_frame(self) -> Optional[np.ndarray]:
        """获取显示帧"""
        with self._lock:
            if self.latest_frame is None:
                return None

            # 复制帧用于添加标注
            display_frame = self.latest_frame.copy()

            # 添加分数标注
            score_text = f"Score: {self.latest_jigsaw_score:.3f}"
            cv2.putText(display_frame, score_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 添加决策标注
            decision_color = {
                "normal": (0, 255, 0),
                "suspicious": (255, 165, 0),
                "confirmed_anomaly": (255, 0, 0),
            }
            decision_text = f"Status: {self.latest_decision.upper()}"
            color = decision_color.get(self.latest_decision, (255, 255, 255))
            cv2.putText(display_frame, decision_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 添加帧ID
            frame_text = f"Frame: {self.latest_frame_id}"
            cv2.putText(display_frame, frame_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            return display_frame

    def get_score_plot_data(self) -> Dict[str, List]:
        """获取分数曲线数据"""
        with self._lock:
            if not self.score_history:
                return {"frames": [], "scores": []}

            frames = [h["frame_id"] for h in self.score_history]
            scores = [h["score"] for h in self.score_history]
            return {"frames": frames, "scores": scores}

    def get_status_text(self) -> str:
        """获取状态文本"""
        with self._lock:
            status = f"帧ID: {self.latest_frame_id}\n"
            status += f"Jigsaw分数: {self.latest_jigsaw_score:.3f}\n"
            status += f"决策: {self.latest_decision}\n"

            if self.deep_analysis_result:
                status += f"\n深度分析:\n"
                status += f"异常类型: {self.deep_analysis_result.anomaly_type}\n"
                status += f"描述: {self.deep_analysis_result.description}\n"

            if self.latest_deterrence:
                status += f"\n劝阻文本:\n{self.latest_deterrence}\n"

            return status

    def get_anomaly_history(self) -> str:
        """获取异常历史"""
        with self._lock:
            if not self.anomaly_events:
                return "暂无异常事件"

            text = "异常事件历史:\n"
            for event in self.anomaly_events[-10:]:  # 最近10个
                text += f"[{event['time']}] Frame {event['frame_id']}: {event['type']}\n"
            return text


class GradioApp:
    """Gradio应用主类"""

    def __init__(self, state: VisualizationState = None, host: str = None, port: int = None):
        """
        初始化Gradio应用

        Args:
            state: 可视化状态对象
            host: 服务主机
            port: 服务端口
        """
        from config import GRADIO_SERVER

        self.state = state or VisualizationState()
        self.host = host or GRADIO_SERVER["host"]
        self.port = port or GRADIO_SERVER["port"]
        self.share = GRADIO_SERVER["share"]

        self.app = None
        self._build_interface()

    def _build_interface(self):
        """构建Gradio界面"""
        with gr.Blocks(title="异常检测监控系统") as self.app:
            gr.Markdown("# 异常检测监控系统")
            gr.Markdown("实时显示来自机器狗的检测结果")

            # 主显示区域
            with gr.Row():
                # 视频显示
                with gr.Column(scale=2):
                    video_output = gr.Image(label="实时视频", height=400)
                    status_output = gr.Textbox(label="检测状态", lines=5)

                # 分数曲线和异常历史
                with gr.Column(scale=1):
                    score_plot = gr.LinePlot(
                        label="Jigsaw分数曲线",
                        x_title="帧ID",
                        y_title="分数",
                        height=300
                    )
                    anomaly_history = gr.Textbox(label="异常历史", lines=10)

            # 控制区域
            with gr.Row():
                refresh_btn = gr.Button("刷新显示", variant="primary")
                reset_btn = gr.Button("重置系统")

            # 定时刷新函数
            def refresh_display():
                frame = self.state.get_display_frame()
                status = self.state.get_status_text()
                history = self.state.get_anomaly_history()
                plot_data = self.state.get_score_plot_data()

                if plot_data["frames"]:
                    plot_value = gr.LinePlotData(
                        x=plot_data["frames"],
                        y=plot_data["scores"]
                    )
                else:
                    plot_value = None

                return frame, status, history, plot_value

            # 绑定刷新按钮
            refresh_btn.click(
                fn=refresh_display,
                outputs=[video_output, status_output, anomaly_history, score_plot]
            )

            # 自动刷新（使用gradio的定时器）
            self.app.load(
                fn=refresh_display,
                outputs=[video_output, status_output, anomaly_history, score_plot],
                every=0.5  # 每0.5秒刷新
            )

    def run(self):
        """启动Gradio服务"""
        print(f"启动Gradio界面: http://{self.host}:{self.port}")
        self.app.launch(
            server_name=self.host,
            server_port=self.port,
            share=self.share
        )


def main():
    """测试Gradio应用"""
    state = VisualizationState()

    # 添加测试数据
    from receiver.edge_receiver import FrameResult
    from aggregator.result_aggregator import FusionResult

    for i in range(5):
        frame = FrameResult(
            frame_id=i,
            timestamp=i * 0.2,
            color_base64="",
            depth_base64=None,
            jigsaw_score=0.5,
            spatial_score=0.3,
            temporal_score=0.7,
            is_anomalous=False,
            received_time=time.time()
        )
        result = FusionResult(
            frame_id=i,
            timestamp=i * 0.2,
            jigsaw_score=0.5,
            final_decision="normal"
        )
        state.update_frame(frame, result)

    app = GradioApp(state)
    app.run()


if __name__ == "__main__":
    main()