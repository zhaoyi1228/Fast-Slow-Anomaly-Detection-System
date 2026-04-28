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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
from matplotlib.figure import Figure
from typing import Dict, Any, List, Optional
from collections import deque


class VisualizationState:
    """可视化状态管理"""

    def __init__(self):
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_anomaly_score = 0.0
        self.latest_decision = "normal"
        self.latest_deterrence = ""
        self.score_history: deque = deque(maxlen=300)  # 缓存更多数据，绘图时筛选
        self.anomaly_events: List[Dict[str, Any]] = []  # 深度分析后的最终判定记录
        self.deep_analysis_result = None
        self.deep_analysis_history: List[Dict[str, Any]] = []  # 最近5条深度分析记录（含帧图像）
        self._last_deep_analysis_key = None
        self._last_event_key = None
        self._lock = threading.Lock()

    def update_frame(self, frame_result, fusion_result):
        """更新帧显示"""
        with self._lock:
            # 解码图像
            if frame_result.color_base64:
                self.latest_frame = self._decode_base64(frame_result.color_base64)
            self.latest_frame_id = frame_result.frame_id
            self.latest_anomaly_score = frame_result.anomaly_score

            # 更新决策
            self.latest_decision = fusion_result.final_decision
            self.latest_deterrence = fusion_result.deterrence_text or ""

            # 更新分数历史
            self.score_history.append({
                "frame_id": frame_result.frame_id,
                "score": frame_result.anomaly_score,
                "time": time.time()
            })

            # 深度分析结果（用于状态显示，详细记录在 update_deep_analysis）
            self.deep_analysis_result = fusion_result.deep_analysis

    def update_deep_analysis(self, frames: List, fusion_result):
        """记录深度分析结果及上传的帧"""
        with self._lock:
            deep_key = fusion_result.frame_id
            if deep_key != self._last_deep_analysis_key:
                # 解码帧图像
                decoded_frames = [self._decode_base64(f.color_base64) for f in frames if f.color_base64]
                frame_ids = [f.frame_id for f in frames]

                self.deep_analysis_history.append({
                    "frames": decoded_frames,
                    "frame_ids": frame_ids,
                    "result": fusion_result.deep_analysis,
                    "time": time.strftime("%H:%M:%S"),
                    "final_decision": "异常确认" if fusion_result.deep_analysis.is_anomaly else "正常确认"
                })
                if len(self.deep_analysis_history) > 5:
                    self.deep_analysis_history.pop(0)

                self._last_deep_analysis_key = deep_key

                # 记录到异常事件历史（只记录云端最终判定）
                self.anomaly_events.append({
                    "time": time.strftime("%H:%M:%S"),
                    "frame_ids": frame_ids,
                    "decision": "异常确认" if fusion_result.deep_analysis.is_anomaly else "正常确认",
                    "anomaly_type": fusion_result.deep_analysis.anomaly_type,
                })

    def reset(self):
        """重置可视化状态"""
        with self._lock:
            self.latest_frame = None
            self.latest_frame_id = 0
            self.latest_anomaly_score = 0.0
            self.latest_decision = "normal"
            self.latest_deterrence = ""
            self.score_history.clear()
            self.anomaly_events.clear()
            self.deep_analysis_result = None
            self.deep_analysis_history.clear()
            self._last_deep_analysis_key = None
            self._last_event_key = None

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
            score_text = f"Score: {self.latest_anomaly_score:.3f}"
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

    def get_score_plot_figure(self):
        """获取分数曲线图像对象"""
        plot_data = self.get_score_plot_data()

        # 使用Figure类直接创建，避免pyplot缓存导致figure堆积
        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")

        # 手动筛选：只显示当前帧及前200帧
        current_frame_id = self.latest_frame_id
        display_start = max(0, current_frame_id - 200)

        if plot_data["frames"]:
            # 筛选范围内的数据
            frames = [f for f in plot_data["frames"] if f >= display_start]
            scores = [plot_data["scores"][i] for i, f in enumerate(plot_data["frames"]) if f >= display_start]

            if frames:
                ax.plot(
                    frames,
                    scores,
                    color="#2563eb",
                    linewidth=2,
                    marker="o",
                    markersize=3,
                )
                # 明确设置 xlim，确保不会连接到原点
                ax.set_xlim(display_start, current_frame_id)
            else:
                ax.text(
                    0.5, 0.5, "暂无数据",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="#6b7280"
                )
        else:
            ax.text(
                0.5, 0.5, "暂无数据",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#6b7280"
            )

        ax.set_title("异常分数曲线")
        ax.set_xlabel("帧ID")
        ax.set_ylabel("分数")
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        return fig

    def get_status_text(self) -> str:
        """获取状态文本（简化）"""
        with self._lock:
            status = f"帧ID: {self.latest_frame_id}\n"
            status += f"异常分数: {self.latest_anomaly_score:.3f}\n"
            status += f"决策: {self.latest_decision}"
            return status

    def get_anomaly_history(self) -> str:
        """获取异常历史（只记录云端最终判定）"""
        with self._lock:
            if not self.anomaly_events:
                return "暂无深度分析记录"

            text = "深度分析记录:\n"
            for event in reversed(self.anomaly_events[-10:]):
                text += f"[{event['time']}] "
                text += f"帧 {event['frame_ids']} "
                text += f"可疑→{event['decision']}: "
                text += f"{event['anomaly_type']}\n"
            return text

    def get_deep_analysis_display(self):
        """获取深度分析模块的显示数据"""
        from typing import Tuple
        with self._lock:
            if not self.deep_analysis_history:
                return [], "暂无深度分析记录"

            latest = self.deep_analysis_history[-1]
            frames = latest["frames"]
            result = latest["result"]

            text = f"【最新深度分析】\n"
            text += f"时间: {latest['time']}\n"
            text += f"帧ID: {latest['frame_ids']}\n"
            text += f"最终判定: {latest['final_decision']}\n"
            text += f"\n分析详情:\n"
            text += f"异常分数: {result.anomaly_score:.3f}\n"
            text += f"异常类型: {result.anomaly_type}\n"
            text += f"描述: {result.description}\n"
            text += f"解释: {result.explanation}\n"

            return frames, text


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

            # 第一行：实时视频 + 检测状态
            with gr.Row():
                with gr.Column(scale=2):
                    video_output = gr.Image(label="实时视频", height=400)
                    status_output = gr.Textbox(label="检测状态", lines=3)

                with gr.Column(scale=1):
                    score_plot = gr.Plot(label="异常分数曲线")

            # 第二行：深度分析模块
            with gr.Row():
                with gr.Column(scale=2):
                    deep_analysis_gallery = gr.Gallery(label="深度分析帧", columns=5, height=200)
                    deep_analysis_result = gr.Textbox(label="深度分析结果", lines=8)

                with gr.Column(scale=1):
                    anomaly_history = gr.Textbox(label="异常历史", lines=10)

            # 第三行：控制按钮
            with gr.Row():
                refresh_btn = gr.Button("刷新显示", variant="primary")
                reset_btn = gr.Button("重置系统")

            # 定时刷新函数
            def refresh_display():
                frame = self.state.get_display_frame()
                status = self.state.get_status_text()
                history = self.state.get_anomaly_history()
                plot_figure = self.state.get_score_plot_figure()
                deep_frames, deep_text = self.state.get_deep_analysis_display()
                return frame, status, history, plot_figure, deep_frames, deep_text

            def reset_display():
                self.state.reset()
                return refresh_display()

            # 绑定刷新按钮
            refresh_btn.click(
                fn=refresh_display,
                outputs=[video_output, status_output, anomaly_history, score_plot, deep_analysis_gallery, deep_analysis_result]
            )
            reset_btn.click(
                fn=reset_display,
                outputs=[video_output, status_output, anomaly_history, score_plot, deep_analysis_gallery, deep_analysis_result]
            )

            # 页面首次加载
            self.app.load(
                fn=refresh_display,
                outputs=[video_output, status_output, anomaly_history, score_plot, deep_analysis_gallery, deep_analysis_result],
            )

            # 自动刷新：优先使用 Timer，旧版本回退到 load(every=...)
            if hasattr(gr, "Timer"):
                timer = gr.Timer(value=0.5, active=True)
                timer.tick(
                    fn=refresh_display,
                    outputs=[video_output, status_output, anomaly_history, score_plot, deep_analysis_gallery, deep_analysis_result],
                )
            else:
                try:
                    self.app.load(
                        fn=refresh_display,
                        outputs=[video_output, status_output, anomaly_history, score_plot, deep_analysis_gallery, deep_analysis_result],
                        every=0.5
                    )
                except TypeError:
                    print("警告: 当前 gradio 版本不支持自动刷新，页面将仅支持手动刷新。")

    def run(self):
        """启动Gradio服务"""
        print(f"启动 Gradio 界面: http://{self.host}:{self.port}")
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
            anomaly_score=0.5,
            spatial_score=0.3,
            temporal_score=0.7,
            is_anomalous=False,
            received_time=time.time()
        )
        result = FusionResult(
            frame_id=i,
            timestamp=i * 0.2,
            anomaly_score=0.5,
            final_decision="normal"
        )
        state.update_frame(frame, result)

    app = GradioApp(state)
    app.run()


if __name__ == "__main__":
    main()