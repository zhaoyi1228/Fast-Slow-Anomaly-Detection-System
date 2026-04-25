"""
Jigsaw Service
Jigsaw快速检测服务 - Flask HTTP服务端
从Unified_Jigsaw移植，提供轻量级异常检测
"""

import io
import base64
import time
import threading
import os
import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class JigsawResult:
    """Jigsaw检测结果"""
    anomaly_score: float
    spatial_score: float
    temporal_score: float
    elapsed_ms: float


class JigsawDetector:
    """Jigsaw检测器核心类"""

    def __init__(self, checkpoint_path: str, sample_num: int = 7, gpu_id: int = 0):
        """
        初始化Jigsaw检测器

        Args:
            checkpoint_path: 模型权重路径
            sample_num: 帧数
            gpu_id: GPU ID
        """
        self.sample_num = sample_num
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu")

        self._model = None
        self._model_error: Optional[str] = None
        self._model_lock = threading.Lock()
        self._frame_buffer: List[np.ndarray] = []
        self._buffer_lock = threading.Lock()

        # 加载模型
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """加载模型权重"""
        try:
            from .models.model import WideBranchNet

            spatial_classes = self.sample_num ** 2
            temporal_classes = self.sample_num
            self._model = WideBranchNet(
                time_length=self.sample_num,
                num_classes=[spatial_classes, temporal_classes],
            )

            # 加载权重
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._model.load_state_dict(checkpoint)

            self._model.to(self.device)
            self._model.eval()

            print(f"Jigsaw模型加载成功，设备: {self.device}")
            self._model_error = None

        except ImportError:
            self._model_error = "models.model模块未找到"
            print(f"警告: {self._model_error}")
            self._model = None

        except Exception as e:
            self._model_error = str(e)
            print(f"模型加载失败: {e}")
            self._model = None

    def detect_single(self, image_base64: str) -> JigsawResult:
        """
        检测单张图片

        Args:
            image_base64: base64编码的图像

        Returns:
            JigsawResult: 检测结果
        """
        start_time = time.time()

        # 解码图像
        image = self._decode_base64(image_base64)

        # 添加到帧缓冲区
        with self._buffer_lock:
            self._frame_buffer.append(image)
            if len(self._frame_buffer) > self.sample_num:
                self._frame_buffer = self._frame_buffer[-self.sample_num:]

        # 如果缓冲区帧数不够，返回默认值
        if len(self._frame_buffer) < self.sample_num:
            elapsed_ms = (time.time() - start_time) * 1000
            return JigsawResult(
                anomaly_score=1.0,
                spatial_score=1.0,
                temporal_score=1.0,
                elapsed_ms=elapsed_ms
            )

        # 执行检测
        result = self._run_detection()

        elapsed_ms = (time.time() - start_time) * 1000
        result.elapsed_ms = elapsed_ms

        return result

    def detect_batch(self, images_base64: List[str]) -> List[JigsawResult]:
        """批量检测"""
        results = []
        for img_b64 in images_base64:
            result = self.detect_single(img_b64)
            results.append(result)
        return results

    def _decode_base64(self, image_base64: str) -> np.ndarray:
        """解码base64图像"""
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return image

    def _run_detection(self) -> JigsawResult:
        """执行检测"""
        if self._model is None:
            raise RuntimeError(self._model_error or "Jigsaw model is not loaded")

        with self._model_lock:
            # 预处理帧序列
            frames_tensor = self._preprocess_frames()

            if frames_tensor is None:
                raise RuntimeError("Jigsaw frame preprocessing failed")

            # 推理
            with torch.no_grad():
                frames_tensor = frames_tensor.to(self.device)
                spatial_logits, temporal_logits = self._model(frames_tensor.unsqueeze(0))

                spatial_score = self._compute_spatial_score(spatial_logits)
                temporal_score = self._compute_temporal_score(temporal_logits)

                # ref 验证逻辑更接近保守聚合：两个分支中更差者作为主分数
                anomaly_score = min(spatial_score, temporal_score)

            return JigsawResult(
                anomaly_score=anomaly_score,
                spatial_score=spatial_score,
                temporal_score=temporal_score,
                elapsed_ms=0
            )

    def _compute_spatial_score(self, spatial_logits: torch.Tensor) -> float:
        logits = spatial_logits.reshape(-1, self.sample_num, self.sample_num)
        probs = torch.softmax(logits, dim=-1)
        diagonal = torch.diagonal(probs, dim1=1, dim2=2)
        return diagonal.min(dim=1).values[0].item()

    def _compute_temporal_score(self, temporal_logits: torch.Tensor) -> float:
        probs = torch.softmax(temporal_logits, dim=-1)
        diagonal = torch.diagonal(torch.diag_embed(probs), dim1=1, dim2=2)
        return diagonal.min(dim=1).values[0].item()

    def _preprocess_frames(self) -> Optional[torch.Tensor]:
        """预处理帧序列"""
        try:
            frames = []
            for img in self._frame_buffer:
                # resize到64x64
                img_resized = cv2.resize(img, (64, 64))
                # 转换为RGB并归一化
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                # 转换为CHW格式
                img_chw = np.transpose(img_normalized, (2, 0, 1))
                frames.append(img_chw)

            # 堆叠为 [T, C, H, W] -> [C, T, H, W]
            frames_array = np.stack(frames, axis=0)  # [T, C, H, W]
            frames_tensor = torch.from_numpy(frames_array).float()
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]

            return frames_tensor

        except Exception as e:
            print(f"预处理失败: {e}")
            return None

    def reset_buffer(self):
        """重置帧缓冲区"""
        with self._buffer_lock:
            self._frame_buffer.clear()


# Flask服务
class JigsawService:
    """Jigsaw Flask服务"""

    def __init__(self, port: int = 8000, checkpoint_path: str = None,
                 sample_num: int = 7, gpu_id: int = 0):
        """
        初始化Jigsaw服务

        Args:
            port: 服务端口
            checkpoint_path: 模型权重路径
            sample_num: 帧数
            gpu_id: GPU ID
        """
        from config import JIGSAW_MODEL_CONFIG

        self.port = port
        checkpoint_path = checkpoint_path or JIGSAW_MODEL_CONFIG["checkpoint_path"]
        if checkpoint_path and not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
        sample_num = sample_num or JIGSAW_MODEL_CONFIG["sample_num"]
        gpu_id = gpu_id if gpu_id is not None else JIGSAW_MODEL_CONFIG["gpu_id"]

        self.app = Flask(__name__)
        self.detector = None

        # 初始化检测器
        try:
            self.detector = JigsawDetector(checkpoint_path, sample_num, gpu_id)
        except Exception as e:
            print(f"检测器初始化失败: {e}")

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "ok",
                "model_loaded": self.detector is not None and self.detector._model is not None,
                "device": str(self.detector.device) if self.detector else "none",
                "sample_num": self.detector.sample_num if self.detector else None,
                "model_error": self.detector._model_error if self.detector else "detector not initialized",
            })

        @self.app.route('/detect', methods=['POST'])
        def detect():
            try:
                if self.detector is None or self.detector._model is None:
                    return jsonify({"error": self.detector._model_error if self.detector else "detector not initialized"}), 503

                data = request.get_json()
                image_base64 = data.get('image')

                if not image_base64:
                    return jsonify({"error": "缺少image参数"}), 400

                result = self.detector.detect_single(image_base64)

                return jsonify({
                    "anomaly_score": result.anomaly_score,
                    "spatial_score": result.spatial_score,
                    "temporal_score": result.temporal_score,
                    "elapsed_ms": result.elapsed_ms
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/batch_detect', methods=['POST'])
        def batch_detect():
            try:
                if self.detector is None or self.detector._model is None:
                    return jsonify({"error": self.detector._model_error if self.detector else "detector not initialized"}), 503

                data = request.get_json()
                images = data.get('images', [])

                if not images:
                    return jsonify({"error": "缺少images参数"}), 400

                results = self.detector.detect_batch(images)

                return jsonify({
                    "results": [{
                        "anomaly_score": r.anomaly_score,
                        "spatial_score": r.spatial_score,
                        "temporal_score": r.temporal_score,
                        "elapsed_ms": r.elapsed_ms
                    } for r in results]
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/reset', methods=['POST'])
        def reset():
            if self.detector:
                self.detector.reset_buffer()
            return jsonify({"status": "ok"})

    def run(self, host: str = "0.0.0.0"):
        """启动服务"""
        print(f"启动Jigsaw服务，端口: {self.port}")
        self.app.run(host=host, port=self.port, threaded=True)


def create_jigsaw_client(host: str = "localhost", port: int = 8000):
    """
    创建Jigsaw客户端（用于本地调用）

    Args:
        host: 服务主机
        port: 服务端口

    Returns:
        JigsawClient: 客户端实例
    """
    import requests

    class JigsawClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def detect_single(self, image_base64: str) -> Dict[str, Any]:
            url = f"{self.base_url}/detect"
            try:
                response = requests.post(
                    url,
                    json={"image": image_base64},
                    timeout=5.0
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "anomaly_score": None}

        def health_check(self) -> bool:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2.0)
                return response.status_code == 200
            except:
                return False

    return JigsawClient(f"http://{host}:{port}")


def main():
    """启动Jigsaw服务"""
    import argparse
    parser = argparse.ArgumentParser(description='Jigsaw异常检测服务')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    parser.add_argument('--checkpoint', type=str, default='pre_trained/stc_78.76_sample7.pth',
                        help='模型权重路径')
    parser.add_argument('--sample-num', type=int, default=7, help='帧数')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    service = JigsawService(
        port=args.port,
        checkpoint_path=args.checkpoint,
        sample_num=args.sample_num,
        gpu_id=args.gpu_id
    )
    service.run()


if __name__ == "__main__":
    main()