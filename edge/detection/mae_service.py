"""
MAE Service
MAE异常检测服务 - Flask HTTP服务端
从aed-mae移植，提供实时视频异常检测

基于Self-Distilled Masked Auto-Encoders (CVPR 2024)
"""

import io
import base64
import time
import threading
import os
import sys
import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Ensure imports work when launched from edge directory
EDGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EDGE_ROOT not in sys.path:
    sys.path.insert(0, EDGE_ROOT)


@dataclass
class MAEResult:
    """MAE检测结果"""
    anomaly_score: float       # 综合异常得分 (teacher + student-teacher)
    teacher_score: float       # Teacher重建误差分量
    student_teacher_score: float  # Student-Teacher差异分量
    elapsed_ms: float          # 推理耗时(ms)
    buffer_ready: bool         # 帧缓冲是否就绪
    buffer_size: int           # 当前缓冲区大小


class MAEDetector:
    """MAE检测器核心类"""

    def __init__(self, checkpoint_dir: str, teacher_ckpt: str, student_ckpt: str,
                 input_size: tuple = (160, 320), patch_size: int = 8,
                 buffer_size: int = 7, mask_ratio: float = 0.5, gpu_id: int = 0):
        """
        初始化MAE检测器

        Args:
            checkpoint_dir: 模型权重目录
            teacher_ckpt: Teacher模型权重文件名
            student_ckpt: Student模型权重文件名
            input_size: 输入尺寸 (H, W)
            patch_size: Patch大小
            buffer_size: 帧缓冲大小
            mask_ratio: MAE mask比例
            gpu_id: GPU ID (-1表示CPU)
        """
        # 设备选择 (GPU优先，CPU兼容)
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu")

        # 模型参数
        self.input_size = input_size  # (H, W)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 帧缓冲
        self.buffer_size = buffer_size
        self._frame_buffer: List[np.ndarray] = []
        self._buffer_lock = threading.Lock()

        # 模型
        self._model = None
        self._model_error: Optional[str] = None
        self._model_lock = threading.Lock()

        # 加载模型
        self._load_model(checkpoint_dir, teacher_ckpt, student_ckpt)

    def _load_model(self, checkpoint_dir: str, teacher_ckpt: str, student_ckpt: str):
        """加载MAE模型 (合并teacher+student权重)"""
        try:
            from models.model_factory import mae_cvt_patch8

            # 创建模型
            self._model = mae_cvt_patch8(
                norm_pix_loss=False,
                img_size=self.input_size,
                use_only_masked_tokens_ab=False,
                abnormal_score_func=['L1', 'L1'],
                masking_method="random_masking",
                grad_weighted_loss=False
            ).float()

            # 加载权重路径
            teacher_path = os.path.join(checkpoint_dir, teacher_ckpt)
            student_path = os.path.join(checkpoint_dir, student_ckpt)

            if not os.path.exists(teacher_path):
                raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
            if not os.path.exists(student_path):
                raise FileNotFoundError(f"Student checkpoint not found: {student_path}")

            # 加载权重
            teacher_state = torch.load(teacher_path, map_location=self.device)['model']
            student_state = torch.load(student_path, map_location=self.device)['model']

            # 合并student权重到teacher
            for key in student_state:
                if 'student' in key:
                    teacher_state[key] = student_state[key]

            self._model.load_state_dict(teacher_state, strict=False)
            self._model.to(self.device)
            self._model.eval()
            self._model.train_TS = True  # 启用student-teacher模式
            self._model.abnormal_score_func_TS = 'L1'  # ShanghaiTech使用L1

            print(f"MAE模型加载成功，设备: {self.device}")
            self._model_error = None

        except ImportError as e:
            self._model_error = f"models.model_factory模块未找到: {e}"
            print(f"警告: {self._model_error}")
            self._model = None

        except FileNotFoundError as e:
            self._model_error = str(e)
            print(f"模型文件未找到: {e}")
            self._model = None

        except Exception as e:
            self._model_error = str(e)
            print(f"模型加载失败: {e}")
            self._model = None

    def detect_single(self, image_base64: str, frame_id: int = None) -> MAEResult:
        """
        检测单帧

        Args:
            image_base64: base64编码的图像
            frame_id: 帧序号 (可选)

        Returns:
            MAEResult: 检测结果
        """
        start_time = time.time()

        # 解码图像
        image = self._decode_base64(image_base64)

        # 添加到帧缓冲
        with self._buffer_lock:
            self._frame_buffer.append(image)
            if len(self._frame_buffer) > self.buffer_size:
                self._frame_buffer.pop(0)

        buffer_ready = len(self._frame_buffer) >= 4  # 至少4帧才能处理(-3到+3范围)

        if not buffer_ready:
            elapsed_ms = (time.time() - start_time) * 1000
            return MAEResult(
                anomaly_score=0.0,
                teacher_score=0.0,
                student_teacher_score=0.0,
                elapsed_ms=elapsed_ms,
                buffer_ready=False,
                buffer_size=len(self._frame_buffer)
            )

        # 执行检测
        result = self._run_detection()
        result.elapsed_ms = (time.time() - start_time) * 1000
        result.buffer_ready = True
        result.buffer_size = len(self._frame_buffer)

        return result

    def _run_detection(self) -> MAEResult:
        """执行检测推理"""
        if self._model is None:
            raise RuntimeError(self._model_error or "MAE model is not loaded")

        with self._model_lock:
            # 当前帧索引在缓冲区末尾
            curr_idx = len(self._frame_buffer) - 1

            # 准备输入数据
            samples, grads, targets = self._prepare_inputs(curr_idx)

            # 推理
            with torch.no_grad():
                samples = samples.to(self.device)
                grads = grads.to(self.device)
                targets = targets.to(self.device)

                _, _, _, scores = self._model(
                    samples, targets=targets,
                    grad_mask=grads, mask_ratio=self.mask_ratio
                )

                # scores[0]: student-teacher差异
                # scores[1]: teacher重建误差
                student_teacher_score = scores[0].item()
                teacher_score = scores[1].item()
                anomaly_score = teacher_score + student_teacher_score

            return MAEResult(
                anomaly_score=anomaly_score,
                teacher_score=teacher_score,
                student_teacher_score=student_teacher_score,
                elapsed_ms=0,
                buffer_ready=True,
                buffer_size=len(self._frame_buffer)
            )

    def _prepare_inputs(self, curr_idx: int):
        """
        准备模型输入 (帧堆叠 + 梯度计算 + 预处理)

        Args:
            curr_idx: 当前帧在缓冲区中的索引

        Returns:
            samples: 堆叠帧 (1, 9, H, W)
            grads: 梯度图 (1, 3, H, W)
            targets: 目标帧 (1, 4, H, W)
        """
        H, W = self.input_size

        # 帧堆叠: prev(-3), curr(0), next(+3)
        prev_idx = max(curr_idx - 3, 0)
        next_idx = min(curr_idx + 3, len(self._frame_buffer) - 1)

        prev_frame = self._frame_buffer[prev_idx]
        curr_frame = self._frame_buffer[curr_idx]
        next_frame = self._frame_buffer[next_idx]

        # 拼接为9通道
        stacked = np.concatenate([prev_frame, curr_frame, next_frame], axis=-1)

        # 梯度计算: absdiff(prev, next) - 使用相邻帧 (step=1)
        grad_prev_idx = max(curr_idx - 1, 0)
        grad_next_idx = min(curr_idx + 1, len(self._frame_buffer) - 1)
        gradient = cv2.absdiff(self._frame_buffer[grad_prev_idx], self._frame_buffer[grad_next_idx])
        gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB)

        # 缩放
        stacked = cv2.resize(stacked, (W, H))
        gradient = cv2.resize(gradient, (W, H))
        current = cv2.resize(curr_frame, (W, H))

        # 目准备 (当前帧 + 零mask)
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        target = np.concatenate([current, mask], axis=-1)

        # 归一化 [-1, 1] for samples and targets
        stacked = (stacked.astype(np.float32) - 127.5) / 127.5
        # gradient保持原始值范围 (用于加权计算)
        gradient = gradient.astype(np.float32)
        target = (target.astype(np.float32) - 127.5) / 127.5

        # HWC → CHW + batch dimension
        stacked = np.transpose(stacked, (2, 0, 1))[np.newaxis, ...]  # (1, 9, H, W)
        gradient = np.transpose(gradient, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
        target = np.transpose(target, (2, 0, 1))[np.newaxis, ...]  # (1, 4, H, W)

        return torch.from_numpy(stacked), torch.from_numpy(gradient), torch.from_numpy(target)

    def _decode_base64(self, image_base64: str) -> np.ndarray:
        """解码base64图像"""
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return image

    def reset_buffer(self):
        """重置帧缓冲区"""
        with self._buffer_lock:
            self._frame_buffer.clear()

    def get_buffer_size(self) -> int:
        """获取当前缓冲区大小"""
        with self._buffer_lock:
            return len(self._frame_buffer)


class MAEService:
    """MAE Flask服务"""

    def __init__(self, port: int = 8001, checkpoint_dir: str = None,
                 gpu_id: int = None, **kwargs):
        """
        初始化MAE服务

        Args:
            port: 服务端口
            checkpoint_dir: 模型权重目录
            gpu_id: GPU ID
            **kwargs: 其他参数传递给MAEDetector
        """
        from config import MAE_MODEL_CONFIG

        self.port = port
        checkpoint_dir = checkpoint_dir or MAE_MODEL_CONFIG["checkpoint_dir"]
        gpu_id = gpu_id if gpu_id is not None else MAE_MODEL_CONFIG["gpu_id"]

        self.app = Flask(__name__)
        self.detector = None

        # 初始化检测器
        try:
            self.detector = MAEDetector(
                checkpoint_dir=checkpoint_dir,
                teacher_ckpt=MAE_MODEL_CONFIG["teacher_ckpt"],
                student_ckpt=MAE_MODEL_CONFIG["student_ckpt"],
                input_size=MAE_MODEL_CONFIG["input_size"],
                patch_size=MAE_MODEL_CONFIG["patch_size"],
                buffer_size=MAE_MODEL_CONFIG["buffer_size"],
                mask_ratio=MAE_MODEL_CONFIG["mask_ratio"],
                gpu_id=gpu_id
            )
        except Exception as e:
            print(f"检测器初始化失败: {e}")
            self.detector = None
            self._init_error = str(e)

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/health', methods=['GET'])
        def health():
            if self.detector is None:
                return jsonify({
                    "status": "error",
                    "model_loaded": False,
                    "error": getattr(self, '_init_error', 'detector not initialized'),
                })
            return jsonify({
                "status": "ok",
                "model_loaded": self.detector._model is not None,
                "device": str(self.detector.device),
                "input_size": list(self.detector.input_size),
                "buffer_size": self.detector.get_buffer_size(),
                "patch_size": self.detector.patch_size,
                "model_error": self.detector._model_error,
            })

        @self.app.route('/detect', methods=['POST'])
        def detect():
            try:
                if self.detector is None or self.detector._model is None:
                    error_msg = self.detector._model_error if self.detector else getattr(self, '_init_error', 'detector not initialized')
                    return jsonify({"error": error_msg}), 503

                data = request.get_json()
                image_base64 = data.get('image')
                frame_id = data.get('frame_id')

                if not image_base64:
                    return jsonify({"error": "缺少image参数"}), 400

                result = self.detector.detect_single(image_base64, frame_id)

                return jsonify({
                    "anomaly_score": result.anomaly_score,
                    "teacher_score": result.teacher_score,
                    "student_teacher_score": result.student_teacher_score,
                    "elapsed_ms": result.elapsed_ms,
                    "buffer_ready": result.buffer_ready,
                    "buffer_size": result.buffer_size,
                    "frame_id": frame_id,
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/batch_detect', methods=['POST'])
        def batch_detect():
            try:
                if self.detector is None or self.detector._model is None:
                    error_msg = self.detector._model_error if self.detector else getattr(self, '_init_error', 'detector not initialized')
                    return jsonify({"error": error_msg}), 503

                data = request.get_json()
                images = data.get('images', [])

                if not images:
                    return jsonify({"error": "缺少images参数"}), 400

                results = []
                for img_b64 in images:
                    result = self.detector.detect_single(img_b64)
                    results.append({
                        "anomaly_score": result.anomaly_score,
                        "teacher_score": result.teacher_score,
                        "student_teacher_score": result.student_teacher_score,
                        "elapsed_ms": result.elapsed_ms,
                        "buffer_ready": result.buffer_ready,
                    })

                return jsonify({"results": results})

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/reset', methods=['POST'])
        def reset():
            if self.detector:
                self.detector.reset_buffer()
            return jsonify({"status": "ok", "message": "Buffer reset successfully"})

        @self.app.route('/stats', methods=['GET'])
        def stats():
            if self.detector is None:
                return jsonify({"error": "detector not initialized"}), 503
            return jsonify({
                "buffer_size": self.detector.get_buffer_size(),
                "buffer_ready": self.detector.get_buffer_size() >= 4,
                "input_size": list(self.detector.input_size),
                "device": str(self.detector.device),
            })

    def run(self, host: str = "0.0.0.0"):
        """启动服务"""
        print(f"启动MAE服务，端口: {self.port}")
        self.app.run(host=host, port=self.port, threaded=True)


def create_mae_client(host: str = "localhost", port: int = 8001):
    """
    创建MAE客户端 (用于本地调用)

    Args:
        host: 服务主机
        port: 服务端口

    Returns:
        MAEClient: 客户端实例
    """
    import requests

    class MAEClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def detect_single(self, image_base64: str, frame_id: int = None) -> Dict[str, Any]:
            """单帧检测"""
            url = f"{self.base_url}/detect"
            try:
                response = requests.post(
                    url,
                    json={"image": image_base64, "frame_id": frame_id},
                    timeout=5.0
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "anomaly_score": None}

        def batch_detect(self, images_base64: List[str]) -> Dict[str, Any]:
            """批量检测"""
            url = f"{self.base_url}/batch_detect"
            try:
                response = requests.post(
                    url,
                    json={"images": images_base64},
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}

        def health_check(self) -> bool:
            """健康检查"""
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2.0)
                return response.status_code == 200
            except:
                return False

        def reset_buffer(self) -> bool:
            """重置缓冲区"""
            try:
                response = requests.post(f"{self.base_url}/reset", timeout=2.0)
                return response.status_code == 200
            except:
                return False

        def get_stats(self) -> Dict[str, Any]:
            """获取统计信息"""
            try:
                response = requests.get(f"{self.base_url}/stats", timeout=2.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}

    return MAEClient(f"http://{host}:{port}")


def main():
    """启动MAE服务"""
    import argparse
    parser = argparse.ArgumentParser(description='MAE异常检测服务')
    parser.add_argument('--port', type=int, default=8001, help='服务端口')
    parser.add_argument('--checkpoint-dir', type=str, default='/home/zhaoyi/aed-mae/ckpt/shanghai',
                        help='模型权重目录')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID (-1表示CPU)')

    args = parser.parse_args()

    service = MAEService(
        port=args.port,
        checkpoint_dir=args.checkpoint_dir,
        gpu_id=args.gpu_id
    )
    service.run()


if __name__ == "__main__":
    main()