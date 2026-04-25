"""
Cloud Client
云侧通信客户端 - 通过VPN向云侧Agent发送请求
"""

import time
import threading
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class DeepAnalysisResult:
    """深度分析结果"""
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str
    description: str
    explanation: str
    frame_scores: List[float]
    processing_time_ms: int
    raw_response: Dict[str, Any]


class CloudClient:
    """云侧Agent通信客户端"""

    def __init__(self, host: str = None, port: int = None):
        """
        初始化云侧客户端

        Args:
            host: 云侧服务器IP（VPN IP）
            port: 云侧服务端口
        """
        from config import CLOUD_SERVER, NETWORK_CONFIG

        self.host = host or CLOUD_SERVER["host"]
        self.port = port or CLOUD_SERVER["port"]
        self.base_url = f"http://{self.host}:{self.port}"

        self.network_config = NETWORK_CONFIG
        self.max_retries = NETWORK_CONFIG["max_retries"]
        self.retry_base = NETWORK_CONFIG["retry_interval_base"]
        self.timeout = NETWORK_CONFIG["request_timeout"]

        # 连接状态
        self._is_connected = False
        self._last_success_time = 0
        self._consecutive_failures = 0

        # 健康检查线程
        self._health_thread = None
        self._health_stop = threading.Event()
        self._start_health_check_thread()

    def _start_health_check_thread(self):
        """启动健康检查线程"""
        from config import NETWORK_CONFIG

        interval = NETWORK_CONFIG["health_check_interval"]

        def health_loop():
            while not self._health_stop.is_set():
                self.health_check()
                time.sleep(interval)

        self._health_thread = threading.Thread(target=health_loop, daemon=True)
        self._health_thread.start()

    def detect(self, frames: List[Dict[str, Any]], scene_type: str = "general",
               dataset: str = "ped2") -> DeepAnalysisResult:
        """
        发送帧到云侧进行深度分析

        Args:
            frames: 帧数据列表，每个包含:
                - frame_id: 帧ID
                - color_base64: 彩色图像base64
                - jigsaw_score: Jigsaw分数
            scene_type: 场景类型
            dataset: 数据集类型

        Returns:
            DeepAnalysisResult: 深度分析结果
        """
        start_time = time.time()

        result = self._detect_with_retry(frames, scene_type, dataset)

        processing_time = int((time.time() - start_time) * 1000)

        if result.get("success", False):
            result_data = result.get("result", {})
            return DeepAnalysisResult(
                anomaly_score=result_data.get("anomaly_score", 0.0),
                is_anomaly=result_data.get("is_anomaly", False),
                anomaly_type=result_data.get("anomaly_type", "unknown"),
                description=result_data.get("description", ""),
                explanation=result_data.get("explanation", ""),
                frame_scores=result_data.get("frame_scores", []),
                processing_time_ms=processing_time,
                raw_response=result
            )

        # 失败返回默认结果
        error_message = result.get("error", "Unknown error")
        if not isinstance(error_message, str):
            error_message = str(error_message)
        return DeepAnalysisResult(
            anomaly_score=0.0,
            is_anomaly=False,
            anomaly_type="error",
            description="",
            explanation=error_message,
            frame_scores=[],
            processing_time_ms=processing_time,
            raw_response=result
        )

    def _detect_with_retry(self, frames: List[Dict[str, Any]],
                           scene_type: str, dataset: str) -> Dict[str, Any]:
        """带重试机制的检测"""
        from config import CLOUD_SERVER, FUSION_CONFIG

        url = f"{self.base_url}{CLOUD_SERVER['detect_endpoint']}"

        payload = {
            "frames": [
                {
                    "image_base64": f.get("image_base64") or f.get("color_base64"),
                    "frame_id": f.get("frame_id"),
                    "timestamp": f.get("timestamp"),
                    "jigsaw_score": f.get("jigsaw_score"),
                }
                for f in frames
            ],
            "scene_type": scene_type,
            "dataset": dataset,
            "batch_size": FUSION_CONFIG["deep_analysis_batch_size"],
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                result = response.json()
                response.raise_for_status()

                self._is_connected = True
                self._last_success_time = time.time()
                self._consecutive_failures = 0

                return result

            except requests.exceptions.ConnectionError as e:
                self._consecutive_failures += 1
                self._is_connected = False

                if attempt < self.max_retries - 1:
                    wait_time = min(self.retry_base * (attempt + 1), self.retry_base * 5)
                    time.sleep(wait_time)

            except requests.exceptions.Timeout as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_base)

            except requests.exceptions.HTTPError as e:
                try:
                    detail = e.response.json()
                except Exception:
                    detail = {"detail": str(e)}
                return {"success": False, "error": detail}

            except Exception as e:
                print(f"云侧请求异常: {e}")
                break

        return {
            "success": False,
            "error": f"达到最大重试次数 ({self.max_retries})"
        }

    def health_check(self) -> bool:
        """检查与云侧的连接状态"""
        from config import CLOUD_SERVER

        try:
            url = f"{self.base_url}{CLOUD_SERVER['health_endpoint']}"
            response = requests.get(url, timeout=5.0)
            if response.status_code == 200:
                self._is_connected = True
                return True
        except Exception:
            self._is_connected = False
        return False

    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            "cloud_host": self.host,
            "cloud_port": self.port,
            "is_connected": self._is_connected,
            "consecutive_failures": self._consecutive_failures,
            "last_success_time": self._last_success_time,
        }

    def stop(self):
        """停止客户端"""
        self._health_stop.set()
        if self._health_thread:
            self._health_thread.join(timeout=2.0)


def main():
    """测试云侧客户端"""
    client = CloudClient()

    print(f"健康检查: {client.health_check()}")
    print(f"状态: {client.get_status()}")

    client.stop()


if __name__ == "__main__":
    main()