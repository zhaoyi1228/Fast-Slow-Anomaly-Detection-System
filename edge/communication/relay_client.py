"""
Relay Client
中间机器通信客户端 - 将检测数据发送到中间节点
"""

import time
import json
import threading
import requests
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass


@dataclass
class CachedFrame:
    """缓存的帧数据"""
    frame_data: Dict[str, Any]
    retry_count: int
    timestamp: float


class RelayClient:
    """中间机器通信客户端，支持断线重连"""

    def __init__(self, host: str = None, port: int = None):
        """
        初始化Relay客户端

        Args:
            host: 中间机器IP地址
            port: 中间机器端口
        """
        from config import RELAY_SERVER, NETWORK_CONFIG

        self.host = host or RELAY_SERVER["host"]
        self.port = port or RELAY_SERVER["port"]
        self.base_url = f"http://{self.host}:{self.port}"

        self.network_config = NETWORK_CONFIG
        self.max_retries = NETWORK_CONFIG["max_retries"]
        self.retry_base = NETWORK_CONFIG["retry_interval_base"]
        self.retry_max = NETWORK_CONFIG["retry_interval_max"]
        self.timeout = NETWORK_CONFIG["request_timeout"]
        self.cache_max = NETWORK_CONFIG["cache_max_frames"]

        # 本地帧缓存队列
        self._frame_cache: deque = deque(maxlen=self.cache_max)
        self._cache_lock = threading.Lock()

        # 连接状态
        self._is_connected = False
        self._last_success_time = 0
        self._consecutive_failures = 0

        # 后台重试线程
        self._retry_thread = None
        self._retry_stop_event = threading.Event()
        self._start_retry_thread()

    def _start_retry_thread(self):
        """启动后台重试线程"""
        self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
        self._retry_thread.start()

    def _retry_loop(self):
        """后台重试循环，尝试发送缓存的帧"""
        while not self._retry_stop_event.is_set():
            time.sleep(2.0)  # 每2秒检查一次缓存

            with self._cache_lock:
                if self._frame_cache:
                    cached = self._frame_cache[0]
                    try:
                        result = self._send_request(cached.frame_data)
                        if result.get("status") == "success":
                            self._frame_cache.popleft()
                            self._is_connected = True
                            self._consecutive_failures = 0
                    except Exception:
                        pass

    def send_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        发送帧数据到中间机器，支持重试和缓存

        Args:
            frame_data: 帧数据字典，包含:
                - frame_id: 帧ID
                - timestamp: 时间戳
                - color_base64: 彩色图像base64
                - depth_base64: 深度图像base64（可选）
                - jigsaw_score: Jigsaw异常分数
                - spatial_score: 空间分数
                - temporal_score: 时间分数

        Returns:
            Dict: 发送结果
        """
        # 如果连接正常，直接发送
        if self._is_connected or self._consecutive_failures < 3:
            result = self._send_with_retry(frame_data)
            if result.get("status") == "success":
                return result

        # 发送失败，缓存帧数据
        self._cache_frame(frame_data)
        return {"status": "cached", "message": "帧已缓存，等待网络恢复后发送"}

    def _send_with_retry(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """带重试机制的发送"""
        for attempt in range(self.max_retries):
            try:
                result = self._send_request(frame_data)
                self._is_connected = True
                self._last_success_time = time.time()
                self._consecutive_failures = 0
                return result

            except requests.exceptions.ConnectionError as e:
                self._consecutive_failures += 1
                self._is_connected = False

                if attempt < self.max_retries - 1:
                    wait_time = min(self.retry_base * (attempt + 1), self.retry_max)
                    time.sleep(wait_time)

            except requests.exceptions.Timeout as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_base)

            except Exception as e:
                print(f"发送异常: {e}")
                break

        return {"status": "failed", "error": "达到最大重试次数"}

    def _send_request(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """发送HTTP请求"""
        from config import RELAY_SERVER

        url = f"{self.base_url}{RELAY_SERVER['frame_endpoint']}"

        response = requests.post(
            url,
            json=frame_data,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def _cache_frame(self, frame_data: Dict[str, Any]):
        """缓存帧数据"""
        with self._cache_lock:
            cached = CachedFrame(
                frame_data=frame_data,
                retry_count=0,
                timestamp=time.time()
            )
            self._frame_cache.append(cached)

    def send_batch(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量发送帧数据"""
        results = []
        for frame in frames:
            result = self.send_frame(frame)
            results.append(result)

        success_count = sum(1 for r in results if r.get("status") == "success")
        cached_count = sum(1 for r in results if r.get("status") == "cached")

        return {
            "status": "batch_complete",
            "total": len(frames),
            "success": success_count,
            "cached": cached_count,
        }

    def health_check(self) -> bool:
        """检查与中间机器的连接状态"""
        from config import RELAY_SERVER

        try:
            url = f"{self.base_url}{RELAY_SERVER['status_endpoint']}"
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                self._is_connected = True
                return True
        except Exception:
            self._is_connected = False
        return False

    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        with self._cache_lock:
            cache_size = len(self._frame_cache)

        return {
            "relay_host": self.host,
            "relay_port": self.port,
            "is_connected": self._is_connected,
            "consecutive_failures": self._consecutive_failures,
            "last_success_time": self._last_success_time,
            "cached_frames": cache_size,
        }

    def flush_cache(self) -> int:
        """
        强制发送所有缓存的帧

        Returns:
            int: 成功发送的帧数
        """
        sent_count = 0
        with self._cache_lock:
            while self._frame_cache:
                cached = self._frame_cache.popleft()
                try:
                    result = self._send_request(cached.frame_data)
                    if result.get("status") == "success":
                        sent_count += 1
                except Exception:
                    # 重新放回缓存
                    self._frame_cache.appendleft(cached)
                    break

        return sent_count

    def stop(self):
        """停止客户端"""
        self._retry_stop_event.set()
        if self._retry_thread:
            self._retry_thread.join(timeout=2.0)


def main():
    """测试Relay客户端"""
    client = RelayClient()

    # 测试健康检查
    print(f"健康检查: {client.health_check()}")

    # 测试发送假数据
    test_data = {
        "frame_id": 1,
        "timestamp": 0.0,
        "color_base64": "test_base64_string",
        "jigsaw_score": 0.5,
        "spatial_score": 0.3,
        "temporal_score": 0.7,
    }
    result = client.send_frame(test_data)
    print(f"发送结果: {result}")

    print(f"客户端状态: {client.get_status()}")
    client.stop()


if __name__ == "__main__":
    main()