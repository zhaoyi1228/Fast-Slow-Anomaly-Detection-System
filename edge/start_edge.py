"""
Start Edge Services
端侧服务启动脚本 - 同时启动Jigsaw服务和检测程序
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from threading import Thread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import JIGSAW_SERVER, JIGSAW_MODEL_CONFIG, RELAY_SERVER


def start_jigsaw_service(port: int = None, checkpoint: str = None,
                         sample_num: int = None, gpu_id: int = None):
    """
    启动Jigsaw服务

    Args:
        port: 服务端口
        checkpoint: 模型权重路径
        sample_num: 帧数
        gpu_id: GPU ID
    """
    port = port or JIGSAW_SERVER["port"]
    checkpoint = checkpoint or JIGSAW_MODEL_CONFIG["checkpoint_path"]
    sample_num = sample_num or JIGSAW_MODEL_CONFIG["sample_num"]
    gpu_id = gpu_id if gpu_id is not None else JIGSAW_MODEL_CONFIG["gpu_id"]

    print(f"启动Jigsaw服务...")
    print(f"  端口: {port}")
    print(f"  模型: {checkpoint}")
    print(f"  帧数: {sample_num}")
    print(f"  GPU: {gpu_id if gpu_id >= 0 else 'CPU'}")

    # 启动Jigsaw服务进程
    cmd = [
        sys.executable,
        "detection/jigsaw_service.py",
        "--port", str(port),
        "--checkpoint", checkpoint,
        "--sample-num", str(sample_num),
        "--gpu-id", str(gpu_id),
    ]

    process = subprocess.Popen(cmd)
    return process


def start_detection(relay_host: str = None, relay_port: int = None):
    """
    启动检测程序

    Args:
        relay_host: 中间机器IP
        relay_port: 中间机器端口
    """
    relay_host = relay_host or RELAY_SERVER["host"]
    relay_port = relay_port or RELAY_SERVER["port"]

    print(f"\n启动检测程序...")
    print(f"  中间机器: {relay_host}:{relay_port}")

    # 启动检测进程
    cmd = [
        sys.executable,
        "run_edge.py",
        "--relay-host", relay_host,
        "--relay-port", str(relay_port),
    ]

    process = subprocess.Popen(cmd)
    return process


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='端侧服务启动')
    parser.add_argument('--start-jigsaw', action='store_true',
                        help='仅启动Jigsaw服务')
    parser.add_argument('--start-detection', action='store_true',
                        help='仅启动检测程序')
    parser.add_argument('--relay-host', type=str, default=RELAY_SERVER["host"],
                        help='中间机器IP地址')
    parser.add_argument('--relay-port', type=int, default=RELAY_SERVER["port"],
                        help='中间机器端口')
    parser.add_argument('--jigsaw-port', type=int, default=JIGSAW_SERVER["port"],
                        help='Jigsaw服务端口')
    parser.add_argument('--checkpoint', type=str,
                        default=JIGSAW_MODEL_CONFIG["checkpoint_path"],
                        help='Jigsaw模型权重路径')
    parser.add_argument('--gpu-id', type=int, default=JIGSAW_MODEL_CONFIG["gpu_id"],
                        help='GPU ID')

    args = parser.parse_args()

    processes = []

    # 根据参数决定启动哪些服务
    if args.start_jigsaw:
        # 仅启动Jigsaw
        p = start_jigsaw_service(
            port=args.jigsaw_port,
            checkpoint=args.checkpoint,
            gpu_id=args.gpu_id
        )
        processes.append(p)

    elif args.start_detection:
        # 仅启动检测（需要Jigsaw已运行）
        p = start_detection(
            relay_host=args.relay_host,
            relay_port=args.relay_port
        )
        processes.append(p)

    else:
        # 默认：同时启动两个服务
        p1 = start_jigsaw_service(
            port=args.jigsaw_port,
            checkpoint=args.checkpoint,
            gpu_id=args.gpu_id
        )
        processes.append(p1)

        # 等待Jigsaw启动
        print("\n等待Jigsaw服务启动...")
        time.sleep(3.0)

        p2 = start_detection(
            relay_host=args.relay_host,
            relay_port=args.relay_port
        )
        processes.append(p2)

    # 注册信号处理
    def signal_handler(sig, frame):
        print("\n停止所有服务...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 等待进程
    try:
        while True:
            # 检查进程状态
            for i, p in enumerate(processes):
                if p.poll() is not None:
                    print(f"进程 {i} 已退出")

            time.sleep(1.0)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()