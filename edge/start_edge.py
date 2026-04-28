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

from config import JIGSAW_SERVER, JIGSAW_MODEL_CONFIG, RELAY_SERVER, MAE_SERVER, MAE_MODEL_CONFIG


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


def start_mae_service(port: int = None, checkpoint_dir: str = None, gpu_id: int = None):
    """
    启动MAE服务

    Args:
        port: 服务端口
        checkpoint_dir: 模型权重目录
        gpu_id: GPU ID
    """
    port = port or MAE_SERVER["port"]
    checkpoint_dir = checkpoint_dir or MAE_MODEL_CONFIG["checkpoint_dir"]
    gpu_id = gpu_id if gpu_id is not None else MAE_MODEL_CONFIG["gpu_id"]

    print(f"启动MAE服务...")
    print(f"  端口: {port}")
    print(f"  模型目录: {checkpoint_dir}")
    print(f"  GPU: {gpu_id if gpu_id >= 0 else 'CPU'}")

    # 启动MAE服务进程
    cmd = [
        sys.executable,
        "detection/mae_service.py",
        "--port", str(port),
        "--checkpoint-dir", checkpoint_dir,
        "--gpu-id", str(gpu_id),
    ]

    process = subprocess.Popen(cmd)
    return process


def start_detection(relay_host: str = None, relay_port: int = None,
                    detector_type: str = None, detector_port: int = None):
    """
    启动检测程序

    Args:
        relay_host: 中间机器IP
        relay_port: 中间机器端口
        detector_type: 检测器类型
        detector_port: 检测服务端口
    """
    relay_host = relay_host or RELAY_SERVER["host"]
    relay_port = relay_port or RELAY_SERVER["port"]
    detector_type = detector_type or 'jigsaw'

    # 根据detector-type确定默认端口
    if detector_port is None:
        if detector_type == 'mae':
            detector_port = MAE_SERVER["port"]
        else:
            detector_port = JIGSAW_SERVER["port"]

    print(f"\n启动检测程序...")
    print(f"  检测器类型: {detector_type}")
    print(f"  检测服务端口: {detector_port}")
    print(f"  中间机器: {relay_host}:{relay_port}")

    # 启动检测进程
    cmd = [
        sys.executable,
        "run_edge.py",
        "--relay-host", relay_host,
        "--relay-port", str(relay_port),
        "--detector-type", detector_type,
        "--port", str(detector_port),
    ]

    process = subprocess.Popen(cmd)
    return process


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='端侧服务启动')
    parser.add_argument('--start-service', action='store_true',
                        help='仅启动检测服务（根据detector-type）')
    parser.add_argument('--start-detection', action='store_true',
                        help='仅启动检测程序（需要服务已运行）')
    parser.add_argument('--detector-type', type=str, default='jigsaw',
                        choices=['jigsaw', 'mae'],
                        help='检测器类型 (默认jigsaw)')
    parser.add_argument('--port', type=int, default=None,
                        help='检测服务端口（默认根据detector-type: jigsaw=8000, mae=8001）')
    parser.add_argument('--relay-host', type=str, default=RELAY_SERVER["host"],
                        help='中间机器IP地址')
    parser.add_argument('--relay-port', type=int, default=RELAY_SERVER["port"],
                        help='中间机器端口')
    parser.add_argument('--checkpoint', type=str,
                        default=JIGSAW_MODEL_CONFIG["checkpoint_path"],
                        help='Jigsaw模型权重路径')
    parser.add_argument('--mae-checkpoint-dir', type=str,
                        default=MAE_MODEL_CONFIG["checkpoint_dir"],
                        help='MAE模型权重目录')
    parser.add_argument('--gpu-id', type=int, default=JIGSAW_MODEL_CONFIG["gpu_id"],
                        help='GPU ID')

    args = parser.parse_args()

    # 根据detector-type确定默认端口
    if args.port is None:
        if args.detector_type == 'mae':
            args.port = MAE_SERVER["port"]
        else:
            args.port = JIGSAW_SERVER["port"]

    processes = []

    # 根据参数决定启动哪些服务
    if args.start_service:
        # 仅启动检测服务
        if args.detector_type == 'mae':
            p = start_mae_service(
                port=args.port,
                checkpoint_dir=args.mae_checkpoint_dir,
                gpu_id=args.gpu_id
            )
        else:
            p = start_jigsaw_service(
                port=args.port,
                checkpoint=args.checkpoint,
                gpu_id=args.gpu_id
            )
        processes.append(p)

    elif args.start_detection:
        # 仅启动检测程序（需要服务已运行）
        p = start_detection(
            relay_host=args.relay_host,
            relay_port=args.relay_port,
            detector_type=args.detector_type,
            detector_port=args.port
        )
        processes.append(p)

    else:
        # 默认：同时启动服务+检测程序
        if args.detector_type == 'mae':
            p1 = start_mae_service(
                port=args.port,
                checkpoint_dir=args.mae_checkpoint_dir,
                gpu_id=args.gpu_id
            )
            processes.append(p1)
            print("\n等待MAE服务启动...")
        else:
            p1 = start_jigsaw_service(
                port=args.port,
                checkpoint=args.checkpoint,
                gpu_id=args.gpu_id
            )
            processes.append(p1)
            print("\n等待Jigsaw服务启动...")

        time.sleep(3.0)

        p2 = start_detection(
            relay_host=args.relay_host,
            relay_port=args.relay_port,
            detector_type=args.detector_type,
            detector_port=args.port
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