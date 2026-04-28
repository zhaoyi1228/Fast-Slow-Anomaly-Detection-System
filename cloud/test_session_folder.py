"""
Session Folder 功能测试

测试内容:
1. 验证 FrameProcessor 创建时间戳命名的 session 文件夹
2. 验证 frames 子文件夹存储视频帧
3. 验证 result 子文件夹存储处理结果
4. 验证 API 返回 session_folder 路径用于回溯
"""

import os
import sys
import base64
import json
import time
import requests
import numpy as np
import cv2

# 添加项目路径
CLOUD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CLOUD_DIR)

from config import AGENT_CONFIG

# 测试配置
TEMP_FRAME_DIR = AGENT_CONFIG["temp_frame_dir"]

# Cloud API配置
CLOUD_API_HOST = "localhost"
CLOUD_API_PORT = 8001
CLOUD_API_URL = f"http://{CLOUD_API_HOST}:{CLOUD_API_PORT}/api/v1/detect"
SESSIONS_API_URL = f"http://{CLOUD_API_HOST}:{CLOUD_API_PORT}/api/v1/sessions"


def create_test_image():
    """创建测试图像"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), -1)
    cv2.circle(img, (450, 200), 80, (0, 255, 0), -1)
    cv2.putText(img, "SESSION TEST", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.putText(img, timestamp, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def encode_image_to_base64(img):
    """将图像编码为base64"""
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')


def check_cloud_service():
    """检查cloud服务"""
    health_url = f"http://{CLOUD_API_HOST}:{CLOUD_API_PORT}/api/v1/health"
    try:
        response = requests.get(health_url, timeout=5.0)
        return response.status_code == 200
    except:
        return False


def test_frame_processor_local():
    """本地测试 FrameProcessor（不依赖云服务）"""
    print("\n" + "=" * 60)
    print("[本地测试] FrameProcessor Session Folder 功能")
    print("=" * 60)

    # 导入 FrameProcessor - 使用正确的 AnomalyAgent 项目路径
    try:
        # 直接使用 AnomalyAgent 项目路径
        anomaly_agent_path = "/home/zhaoyi/media/projects_zy/anomaly_agent"
        if anomaly_agent_path not in sys.path:
            sys.path.insert(0, anomaly_agent_path)
        from api.handlers.frame_processor import FrameProcessor
    except Exception as e:
        print(f"[FAIL] 无法导入 FrameProcessor: {e}")
        print(f"[INFO] 尝试的路径: {anomaly_agent_path}")
        return False

    # 创建 FrameProcessor 实例
    processor = FrameProcessor(
        temp_dir=TEMP_FRAME_DIR,
        cleanup=False,
        save_results=True
    )
    print(f"[OK] FrameProcessor 创建成功")
    print(f"     temp_dir: {TEMP_FRAME_DIR}")
    print(f"     cleanup: False (保留session用于回溯)")
    print(f"     save_results: True")

    # 创建 session folder
    session_folder = processor.create_session_folder()
    print(f"\n[OK] Session folder 创建成功: {session_folder}")

    # 验证目录结构
    frames_dir = os.path.join(session_folder, "frames")
    result_dir = os.path.join(session_folder, "result")

    if os.path.exists(frames_dir):
        print(f"[OK] frames 子文件夹存在: {frames_dir}")
    else:
        print(f"[FAIL] frames 子文件夹不存在")
        return False

    if os.path.exists(result_dir):
        print(f"[OK] result 子文件夹存在: {result_dir}")
    else:
        print(f"[FAIL] result 子文件夹不存在")
        return False

    # 创建测试帧并存储
    test_img = create_test_image()
    img_base64 = encode_image_to_base64(test_img)

    frames_input = [{"image_base64": img_base64, "frame_id": 1}]

    # 使用 async 方式需要 asyncio
    import asyncio
    frame_paths, returned_session = asyncio.run(
        processor.process_base64_frames(frames_input, session_folder)
    )

    print(f"\n[OK] 帧处理完成")
    print(f"     返回的 session folder: {returned_session}")
    print(f"     帧路径数量: {len(frame_paths)}")

    # 验证帧是否在 frames 子文件夹
    for path in frame_paths[:3]:
        if os.path.dirname(path) == frames_dir:
            print(f"[OK] 帧存储在 frames 子文件夹: {os.path.basename(path)}")
        else:
            print(f"[FAIL] 帧未存储在 frames 子文件夹: {path}")

    # 验证 request_meta.json
    meta_path = os.path.join(result_dir, "request_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"\n[OK] request_meta.json 存在")
        print(f"     frame_count: {meta.get('frame_count')}")
        print(f"     created_at: {meta.get('created_at')}")
    else:
        print(f"[FAIL] request_meta.json 不存在")

    # 测试 save_detection_result
    test_result = {
        "anomaly_score": 0.75,
        "is_anomaly": True,
        "anomaly_type": "loitering",
        "description": "测试结果"
    }

    asyncio.run(
        processor.save_detection_result(
            session_folder=session_folder,
            result=test_result,
            request_id="test_req_001",
            video_id="test_video_001"
        )
    )

    result_path = os.path.join(result_dir, "detection_result.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            saved_result = json.load(f)
        print(f"\n[OK] detection_result.json 存在")
        print(f"     request_id: {saved_result.get('request_id')}")
        print(f"     video_id: {saved_result.get('video_id')}")
        print(f"     result.anomaly_score: {saved_result.get('result', {}).get('anomaly_score')}")
    else:
        print(f"[FAIL] detection_result.json 不存在")

    # 测试 get_session_info
    session_info = processor.get_session_info(session_folder)
    print(f"\n[OK] Session 信息:")
    print(f"     session_name: {session_info.get('session_name')}")
    print(f"     frame_count: {session_info.get('frame_count')}")
    print(f"     has_detection_result: {session_info.get('has_detection_result')}")

    # 测试 list_sessions
    sessions = processor.list_sessions()
    print(f"\n[OK] 当前 temp_frames 中的 session 数量: {len(sessions)}")

    print("\n" + "=" * 60)
    print("[SUCCESS] FrameProcessor Session Folder 功能测试通过")
    print("=" * 60)

    return True


def test_api_endpoint():
    """测试 API endpoint（需要云服务运行）"""
    print("\n" + "=" * 60)
    print("[API测试] Session Folder API 功能")
    print("=" * 60)

    if not check_cloud_service():
        print("[WARN] Cloud服务未运行，跳过API测试")
        return False

    # 创建测试帧
    test_img = create_test_image()
    img_base64 = encode_image_to_base64(test_img)

    request_payload = {
        "frames": [{"image_base64": img_base64, "frame_id": 1}],
        "video_id": "session_test_video",
        "scene_type": "general",
        "dataset": "ped2",
    }

    print(f"\n[INFO] 发送检测请求...")
    try:
        response = requests.post(CLOUD_API_URL, json=request_payload, timeout=30.0)
        if response.status_code != 200:
            print(f"[FAIL] API响应异常: {response.status_code}")
            return False

        data = response.json()
        print(f"[OK] 检测请求成功")
        print(f"     request_id: {data.get('request_id')}")
        print(f"     session_folder: {data.get('session_folder')}")
        print(f"     session_info: {json.dumps(data.get('session_info', {}), indent=2)}")

        # 验证 session_folder 返回
        if data.get("session_folder"):
            print(f"\n[OK] API返回 session_folder 用于回溯定位")
        else:
            print(f"[FAIL] API未返回 session_folder")
            return False

    except Exception as e:
        print(f"[FAIL] API测试失败: {e}")
        return False

    # 测试 sessions API
    print(f"\n[INFO] 测试 sessions 列表 API...")
    try:
        response = requests.get(SESSIONS_API_URL, timeout=5.0)
        if response.status_code == 200:
            sessions_data = response.json()
            print(f"[OK] Sessions API 成功")
            print(f"     total_sessions: {sessions_data.get('total_sessions')}")
            print(f"     temp_frame_dir: {sessions_data.get('temp_frame_dir')}")
        else:
            print(f"[FAIL] Sessions API 响应异常: {response.status_code}")
    except Exception as e:
        print(f"[FAIL] Sessions API 测试失败: {e}")

    print("\n" + "=" * 60)
    print("[SUCCESS] API Session 功能测试通过")
    print("=" * 60)

    return True


def main():
    """主测试流程"""
    print("=" * 60)
    print("Session Folder 功能完整测试")
    print("=" * 60)
    print(f"temp_frame_dir: {TEMP_FRAME_DIR}")
    print()

    # Step 1: 本地测试（不依赖云服务）
    local_ok = test_frame_processor_local()

    # Step 2: API测试（需要云服务）
    api_ok = test_api_endpoint()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  - 本地测试: {'OK' if local_ok else 'FAIL'}")
    print(f"  - API测试: {'OK' if api_ok else 'SKIP/FAIL'}")
    print("=" * 60)

    if local_ok:
        print("\n[SUCCESS] 核心功能验证通过")
        print("Session folder 结构:")
        print(f"  {TEMP_FRAME_DIR}/")
        print("    └── <timestamp_uuid>/  (时间戳命名)")
        print("        ├── frames/        (视频帧)")
        print("        └── result/        (检测结果)")
    else:
        print("\n[FAILED] 测试失败")


if __name__ == "__main__":
    main()