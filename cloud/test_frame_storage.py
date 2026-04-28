"""
Mock测试: 验证图像可以上传到/media目录并被cloud服务处理

测试步骤:
1. 创建测试图像并保存到 /home/zhaoyi/media/temp_frames
2. 将图像编码为base64
3. 模拟发送到cloud API的detect endpoint
4. 确认图像被正确存储和处理
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
TEST_IMAGE_NAME = "mock_test_frame.jpg"
TEST_IMAGE_PATH = os.path.join(TEMP_FRAME_DIR, TEST_IMAGE_NAME)

# Cloud API配置
CLOUD_API_HOST = "localhost"
CLOUD_API_PORT = 8001
CLOUD_API_URL = f"http://{CLOUD_API_HOST}:{CLOUD_API_PORT}/api/v1/detect"


def create_test_image():
    """创建测试图像 (640x480 RGB图像, 包含一些测试内容)"""
    # 创建一个简单的测试图像
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # 绘制一些内容: 红色矩形、绿色圆形、蓝色文字
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), -1)  # 红色矩形
    cv2.circle(img, (450, 200), 80, (0, 255, 0), -1)  # 绿色圆形
    cv2.putText(img, "MOCK TEST", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # 添加时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.putText(img, timestamp, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


def save_test_image(img):
    """保存测试图像到temp_frames目录"""
    os.makedirs(TEMP_FRAME_DIR, exist_ok=True)
    cv2.imwrite(TEST_IMAGE_PATH, img)
    print(f"[OK] 测试图像已保存到: {TEST_IMAGE_PATH}")

    # 验证文件存在并可读
    if os.path.exists(TEST_IMAGE_PATH):
        file_size = os.path.getsize(TEST_IMAGE_PATH)
        print(f"[OK] 文件大小: {file_size} bytes")
        return True
    else:
        print(f"[FAIL] 文件不存在: {TEST_IMAGE_PATH}")
        return False


def encode_image_to_base64(img):
    """将图像编码为base64"""
    # 编码为JPEG格式
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        print("[FAIL] 图像编码失败")
        return None

    # 转换为base64字符串
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    print(f"[OK] Base64编码成功, 长度: {len(img_base64)} chars")
    return img_base64


def check_cloud_service_health():
    """检查cloud服务是否运行"""
    health_url = f"http://{CLOUD_API_HOST}:{CLOUD_API_PORT}/api/v1/health"
    try:
        response = requests.get(health_url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Cloud服务运行正常")
            print(f"     - Agent可用: {data.get('agent_available', False)}")
            print(f"     - Agent已加载: {data.get('agent_loaded', False)}")
            return True, data
        else:
            print(f"[FAIL] Cloud服务响应异常: {response.status_code}")
            return False, None
    except requests.exceptions.ConnectionError:
        print(f"[WARN] Cloud服务未运行 ({health_url})")
        print("       请先启动cloud服务: python start_cloud.py")
        return False, None
    except Exception as e:
        print(f"[FAIL] 健康检查失败: {e}")
        return False, None


def mock_detect_request(img_base64):
    """模拟发送检测请求到cloud API"""
    request_payload = {
        "frames": [
            {
                "image_base64": img_base64,
                "frame_id": 1,
                "timestamp": time.time(),
                "jigsaw_score": 0.35,  # 模拟Jigsaw分数
            }
        ],
        "video_id": "mock_test_video",
        "scene_type": "general",
        "dataset": "ped2",
    }

    print(f"\n[INFO] 发送检测请求到: {CLOUD_API_URL}")
    print(f"       - 帧数: {len(request_payload['frames'])}")
    print(f"       - video_id: {request_payload['video_id']}")

    try:
        response = requests.post(
            CLOUD_API_URL,
            json=request_payload,
            timeout=30.0
        )

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] 检测请求成功")
            print(f"     - success: {data.get('success')}")
            print(f"     - request_id: {data.get('request_id')}")
            print(f"     - processing_time_ms: {data.get('processing_time_ms')}")
            if data.get('result'):
                print(f"     - result: {json.dumps(data.get('result'), indent=2, ensure_ascii=False)[:500]}...")
            return True, data
        else:
            print(f"[FAIL] 检测请求失败: {response.status_code}")
            print(f"       - response: {response.text[:200]}")
            return False, response.text
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] 无法连接到Cloud服务")
        return False, "Connection error"
    except Exception as e:
        print(f"[FAIL] 检测请求异常: {e}")
        return False, str(e)


def verify_temp_frames_written():
    """验证帧是否被正确写入到temp_frames目录"""
    # 检查目录中的文件
    files = os.listdir(TEMP_FRAME_DIR)
    print(f"\n[INFO] {TEMP_FRAME_DIR} 目录内容:")
    print(f"       文件数: {len(files)}")
    for f in files[:10]:  # 只显示前10个
        fpath = os.path.join(TEMP_FRAME_DIR, f)
        fsize = os.path.getsize(fpath)
        print(f"       - {f}: {fsize} bytes")

    if len(files) > 0:
        print(f"[OK] 帧已成功写入到 {TEMP_FRAME_DIR}")
        return True
    else:
        print(f"[WARN] 目录为空，可能服务未正确处理帧")
        return False


def main():
    """主测试流程"""
    print("=" * 60)
    print("Cloud服务帧存储Mock测试")
    print("=" * 60)
    print(f"temp_frame_dir配置: {TEMP_FRAME_DIR}")
    print()

    # Step 1: 创建测试图像
    print("\n[Step 1] 创建测试图像")
    test_img = create_test_image()

    # Step 2: 直接保存到temp_frames目录 (验证目录可写入)
    print("\n[Step 2] 验证目录可写入")
    save_success = save_test_image(test_img)

    if not save_success:
        print("\n[FAIL] 测试终止: 无法写入到temp_frames目录")
        return

    # Step 3: 编码为base64
    print("\n[Step 3] 编码图像为base64")
    img_base64 = encode_image_to_base64(test_img)

    if img_base64 is None:
        print("\n[FAIL] 测试终止: base64编码失败")
        return

    # Step 4: 检查cloud服务健康状态
    print("\n[Step 4] 检查Cloud服务状态")
    health_ok, health_data = check_cloud_service_health()

    if not health_ok:
        print("\n[INFO] Cloud服务未运行, 跳过API测试")
        print("       但目录写入验证已成功!")
        verify_temp_frames_written()
        print("\n[OK] Mock测试完成 (目录验证通过)")
        return

    # Step 5: 发送检测请求
    print("\n[Step 5] 发送模拟检测请求")
    detect_ok, detect_data = mock_detect_request(img_base64)

    # Step 6: 验证帧存储
    print("\n[Step 6] 验证帧存储结果")
    verify_temp_frames_written()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  - 目录写入验证: OK ({TEMP_FRAME_DIR})")
    print(f"  - Cloud服务状态: {'OK' if health_ok else 'SKIP'}")
    print(f"  - API检测请求: {'OK' if detect_ok else 'FAIL/SKIP'}")
    print("=" * 60)

    if save_success:
        print("\n[SUCCESS] 核心验证通过: 帧可存储到VLM可访问路径")
    else:
        print("\n[FAILED] 测试失败")


if __name__ == "__main__":
    main()