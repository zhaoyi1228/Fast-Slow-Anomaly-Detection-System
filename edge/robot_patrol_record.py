"""
Robot Patrol with Video Recording
机器狗巡逻录制脚本 - 控制机器狗行走并录制视频

功能：
1. 开启避障开关
2. 机器狗向前前进指定距离（可通过参数调节）
3. 360度转身
4. 往回走相同距离
5. 全程录制高质量视频
"""

import os
import sys
import time
import signal
import argparse
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from dataclasses import dataclass
from typing import Optional, Dict, Any

# 添加 unitree SDK 路径
unitree_sdk_path = "/Users/zhaoyi/Documents/projects/mcislab-projects/unitree_sdk2_python"
if os.path.exists(unitree_sdk_path):
    sys.path.insert(0, unitree_sdk_path)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient


@dataclass
class PatrolConfig:
    """巡逻配置"""
    forward_distance: float = 10.0  # 前进距离（米）
    walk_speed: float = 0.3  # 行走速度（米/秒）
    turn_speed: float = 0.5  # 转身速度（弧度/秒）
    turn_angle: float = 2 * np.pi  # 转身角度（360度 = 2π）
    video_resolution: tuple = (640, 480)  # 视频分辨率
    video_fps: int = 30  # 视频帧率


class RealSenseRecorder:
    """RealSense视频录制器"""

    def __init__(self, config: PatrolConfig):
        self.config = config
        self.pipeline = None
        self.is_recording = False
        self.video_writer = None
        self.record_thread = None
        self._stop_event = threading.Event()
        self.frame_count = 0
        self.output_path = None

    def start_recording(self, output_path: str) -> bool:
        """
        开始录制视频

        Args:
            output_path: 输出视频路径

        Returns:
            bool: 是否成功启动
        """
        try:
            # 初始化 RealSense
            self.pipeline = rs.pipeline()
            config_rs = rs.config()
            config_rs.enable_stream(
                rs.stream.color,
                self.config.video_resolution[0],
                self.config.video_resolution[1],
                rs.format.bgr8,
                self.config.video_fps
            )
            self.pipeline.start(config_rs)

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config.video_fps,
                self.config.video_resolution
            )
            self.output_path = output_path

            self.is_recording = True
            self._stop_event.clear()
            self.frame_count = 0

            # 启动录制线程
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()

            print(f"视频录制已启动: {output_path}")
            return True

        except Exception as e:
            print(f"视频录制启动失败: {e}")
            return False

    def _record_loop(self):
        """录制循环"""
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    self.video_writer.write(color_image)
                    self.frame_count += 1

            except Exception as e:
                print(f"录制错误: {e}")
                time.sleep(0.01)

    def stop_recording(self):
        """停止录制"""
        self._stop_event.set()
        self.is_recording = False

        if self.record_thread:
            self.record_thread.join(timeout=2.0)
            self.record_thread = None

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

        print(f"视频录制已停止，共录制 {self.frame_count} 帧")
        print(f"视频保存路径: {self.output_path}")

    def get_status(self) -> Dict[str, Any]:
        """获取录制状态"""
        return {
            "is_recording": self.is_recording,
            "frame_count": self.frame_count,
            "output_path": self.output_path,
        }


class RobotController:
    """机器狗控制器"""

    def __init__(self, config: PatrolConfig):
        self.config = config
        self.sport_client = None
        self.oa_client = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """初始化机器狗连接"""
        try:
            # 初始化 DDS 通信
            ChannelFactoryInitialize(0)

            # 初始化运动控制客户端
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()

            # 初始化避障客户端
            self.oa_client = ObstaclesAvoidClient()
            self.oa_client.SetTimeout(5.0)
            self.oa_client.Init()

            self.is_initialized = True
            print("机器狗连接初始化成功")
            return True

        except Exception as e:
            print(f"机器狗连接初始化失败: {e}")
            return False

    def stand_up(self):
        """机器狗站起"""
        print("机器狗站起...")
        self.sport_client.StandUp()
        time.sleep(2.0)

    def enable_obstacle_avoidance(self, enable: bool = True):
        """开启/关闭避障"""
        print(f"避障开关: {'开启' if enable else '关闭'}")
        self.oa_client.SwitchSet(enable)
        self.oa_client.UseRemoteCommandFromApi(True)
        time.sleep(0.5)

    def move_forward(self, distance: float):
        """
        向前移动指定距离

        Args:
            distance: 移动距离（米）
        """
        # 计算移动时间
        move_time = distance / self.config.walk_speed

        print(f"向前移动 {distance:.1f} 米，速度 {self.config.walk_speed:.2f} m/s，预计时间 {move_time:.1f} 秒")

        # 使用避障移动
        self.oa_client.Move(self.config.walk_speed, 0.0, 0.0)
        time.sleep(move_time)

        # 停止移动
        self.oa_client.Move(0.0, 0.0, 0.0)
        print("前进完成")

    def move_backward(self, distance: float):
        """
        向后移动指定距离

        Args:
            distance: 移动距离（米）
        """
        move_time = distance / self.config.walk_speed

        print(f"向后移动 {distance:.1f} 米，速度 {self.config.walk_speed:.2f} m/s，预计时间 {move_time:.1f} 秒")

        # 使用避障移动（负速度表示后退）
        self.oa_client.Move(-self.config.walk_speed, 0.0, 0.0)
        time.sleep(move_time)

        # 停止移动
        self.oa_client.Move(0.0, 0.0, 0.0)
        print("后退完成")

    def turn(self, angle: float):
        """
        转身指定角度

        Args:
            angle: 转身角度（弧度），正值左转，负值右转
        """
        turn_time = abs(angle) / self.config.turn_speed
        vyaw = self.config.turn_speed if angle > 0 else -self.config.turn_speed

        print(f"转身 {angle:.2f} 弧度 ({angle * 180 / np.pi:.1f} 度)，预计时间 {turn_time:.1f} 秒")

        self.oa_client.Move(0.0, 0.0, vyaw)
        time.sleep(turn_time)

        # 停止转身
        self.oa_client.Move(0.0, 0.0, 0.0)
        print("转身完成")

    def turn_360(self):
        """360度转身"""
        self.turn(self.config.turn_angle)

    def stand_down(self):
        """机器狗趴下"""
        print("机器狗趴下...")
        self.sport_client.StandDown()
        time.sleep(2.0)


class PatrolMission:
    """巡逻任务执行器"""

    def __init__(self, config: PatrolConfig):
        self.config = config
        self.robot = RobotController(config)
        self.recorder = RealSenseRecorder(config)
        self.is_running = False

    def run(self, video_output: str) -> bool:
        """
        执行巡逻任务

        Args:
            video_output: 视频输出路径

        Returns:
            bool: 任务是否成功
        """
        # 初始化机器狗
        if not self.robot.initialize():
            print("无法初始化机器狗，任务终止")
            return False

        self.is_running = True

        try:
            # 开始录制视频
            if not self.recorder.start_recording(video_output):
                print("视频录制启动失败，但继续执行任务")

            # 机器狗站起
            self.robot.stand_up()
            time.sleep(1.0)

            # 开启避障
            self.robot.enable_obstacle_avoidance(True)

            # 向前前进
            self.robot.move_forward(self.config.forward_distance)

            # 稍作停顿
            time.sleep(2.0)

            # 360度转身
            self.robot.turn_360()

            # 稍作停顿
            time.sleep(2.0)

            # 往回走
            self.robot.move_forward(self.config.forward_distance)

            # 关闭避障
            self.robot.enable_obstacle_avoidance(False)

            # 机器狗趴下
            self.robot.stand_down()

            print("\n巡逻任务完成!")

        except Exception as e:
            print(f"任务执行错误: {e}")
            self.is_running = False
            return False

        finally:
            # 停止录制
            self.recorder.stop_recording()
            self.is_running = False

        return True

    def stop(self):
        """停止任务"""
        self.is_running = False
        self.recorder.stop_recording()
        self.robot.enable_obstacle_avoidance(False)
        self.robot.stand_down()


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='机器狗巡逻录制脚本')

    parser.add_argument('--distance', type=float, default=10.0,
                        help='前进距离（米），默认10米')
    parser.add_argument('--speed', type=float, default=0.3,
                        help='行走速度（米/秒），默认0.3')
    parser.add_argument('--turn-speed', type=float, default=0.5,
                        help='转身速度（弧度/秒），默认0.5')
    parser.add_argument('--output', type=str, default='patrol_video.mp4',
                        help='视频输出路径，默认 patrol_video.mp4')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='视频分辨率，默认 640x480')
    parser.add_argument('--fps', type=int, default=30,
                        help='视频帧率，默认 30')

    args = parser.parse_args()

    # 解析分辨率
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        resolution = (640, 480)
        print(f"分辨率格式错误，使用默认值 {resolution}")

    # 创建配置
    config = PatrolConfig(
        forward_distance=args.distance,
        walk_speed=args.speed,
        turn_speed=args.turn_speed,
        video_resolution=resolution,
        video_fps=args.fps,
    )

    print("=" * 50)
    print("机器狗巡逻录制任务")
    print("=" * 50)
    print(f"前进距离: {config.forward_distance} 米")
    print(f"行走速度: {config.walk_speed} 米/秒")
    print(f"转身速度: {config.turn_speed} 弧度/秒")
    print(f"视频分辨率: {config.video_resolution}")
    print(f"视频帧率: {config.video_fps}")
    print(f"视频输出: {args.output}")
    print("=" * 50)

    # 创建任务执行器
    mission = PatrolMission(config)

    # 注册信号处理
    def signal_handler(sig, frame):
        print("\n收到中断信号，停止任务...")
        mission.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 执行任务
    success = mission.run(args.output)

    if success:
        print("\n任务成功完成!")
    else:
        print("\n任务执行失败!")


if __name__ == "__main__":
    main()