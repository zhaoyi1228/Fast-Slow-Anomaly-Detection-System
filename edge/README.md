# 端侧异常检测系统 (Edge Anomaly Detection)

机器狗端侧视频异常检测系统，运行在机器狗本地。

## 功能

- RealSense深度相机视频采集
- Jigsaw快速异常检测（本地推理）
- 检测结果发送到中间机器

## 目录结构

```
edge_anomaly_detection/
├── config.py              # 配置文件
├── camera/
│   └── realsense_client.py   # RealSense相机客户端
├── detection/
│   └── jigsaw_service.py     # Jigsaw检测服务
├── communication/
│   └ relay_client.py         # 中间机器通信客户端
├── run_edge.py             # 主运行脚本
├── start_edge.py           # 服务启动入口
└── requirements.txt        # 依赖列表
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 确保RealSense相机驱动已安装
```

## 配置

编辑 `config.py`，修改以下配置：

```python
# 中间机器IP地址（局域网IP）
RELAY_SERVER = {
    "host": "192.168.1.100",  # 修改为实际IP
    "port": 9000,
}

# Jigsaw模型路径
JIGSAW_MODEL_CONFIG = {
    "checkpoint_path": "pre_trained/stc_78.76_sample7.pth",
    "sample_num": 7,
    "gpu_id": 0,  # GPU ID，空字符串表示CPU
}
```

## Jigsaw 集成说明（修正后）

- `WideBranchNet` 必须按二元类别结构初始化：`num_classes=[sample_num**2, sample_num]`。
- 当前默认 checkpoint `stc_78.76_sample7.pth` 与 `sample_num=7` 绑定。
- 推理分数不再使用简单 `softmax(...).mean()`；当前实现按更接近 ref 验证逻辑的方式：
  - spatial: reshape 为 `sample_num x sample_num` 后取 softmax 对角线最小值
  - temporal: 取时间分支 softmax 的保守最小值
  - anomaly_score: 取 `min(spatial_score, temporal_score)`
- 模型加载失败时：
  - `/health` 会返回 `model_loaded=false`
  - `/detect` / `/batch_detect` 返回 503
  - 不再返回固定 fake score

## 采样与状态说明

- 深度图编码判断使用 `depth_image is not None`，避免 numpy truth-value 异常。
- 当 `sample_fps > actual_fps` 时，采样逻辑会自动将 `frame_skip` 约束为至少 1。
- `run_edge.py` 不再直接访问 relay client 的私有属性，而是通过 `relay_client.get_status()` 获取连接与缓存状态。

## 使用

### 方式1: 分步启动

```bash
# 1. 启动Jigsaw服务
python start_edge.py --start-jigsaw --gpu-id 0

# 2. 启动检测程序（另一个终端）
python start_edge.py --start-detection --relay-host 192.168.1.100
```

### 方式2: 一键启动

```bash
python start_edge.py --relay-host 192.168.1.100 --gpu-id 0
```

### 测试相机

```bash
python -m camera.realsense_client --test
```

## 数据流

```
RealSense相机 → Jigsaw检测 → 发送到中间机器(192.168.1.100:9000)
```

## 断线重连

当网络断开时，帧数据会缓存到本地队列，网络恢复后自动发送。
最大缓存100帧，防止数据丢失。