# 中间节点异常检测系统 (Relay Anomaly Detection)

中间转发节点，运行在转发机器上，连接端侧和云侧。

## 功能

- 接收机器狗端侧的帧数据
- 转发到云侧Agent进行深度分析
- 聚合检测结果，生成劝阻文本
- Gradio可视化界面（本地查看）

## 目录结构

```
relay_anomaly_detection/
├── config.py              # 配置文件
├── receiver/
│   ├── edge_receiver.py   # 端侧数据接收服务
│   └── frame_buffer.py    # 帧缓冲管理
├── sender/
│   └── cloud_client.py    # 云侧通信客户端
├── aggregator/
│   └── result_aggregator.py   # 结果聚合器
├── visualization/
│   └── gradio_app.py      # Gradio可视化界面
├── start_relay.py         # 启动入口
└ requirements.txt         # 依赖列表
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置

编辑 `config.py`，修改以下配置：

```python
# 云侧服务器IP（VPN IP）
CLOUD_SERVER = {
    "host": "10.8.0.1",  # 云侧VPN IP
    "port": 8001,
}

# 端侧接收端口（监听机器狗）
EDGE_RECEIVER = {
    "host": "0.0.0.0",
    "port": 9000,
}

# Gradio本地界面端口
GRADIO_SERVER = {
    "host": "127.0.0.1",  # 仅本地访问
    "port": 7860,
}
```

## 使用

```bash
# 启动中间节点
python start_relay.py --cloud-host 10.8.0.1 --gradio-port 7860
```

启动后访问: http://127.0.0.1:7860

## 数据流

```
机器狗(局域网) → 本机(接收端口9000) → 云侧(VPN:10.8.0.1:8001) → 本机(Gradio显示)
```

## 功能说明

### 接收服务 (端口9000)
- 接收端侧发送的帧数据和Jigsaw检测结果
- 维护帧缓冲区，支持滑动窗口分析

### 结果聚合
- 维护5秒滑动窗口
- 当窗口内异常帧比例超过30%，触发云侧深度分析
- 生成劝阻文本
- 深度分析网络调用在锁外执行，避免聚合器因重复加锁或锁内阻塞而卡死
- UI 展示通过 `ResultAggregator(frame_callback=...)` 回调更新

## Relay / Cloud 边界说明（修正后）

- relay 向 cloud 上传的帧结构至少包含：
  - `image_base64`
  - `frame_id`
  - `timestamp`
  - `jigsaw_score`
- cloud 可以接受这些扩展元数据并忽略不需要的字段。
- 当 cloud 不可用时：
  - relay 不会崩溃
  - 当前帧仍会产生本地 `suspicious/normal` 决策
  - 深度分析结果退化为错误态，不影响后续窗口继续处理

### Gradio界面
- 实时视频显示
- Jigsaw分数曲线
- 异常事件历史
- 深度分析结果展示