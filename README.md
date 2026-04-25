# 异常检测系统 (Anomaly Detection System)

三节点分离部署的视频异常检测系统，用于机器狗RealSense相机实时检测。

## 系统架构

```
机器狗(端侧) ←局域网→ 中间机器(转发) ←VPN→ 云侧服务器
     ↓                  ↓                      ↓
 RealSense采集        视频帧转发              Agent深度分析
 Jigsaw快速检测       结果聚合                VLM+LLM推理
 发送帧+分数          Gradio可视化            Memory增强
```

## 目录结构

```
anomaly_detection_system/
├── edge/           # 端侧仓库（机器狗）
│   ├── camera/         # RealSense相机客户端
│   ├── detection/      # Jigsaw检测服务
│   │   └── models/     # WideBranchNet模型
│   ├── communication/  # 中间机器通信客户端
│   ├── config.py       # 配置文件
│   ├── run_edge.py     # 主运行脚本
│   ├── start_edge.py   # 启动入口
│   └── README.md       # 使用说明
│
├── relay/          # 中间节点仓库（转发机器）
│   ├── receiver/       # 端侧数据接收服务
│   ├── sender/         # 云侧通信客户端
│   ├── aggregator/     # 结果聚合器
│   ├── visualization/  # Gradio可视化
│   ├── config.py       # 配置文件
│   ├── start_relay.py  # 启动入口
│   └── README.md       # 使用说明
│
├── cloud/          # 云侧仓库（服务器）
│   ├── service/        # FastAPI服务
│   ├── agent/          # Agent组件（从原项目导入）
│   ├── config/
│   │   └── initial_rules.json  # 异常规则库
│   ├── config.py       # 配置文件
│   ├── start_cloud.py  # 启动入口
│   └── README.md       # 使用说明
│
└── README.md       # 本文件
```

## 快速启动

### 1. 端侧（机器狗）
```bash
cd edge

# 配置中间机器IP（修改config.py）
RELAY_SERVER = {"host": "中间机器局域网IP", "port": 9000}

# 启动
python start_edge.py --relay-host 192.168.x.x --gpu-id 0
```

### 2. 中间机器
```bash
cd relay

# 配置云侧VPN IP（修改config.py）
CLOUD_SERVER = {"host": "云侧VPN_IP", "port": 8001}

# 启动（访问 http://127.0.0.1:7860 查看结果）
python start_relay.py --cloud-host 10.8.0.x
```

### 3. 云侧服务器
```bash
cd cloud

# 启动Agent服务
python start_cloud.py --port 8001
```

## 依赖说明

### 端侧
- pyrealsense2 (RealSense相机)
- torch (Jigsaw模型推理)
- flask (本地Jigsaw服务)
- requests (HTTP通信)

### 中间节点
- flask (接收服务)
- gradio (可视化界面)
- requests (HTTP通信)

### 云侧
- fastapi (API服务)
- uvicorn (ASGI服务器)
- 从 `/home/zhaoyi/media/projects_zy/anomaly_agent` 导入Agent组件

## 数据流

1. **端侧采集**: RealSense相机采集视频帧 → 转base64
2. **本地检测**: Jigsaw快速检测（~45ms） → 生成异常分数
3. **发送中间**: HTTP POST到中间机器端口9000
4. **结果聚合**: 滑动窗口分析 → 判断是否触发深度分析
5. **转发云侧**: 通过VPN发送帧到云侧Agent
6. **深度分析**: VLM+LLM分析 → 返回异常类型和描述
7. **可视化**: Gradio实时显示检测结果

## 当前集成契约（修正后）

- **Edge / Jigsaw**
  - `WideBranchNet` 按 `num_classes=[spatial_classes, temporal_classes]` 初始化，不再传单个整数。
  - `sample7` checkpoint 默认绑定 `sample_num=7`，空间分支按 `7x7` reshape 后取 softmax 对角线最小值作为主置信分数。
  - `/health` 的 `model_loaded` 才表示模型真实可用；模型未加载时 `/detect` 返回错误，不再返回 fake score。
- **Relay**
  - 触发策略是“滑动窗口内 Jigsaw 异常帧比例超过阈值”。
  - 深度分析请求在聚合器锁外执行，避免触发链路卡死。
  - UI 更新通过 `frame_callback` 正式参数接入，而不是写私有属性。
- **Cloud**
  - cloud 实际包装的是 `MemoryEnhancedVADAgent.run_with_memory()`，不再调用不存在的方法。
  - cloud 复用 `AnomalyAgent/main_memory_vad.py` 与 `AnomalyAgent/api/handlers/*` 的建模/帧落盘链路。
  - 非法输入返回 4xx，Agent 未就绪返回 503，内部异常返回 500。

## 配置要点

### IP配置
- 端侧 `config.py`: 设置中间机器局域网IP
- 中间 `config.py`: 设置云侧VPN IP
- 所有IP通过配置文件设置，部署时手动修改

### 断线重连
- 端侧到中间: 最大重试5次，本地缓存100帧
- 中间到云侧: 最大重试3次，后台健康检查

### 检测参数
- Jigsaw阈值: 0.4（低于此值认为可能异常）
- 滑动窗口: 5秒
- 触发深度分析: 窗口内异常帧比例 > 30%

## 模型文件

端侧Jigsaw模型需要从Unified_Jigsaw项目复制：
```bash
# 复制预训练权重到 edge/detection/pre_trained/
cp /home/zhaoyi/media/data3/Unified_Jigsaw/pre_trained/stc_78.76_sample7.pth \
   anomaly_detection_system/edge/detection/pre_trained/
```

## 注意事项

1. **Agent组件**: 云侧服务依赖原始 `anomaly_agent` 项目，请确保路径正确
2. **VPN连接**: 中间机器需要通过VPN连接云侧服务器
3. **GPU**: 端侧Jigsaw推理需要GPU（或CPU模式）
4. **VLM/LLM**: 云侧Agent需要配置VLM和LLM服务地址
5. **环境变量**: cloud 侧优先使用 `ANOMALY_AGENT_PROJECT_PATH`、`ANOMALY_AGENT_CONFIG_PATH`、`SEMANTIC_SIMILARITY_MODEL_PATH` 配置外部依赖路径