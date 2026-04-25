# 云侧异常检测系统 (Cloud Anomaly Detection)

云侧服务器 Agent 深度分析服务。

## 功能

- 接收中间节点发送的帧数据
- VLM+LLM深度异常分析
- Memory增强检测
- 三层Memory系统（Working/Episodic/Semantic）

## 目录结构

```
cloud_anomaly_detection/
├── config.py              # 配置文件
├── service/
│   └── api_server.py      # FastAPI服务
├── agent/
│   └── __init__.py        # Agent组件（从原项目导入）
├── models/
│   └── __init__.py        # 模型配置
├── start_cloud.py         # 启动入口
├── requirements.txt       # 依赖列表
```

## 安装

```bash
pip install -r requirements.txt
```

**注意**: Agent核心组件从原始项目导入，请确保路径正确。

## 配置

优先通过环境变量配置，而不是在 API 层硬编码外部依赖路径：

```python
API_SERVER = {
    "host": "0.0.0.0",
    "port": 8001,
}

AGENT_CONFIG = {
    "config_path": "AnomalyAgent/config/api_config.yaml",
    "temp_frame_dir": "./temp_frames",
    "memory_checkpoint_dir": "./memory_checkpoints",
}
```

推荐环境变量：

```bash
export ANOMALY_AGENT_PROJECT_PATH=/abs/path/to/AnomalyAgent
export ANOMALY_AGENT_CONFIG_PATH=/abs/path/to/api_config.yaml
export SEMANTIC_SIMILARITY_MODEL_PATH=/abs/path/to/all-MiniLM-L6-v2
export API_HOST=0.0.0.0
export API_PORT=8001
```

## Cloud 包装契约（修正后）

- cloud 不再自己手搓 `UnifiedMemorySystem(...)`，而是复用：
  - `AnomalyAgent/main_memory_vad.py`
  - `AnomalyAgent/api/handlers/frame_processor.py`
  - `AnomalyAgent/api/handlers/detection_handler.py`
- 实际调用链路是 `MemoryEnhancedVADAgent.run_with_memory()`。
- base64 帧会先写入请求级临时目录，再以目录路径作为 `source` 传入 agent。
- API 错误语义：
  - 非法请求 / 非法帧输入：`400`
  - Agent 未初始化或依赖不可用：`503`
  - 运行期内部异常：`500`
- relay 传入的额外字段如 `frame_id / timestamp / jigsaw_score` 会被接受，并可用于追踪调试。

## 使用

```bash
# 启动云侧服务
python start_cloud.py --port 8001 --agent-path /abs/path/to/AnomalyAgent --config-path /abs/path/to/api_config.yaml

# 或仅依赖环境变量
python start_cloud.py --port 8001
```

## API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/detect` | POST | 异常检测 |
| `/api/v1/memory/stats` | GET | Memory统计 |
| `/api/v1/memory/reset` | POST | 重置Memory |
| `/api/v1/memory/save` | POST | 保存Memory |
| `/api/v1/config` | GET | 获取配置 |

### 检测请求示例

```bash
curl -X POST http://localhost:8001/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "frames": [{"image_base64": "..."}],
    "scene_type": "general",
    "dataset": "ped2"
  }'
```

## 响应格式

```json
{
  "success": true,
  "request_id": "uuid",
  "result": {
    "anomaly_score": 0.85,
    "is_anomaly": true,
    "anomaly_type": "running",
    "description": "检测到奔跑行为",
    "explanation": "人物在视频中快速移动...",
    "frame_scores": [0.8, 0.85, 0.9]
  },
  "processing_time_ms": 5000
}
```

## Memory系统

三层Memory架构：
- **Working Memory**: 短期场景历史
- **Episodic Memory**: 经验和反思历史
- **Semantic Memory**: 动态规则库

Memory 跨请求持久化，持续学习改进检测准确率。