"""
Start Cloud Service
云侧服务启动脚本 - Agent深度分析服务
"""

import os
import sys
import argparse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import API_SERVER, RESOURCE_PATHS, AGENT_CONFIG


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='云侧Agent服务启动')
    parser.add_argument('--port', type=int, default=API_SERVER["port"],
                        help='服务端口')
    parser.add_argument('--host', type=str, default=API_SERVER["host"],
                        help='服务主机')
    parser.add_argument('--workers', type=int, default=API_SERVER["workers"],
                        help='Worker数量')
    parser.add_argument('--agent-path', type=str,
                        default=RESOURCE_PATHS["anomaly_agent_project_path"],
                        help='AnomalyAgent项目路径')
    parser.add_argument('--config-path', type=str,
                        default=AGENT_CONFIG["config_path"],
                        help='AnomalyAgent API/YAML 配置路径')

    args = parser.parse_args()

    # 将 CLI 覆盖同步回共享配置，确保后续导入的 api_server
    # 读取到的是本次启动参数，而不是模块加载时的默认值。
    API_SERVER["host"] = args.host
    API_SERVER["port"] = args.port
    RESOURCE_PATHS["anomaly_agent_project_path"] = args.agent_path
    AGENT_CONFIG["config_path"] = args.config_path

    # 设置Agent路径环境变量
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["ANOMALY_AGENT_PROJECT_PATH"] = args.agent_path
    os.environ["ANOMALY_AGENT_CONFIG_PATH"] = args.config_path

    print("=" * 50)
    print("云侧Agent服务启动")
    print("=" * 50)
    print(f"服务地址: {args.host}:{args.port}")
    print(f"Worker数量: {args.workers}")
    print(f"Agent项目路径: {args.agent_path}")
    print(f"Agent配置路径: {args.config_path}")
    print("=" * 50)

    # 导入并启动API服务
    from service.api_server import app
    import time
    app.state.start_time = time.time()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers
    )


if __name__ == "__main__":
    main()