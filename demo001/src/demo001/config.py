from mcp import Config

config = Config(
    name="bilibili_analyzer",
    version="0.1.0",
    description="B站用户画像分析服务",
    host="0.0.0.0",
    port=8000,
    workers=4,
    timeout=30,
    max_requests=1000,
    max_requests_jitter=50
) 