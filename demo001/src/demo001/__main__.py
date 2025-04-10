from mcp import MCP
from .bilibili_analyzer import BilibiliAnalyzer
from .config import config
import json

def analyze_bilibili_user(uid: str) -> dict:
    """分析B站用户画像"""
    analyzer = BilibiliAnalyzer()
    return analyzer.analyze_user_profile(uid)

def main():
    # 创建MCP实例
    mcp = MCP(config=config)
    
    # 注册服务
    @mcp.service("analyze_bilibili_user")
    def handle_analyze_bilibili_user(request):
        uid = request.get("uid")
        if not uid:
            return {"error": "缺少uid参数"}
        
        try:
            result = analyze_bilibili_user(uid)
            return result
        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}
    
    # 启动服务
    mcp.run()

if __name__ == "__main__":
    main() 