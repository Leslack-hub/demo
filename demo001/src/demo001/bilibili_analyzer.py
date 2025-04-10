import requests
from bs4 import BeautifulSoup
from typing import Dict, Any
import json
import time

class BilibiliAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com/'
        }
    
    def get_user_info(self, uid: str) -> Dict[str, Any]:
        """获取用户信息"""
        url = f'https://api.bilibili.com/x/space/wbi/acc/info?mid={uid}'
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if data['code'] == 0:
                return data['data']
            return {}
        except Exception as e:
            print(f"获取用户信息失败: {str(e)}")
            return {}
    
    def get_user_videos(self, uid: str) -> Dict[str, Any]:
        """获取用户视频列表"""
        url = f'https://api.bilibili.com/x/space/wbi/arc/search?mid={uid}'
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if data['code'] == 0:
                return data['data']
            return {}
        except Exception as e:
            print(f"获取用户视频列表失败: {str(e)}")
            return {}
    
    def analyze_user_profile(self, uid: str) -> Dict[str, Any]:
        """分析用户画像"""
        user_info = self.get_user_info(uid)
        if not user_info:
            return {"error": "无法获取用户信息"}
        
        videos_info = self.get_user_videos(uid)
        if not videos_info:
            return {"error": "无法获取用户视频信息"}
        
        # 基础信息
        profile = {
            "uid": uid,
            "name": user_info.get('name', ''),
            "sign": user_info.get('sign', ''),
            "level": user_info.get('level', 0),
            "video_count": videos_info.get('page', {}).get('count', 0),
            "total_views": 0,
            "total_likes": 0,
            "total_favorites": 0,
            "total_coins": 0,
            "total_shares": 0
        }
        
        # 分析视频数据
        videos = videos_info.get('list', {}).get('vlist', [])
        for video in videos:
            profile['total_views'] += video.get('play', 0)
            profile['total_likes'] += video.get('like', 0)
            profile['total_favorites'] += video.get('favorite', 0)
            profile['total_coins'] += video.get('coins', 0)
            profile['total_shares'] += video.get('share', 0)
        
        # 用户画像分析
        profile['user_type'] = self._determine_user_type(profile)
        profile['content_quality'] = self._evaluate_content_quality(profile)
        
        return profile
    
    def _determine_user_type(self, profile: Dict[str, Any]) -> str:
        """确定用户类型"""
        video_count = profile['video_count']
        total_views = profile['total_views']
        
        if video_count == 0:
            return "普通用户"
        elif video_count < 10:
            return "轻度创作者"
        elif video_count < 50:
            return "中度创作者"
        elif video_count < 200:
            return "重度创作者"
        else:
            return "专业UP主"
    
    def _evaluate_content_quality(self, profile: Dict[str, Any]) -> str:
        """评估内容质量"""
        if profile['video_count'] == 0:
            return "无内容"
            
        avg_views = profile['total_views'] / profile['video_count']
        avg_likes = profile['total_likes'] / profile['video_count']
        
        if avg_views < 1000:
            return "一般"
        elif avg_views < 10000:
            return "良好"
        elif avg_views < 100000:
            return "优秀"
        else:
            return "极佳" 