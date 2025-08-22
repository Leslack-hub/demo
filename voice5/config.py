#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANNs音频检测器配置文件
统一管理所有配置参数，支持不同场景的配置切换
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional


class PANNsConfig:
    """PANNs音频检测器配置类"""
    
    def __init__(self, config_name: str = "default"):
        """初始化配置
        
        Args:
            config_name: 配置名称 (default, performance, accuracy, mobile)
        """
        self.config_name = config_name
        self._load_config()
    
    def _load_config(self):
        """加载指定配置"""
        configs = {
            "default": self._get_default_config(),
            "performance": self._get_performance_config(),
            "accuracy": self._get_accuracy_config(),
            "mobile": self._get_mobile_config()
        }
        
        if self.config_name not in configs:
            raise ValueError(f"未知配置: {self.config_name}")
        
        config = configs[self.config_name]
        for key, value in config.items():
            setattr(self, key, value)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """默认配置 - 平衡性能和准确性"""
        return {
            # 音频参数
            'sample_rate': 32000,
            'chunk_size': 1024,
            'buffer_duration': 10.0,  # 缓冲区时长(秒)
            'window_duration': 3.0,   # 检测窗口时长(秒)
            
            # PANNs模型配置
            'model_arch': 'cnn14',
            'model_path': None,  # 使用预训练模型
            'device': 'auto',    # 自动选择设备
            
            # 检测参数
            'detection_threshold': 0.7,
            'confidence_smoothing': 0.1,
            'min_detection_interval': 0.5,  # 最小检测间隔(秒)
            
            # 特征提取配置
            'normalize': True,
            'trim_silence': True,
            'embedding_weight': 0.5,
            'logits_weight': 0.3,
            'topk_weight': 0.2,
            'top_k': 10,
            
            # 缓存配置
            'cache_size': 100,
            'enable_cache': True,
            
            # 可视化配置
            'enable_visualization': True,
            'plot_update_interval': 0.1,
            'waveform_display_samples': 2000,
            
            # 日志配置
            'log_level': 'INFO',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            
            # 音频设备配置
            'input_device': None,  # 使用默认设备
            'channels': 1,         # 单声道
            'dtype': 'float32'
        }
    
    def _get_performance_config(self) -> Dict[str, Any]:
        """性能优化配置 - 优先考虑速度"""
        config = self._get_default_config()
        config.update({
            'model_arch': 'cnn6',  # 使用轻量级模型
            'detection_threshold': 0.6,
            'cache_size': 50,
            'trim_silence': False,  # 跳过静音裁剪
            'top_k': 5,
            'waveform_display_samples': 1000,
            'plot_update_interval': 0.2
        })
        return config
    
    def _get_accuracy_config(self) -> Dict[str, Any]:
        """准确性优化配置 - 优先考虑检测精度"""
        config = self._get_default_config()
        config.update({
            'model_arch': 'cnn14',
            'detection_threshold': 0.8,
            'confidence_smoothing': 0.05,
            'embedding_weight': 0.6,
            'logits_weight': 0.25,
            'topk_weight': 0.15,
            'top_k': 15,
            'cache_size': 200,
            'window_duration': 5.0,  # 更长的检测窗口
            'min_detection_interval': 0.2
        })
        return config
    
    def _get_mobile_config(self) -> Dict[str, Any]:
        """移动设备配置 - 适用于资源受限环境"""
        config = self._get_default_config()
        config.update({
            'model_arch': 'cnn6',
            'sample_rate': 16000,  # 降低采样率
            'chunk_size': 512,
            'buffer_duration': 5.0,
            'window_duration': 2.0,
            'detection_threshold': 0.5,
            'cache_size': 20,
            'trim_silence': False,
            'top_k': 3,
            'enable_visualization': False,  # 关闭可视化
            'waveform_display_samples': 500
        })
        return config
    
    def get_device(self) -> torch.device:
        """获取计算设备"""
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(self.device)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_config(self, file_path: str):
        """保存配置到文件"""
        import json
        config_dict = self.to_dict()
        # 处理不能序列化的对象
        config_dict['device'] = str(self.get_device())
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, file_path: str) -> 'PANNsConfig':
        """从文件加载配置"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if key != 'config_name':
                setattr(config, key, value)
        
        return config


# 预定义配置实例
DEFAULT_CONFIG = PANNsConfig("default")
PERFORMANCE_CONFIG = PANNsConfig("performance")
ACCURACY_CONFIG = PANNsConfig("accuracy")
MOBILE_CONFIG = PANNsConfig("mobile")


def get_config(config_name: str = "default") -> PANNsConfig:
    """获取指定配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        配置实例
    """
    return PANNsConfig(config_name)


def create_custom_config(**kwargs) -> PANNsConfig:
    """创建自定义配置
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        自定义配置实例
    """
    config = PANNsConfig("default")
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"未知配置参数: {key}")
    
    return config


if __name__ == "__main__":
    # 配置使用示例
    print("=== PANNs配置示例 ===")
    
    # 使用默认配置
    config = get_config("default")
    print(f"默认配置设备: {config.get_device()}")
    print(f"默认检测阈值: {config.detection_threshold}")
    
    # 使用性能配置
    perf_config = get_config("performance")
    print(f"性能配置模型: {perf_config.model_arch}")
    
    # 创建自定义配置
    custom_config = create_custom_config(
        detection_threshold=0.9,
        model_arch='cnn10'
    )
    print(f"自定义配置阈值: {custom_config.detection_threshold}")
    
    # 保存配置
    config.save_config("panns_config.json")
    print("配置已保存到 panns_config.json")