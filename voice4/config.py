# -*- coding: utf-8 -*-
"""
音频检测系统配置文件
"""

import os

# 音频参数
AUDIO_CONFIG = {
    'sample_rate': 22050,  # 采样率
    'chunk_size': 1024,    # 音频块大小
    'channels': 1,         # 单声道
    'format': 'float32',   # 音频格式
    'overlap_ratio': 0.5,  # 重叠比例
}

# 特征提取参数
FEATURE_CONFIG = {
    'mfcc_n_mfcc': 13,     # MFCC系数数量
    'mfcc_n_fft': 2048,    # FFT窗口大小
    'mfcc_hop_length': 512, # 跳跃长度
    'spectral_n_fft': 2048,
    'spectral_hop_length': 512,
    'chroma_n_fft': 2048,
    'chroma_hop_length': 512,
    'mel_n_mels': 128,     # Mel频谱带数
}

# 检测参数
DETECTION_CONFIG = {
    'mfcc_threshold': 0.65,      # MFCC相似度阈值
    'correlation_threshold': 0.5, # 互相关阈值
    'spectral_threshold': 0.6,   # 频谱质心阈值
    'chroma_threshold': 0.55,    # 色度特征阈值
    'mel_threshold': 0.6,        # Mel频谱阈值
    'min_detection_interval': 1.0, # 最小检测间隔(秒)
    'confidence_weight': {       # 各特征权重
        'mfcc': 0.4,
        'correlation': 0.2,
        'spectral': 0.2,
        'chroma': 0.1,
        'mel': 0.1
    },
    'min_confidence': 0.65,      # 最小置信度 (调整为0.65)
}

# 性能优化参数
PERFORMANCE_CONFIG = {
    'buffer_size': 5,            # 音频缓冲区大小(秒)
    'feature_cache_size': 100,   # 特征缓存大小
    'enable_threading': True,    # 启用多线程
    'max_workers': 2,           # 最大工作线程数
    'enable_gpu': False,        # 启用GPU加速(如果可用)
}

# 文件路径
FILE_CONFIG = {
    'target_audio_path': '/Users/leslack/lsc/300_study/python/demo/voice4/target.wav',
    'log_file': 'detection.log',
    'output_dir': './output',
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'enable_file_log': True,
    'enable_console_log': True,
}

# 调试配置
DEBUG_CONFIG = {
    'enable_debug': True,        # 启用调试模式
    'save_audio_chunks': False,
    'save_features': False,
    'plot_realtime': False,
    'verbose': True,
}

def get_config():
    """获取完整配置"""
    return {
        'audio': AUDIO_CONFIG,
        'feature': FEATURE_CONFIG,
        'detection': DETECTION_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'file': FILE_CONFIG,
        'log': LOG_CONFIG,
        'debug': DEBUG_CONFIG,
    }

def validate_config():
    """验证配置参数"""
    # 检查目标音频文件是否存在
    if not os.path.exists(FILE_CONFIG['target_audio_path']):
        raise FileNotFoundError(f"目标音频文件不存在: {FILE_CONFIG['target_audio_path']}")
    
    # 检查阈值范围
    for key, value in DETECTION_CONFIG.items():
        if 'threshold' in key and not (0 <= value <= 1):
            raise ValueError(f"阈值 {key} 必须在 0-1 范围内: {value}")
    
    # 检查权重总和
    weight_sum = sum(DETECTION_CONFIG['confidence_weight'].values())
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(f"权重总和必须为1.0: {weight_sum}")
    
    return True