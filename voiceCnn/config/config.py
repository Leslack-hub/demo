# -*- coding: utf-8 -*-
"""
声音识别系统配置文件
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径配置
DATA_CONFIG = {
    'positive_samples': PROJECT_ROOT / 'data' / 'positive_samples',
    'negative_samples': PROJECT_ROOT / 'data' / 'negative_samples', 
    'background_noise': PROJECT_ROOT / 'data' / 'background_noise',
    'processed': PROJECT_ROOT / 'data' / 'processed'
}

# 模型路径配置
MODEL_CONFIG = {
    'save_dir': PROJECT_ROOT / 'models',
    'checkpoint_dir': PROJECT_ROOT / 'models' / 'checkpoints',
    'best_model': PROJECT_ROOT / 'models' / 'best_model.h5',
    'lite_model': PROJECT_ROOT / 'models' / 'model.tflite'
}

# 音频处理配置
AUDIO_CONFIG = {
    'sample_rate': 22050,  # 采样率
    'duration': 2.0,       # 音频片段长度(秒)
    'hop_length': 512,     # 跳跃长度
    'n_mels': 128,         # 梅尔频谱图的频率bins数量
    'n_fft': 2048,         # FFT窗口大小
    'window_size': 0.1,    # 实时检测窗口大小(秒)
    'overlap': 0.05        # 窗口重叠(秒)
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'time_stretch_factors': [0.8, 0.9, 1.1, 1.2],  # 时间拉伸因子
    'pitch_shift_steps': [-2, -1, 1, 2],           # 音调变换步数
    'noise_factor': 0.005,                          # 噪声因子
    'snr_range': [5, 20],                          # 信噪比范围(dB)
    'mix_probability': 0.8                          # 混音概率
}

# 模型训练配置
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7
}

# 实时检测配置
DETECTION_CONFIG = {
    'confidence_threshold': 0.95,  # 置信度阈值
    'buffer_size': 1024,           # 音频缓冲区大小
    'device_index': None,          # 音频设备索引(None为默认)
    'channels': 1,                 # 声道数
    'format': 'float32'            # 音频格式
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': PROJECT_ROOT / 'logs'
}

# 输出配置
OUTPUT_CONFIG = {
    'output_dir': PROJECT_ROOT / 'output',
    'save_predictions': True,
    'save_spectrograms': False
}

# 创建必要的目录
for config in [DATA_CONFIG, MODEL_CONFIG, LOGGING_CONFIG, OUTPUT_CONFIG]:
    for path in config.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)