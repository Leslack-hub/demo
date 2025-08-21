#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标音频分析脚本
分析target.wav的详细特征，用于优化检测参数
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from audio_features import extract_audio_features
from config import get_config

def analyze_target_audio():
    """分析目标音频的详细特征"""
    config = get_config()
    target_path = config['file']['target_audio_path']
    
    print(f"=== 分析目标音频: {target_path} ===")
    
    # 加载音频
    y, sr = librosa.load(target_path, sr=config['audio']['sample_rate'])
    
    print(f"\n基本信息:")
    print(f"  采样率: {sr} Hz")
    print(f"  时长: {len(y)/sr:.2f} 秒")
    print(f"  样本数: {len(y)}")
    print(f"  音频范围: {np.min(y):.3f} 到 {np.max(y):.3f}")
    print(f"  RMS能量: {np.sqrt(np.mean(y**2)):.3f}")
    print(f"  峰值: {np.max(np.abs(y)):.3f}")
    
    # 提取特征
    features = extract_audio_features(y, sr, config['feature'])
    
    print(f"\n特征分析:")
    print(f"  MFCC形状: {features['mfcc'].shape}")
    print(f"  MFCC均值: {np.mean(features['mfcc'], axis=1)[:5]}")
    print(f"  MFCC标准差: {np.std(features['mfcc'], axis=1)[:5]}")
    
    print(f"  频谱质心均值: {np.mean(features['spectral_centroid']):.1f}")
    print(f"  频谱质心范围: {np.min(features['spectral_centroid']):.1f} - {np.max(features['spectral_centroid']):.1f}")
    
    print(f"  色度特征形状: {features['chroma'].shape}")
    print(f"  色度特征均值: {np.mean(features['chroma'], axis=1)[:5]}")
    
    print(f"  Mel频谱形状: {features['mel_spectrogram'].shape}")
    print(f"  Mel频谱均值: {np.mean(features['mel_spectrogram'], axis=1)[:5]}")
    
    # 分析音频活动区域
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 找到有效音频区域（RMS > 阈值）
    rms_threshold = np.mean(rms) * 0.1
    active_frames = rms > rms_threshold
    active_ratio = np.sum(active_frames) / len(active_frames)
    
    print(f"\n音频活动分析:")
    print(f"  RMS阈值: {rms_threshold:.4f}")
    print(f"  活动帧比例: {active_ratio:.2%}")
    print(f"  平均RMS: {np.mean(rms):.4f}")
    print(f"  最大RMS: {np.max(rms):.4f}")
    
    # 建议的检测参数
    print(f"\n=== 建议的检测参数 ===")
    
    # 基于实际特征调整阈值
    mfcc_var = np.var(features['mfcc'])
    spectral_var = np.var(features['spectral_centroid'])
    
    print(f"建议MFCC阈值: 0.6-0.8 (当前方差: {mfcc_var:.3f})")
    print(f"建议频谱阈值: 0.4-0.7 (当前方差: {spectral_var:.1f})")
    print(f"建议Mel阈值: 0.6-0.8")
    print(f"建议总体置信度阈值: 0.6-0.7")
    
    # 基于音频能量调整
    avg_energy = np.sqrt(np.mean(y**2))
    if avg_energy < 0.1:
        print(f"\n注意: 音频能量较低 ({avg_energy:.3f})，建议:")
        print(f"  - 降低能量阈值")
        print(f"  - 增加音频增益")
        print(f"  - 调整录音设备音量")
    
    return features

if __name__ == "__main__":
    analyze_target_audio()