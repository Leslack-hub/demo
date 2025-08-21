#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试噪声敏感性问题
"""

import numpy as np
from audio_features import extract_audio_features
from config import get_config
import librosa

def debug_noise_sensitivity():
    """调试噪声敏感性"""
    config = get_config()
    target_path = config['file']['target_audio_path']
    
    print(f"=== 调试噪声敏感性 ===")
    
    # 加载目标音频
    y, sr = librosa.load(target_path, sr=config['audio']['sample_rate'])
    target_features = extract_audio_features(y, sr, config['feature'])
    
    # 测试不同噪声级别
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    for noise_level in noise_levels:
        print(f"\n--- 噪声级别: {noise_level} ---")
        
        # 添加噪声
        noisy_audio = y + np.random.normal(0, noise_level, y.shape)
        noisy_features = extract_audio_features(noisy_audio, sr, config['feature'])
        
        # 计算特征距离
        target_mfcc = np.mean(target_features['mfcc'], axis=1)
        noisy_mfcc = np.mean(noisy_features['mfcc'], axis=1)
        mfcc_distance = np.linalg.norm(target_mfcc - noisy_mfcc)
        
        target_mel = np.mean(target_features['mel_spectrogram'], axis=1)
        noisy_mel = np.mean(noisy_features['mel_spectrogram'], axis=1)
        mel_distance = np.linalg.norm(target_mel - noisy_mel)
        
        target_centroid = np.mean(target_features['spectral_centroid'])
        noisy_centroid = np.mean(noisy_features['spectral_centroid'])
        spectral_diff = abs(target_centroid - noisy_centroid)
        
        print(f"  MFCC距离: {mfcc_distance:.2f}")
        print(f"  Mel距离: {mel_distance:.2f}")
        print(f"  频谱质心差异: {spectral_diff:.1f}")
        
        # 计算相似度
        mfcc_sim = max(0.0, 1.0 - mfcc_distance / 100.0)
        mel_sim = max(0.0, 1.0 - mel_distance / 40.0)
        spectral_sim = 1 - spectral_diff / max(target_centroid, noisy_centroid, 1)
        
        print(f"  MFCC相似度: {mfcc_sim:.3f}")
        print(f"  Mel相似度: {mel_sim:.3f}")
        print(f"  频谱相似度: {spectral_sim:.3f}")
        
        # 检查预筛选
        passes_filter = (mfcc_sim >= 0.3 and spectral_sim >= 0.15 and mel_sim >= 0.3)
        print(f"  预筛选: {'✅' if passes_filter else '❌'}")
        
        if passes_filter:
            # 计算色度相似度
            target_chroma = np.mean(target_features['chroma'], axis=1)
            noisy_chroma = np.mean(noisy_features['chroma'], axis=1)
            chroma_sim = np.corrcoef(target_chroma, noisy_chroma)[0, 1]
            if np.isnan(chroma_sim):
                chroma_sim = 0.0
            
            weights = config['detection']['confidence_weight']
            confidence = (
                weights['mfcc'] * max(0, mfcc_sim) +
                weights['spectral'] * max(0, spectral_sim) +
                weights['chroma'] * max(0, chroma_sim) +
                weights['mel'] * max(0, mel_sim)
            )
            
            print(f"  色度相似度: {chroma_sim:.3f}")
            print(f"  最终置信度: {confidence:.3f}")
            print(f"  检测结果: {'✅' if confidence >= 0.65 else '❌'}")
    
    # 分析特征的典型变化范围
    print(f"\n=== 特征变化范围分析 ===")
    
    # 生成多个噪声样本来分析变化范围
    distances = {'mfcc': [], 'mel': [], 'spectral': []}
    
    for _ in range(50):
        noisy_audio = y + np.random.normal(0, 0.01, y.shape)  # 1%噪声
        noisy_features = extract_audio_features(noisy_audio, sr, config['feature'])
        
        mfcc_dist = np.linalg.norm(np.mean(target_features['mfcc'], axis=1) - 
                                  np.mean(noisy_features['mfcc'], axis=1))
        mel_dist = np.linalg.norm(np.mean(target_features['mel_spectrogram'], axis=1) - 
                                 np.mean(noisy_features['mel_spectrogram'], axis=1))
        spectral_dist = abs(np.mean(target_features['spectral_centroid']) - 
                           np.mean(noisy_features['spectral_centroid']))
        
        distances['mfcc'].append(mfcc_dist)
        distances['mel'].append(mel_dist)
        distances['spectral'].append(spectral_dist)
    
    print(f"1%噪声下的距离统计:")
    print(f"  MFCC距离: 均值={np.mean(distances['mfcc']):.2f}, 标准差={np.std(distances['mfcc']):.2f}")
    print(f"  Mel距离: 均值={np.mean(distances['mel']):.2f}, 标准差={np.std(distances['mel']):.2f}")
    print(f"  频谱距离: 均值={np.mean(distances['spectral']):.1f}, 标准差={np.std(distances['spectral']):.1f}")
    
    # 建议新的归一化参数
    mfcc_95th = np.percentile(distances['mfcc'], 95)
    mel_95th = np.percentile(distances['mel'], 95)
    
    print(f"\n建议的归一化参数:")
    print(f"  MFCC: {mfcc_95th * 2:.0f} (当前: 100)")
    print(f"  Mel: {mel_95th * 2:.0f} (当前: 40)")

if __name__ == "__main__":
    debug_noise_sensitivity()