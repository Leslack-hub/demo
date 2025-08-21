#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试检测器对目标音频的识别能力
"""

import numpy as np
from audio_features import extract_audio_features
from config import get_config
import librosa

def test_self_detection():
    """测试检测器对目标音频自身的识别能力"""
    config = get_config()
    target_path = config['file']['target_audio_path']
    
    print(f"=== 测试目标音频自检测 ===")
    
    # 加载目标音频
    y, sr = librosa.load(target_path, sr=config['audio']['sample_rate'])
    
    # 提取特征
    target_features = extract_audio_features(y, sr, config['feature'])
    
    # 模拟检测器的相似度计算
    def calculate_similarity(features1, features2):
        try:
            # MFCC相似度（使用欧氏距离）
            mfcc1_mean = np.mean(features1['mfcc'], axis=1)
            mfcc2_mean = np.mean(features2['mfcc'], axis=1)
            mfcc_distance = np.linalg.norm(mfcc1_mean - mfcc2_mean)
            mfcc_sim = max(0.0, 1.0 - mfcc_distance / 93.0)
            
            # 频谱质心相似度
            centroid1 = np.mean(features1['spectral_centroid'])
            centroid2 = np.mean(features2['spectral_centroid'])
            spectral_sim = 1 - abs(centroid1 - centroid2) / max(centroid1, centroid2, 1)
            
            # 色度特征相似度
            chroma1_mean = np.mean(features1['chroma'], axis=1)
            chroma2_mean = np.mean(features2['chroma'], axis=1)
            chroma_sim = np.corrcoef(chroma1_mean, chroma2_mean)[0, 1]
            if np.isnan(chroma_sim):
                chroma_sim = 0.0
            
            # Mel频谱相似度（使用欧氏距离）
            mel1_mean = np.mean(features1['mel_spectrogram'], axis=1)
            mel2_mean = np.mean(features2['mel_spectrogram'], axis=1)
            mel_distance = np.linalg.norm(mel1_mean - mel2_mean)
            mel_sim = max(0.0, 1.0 - mel_distance / 94.0)
            
            return mfcc_sim, spectral_sim, chroma_sim, mel_sim
            
        except Exception as e:
            print(f"相似度计算错误: {e}")
            return 0, 0, 0, 0
    
    # 测试自身相似度
    mfcc_sim, spectral_sim, chroma_sim, mel_sim = calculate_similarity(target_features, target_features)
    
    print(f"\n自身相似度测试:")
    print(f"  MFCC相似度: {mfcc_sim:.3f}")
    print(f"  频谱质心相似度: {spectral_sim:.3f}")
    print(f"  色度特征相似度: {chroma_sim:.3f}")
    print(f"  Mel频谱相似度: {mel_sim:.3f}")
    
    # 计算置信度
    weights = config['detection']['confidence_weight']
    
    # 检查预筛选条件
    if (mfcc_sim < 0.25 or spectral_sim < 0.1 or mel_sim < 0.25):
        confidence = 0.0
        print(f"\n❌ 预筛选失败")
    else:
        confidence = (
            weights['mfcc'] * max(0, mfcc_sim) +
            weights['spectral'] * max(0, spectral_sim) +
            weights['chroma'] * max(0, chroma_sim) +
            weights['mel'] * max(0, mel_sim)
        )
        print(f"\n✅ 预筛选通过")
    
    print(f"\n最终置信度: {confidence:.3f}")
    print(f"检测阈值: {config['detection']['min_confidence']}")
    
    if confidence >= config['detection']['min_confidence']:
        print(f"\n🎯 检测成功！")
    else:
        print(f"\n❌ 检测失败 - 置信度不足")
        
        # 建议调整参数
        print(f"\n=== 参数调整建议 ===")
        if mfcc_sim < 0.5:
            print(f"- MFCC预筛选阈值过高，建议降低到 {mfcc_sim*0.8:.2f}")
        if spectral_sim < 0.2:
            print(f"- 频谱预筛选阈值过高，建议降低到 {spectral_sim*0.8:.2f}")
        if mel_sim < 0.5:
            print(f"- Mel预筛选阈值过高，建议降低到 {mel_sim*0.8:.2f}")
        if confidence < config['detection']['min_confidence']:
            print(f"- 总体阈值过高，建议降低到 {confidence*0.9:.2f}")
    
    # 测试添加噪声后的识别能力
    print(f"\n=== 噪声鲁棒性测试 ===")
    noise_levels = [0.01, 0.05, 0.1]
    
    for noise_level in noise_levels:
        # 添加高斯噪声
        noisy_audio = y + np.random.normal(0, noise_level, y.shape)
        noisy_features = extract_audio_features(noisy_audio, sr, config['feature'])
        
        mfcc_sim, spectral_sim, chroma_sim, mel_sim = calculate_similarity(target_features, noisy_features)
        
        if (mfcc_sim >= 0.25 and spectral_sim >= 0.1 and mel_sim >= 0.25):
            confidence = (
                weights['mfcc'] * max(0, mfcc_sim) +
                weights['spectral'] * max(0, spectral_sim) +
                weights['chroma'] * max(0, chroma_sim) +
                weights['mel'] * max(0, mel_sim)
            )
        else:
            confidence = 0.0
        
        status = "✅" if confidence >= config['detection']['min_confidence'] else "❌"
        print(f"噪声级别 {noise_level:.2f}: 置信度 {confidence:.3f} {status}")

if __name__ == "__main__":
    test_self_detection()