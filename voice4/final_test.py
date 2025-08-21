#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终测试脚本 - 验证检测器的完整功能
"""

import numpy as np
from audio_features import extract_audio_features
from config import get_config
import librosa
import time

def final_test():
    """最终功能测试"""
    config = get_config()
    target_path = config['file']['target_audio_path']
    
    print(f"=== 音频检测器最终测试 ===")
    print(f"目标音频: {target_path}")
    print(f"检测阈值: {config['detection']['min_confidence']}")
    
    # 加载目标音频
    y, sr = librosa.load(target_path, sr=config['audio']['sample_rate'])
    target_features = extract_audio_features(y, sr, config['feature'])
    
    def calculate_similarity(features1, features2):
        """计算相似度（与检测器保持一致）"""
        try:
            # MFCC相似度
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
            
            # Mel频谱相似度
            mel1_mean = np.mean(features1['mel_spectrogram'], axis=1)
            mel2_mean = np.mean(features2['mel_spectrogram'], axis=1)
            mel_distance = np.linalg.norm(mel1_mean - mel2_mean)
            mel_sim = max(0.0, 1.0 - mel_distance / 94.0)
            
            return mfcc_sim, spectral_sim, chroma_sim, mel_sim
            
        except Exception as e:
            print(f"相似度计算错误: {e}")
            return 0, 0, 0, 0
    
    def test_detection(audio_data, test_name):
        """测试检测功能"""
        features = extract_audio_features(audio_data, sr, config['feature'])
        mfcc_sim, spectral_sim, chroma_sim, mel_sim = calculate_similarity(target_features, features)
        
        # 预筛选检查
        passes_filter = (mfcc_sim >= 0.25 and spectral_sim >= 0.1 and mel_sim >= 0.25)
        
        if passes_filter:
            weights = config['detection']['confidence_weight']
            confidence = (
                weights['mfcc'] * max(0, mfcc_sim) +
                weights['spectral'] * max(0, spectral_sim) +
                weights['chroma'] * max(0, chroma_sim) +
                weights['mel'] * max(0, mel_sim)
            )
        else:
            confidence = 0.0
        
        detected = confidence >= config['detection']['min_confidence']
        status = "✅ 检测成功" if detected else "❌ 未检测到"
        
        print(f"\n{test_name}:")
        print(f"  相似度: MFCC={mfcc_sim:.3f}, 频谱={spectral_sim:.3f}, 色度={chroma_sim:.3f}, Mel={mel_sim:.3f}")
        print(f"  预筛选: {'✅' if passes_filter else '❌'}")
        print(f"  置信度: {confidence:.3f}")
        print(f"  结果: {status}")
        
        return detected, confidence
    
    # 测试1: 目标音频自身
    print(f"\n=== 测试1: 目标音频自检测 ===")
    detected, confidence = test_detection(y, "目标音频自身")
    assert detected, "目标音频自检测失败！"
    
    # 测试2: 轻微噪声
    print(f"\n=== 测试2: 噪声鲁棒性测试 ===")
    noise_levels = [0.001, 0.005, 0.01]
    success_count = 0
    
    for noise_level in noise_levels:
        noisy_audio = y + np.random.normal(0, noise_level, y.shape)
        detected, confidence = test_detection(noisy_audio, f"噪声级别 {noise_level}")
        if detected:
            success_count += 1
    
    print(f"\n噪声测试结果: {success_count}/{len(noise_levels)} 成功")
    
    # 测试3: 音量变化
    print(f"\n=== 测试3: 音量变化测试 ===")
    volume_factors = [0.5, 0.8, 1.2, 1.5]
    volume_success = 0
    
    for factor in volume_factors:
        scaled_audio = y * factor
        # 防止削波
        if np.max(np.abs(scaled_audio)) > 1.0:
            scaled_audio = scaled_audio / np.max(np.abs(scaled_audio)) * 0.95
        
        detected, confidence = test_detection(scaled_audio, f"音量 {factor}x")
        if detected:
            volume_success += 1
    
    print(f"\n音量测试结果: {volume_success}/{len(volume_factors)} 成功")
    
    # 测试4: 时间偏移（模拟部分匹配）
    print(f"\n=== 测试4: 部分音频测试 ===")
    segment_length = len(y) // 2  # 取一半长度
    start_positions = [0, len(y) // 4, len(y) // 2]
    segment_success = 0
    
    for start_pos in start_positions:
        end_pos = min(start_pos + segment_length, len(y))
        if end_pos - start_pos < segment_length // 2:  # 确保有足够的音频
            continue
        
        segment = y[start_pos:end_pos]
        detected, confidence = test_detection(segment, f"片段 {start_pos}-{end_pos}")
        if detected:
            segment_success += 1
    
    print(f"\n片段测试结果: {segment_success}/{len(start_positions)} 成功")
    
    # 总结
    print(f"\n=== 测试总结 ===")
    print(f"✅ 目标音频自检测: 成功")
    print(f"📊 噪声鲁棒性: {success_count}/{len(noise_levels)} ({success_count/len(noise_levels)*100:.0f}%)")
    print(f"🔊 音量适应性: {volume_success}/{len(volume_factors)} ({volume_success/len(volume_factors)*100:.0f}%)")
    print(f"⏱️ 部分匹配: {segment_success}/{len(start_positions)} ({segment_success/len(start_positions)*100:.0f}%)")
    
    total_tests = 1 + len(noise_levels) + len(volume_factors) + len(start_positions)
    total_success = 1 + success_count + volume_success + segment_success
    overall_success_rate = total_success / total_tests * 100
    
    print(f"\n🎯 总体成功率: {total_success}/{total_tests} ({overall_success_rate:.1f}%)")
    
    if overall_success_rate >= 70:
        print(f"\n🎉 检测器性能良好！")
    elif overall_success_rate >= 50:
        print(f"\n⚠️ 检测器性能一般，建议进一步优化")
    else:
        print(f"\n❌ 检测器性能不佳，需要重新调整参数")
    
    print(f"\n=== 使用建议 ===")
    print(f"1. 确保录音环境相对安静")
    print(f"2. 目标音频应清晰可辨")
    print(f"3. 避免过大的背景噪声")
    print(f"4. 当前检测阈值: {config['detection']['min_confidence']}")
    print(f"5. 如需调整灵敏度，可修改config.py中的min_confidence值")

if __name__ == "__main__":
    final_test()