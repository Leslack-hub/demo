#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试音频文件
用于测试PANNs音频检测器
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse

def create_sine_wave(frequency: float, duration: float, sample_rate: int = 32000) -> np.ndarray:
    """
    创建正弦波音频
    
    Args:
        frequency: 频率 (Hz)
        duration: 持续时间 (秒)
        sample_rate: 采样率
        
    Returns:
        音频数据
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # 添加淡入淡出效果
    fade_samples = int(0.1 * sample_rate)  # 0.1秒淡入淡出
    if len(audio) > 2 * fade_samples:
        # 淡入
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        # 淡出
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return audio

def create_chirp(start_freq: float, end_freq: float, duration: float, sample_rate: int = 32000) -> np.ndarray:
    """
    创建线性扫频音频
    
    Args:
        start_freq: 起始频率 (Hz)
        end_freq: 结束频率 (Hz)
        duration: 持续时间 (秒)
        sample_rate: 采样率
        
    Returns:
        音频数据
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # 线性扫频
    freq = start_freq + (end_freq - start_freq) * t / duration
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    audio = np.sin(phase)
    
    # 添加淡入淡出效果
    fade_samples = int(0.1 * sample_rate)
    if len(audio) > 2 * fade_samples:
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return audio

def create_white_noise(duration: float, sample_rate: int = 32000) -> np.ndarray:
    """
    创建白噪声
    
    Args:
        duration: 持续时间 (秒)
        sample_rate: 采样率
        
    Returns:
        音频数据
    """
    num_samples = int(sample_rate * duration)
    audio = np.random.normal(0, 0.1, num_samples)
    
    # 添加淡入淡出效果
    fade_samples = int(0.1 * sample_rate)
    if len(audio) > 2 * fade_samples:
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return audio

def create_mixed_audio(duration: float = 3.0, sample_rate: int = 32000) -> np.ndarray:
    """
    创建混合音频（包含多种声音成分）
    
    Args:
        duration: 持续时间 (秒)
        sample_rate: 采样率
        
    Returns:
        音频数据
    """
    # 创建基础音调
    base_tone = create_sine_wave(440, duration, sample_rate)  # A4音符
    
    # 添加和声
    harmony1 = create_sine_wave(554.37, duration, sample_rate) * 0.5  # C#5
    harmony2 = create_sine_wave(659.25, duration, sample_rate) * 0.3  # E5
    
    # 添加扫频效果
    chirp = create_chirp(200, 800, duration, sample_rate) * 0.2
    
    # 添加少量噪声
    noise = create_white_noise(duration, sample_rate) * 0.05
    
    # 混合所有成分
    mixed = base_tone + harmony1 + harmony2 + chirp + noise
    
    # 归一化
    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.8  # 留一些余量
    
    return mixed

def main():
    parser = argparse.ArgumentParser(description='创建测试音频文件')
    parser.add_argument('--type', choices=['sine', 'chirp', 'noise', 'mixed'], 
                       default='mixed', help='音频类型')
    parser.add_argument('--frequency', type=float, default=440, 
                       help='正弦波频率 (Hz)')
    parser.add_argument('--duration', type=float, default=3.0, 
                       help='音频持续时间 (秒)')
    parser.add_argument('--output', type=str, default='target.wav', 
                       help='输出文件名')
    parser.add_argument('--sample-rate', type=int, default=32000, 
                       help='采样率')
    
    args = parser.parse_args()
    
    print(f"创建{args.type}音频文件: {args.output}")
    print(f"持续时间: {args.duration}秒, 采样率: {args.sample_rate}Hz")
    
    # 创建音频
    if args.type == 'sine':
        audio = create_sine_wave(args.frequency, args.duration, args.sample_rate)
        print(f"正弦波频率: {args.frequency}Hz")
    elif args.type == 'chirp':
        audio = create_chirp(200, 2000, args.duration, args.sample_rate)
        print("线性扫频: 200Hz -> 2000Hz")
    elif args.type == 'noise':
        audio = create_white_noise(args.duration, args.sample_rate)
        print("白噪声")
    elif args.type == 'mixed':
        audio = create_mixed_audio(args.duration, args.sample_rate)
        print("混合音频 (基础音调 + 和声 + 扫频 + 噪声)")
    
    # 保存音频文件
    sf.write(args.output, audio, args.sample_rate)
    print(f"音频文件已保存: {args.output}")
    
    # 显示音频信息
    print(f"音频长度: {len(audio)} 采样点")
    print(f"音频幅度范围: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
    print(f"音频RMS: {np.sqrt(np.mean(audio**2)):.3f}")

if __name__ == '__main__':
    main()