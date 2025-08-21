#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频分析工具
用于分析目标音频文件的基本参数和特征
"""

import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import os

def analyze_wav_file(file_path):
    """
    分析WAV文件的基本参数
    """
    print(f"正在分析音频文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None
    
    try:
        # 使用wave库读取基本信息
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频参数
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = frames / sample_rate
            
            print("=== 基本音频参数 ===")
            print(f"采样率: {sample_rate} Hz")
            print(f"声道数: {channels}")
            print(f"采样位深: {sample_width * 8} bits")
            print(f"总帧数: {frames}")
            print(f"时长: {duration:.2f} 秒")
            
            # 读取音频数据
            audio_data = wav_file.readframes(frames)
            
        # 使用librosa进行更详细的分析
        y, sr = librosa.load(file_path, sr=None)
        
        print("\n=== 音频特征分析 ===")
        print(f"音频数据形状: {y.shape}")
        print(f"最大振幅: {np.max(np.abs(y)):.4f}")
        print(f"RMS能量: {np.sqrt(np.mean(y**2)):.4f}")
        
        # 计算频谱特征
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        magnitude = np.abs(fft)
        
        # 找到主要频率成分
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        dominant_freq_idx = np.argmax(positive_magnitude)
        dominant_freq = positive_freqs[dominant_freq_idx]
        
        print(f"主要频率: {dominant_freq:.2f} Hz")
        
        # 计算零交叉率
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        print(f"平均零交叉率: {np.mean(zcr):.4f}")
        
        # 计算频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        print(f"平均频谱质心: {np.mean(spectral_centroids):.2f} Hz")
        
        # 计算MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"MFCC特征形状: {mfccs.shape}")
        
        return {
            'sample_rate': sample_rate,
            'channels': channels,
            'duration': duration,
            'audio_data': y,
            'dominant_freq': dominant_freq,
            'zcr': np.mean(zcr),
            'spectral_centroid': np.mean(spectral_centroids),
            'mfccs': mfccs
        }
        
    except Exception as e:
        print(f"分析音频文件时出错: {e}")
        return None

def plot_audio_analysis(file_path, audio_info):
    """
    绘制音频分析图表
    """
    if audio_info is None:
        return
    
    y = audio_info['audio_data']
    sr = audio_info['sample_rate']
    
    plt.figure(figsize=(15, 10))
    
    # 时域波形
    plt.subplot(3, 2, 1)
    time = np.linspace(0, len(y)/sr, len(y))
    plt.plot(time, y)
    plt.title('时域波形')
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')
    plt.grid(True)
    
    # 频谱
    plt.subplot(3, 2, 2)
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    magnitude = np.abs(fft)
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]
    plt.plot(positive_freqs, positive_magnitude)
    plt.title('频谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度')
    plt.grid(True)
    plt.xlim(0, sr//2)
    
    # 频谱图
    plt.subplot(3, 2, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.title('频谱图')
    plt.colorbar(format='%+2.0f dB')
    
    # MFCC特征
    plt.subplot(3, 2, 4)
    mfccs = audio_info['mfccs']
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title('MFCC特征')
    plt.colorbar()
    
    # 零交叉率
    plt.subplot(3, 2, 5)
    zcr = librosa.feature.zero_crossing_rate(y)
    frames = range(len(zcr[0]))
    t = librosa.frames_to_time(frames, sr=sr)
    plt.plot(t, zcr[0])
    plt.title('零交叉率')
    plt.xlabel('时间 (秒)')
    plt.ylabel('ZCR')
    plt.grid(True)
    
    # 频谱质心
    plt.subplot(3, 2, 6)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    frames = range(len(spectral_centroids[0]))
    t = librosa.frames_to_time(frames, sr=sr)
    plt.plot(t, spectral_centroids[0])
    plt.title('频谱质心')
    plt.xlabel('时间 (秒)')
    plt.ylabel('Hz')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/leslack/lsc/300_study/python/demo/voice4/audio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    target_file = "/Users/leslack/lsc/300_study/python/demo/voice4/target.wav"
    
    # 分析音频文件
    audio_info = analyze_wav_file(target_file)
    
    if audio_info:
        # 绘制分析图表
        plot_audio_analysis(target_file, audio_info)
        
        print("\n=== 分析完成 ===")
        print("音频分析图表已保存为 audio_analysis.png")
    else:
        print("音频分析失败")