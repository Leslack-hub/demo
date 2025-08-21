#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio
import numpy as np
import time

def test_audio_input():
    """测试音频输入设备"""
    
    # 音频参数
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 22050
    
    # 初始化PyAudio
    p = pyaudio.PyAudio()
    
    # 列出所有音频设备
    print("可用音频设备:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"  {i} {info['name']}, {info['hostApi']} ({info['maxInputChannels']} in, {info['maxOutputChannels']} out)")
    
    # 尝试使用外置麦克风
    input_device = 2  # 外置麦克风
    
    try:
        # 打开音频流
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=CHUNK
        )
        
        print(f"\n使用设备 {input_device}: 外置麦克风")
        print("开始监听音频输入...")
        print("请说话或播放声音，观察音频电平变化")
        print("按 Ctrl+C 停止测试\n")
        
        try:
            while True:
                # 读取音频数据
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # 计算音频电平
                audio_level = np.sqrt(np.mean(audio_data**2))
                
                # 显示音频电平
                bar_length = int(audio_level * 50)  # 音频电平条
                bar = '█' * bar_length + '░' * (50 - bar_length)
                
                print(f"\r音频电平: {audio_level:.4f} [{bar}]", end="", flush=True)
                
                # 如果有明显的音频输入，显示提示
                if audio_level > 0.01:
                    print(f"  ✓ 检测到音频输入!")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n测试停止")
            
    except Exception as e:
        print(f"音频设备错误: {e}")
        print("尝试使用默认输入设备...")
        
        try:
            # 使用默认输入设备
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print("使用默认音频输入设备")
            print("开始监听音频输入...")
            print("请说话或播放声音，观察音频电平变化")
            print("按 Ctrl+C 停止测试\n")
            
            try:
                while True:
                    # 读取音频数据
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # 计算音频电平
                    audio_level = np.sqrt(np.mean(audio_data**2))
                    
                    # 显示音频电平
                    bar_length = int(audio_level * 50)  # 音频电平条
                    bar = '█' * bar_length + '░' * (50 - bar_length)
                    
                    print(f"\r音频电平: {audio_level:.4f} [{bar}]", end="", flush=True)
                    
                    # 如果有明显的音频输入，显示提示
                    if audio_level > 0.01:
                        print(f"  ✓ 检测到音频输入!")
                    
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n\n测试停止")
                
        except Exception as e2:
            print(f"默认设备也无法使用: {e2}")
    
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    test_audio_input()