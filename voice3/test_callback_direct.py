#!/usr/bin/env python3
"""直接测试回调函数调用"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from audio_waveform_visualizer import RealTimeAudioVisualizer
import numpy as np

# 全局变量用于跟踪回调是否被调用
callback_called = False

def test_callback():
    """测试回调函数"""
    global callback_called
    callback_called = True
    print("回调函数被调用了！")

def main():
    """测试主函数"""
    global callback_called
    
    # 创建可视化器实例，传入回调函数
    visualizer = RealTimeAudioVisualizer(sound_detected_callback=test_callback)
    
    # 手动设置 target_sound 和 audio_buffer 以确保检测成功
    visualizer.target_sound = np.array([1.0, 1.0, 1.0])
    visualizer.audio_buffer = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # 降低检测阈值以确保检测成功
    visualizer.sound_detection_threshold = 0.1
    
    print(f"回调函数调用状态 (之前): {callback_called}")
    
    # 创建一个匹配的音频块
    audio_chunk = np.array([1.0, 1.0, 1.0])
    
    # 直接在 audio_callback 中测试回调函数调用
    print("直接调用 audio_callback 方法测试回调函数...")
    
    # 模拟 indata 参数
    indata = np.array([[1.0], [1.0], [1.0]])
    frames = 3
    time = None
    status = None
    
    # 调用 audio_callback 方法，这应该会触发回调函数
    visualizer.audio_callback(indata, frames, time, status)
    
    print(f"回调函数调用状态 (之后): {callback_called}")

if __name__ == "__main__":
    main()