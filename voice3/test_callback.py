#!/usr/bin/env python3
"""测试回调函数功能"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from audio_waveform_visualizer import RealTimeAudioVisualizer
import numpy as np


def test_callback():
    """测试回调函数"""
    print("回调函数被调用了！")


def main():
    """测试主函数"""
    # 创建可视化器实例，传入回调函数
    visualizer = RealTimeAudioVisualizer(sound_detected_callback=test_callback)

    # 手动设置 target_sound
    visualizer.target_sound = np.array([0.1, 0.2, 0.3])

    # 创建一个与目标声音完全匹配的音频块，以确保检测成功
    audio_chunk = np.tile(visualizer.target_sound, 10)  # 重复10次以确保足够长

    # 增加音频缓冲区大小以确保检测能够进行
    visualizer.audio_buffer = np.tile(visualizer.target_sound, 50)  # 创建一个足够大的缓冲区

    print("模拟检测到指定声音...")
    # 调用 detect_sound 方法测试回调函数
    result = visualizer.detect_sound(audio_chunk)
    print(f"声音检测结果: {result}")

    # 再次调用以测试回调函数是否被调用
    result2 = visualizer.detect_sound(audio_chunk)
    print(f"第二次声音检测结果: {result2}")


if __name__ == "__main__":
    main()
