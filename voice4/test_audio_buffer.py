#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频缓冲区测试脚本
用于诊断音频流和缓冲区问题
"""

import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time
import argparse

class AudioBufferTest:
    def __init__(self, input_device=None, duration=10):
        self.input_device = input_device
        self.duration = duration
        self.sample_rate = 22050
        self.chunk_size = 1024
        self.buffer_size = self.sample_rate * 5  # 5秒缓冲区
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        self.is_running = False
        self.callback_count = 0
        self.error_count = 0
        
    def audio_callback(self, indata, frames, time, status):
        """音频回调函数"""
        self.callback_count += 1
        
        if status:
            print(f"\n音频状态警告: {status}")
            self.error_count += 1
            return
            
        try:
            with self.buffer_lock:
                # 转换为单声道
                audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
                
                # 检查音频数据
                if len(audio_chunk) > 0:
                    self.audio_buffer.extend(audio_chunk)
                    
        except Exception as e:
            print(f"\n回调错误: {e}")
            self.error_count += 1
            
    def test_audio_stream(self):
        """测试音频流"""
        print("=== 音频缓冲区测试 ===")
        
        # 显示设备信息
        if self.input_device is not None:
            device_info = sd.query_devices()[self.input_device]
            print(f"使用设备: {self.input_device} - {device_info['name']}")
        else:
            default_device = sd.query_devices()[sd.default.device[0]]
            print(f"使用默认设备: {default_device['name']}")
            
        print(f"采样率: {self.sample_rate} Hz")
        print(f"块大小: {self.chunk_size}")
        print(f"缓冲区大小: {self.buffer_size} 样本 ({self.buffer_size/self.sample_rate:.1f}秒)")
        print(f"测试时长: {self.duration}秒\n")
        
        try:
            # 构建音频流参数
            stream_params = {
                'samplerate': self.sample_rate,
                'channels': 1,
                'dtype': 'float32',
                'blocksize': self.chunk_size,
                'callback': self.audio_callback,
                'latency': 'low'
            }
            
            if self.input_device is not None:
                stream_params['device'] = (self.input_device, None)
                
            # 启动音频流
            with sd.InputStream(**stream_params) as stream:
                print("音频流已启动，开始测试...")
                self.is_running = True
                
                start_time = time.time()
                last_report_time = start_time
                
                while time.time() - start_time < self.duration:
                    current_time = time.time()
                    
                    # 每秒报告一次状态
                    if current_time - last_report_time >= 1.0:
                        with self.buffer_lock:
                            buffer_fill = len(self.audio_buffer)
                            buffer_percent = (buffer_fill / self.buffer_size) * 100
                            
                        # 计算音频电平
                        if buffer_fill > 0:
                            recent_samples = list(self.audio_buffer)[-1000:] if buffer_fill >= 1000 else list(self.audio_buffer)
                            audio_level = np.sqrt(np.mean(np.array(recent_samples)**2)) if recent_samples else 0
                        else:
                            audio_level = 0
                            
                        elapsed = current_time - start_time
                        print(f"时间: {elapsed:.1f}s | 缓冲区: {buffer_fill}/{self.buffer_size} ({buffer_percent:.1f}%) | "
                              f"音频电平: {audio_level:.4f} | 回调次数: {self.callback_count} | 错误: {self.error_count}")
                        
                        last_report_time = current_time
                        
                    time.sleep(0.1)
                    
                self.is_running = False
                print("\n测试完成!")
                
        except Exception as e:
            print(f"音频流错误: {e}")
            return False
            
        # 最终报告
        print("\n=== 测试结果 ===")
        print(f"总回调次数: {self.callback_count}")
        print(f"错误次数: {self.error_count}")
        print(f"最终缓冲区大小: {len(self.audio_buffer)}")
        
        if self.callback_count > 0 and self.error_count == 0:
            print("✅ 音频流和缓冲区工作正常")
            return True
        elif self.callback_count == 0:
            print("❌ 未收到任何音频数据")
            return False
        else:
            print(f"⚠️  有 {self.error_count} 个错误，但收到了 {self.callback_count} 次回调")
            return False

def main():
    parser = argparse.ArgumentParser(description='音频缓冲区测试')
    parser.add_argument('--input', type=str, help='输入设备ID或名称')
    parser.add_argument('--duration', type=int, default=10, help='测试时长(秒)')
    parser.add_argument('--list-devices', action='store_true', help='列出音频设备')
    args = parser.parse_args()
    
    if args.list_devices:
        devices = sd.query_devices()
        print("\n可用的音频设备:")
        print("-" * 60)
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append(f"{device['max_input_channels']}输入")
            if device['max_output_channels'] > 0:
                device_type.append(f"{device['max_output_channels']}输出")
            marker = "*" if i == sd.default.device[0] else " "
            print(f"{marker}{i:2d}: {device['name']:<30} ({', '.join(device_type)})")
        return
        
    # 解析输入设备
    input_device = None
    if args.input:
        try:
            input_device = int(args.input)
        except ValueError:
            # 按名称搜索
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 and args.input.lower() in device['name'].lower():
                    input_device = i
                    break
            if input_device is None:
                print(f"未找到匹配的输入设备: {args.input}")
                return
                
    # 运行测试
    tester = AudioBufferTest(input_device, args.duration)
    success = tester.test_audio_stream()
    
    if not success:
        print("\n建议检查:")
        print("1. 麦克风是否正确连接")
        print("2. 系统音频权限是否已授予")
        print("3. 是否有其他程序占用音频设备")
        print("4. 尝试使用不同的输入设备")

if __name__ == "__main__":
    main()