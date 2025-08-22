#!/usr/bin/env python3
"""
播放音频文件到BlackHole虚拟音频设备的测试脚本
用于测试PANNs音频检测器
"""

import sounddevice as sd
import soundfile as sf
import time
import argparse
import sys

def list_audio_devices():
    """列出所有音频设备"""
    print("可用的音频设备:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} - {device['max_output_channels']} 输出通道")
    return devices

def play_audio_to_device(audio_file, device_id, repeat=1, interval=2.0):
    """播放音频文件到指定设备"""
    try:
        # 读取音频文件
        data, samplerate = sf.read(audio_file)
        print(f"加载音频文件: {audio_file}")
        print(f"采样率: {samplerate} Hz")
        print(f"时长: {len(data)/samplerate:.2f} 秒")
        print(f"输出设备ID: {device_id}")
        
        # 获取设备信息
        device_info = sd.query_devices(device_id)
        print(f"输出设备: {device_info['name']}")
        
        for i in range(repeat):
            print(f"\n播放第 {i+1}/{repeat} 次...")
            
            # 播放音频
            sd.play(data, samplerate, device=device_id)
            
            # 等待播放完成
            sd.wait()
            
            print(f"播放完成")
            
            # 如果不是最后一次播放，等待间隔时间
            if i < repeat - 1:
                print(f"等待 {interval} 秒后继续...")
                time.sleep(interval)
        
        print("\n所有播放完成")
        
    except Exception as e:
        print(f"播放错误: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='播放音频文件到指定设备')
    parser.add_argument('audio_file', help='要播放的音频文件路径')
    parser.add_argument('--device', '-d', type=int, help='输出设备ID')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有音频设备')
    parser.add_argument('--repeat', '-r', type=int, default=1, help='重复播放次数 (默认: 1)')
    parser.add_argument('--interval', '-i', type=float, default=2.0, help='重复播放间隔秒数 (默认: 2.0)')
    
    args = parser.parse_args()
    
    # 列出设备
    if args.list:
        list_audio_devices()
        return
    
    # 检查参数
    if args.device is None:
        print("错误: 请指定输出设备ID")
        print("使用 --list 参数查看可用设备")
        sys.exit(1)
    
    # 播放音频
    success = play_audio_to_device(
        args.audio_file, 
        args.device, 
        args.repeat, 
        args.interval
    )
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()