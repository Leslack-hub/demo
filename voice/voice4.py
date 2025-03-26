import sounddevice as sd
import numpy as np
import wave
import sys

def find_output_device():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            print(f"找到输出设备: {device['name']}")
            return i
    return None

# 设置录音参数
duration = 5  # 录音时长（秒）
sample_rate = 44100  # 采样率
channels = 2  # 声道数

# 查找输出设备
output_device = 1

if output_device is None:
    print("未找到可用的输出设备")
    sys.exit(1)

# 录制音频
print("开始录音...")
try:
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, device=output_device)
    sd.wait()  # 等待录音结束
    print("录音结束")
except sd.PortAudioError as e:
    print(f"录音出错: {e}")
    sys.exit(1)

# 将录音数据保存为 WAV 文件
filename = "output3.wav"
with wave.open(filename, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 16-bit 音频
    wf.setframerate(sample_rate)
    wf.writeframes((recording * 32767).astype(np.int16).tobytes())

print(f"音频已保存为 {filename}")
