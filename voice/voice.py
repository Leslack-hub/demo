from datetime import time

import pyaudio
import numpy as np
import librosa


def match_target_audio(data, target_data, sr):
    # 这里可以使用更复杂的音频相似度检测算法
    # 简单示例：计算两个音频数据的欧氏距离
    distance = np.linalg.norm(data - target_data)
    return distance < 0.1  # 阈值需要根据实际情况调整


# 加载目标声音
target_sound, sr = librosa.load('down_to_cases.wav', sr=None)
target_sound = librosa.resample(target_sound, orig_sr=sr, target_sr=44100)

# 初始化PyAudio
p = pyaudio.PyAudio()

# 打开流
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024)

print("开始实时监听...")

try:
    while True:
        data = stream.read(1024)
        audio_data = np.frombuffer(data, dtype=np.float32)

        # 检查音频数据长度是否足够
        if len(audio_data) >= len(target_sound):
            # 检测音频是否匹配
            if match_target_audio(audio_data[:len(target_sound)], target_sound, 44100):
                print(f"匹配成功！时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
except KeyboardInterrupt:
    print("停止监听")

# 停止流
stream.stop_stream()
stream.close()

# 关闭PyAudio
p.terminate()