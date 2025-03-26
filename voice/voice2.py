import queue

import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile

target_audio_path = '../target/target_audio.wav'
target_rate, target_data = wavfile.read(target_audio_path)

# 转换目标数据为numpy数组
if target_data.ndim > 1:  # 如果是立体声音频，则只取一通道
    target_data = target_data[:, 0]

print(target_data)


def audio_contains(audio_chunk, target_data):
    # 将音频块转换为numpy数组
    chunk_data = np.frombuffer(audio_chunk, dtype=np.int16)

    # 如果目标音频数据比音频块大，直接返回False
    if len(target_data) > len(chunk_data):
        return False

    # 使用滑动窗口检查
    for i in range(len(chunk_data) - len(target_data) + 1):
        if np.array_equal(chunk_data[i:i + len(target_data)], target_data):
            return True
    return False


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_chunk = indata.tobytes()  # 将输入数据转换为字节串

    # 检查当前音频块是否包含目标音频片段
    if audio_contains(audio_chunk, target_data):
        print(f"目标音频在时间 {time.strftime('%H:%M:%S')} 找到！")


def audio_callback2(indata, frames, time, status):
    if status:
        print(status)
    chunk_data = np.frombuffer(indata, dtype=np.int16)
    print(chunk_data)
    for i in range(len(chunk_data) - len(target_data) + 1):
        if np.array_equal(chunk_data[i:i + len(target_data)], target_data):
            print(f"目标音频在时间 {time.strftime('%H:%M:%S')} 找到！")


device_index = 1  # 将这里替换为虚拟设备的索引
duration = 10  # 监听10秒
samplerate = 22050
channels = 1

print(target_data)

with sd.InputStream(device=device_index, dtype='float32', callback=audio_callback2, samplerate=samplerate,
                    channels=channels, blocksize=int(samplerate * 2)):
    sd.sleep(duration * 1000)
