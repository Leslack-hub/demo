import numpy as np
import sounddevice as sd
import queue
from scipy.spatial import distance
import librosa

audio_buffer = queue.Queue()


def callback(indata, frames, time, status):
    if status:
        print(status)
    # 将音频数据放入队列中
    audio_buffer.put(indata.copy())


# 配置音频流参数
duration = 10  # 监听的总时长（秒）
samplerate = 22050  # 设置一个采样率
channels = 1  # 单声道

stream = sd.InputStream(callback=callback, samplerate=samplerate, channels=channels, dtype='float32',device=1)

# 开始捕获音频
with stream:
    sd.sleep(duration * 1000)


def process_audio_buffer(audio_buffer, target_audio):
    audio_data = []
    while not audio_buffer.empty():
        audio_data.append(audio_buffer.get())

    audio_data_np = np.concatenate(audio_data, axis=0).flatten()  # 确保一维

    # 计算余弦相似度
    sim = 1 - distance.cosine(audio_data_np, target_audio.flatten())
    print(f"与目标音频的相似度: {sim}")


def load_target_audio(file_path):
    """
    加载一个音频文件为 NumPy 数组。

    :param file_path: 音频文件的路径
    :return: 音频数据 (NumPy数组), 采样率
    """
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


# 示例加载目标音频，假设已有加载函数
target_audio, sr = load_target_audio("path_to_target_audio.wav")

process_audio_buffer(audio_buffer, target_audio)
