import pyaudio
import numpy as np
import librosa

# 设置参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# 加载目标声音并提取特征
target_audio, _ = librosa.load("target_sound.wav", sr=RATE)
target_mfccs = librosa.feature.mfcc(y=target_audio, sr=RATE, n_mfcc=13)

# 初始化 PyAudio
audio = pyaudio.PyAudio()

# 打开音频输出流
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("开始监听...")

while True:
    # 读取音频数据
    data = stream.read(CHUNK)

    # 将音频数据转换为 NumPy 数组
    audio_data = np.frombuffer(data, dtype=np.int16)

    # 提取 MFCCs 特征
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13)

    # 计算相似度 (使用 DTW 算法)
    distance, _ = librosa.dtw(target_mfccs, mfccs)
    similarity = 1 / distance[-1, -1]

    # 判断是否检测到目标声音
    if similarity > 0.8:  # 设置阈值为 0.8
        print("检测到目标声音!")
        break

# 停止监听
stream.stop_stream()
stream.close()
audio.terminate()
