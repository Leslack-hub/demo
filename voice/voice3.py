import wave
import numpy as np

# 打开wav文件
with wave.open('../target/target_audio.wav', 'rb') as wav_file:
    # 获取音频参数
    params = wav_file.getparams()
    # 读取音频数据
    frames = wav_file.readframes(params.nframes)
    # 将音频数据转换为numpy数组
    audio_data = np.frombuffer(frames, dtype=np.int16)

# 保存为npy文件
np.save('audio_data.npy', audio_data)
