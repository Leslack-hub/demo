import sounddevice as sd
import numpy as np
import datetime
import librosa.display
from scipy.spatial.distance import cosine
import librosa

# 参数设置
samplerate = 44100  # 采样率
duration = 50  # 持续时间（秒），用于监控音频数据流


# 读取音频文件
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # 加载音频文件，sr=None表示使用原采样率
    return y, sr


# 示例：读取录音和歌曲
recording_y, recording_sr = load_audio('../target/target_audio.wav')
# 提取MFCC特征
mfcc_recording = librosa.feature.mfcc(y=recording_y, sr=recording_sr, n_mfcc=40)


def match_audio(input_data):
    # song_y, song_sr = load_audio('output3.wav')
    # # 确保两者采样率相同，以便后续处理
    # if recording_sr != song_sr:
    #     song_y = librosa.resample(y=song_y, orig_sr=song_sr, target_sr=recording_sr)
    # 将录制的音频数据合并成二维数组
    audio_data = np.frombuffer(input_data, dtype=np.float32)

    mfcc_song = librosa.feature.mfcc(y=audio_data, sr=recording_sr, n_mfcc=40)

    vector1 = mfcc_recording[:, 0]
    vector2 = mfcc_song[:, 0]

    similarity = 1 - cosine(vector1, vector2)
    print('Cosine Similarity:', similarity)
    return similarity > 0.98


def callback(indata, frames, time, status):
    """每次从麦克风收到音频缓冲区时调用的回调函数"""
    if status:
        print(status)
    if match_audio(indata):
        print(f"Match found! Current time: {datetime.datetime.now()}")


with sd.InputStream(callback=callback, channels=1, samplerate=samplerate, blocksize=int(samplerate * 2),
                    device=1, dtype='float32'):
    print(f"Listening for {duration} seconds per buffer...")
    sd.sleep(int(duration * 1000))
