import librosa.display
from scipy.spatial.distance import cosine
import librosa


# 读取音频文件
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # 加载音频文件，sr=None表示使用原采样率
    return y, sr


# 示例：读取录音和歌曲
recording_y, recording_sr = load_audio('../target/target_audio.wav')
song_y, song_sr = load_audio('output3.wav')
# 确保两者采样率相同，以便后续处理
if recording_sr != song_sr:
    song_y = librosa.resample(y=song_y, orig_sr=song_sr, target_sr=recording_sr)

# 提取MFCC特征
mfcc_recording = librosa.feature.mfcc(y=recording_y, sr=recording_sr, n_mfcc=40)
mfcc_song = librosa.feature.mfcc(y=song_y, sr=recording_sr, n_mfcc=40)

vector1 = mfcc_recording[:, 0]
vector2 = mfcc_song[:, 0]

similarity = 1 - cosine(vector1, vector2)
print('Cosine Similarity:', similarity)
