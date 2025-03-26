import sounddevice as sd
import numpy as np
import librosa
import time
import pickle  # 用于加载保存的模型

# 加载预先训练好的分类器
with open('../demo/model.pkl', 'rb') as file:  # 确保已经创建了模型并保存为clf_model.pkl
    clf = pickle.load(file)


def extract_features(y, sr=22050):
    """ 提取音频特征 """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfcc, axis=1)
    std_mfcc = np.std(mfcc, axis=1)
    feature = np.hstack((mean_mfcc, std_mfcc))
    return feature.reshape(1, -1)


def detect_alarm_sound(data, sr=22050):
    """ 使用模型检测警报声 """
    features = extract_features(np.array(data), sr)
    prediction = clf.predict(features)
    if prediction[0] == 1:
        print("警报声被检测到！时间：", time.strftime("%Y-%m-%d %H:%M:%S"))


def callback(indata, frames, time, status):
    """ 这是一个回调函数，用于处理实时音频数据 """
    if status:
        print(status)
    detect_alarm_sound(indata[:, 0])


try:
    # 使用默认设备以44100 Hz的频率录制声音
    with sd.InputStream(callback=callback, channels=1, device=1, samplerate=44100,
                        blocksize=int(44100 * 5), dtype='float32'):
        print("开始监听...")
        while True:
            time.sleep(1)  # 留空此循环让系统有机会处理音频
except KeyboardInterrupt:
    print("停止监听")
except Exception as e:
    print(str(e))
