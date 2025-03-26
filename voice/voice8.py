import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

def load_features(file_path):
    """ 读取文件并提取MFCC特征 """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfcc, axis=1)
    std_mfcc = np.std(mfcc, axis=1)
    feature = np.hstack((mean_mfcc, std_mfcc))
    return feature


def load_data(alarm_dir, non_alarm_dir):
    """ 从给定目录加载数据和标签 """
    features, labels = [], []
    # 处理警报声音
    for file in os.listdir(alarm_dir):
        feature = load_features(os.path.join(alarm_dir, file))
        features.append(feature)
        labels.append(1)  # 1 代表警报
    # 处理非警报声音
    for file in os.listdir(non_alarm_dir):
        feature = load_features(os.path.join(non_alarm_dir, file))
        features.append(feature)
        labels.append(0)  # 0 代表非警报
    return np.array(features), np.array(labels)


# 数据路径
alarm_dir = '../target/'
non_alarm_dir = '../non_target/'

# 加载数据
X, y = load_data(alarm_dir, non_alarm_dir)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 训练随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 测试模型
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 你的模型
# clf = RandomForestClassifier().fit(X_train, y_train)  # 假设你已经训练了这个模型

# 保存模型到文件
with open('../demo/model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# 可以选择一个示例文件来显示其MFCC
y, sr = librosa.load(os.path.join(alarm_dir, os.listdir(alarm_dir)[0]), sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
