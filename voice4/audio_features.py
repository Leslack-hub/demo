#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取模块
提供多种音频特征提取方法，用于音频模式匹配
"""

import numpy as np
import librosa
from scipy import signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """
    音频特征提取器
    """
    
    def __init__(self, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        初始化特征提取器
        
        Args:
            sample_rate: 采样率
            n_mfcc: MFCC特征数量
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_mfcc(self, audio_data):
        """
        提取MFCC特征
        
        Args:
            audio_data: 音频数据数组
            
        Returns:
            MFCC特征矩阵
        """
        try:
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return mfccs
        except Exception as e:
            print(f"MFCC提取错误: {e}")
            return None
    
    def extract_spectral_features(self, audio_data):
        """
        提取频谱特征
        
        Args:
            audio_data: 音频数据数组
            
        Returns:
            包含多种频谱特征的字典
        """
        try:
            features = {}
            
            # 频谱质心
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )
            features['spectral_centroid'] = spectral_centroids[0]
            
            # 频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate
            )
            features['spectral_bandwidth'] = spectral_bandwidth[0]
            
            # 频谱对比度
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_data, sr=self.sample_rate
            )
            features['spectral_contrast'] = spectral_contrast
            
            # 频谱滚降
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )
            features['spectral_rolloff'] = spectral_rolloff[0]
            
            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features['zcr'] = zcr[0]
            
            return features
        except Exception as e:
            print(f"频谱特征提取错误: {e}")
            return None
    
    def extract_chroma_features(self, audio_data):
        """
        提取色度特征
        
        Args:
            audio_data: 音频数据数组
            
        Returns:
            色度特征矩阵
        """
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return chroma
        except Exception as e:
            print(f"色度特征提取错误: {e}")
            return None
    
    def extract_tonnetz_features(self, audio_data):
        """
        提取调性网络特征
        
        Args:
            audio_data: 音频数据数组
            
        Returns:
            调性网络特征矩阵
        """
        try:
            tonnetz = librosa.feature.tonnetz(
                y=audio_data,
                sr=self.sample_rate
            )
            return tonnetz
        except Exception as e:
            print(f"调性网络特征提取错误: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio_data):
        """
        提取梅尔频谱图
        
        Args:
            audio_data: 音频数据数组
            
        Returns:
            梅尔频谱图
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            # 转换为对数刻度
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel_spec
        except Exception as e:
            print(f"梅尔频谱图提取错误: {e}")
            return None
    
    def extract_all_features(self, audio_data):
        """
        提取所有特征
        
        Args:
            audio_data: 音频数据数组
            
        Returns:
            包含所有特征的字典
        """
        features = {}
        
        # MFCC特征
        mfcc = self.extract_mfcc(audio_data)
        if mfcc is not None:
            features['mfcc'] = mfcc
        
        # 频谱特征
        spectral_features = self.extract_spectral_features(audio_data)
        if spectral_features is not None:
            features.update(spectral_features)
        
        # 色度特征
        chroma = self.extract_chroma_features(audio_data)
        if chroma is not None:
            features['chroma'] = chroma
        
        # 调性网络特征
        tonnetz = self.extract_tonnetz_features(audio_data)
        if tonnetz is not None:
            features['tonnetz'] = tonnetz
        
        # 梅尔频谱图
        mel_spec = self.extract_mel_spectrogram(audio_data)
        if mel_spec is not None:
            features['mel_spectrogram'] = mel_spec
        
        return features
    
    def normalize_features(self, features):
        """
        标准化特征
        
        Args:
            features: 特征字典或数组
            
        Returns:
            标准化后的特征
        """
        if isinstance(features, dict):
            normalized = {}
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    # 使用Z-score标准化
                    mean = np.mean(value, axis=-1, keepdims=True)
                    std = np.std(value, axis=-1, keepdims=True)
                    std = np.where(std == 0, 1, std)  # 避免除零
                    normalized[key] = (value - mean) / std
                else:
                    normalized[key] = value
            return normalized
        elif isinstance(features, np.ndarray):
            mean = np.mean(features, axis=-1, keepdims=True)
            std = np.std(features, axis=-1, keepdims=True)
            std = np.where(std == 0, 1, std)
            return (features - mean) / std
        else:
            return features
    
    def compute_feature_statistics(self, features):
        """
        计算特征统计量
        
        Args:
            features: 特征数组或字典
            
        Returns:
            特征统计量字典
        """
        stats = {}
        
        if isinstance(features, dict):
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    stats[f'{key}_mean'] = np.mean(value)
                    stats[f'{key}_std'] = np.std(value)
                    stats[f'{key}_max'] = np.max(value)
                    stats[f'{key}_min'] = np.min(value)
        elif isinstance(features, np.ndarray):
            stats['mean'] = np.mean(features)
            stats['std'] = np.std(features)
            stats['max'] = np.max(features)
            stats['min'] = np.min(features)
        
        return stats

class AudioMatcher:
    """
    音频匹配器，用于比较音频特征
    """
    
    def __init__(self, feature_extractor=None):
        """
        初始化匹配器
        
        Args:
            feature_extractor: 特征提取器实例（可选）
        """
        self.feature_extractor = feature_extractor
    
    def dtw_distance(self, features1, features2):
        """
        计算动态时间规整距离
        
        Args:
            features1: 第一个特征序列
            features2: 第二个特征序列
            
        Returns:
            DTW距离
        """
        try:
            # 如果是多维特征，使用第一维进行DTW
            if len(features1.shape) > 1:
                features1 = features1[0]
            if len(features2.shape) > 1:
                features2 = features2[0]
            
            distance, _ = fastdtw(features1, features2, dist=euclidean)
            return distance
        except Exception as e:
            print(f"DTW距离计算错误: {e}")
            return float('inf')
    
    def cross_correlation(self, signal1, signal2):
        """
        计算互相关
        
        Args:
            signal1: 第一个信号
            signal2: 第二个信号
            
        Returns:
            最大互相关值和位置
        """
        try:
            # 确保信号是一维的
            if len(signal1.shape) > 1:
                signal1 = np.mean(signal1, axis=0)
            if len(signal2.shape) > 1:
                signal2 = np.mean(signal2, axis=0)
            
            correlation = signal.correlate(signal1, signal2, mode='full')
            max_corr = np.max(correlation)
            max_pos = np.argmax(correlation)
            
            # 标准化相关值
            norm_corr = max_corr / (np.linalg.norm(signal1) * np.linalg.norm(signal2))
            
            return norm_corr, max_pos
        except Exception as e:
            print(f"互相关计算错误: {e}")
            return 0, 0
    
    def cosine_similarity(self, features1, features2):
        """
        计算余弦相似度
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            余弦相似度
        """
        try:
            # 展平特征向量
            f1 = features1.flatten()
            f2 = features2.flatten()
            
            # 确保向量长度相同
            min_len = min(len(f1), len(f2))
            f1 = f1[:min_len]
            f2 = f2[:min_len]
            
            # 计算余弦相似度
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            similarity = dot_product / (norm1 * norm2)
            return similarity
        except Exception as e:
            print(f"余弦相似度计算错误: {e}")
            return 0
    
    def euclidean_distance(self, features1, features2):
        """
        计算欧几里得距离
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            欧几里得距离
        """
        try:
            # 展平特征向量
            f1 = features1.flatten()
            f2 = features2.flatten()
            
            # 确保向量长度相同
            min_len = min(len(f1), len(f2))
            f1 = f1[:min_len]
            f2 = f2[:min_len]
            
            distance = np.linalg.norm(f1 - f2)
            return distance
        except Exception as e:
            print(f"欧几里得距离计算错误: {e}")
            return float('inf')

def extract_audio_features(audio_data, sample_rate, feature_config):
    """
    提取音频特征的包装函数
    
    Args:
        audio_data: 音频数据数组
        sample_rate: 采样率
        feature_config: 特征配置字典
        
    Returns:
        特征字典
    """
    extractor = AudioFeatureExtractor(
        sample_rate=sample_rate,
        n_mfcc=feature_config.get('mfcc_n_mfcc', 13),
        n_fft=feature_config.get('mfcc_n_fft', 2048),
        hop_length=feature_config.get('mfcc_hop_length', 512)
    )
    
    return extractor.extract_all_features(audio_data)

if __name__ == "__main__":
    # 测试代码
    import librosa
    
    # 加载测试音频文件
    audio_file = "/Users/leslack/lsc/300_study/python/demo/voice4/target.wav"
    y, sr = librosa.load(audio_file, sr=None)
    
    # 创建特征提取器
    extractor = AudioFeatureExtractor(sample_rate=sr)
    
    # 提取特征
    print("正在提取音频特征...")
    features = extractor.extract_all_features(y)
    
    print("\n=== 特征提取结果 ===")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: 形状 {value.shape}, 类型 {value.dtype}")
        else:
            print(f"{key}: {type(value)}")
    
    # 标准化特征
    normalized_features = extractor.normalize_features(features)
    print("\n特征标准化完成")
    
    # 计算统计量
    stats = extractor.compute_feature_statistics(features)
    print("\n=== 特征统计量 ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")