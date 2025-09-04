# -*- coding: utf-8 -*-
"""
音频预处理和特征提取模块
"""

import warnings
from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings('ignore')

from config.config import AUDIO_CONFIG


class AudioProcessor:
    """
    音频处理器类，负责音频的加载、预处理和特征提取
    """
    
    def __init__(self, 
                 sample_rate: int = AUDIO_CONFIG['sample_rate'],
                 duration: float = AUDIO_CONFIG['duration'],
                 n_mels: int = AUDIO_CONFIG['n_mels'],
                 n_fft: int = AUDIO_CONFIG['n_fft'],
                 hop_length: int = AUDIO_CONFIG['hop_length']):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率
            duration: 音频片段长度(秒)
            n_mels: 梅尔频谱图的频率bins数量
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = int(sample_rate * duration)
        
    def load_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频数据数组
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            raise ValueError(f"无法加载音频文件 {file_path}: {e}")
    
    def normalize_length(self, audio: np.ndarray) -> np.ndarray:
        """
        标准化音频长度
        
        Args:
            audio: 音频数据
            
        Returns:
            标准化长度的音频数据
        """
        if len(audio) > self.target_length:
            # 如果音频过长，随机截取一段
            start = np.random.randint(0, len(audio) - self.target_length + 1)
            audio = audio[start:start + self.target_length]
        elif len(audio) < self.target_length:
            # 如果音频过短，用零填充
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        提取梅尔频谱图特征
        
        Args:
            audio: 音频数据
            
        Returns:
            梅尔频谱图 (n_mels, time_steps)
        """
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # 转换为对数刻度
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            audio: 音频数据
            n_mfcc: MFCC系数数量
            
        Returns:
            MFCC特征 (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def extract_spectral_features(self, audio: np.ndarray) -> dict:
        """
        提取频谱特征
        
        Args:
            audio: 音频数据
            
        Returns:
            包含各种频谱特征的字典
        """
        features = {}
        
        # 频谱质心
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # 频谱带宽
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # 频谱滚降
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # 零交叉率
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )
        
        return features
    
    def process_audio_file(self, file_path: Union[str, Path], 
                          feature_type: str = 'mel_spectrogram') -> np.ndarray:
        """
        处理单个音频文件，提取特征
        
        Args:
            file_path: 音频文件路径
            feature_type: 特征类型 ('mel_spectrogram', 'mfcc', 'spectral')
            
        Returns:
            提取的特征
        """
        # 加载音频
        audio = self.load_audio(file_path)
        
        # 标准化长度
        audio = self.normalize_length(audio)
        
        # 提取特征
        if feature_type == 'mel_spectrogram':
            features = self.extract_mel_spectrogram(audio)
        elif feature_type == 'mfcc':
            features = self.extract_mfcc(audio)
        elif feature_type == 'spectral':
            features = self.extract_spectral_features(audio)
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
        
        return features
    
    def save_audio(self, audio: np.ndarray, file_path: Union[str, Path]):
        """
        保存音频文件
        
        Args:
            audio: 音频数据
            file_path: 保存路径
        """
        sf.write(file_path, audio, self.sample_rate)
    
    def get_audio_info(self, file_path: Union[str, Path]) -> dict:
        """
        获取音频文件信息
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频信息字典
        """
        try:
            info = sf.info(file_path)
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames
            }
        except Exception as e:
            raise ValueError(f"无法获取音频信息 {file_path}: {e}")
    
    def preprocess_for_realtime(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        实时音频预处理
        
        Args:
            audio_chunk: 音频块数据
            
        Returns:
            预处理后的特征
        """
        # 如果音频块长度不足，用零填充
        if len(audio_chunk) < self.target_length:
            padding = self.target_length - len(audio_chunk)
            audio_chunk = np.pad(audio_chunk, (0, padding), mode='constant')
        elif len(audio_chunk) > self.target_length:
            # 如果过长，截取最后一段
            audio_chunk = audio_chunk[-self.target_length:]
        
        # 提取梅尔频谱图
        mel_spec = self.extract_mel_spectrogram(audio_chunk)
        
        return mel_spec


def create_dataset_from_directory(data_dir: Path, 
                                processor: AudioProcessor,
                                feature_type: str = 'mel_spectrogram') -> Tuple[np.ndarray, np.ndarray]:
    """
    从目录创建数据集
    
    Args:
        data_dir: 数据目录路径
        processor: 音频处理器
        feature_type: 特征类型
        
    Returns:
        特征数组和标签数组
    """
    features = []
    labels = []
    
    # 支持的音频格式
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    for audio_file in data_dir.rglob('*'):
        if audio_file.suffix.lower() in audio_extensions:
            try:
                # 提取特征
                feature = processor.process_audio_file(audio_file, feature_type)
                features.append(feature)
                
                # 根据文件夹名称确定标签
                if 'positive' in str(audio_file.parent).lower():
                    labels.append(1)  # 正样本
                else:
                    labels.append(0)  # 负样本
                    
            except Exception as e:
                print(f"处理文件 {audio_file} 时出错: {e}")
                continue
    
    return np.array(features), np.array(labels)