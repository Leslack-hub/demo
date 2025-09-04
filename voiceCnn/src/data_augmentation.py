# -*- coding: utf-8 -*-
"""
数据增强模块
"""

import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings('ignore')

from config.config import AUGMENTATION_CONFIG, AUDIO_CONFIG


class AudioAugmentor:
    """
    音频数据增强器
    """
    
    def __init__(self, 
                 sample_rate: int = AUDIO_CONFIG['sample_rate'],
                 time_stretch_factors: List[float] = AUGMENTATION_CONFIG['time_stretch_factors'],
                 pitch_shift_steps: List[int] = AUGMENTATION_CONFIG['pitch_shift_steps'],
                 noise_factor: float = AUGMENTATION_CONFIG['noise_factor'],
                 snr_range: List[float] = AUGMENTATION_CONFIG['snr_range']):
        """
        初始化数据增强器
        
        Args:
            sample_rate: 采样率
            time_stretch_factors: 时间拉伸因子列表
            pitch_shift_steps: 音调变换步数列表
            noise_factor: 噪声因子
            snr_range: 信噪比范围
        """
        self.sample_rate = sample_rate
        self.time_stretch_factors = time_stretch_factors
        self.pitch_shift_steps = pitch_shift_steps
        self.noise_factor = noise_factor
        self.snr_range = snr_range
    
    def time_stretch(self, audio: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        时间拉伸/压缩
        
        Args:
            audio: 音频数据
            factor: 拉伸因子，None时随机选择
            
        Returns:
            拉伸后的音频
        """
        if factor is None:
            factor = random.choice(self.time_stretch_factors)
        
        # 使用librosa进行时间拉伸
        stretched = librosa.effects.time_stretch(audio, rate=factor)
        
        return stretched
    
    def pitch_shift(self, audio: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        """
        音调变换
        
        Args:
            audio: 音频数据
            steps: 变换步数，None时随机选择
            
        Returns:
            变换后的音频
        """
        if steps is None:
            steps = random.choice(self.pitch_shift_steps)
        
        # 使用librosa进行音调变换
        shifted = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
        
        return shifted
    
    def add_noise(self, audio: np.ndarray, noise_factor: Optional[float] = None) -> np.ndarray:
        """
        添加随机噪声
        
        Args:
            audio: 音频数据
            noise_factor: 噪声因子，None时使用默认值
            
        Returns:
            添加噪声后的音频
        """
        if noise_factor is None:
            noise_factor = self.noise_factor
        
        # 生成随机噪声
        noise = np.random.normal(0, noise_factor, audio.shape)
        
        # 添加噪声
        noisy_audio = audio + noise
        
        # 确保音频值在合理范围内
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        
        return noisy_audio
    
    def mix_with_background(self, 
                           target_audio: np.ndarray, 
                           background_audio: np.ndarray, 
                           snr_db: Optional[float] = None) -> np.ndarray:
        """
        与背景音频混合
        
        Args:
            target_audio: 目标音频
            background_audio: 背景音频
            snr_db: 信噪比(dB)，None时随机选择
            
        Returns:
            混合后的音频
        """
        if snr_db is None:
            snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # 确保背景音频长度足够
        if len(background_audio) < len(target_audio):
            # 如果背景音频太短，重复它
            repeat_times = int(np.ceil(len(target_audio) / len(background_audio)))
            background_audio = np.tile(background_audio, repeat_times)
        
        # 随机选择背景音频的起始位置
        if len(background_audio) > len(target_audio):
            start_idx = random.randint(0, len(background_audio) - len(target_audio))
            background_audio = background_audio[start_idx:start_idx + len(target_audio)]
        
        # 计算功率
        target_power = np.mean(target_audio ** 2)
        background_power = np.mean(background_audio ** 2)
        
        # 根据SNR调整背景音频的音量
        snr_linear = 10 ** (snr_db / 10)
        background_scale = np.sqrt(target_power / (background_power * snr_linear))
        
        # 混合音频
        mixed_audio = target_audio + background_scale * background_audio
        
        # 归一化
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val
        
        return mixed_audio
    
    def volume_change(self, audio: np.ndarray, factor_range: Tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
        """
        音量变化
        
        Args:
            audio: 音频数据
            factor_range: 音量变化因子范围
            
        Returns:
            音量变化后的音频
        """
        factor = random.uniform(factor_range[0], factor_range[1])
        
        # 应用音量变化
        modified_audio = audio * factor
        
        # 确保音频值在合理范围内
        modified_audio = np.clip(modified_audio, -1.0, 1.0)
        
        return modified_audio
    
    def apply_random_augmentation(self, 
                                audio: np.ndarray, 
                                background_audios: Optional[List[np.ndarray]] = None,
                                augmentation_probability: float = 0.8) -> np.ndarray:
        """
        随机应用数据增强
        
        Args:
            audio: 原始音频
            background_audios: 背景音频列表
            augmentation_probability: 增强概率
            
        Returns:
            增强后的音频
        """
        augmented_audio = audio.copy()
        
        # 随机决定是否应用各种增强
        augmentations = [
            ('time_stretch', lambda x: self.time_stretch(x)),
            ('pitch_shift', lambda x: self.pitch_shift(x)),
            ('add_noise', lambda x: self.add_noise(x)),
            ('volume_change', lambda x: self.volume_change(x))
        ]
        
        # 随机选择要应用的增强方法
        for aug_name, aug_func in augmentations:
            if random.random() < augmentation_probability:
                try:
                    augmented_audio = aug_func(augmented_audio)
                except Exception as e:
                    print(f"应用 {aug_name} 增强时出错: {e}")
                    continue
        
        # 如果有背景音频，随机混合
        if background_audios and random.random() < AUGMENTATION_CONFIG['mix_probability']:
            background = random.choice(background_audios)
            try:
                augmented_audio = self.mix_with_background(augmented_audio, background)
            except Exception as e:
                print(f"混合背景音频时出错: {e}")
        
        return augmented_audio
    
    def generate_augmented_dataset(self, 
                                 positive_samples: List[np.ndarray],
                                 negative_samples: List[np.ndarray],
                                 background_audios: List[np.ndarray],
                                 augmentation_factor: int = 5) -> Tuple[List[np.ndarray], List[int]]:
        """
        生成增强数据集
        
        Args:
            positive_samples: 正样本列表
            negative_samples: 负样本列表
            background_audios: 背景音频列表
            augmentation_factor: 增强倍数
            
        Returns:
            增强后的音频列表和对应标签
        """
        augmented_audios = []
        labels = []
        
        # 处理正样本
        for audio in positive_samples:
            # 添加原始样本
            augmented_audios.append(audio)
            labels.append(1)
            
            # 生成增强样本
            for _ in range(augmentation_factor):
                augmented = self.apply_random_augmentation(audio, background_audios)
                augmented_audios.append(augmented)
                labels.append(1)
        
        # 处理负样本
        for audio in negative_samples:
            # 添加原始样本
            augmented_audios.append(audio)
            labels.append(0)
            
            # 生成增强样本（较少的增强）
            for _ in range(augmentation_factor // 2):
                augmented = self.apply_random_augmentation(audio, background_audios, 0.5)
                augmented_audios.append(augmented)
                labels.append(0)
        
        return augmented_audios, labels
    
    def save_augmented_samples(self, 
                             augmented_audios: List[np.ndarray],
                             labels: List[int],
                             output_dir: Path,
                             prefix: str = "aug"):
        """
        保存增强后的样本
        
        Args:
            augmented_audios: 增强后的音频列表
            labels: 标签列表
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (audio, label) in enumerate(zip(augmented_audios, labels)):
            label_name = "positive" if label == 1 else "negative"
            filename = f"{prefix}_{label_name}_{i:04d}.wav"
            filepath = output_dir / filename
            
            try:
                sf.write(filepath, audio, self.sample_rate)
            except Exception as e:
                print(f"保存文件 {filepath} 时出错: {e}")


def load_background_audios(background_dir: Path, 
                          sample_rate: int = AUDIO_CONFIG['sample_rate']) -> List[np.ndarray]:
    """
    加载背景音频
    
    Args:
        background_dir: 背景音频目录
        sample_rate: 采样率
        
    Returns:
        背景音频列表
    """
    background_audios = []
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    for audio_file in background_dir.rglob('*'):
        if audio_file.suffix.lower() in audio_extensions:
            try:
                audio, _ = librosa.load(audio_file, sr=sample_rate)
                background_audios.append(audio)
            except Exception as e:
                print(f"加载背景音频 {audio_file} 时出错: {e}")
                continue
    
    return background_audios