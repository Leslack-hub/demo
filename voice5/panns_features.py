#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANNs特征提取器 - 专为音频检测优化的特征提取模块
提供高效的PANNs特征提取和相似度计算功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import logging
from functools import lru_cache
import threading
from typing import Dict, Tuple, Optional, Union
import time


class PANNsFeatureExtractor:
    """PANNs特征提取器 - 专为实时音频检测优化"""
    
    def __init__(self, 
                 model: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 cache_size: int = 100):
        """
        初始化PANNs特征提取器
        
        Args:
            model: PANNs模型实例
            device: 计算设备
            cache_size: 特征缓存大小
        """
        self.model = model
        self.device = device or self._get_optimal_device()
        self.cache_size = cache_size
        self.feature_cache = {}
        self.cache_lock = threading.Lock()
        
        # 设置日志
        self.logger = logging.getLogger('PANNsFeatureExtractor')
        
        # 性能统计
        self.extraction_times = []
        
    def _get_optimal_device(self) -> torch.device:
        """智能设备选择"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取PANNs特征
        
        Args:
            waveform: 音频波形 [batch_size, num_samples]
            
        Returns:
            特征字典，包含embedding、logits等
        """
        start_time = time.time()
        
        try:
            # 确保输入在正确设备上
            if waveform.device != self.device:
                waveform = waveform.to(self.device)
            
            # 暂时禁用缓存以避免误检测问题
            # TODO: 改进缓存键生成算法，使其更加精确
            # cache_key = self._get_cache_key(waveform)
            # if cache_key in self.feature_cache:
            #     return self.feature_cache[cache_key]
            
            # 提取特征
            with torch.no_grad():
                if self.model is None:
                    raise ValueError("PANNs模型未初始化")
                
                outputs = self.model(waveform)
                
                # 暂时禁用缓存
                # with self.cache_lock:
                #     if len(self.feature_cache) >= self.cache_size:
                #         # 移除最旧的缓存项
                #         oldest_key = next(iter(self.feature_cache))
                #         del self.feature_cache[oldest_key]
                #     
                #     self.feature_cache[cache_key] = outputs
                
                # 记录性能
                extraction_time = time.time() - start_time
                self.extraction_times.append(extraction_time)
                if len(self.extraction_times) > 100:
                    self.extraction_times.pop(0)
                
                return outputs
                
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            raise
    
    def _get_cache_key(self, waveform: torch.Tensor) -> str:
        """生成缓存键"""
        # 使用波形的哈希值作为缓存键
        waveform_hash = hash(waveform.cpu().numpy().tobytes())
        return f"{waveform.shape}_{waveform_hash}"
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if not self.extraction_times:
            return {}
        
        return {
            'avg_extraction_time': np.mean(self.extraction_times),
            'min_extraction_time': np.min(self.extraction_times),
            'max_extraction_time': np.max(self.extraction_times),
            'total_extractions': len(self.extraction_times)
        }
    
    def clear_cache(self):
        """清空特征缓存"""
        with self.cache_lock:
            self.feature_cache.clear()


class PANNsSimilarityCalculator:
    """PANNs相似度计算器"""
    
    def __init__(self, 
                 embedding_weight: float = 0.5,
                 logits_weight: float = 0.3,
                 topk_weight: float = 0.2,
                 top_k: int = 10):
        """
        初始化相似度计算器
        
        Args:
            embedding_weight: 嵌入向量相似度权重
            logits_weight: 分类输出相似度权重
            topk_weight: Top-K重叠度权重
            top_k: Top-K类别数量
        """
        self.embedding_weight = embedding_weight
        self.logits_weight = logits_weight
        self.topk_weight = topk_weight
        self.top_k = top_k
        
        # 确保权重和为1
        total_weight = embedding_weight + logits_weight + topk_weight
        self.embedding_weight /= total_weight
        self.logits_weight /= total_weight
        self.topk_weight /= total_weight
    
    def calculate_similarity(self, 
                           features1: Dict[str, torch.Tensor],
                           features2: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """
        计算两个特征之间的相似度
        
        Args:
            features1: 第一个特征字典
            features2: 第二个特征字典
            
        Returns:
            (总相似度, 详细相似度字典)
        """
        similarities = {}
        
        try:
            # 检查特征是否为零或无效
            def is_zero_features(features):
                if not features:
                    return True
                for key, tensor in features.items():
                    if torch.sum(torch.abs(tensor)) > 1e-8:
                        return False
                return True
            
            # 检查模型输出的有效性（针对未训练模型）
            def is_invalid_model_output(features):
                return self._is_invalid_model_output(features)
            
            # 如果任一特征为零，返回零相似度
            if is_zero_features(features1) or is_zero_features(features2):
                return 0.0, {'embedding': 0.0, 'kl_divergence': 0.0, 'topk_overlap': 0.0, 'total': 0.0, 'confidence': 0.0, 'reason': 'zero_features'}
            
            # 如果检测到无效的模型输出，返回低相似度
            if is_invalid_model_output(features1) or is_invalid_model_output(features2):
                return 0.0, {'embedding': 0.0, 'kl_divergence': 0.0, 'topk_overlap': 0.0, 'total': 0.0, 'confidence': 0.0, 'reason': 'invalid_model_output'}
            
            # 1. 嵌入向量相似度 (余弦相似度)
            if 'embedding' in features1 and 'embedding' in features2:
                emb1 = features1['embedding']
                emb2 = features2['embedding']
                
                # 确保维度匹配
                if emb1.dim() > 1:
                    emb1 = emb1.mean(dim=0, keepdim=True)
                if emb2.dim() > 1:
                    emb2 = emb2.mean(dim=0, keepdim=True)
                
                # 检查嵌入是否为零
                if torch.sum(torch.abs(emb1)) < 1e-8 or torch.sum(torch.abs(emb2)) < 1e-8:
                    similarities['embedding'] = 0.0
                else:
                    cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)
                    embedding_similarity = cos_sim.mean().item()
                    similarities['embedding'] = max(0.0, embedding_similarity)  # 确保非负
            else:
                similarities['embedding'] = 0.0
            
            # 2. 分类输出相似度 (基于KL散度)
            if 'logits' in features1 and 'logits' in features2:
                logits1 = features1['logits']
                logits2 = features2['logits']
                
                # 转换为概率分布
                probs1 = F.softmax(logits1, dim=-1)
                probs2 = F.softmax(logits2, dim=-1)
                
                # 计算KL散度
                kl_div = F.kl_div(probs2.log(), probs1, reduction='batchmean')
                kl_similarity = 1.0 / (1.0 + kl_div.item())  # 转换为相似度
                similarities['kl_divergence'] = kl_similarity
            else:
                similarities['kl_divergence'] = 0.0
            
            # 3. Top-K类别重叠度
            if 'logits' in features1 and 'logits' in features2:
                logits1 = features1['logits']
                logits2 = features2['logits']
                
                # 获取Top-K类别
                _, top_k1 = torch.topk(logits1, self.top_k, dim=-1)
                _, top_k2 = torch.topk(logits2, self.top_k, dim=-1)
                
                # 计算交集
                if top_k1.dim() > 1:
                    top_k1 = top_k1.flatten()
                if top_k2.dim() > 1:
                    top_k2 = top_k2.flatten()
                
                set1 = set(top_k1.cpu().numpy())
                set2 = set(top_k2.cpu().numpy())
                intersection = len(set1 & set2)
                topk_similarity = intersection / self.top_k
                similarities['topk_overlap'] = topk_similarity
            else:
                similarities['topk_overlap'] = 0.0
            
            # 4. 计算加权总相似度
            total_similarity = (
                self.embedding_weight * similarities['embedding'] +
                self.logits_weight * similarities['kl_divergence'] +
                self.topk_weight * similarities['topk_overlap']
            )
            
            # 添加额外的相似度指标
            similarities['total'] = total_similarity
            similarities['confidence'] = self._calculate_confidence(features1, features2)
            
            return total_similarity, similarities
            
        except Exception as e:
            logging.error(f"相似度计算错误: {e}")
            return 0.0, {'error': str(e)}
    
    def _is_invalid_model_output(self, features: Dict[str, torch.Tensor]) -> bool:
        """
        检查模型输出是否有效（针对未训练模型）
        """
        if not features or 'logits' not in features:
            return True
        
        logits = features['logits']
        # 检查logits是否过于随机（方差过大或分布过于均匀）
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        max_entropy = torch.log(torch.tensor(float(probs.shape[-1])))
        
        # 如果熵接近最大值，说明分布过于均匀，可能是未训练模型
        if entropy / max_entropy > 0.95:
            return True
        
        return False
    
    def _calculate_confidence(self, 
                            features1: Dict[str, torch.Tensor],
                            features2: Dict[str, torch.Tensor]) -> float:
        """
        计算检测置信度
        
        Args:
            features1: 目标特征
            features2: 当前特征
            
        Returns:
            置信度分数
        """
        try:
            if 'logits' not in features1 or 'logits' not in features2:
                return 0.0
            
            # 使用目标音频的最高激活类别作为参考
            target_logits = features1['logits']
            current_logits = features2['logits']
            
            # 获取目标音频的主要类别
            target_max_idx = torch.argmax(target_logits, dim=-1)
            
            # 计算当前音频在该类别上的激活强度
            if target_max_idx.dim() == 0:
                current_activation = current_logits[target_max_idx]
            else:
                current_activation = current_logits[0, target_max_idx[0]]
            
            # 转换为概率
            confidence = torch.sigmoid(current_activation).item()
            
            return confidence
            
        except Exception as e:
            logging.error(f"置信度计算错误: {e}")
            return 0.0


class AudioPreprocessor:
    """音频预处理器"""
    
    def __init__(self, 
                 target_sr: int = 32000,
                 normalize: bool = True,
                 trim_silence: bool = True):
        """
        初始化音频预处理器
        
        Args:
            target_sr: 目标采样率
            normalize: 是否归一化
            trim_silence: 是否裁剪静音
        """
        self.target_sr = target_sr
        self.normalize = normalize
        self.trim_silence = trim_silence
    
    def preprocess(self, 
                  audio_data: Union[np.ndarray, torch.Tensor],
                  sample_rate: int) -> torch.Tensor:
        """
        预处理音频数据
        
        Args:
            audio_data: 音频数据
            sample_rate: 原始采样率
            
        Returns:
            预处理后的音频张量
        """
        logger = logging.getLogger('AudioPreprocessor')
        logger.debug(f"开始预处理，输入形状: {audio_data.shape if hasattr(audio_data, 'shape') else len(audio_data)}")
        
        # 转换为numpy数组
        if isinstance(audio_data, torch.Tensor):
            audio_np = audio_data.cpu().numpy()
        else:
            audio_np = audio_data
        logger.debug("转换为numpy数组完成")
        
        # 确保单声道
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=0)
        logger.debug("单声道转换完成")
        
        # 重采样
        if sample_rate != self.target_sr:
            logger.debug(f"开始重采样: {sample_rate} -> {self.target_sr}")
            audio_np = librosa.resample(
                audio_np, 
                orig_sr=sample_rate, 
                target_sr=self.target_sr
            )
            logger.debug("重采样完成")
        
        # 检查音频有效性
        audio_rms = np.sqrt(np.mean(audio_np ** 2))
        logger.debug(f"音频RMS: {audio_rms}")
        
        # 如果音频过于安静，返回零张量避免误检测
        if audio_rms < 1e-6:
            logger.debug("音频过于安静，返回零张量")
            zero_tensor = torch.zeros(1, int(self.target_sr * 1.0))  # 1秒的零音频
            return zero_tensor
        
        # 裁剪静音
        if self.trim_silence:
            logger.debug("开始裁剪静音")
            try:
                audio_np, _ = librosa.effects.trim(audio_np, top_db=20)
                # 检查裁剪后是否还有足够的音频
                if len(audio_np) < self.target_sr * 0.1:  # 少于0.1秒
                    logger.debug("裁剪后音频过短，返回零张量")
                    zero_tensor = torch.zeros(1, int(self.target_sr * 1.0))
                    return zero_tensor
            except Exception as e:
                logger.warning(f"静音裁剪失败: {e}，跳过裁剪")
            logger.debug("静音裁剪完成")
        
        # 归一化
        if self.normalize:
            logger.debug("开始归一化")
            max_val = np.max(np.abs(audio_np))
            if max_val > 1e-8:  # 提高阈值
                audio_np = audio_np / max_val
            else:
                logger.debug("音频幅度过小，返回零张量")
                zero_tensor = torch.zeros(1, int(self.target_sr * 1.0))
                return zero_tensor
            logger.debug("归一化完成")
        
        # 转换为torch张量
        logger.debug("转换为torch张量")
        audio_tensor = torch.from_numpy(audio_np).float()
        
        # 确保有batch维度
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        logger.debug(f"预处理完成，输出形状: {audio_tensor.shape}")
        return audio_tensor


class PANNsFeatureEngine:
    """PANNs特征提取引擎 - 整合所有组件的主要接口"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 config: Optional[Dict] = None):
        """
        初始化PANNs特征引擎
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备
            config: 配置字典
        """
        # 设置日志（首先初始化）
        self.logger = logging.getLogger('PANNsFeatureEngine')
        
        self.config = config or self._get_default_config()
        self.device = device or self._get_optimal_device()
        
        # 初始化组件
        self.preprocessor = AudioPreprocessor(
            target_sr=self.config.get('sample_rate', 32000),
            normalize=self.config.get('normalize', True),
            trim_silence=self.config.get('trim_silence', True)
        )
        
        self.similarity_calculator = PANNsSimilarityCalculator(
            embedding_weight=self.config.get('embedding_weight', 0.5),
            logits_weight=self.config.get('logits_weight', 0.3),
            topk_weight=self.config.get('topk_weight', 0.2),
            top_k=self.config.get('top_k', 10)
        )
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 初始化特征提取器
        self.feature_extractor = PANNsFeatureExtractor(
            model=self.model,
            device=self.device,
            cache_size=self.config.get('cache_size', 100)
        )
        
        # 模型训练状态标记
        self.is_model_trained = model_path is not None and Path(model_path).exists() if model_path else False
        
        # 目标音频特征缓存
        self.target_features = None
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'sample_rate': 32000,
            'normalize': True,
            'trim_silence': True,
            'embedding_weight': 0.5,
            'logits_weight': 0.3,
            'topk_weight': 0.2,
            'top_k': 10,
            'cache_size': 100,
            'model_arch': 'cnn14'
        }
    
    def _get_optimal_device(self) -> torch.device:
        """智能设备选择"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        加载PANNs模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的模型
        """
        try:
            # 这里应该加载真实的PANNs模型
            # 暂时使用简化版本作为占位符
            from panns_implementation import PANNsModel
            
            model = PANNsModel(
                backbone_arch=self.config.get('model_arch', 'cnn14'),
                use_attention=True
            )
            
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.debug(f"已加载预训练模型: {model_path}")
            else:
                self.logger.warning("警告：使用未训练的PANNs模型，可能导致不准确的结果")
                self.logger.warning("未找到预训练模型，使用随机初始化权重")
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def load_target_audio(self, audio_path: str) -> bool:
        """
        加载目标音频并提取特征
        
        Args:
            audio_path: 目标音频文件路径
            
        Returns:
            是否成功加载
        """
        try:
            self.logger.debug(f"开始加载音频文件: {audio_path}")
            # 加载音频文件
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            self.logger.debug(f"音频加载完成，采样率: {sample_rate}, 长度: {len(audio_data)}")
            
            # 预处理
            self.logger.debug("开始音频预处理")
            processed_audio = self.preprocessor.preprocess(audio_data, sample_rate)
            processed_audio = processed_audio.to(self.device)
            self.logger.debug(f"预处理完成，张量形状: {processed_audio.shape}")
            
            # 提取特征
            self.logger.debug("开始特征提取")
            self.target_features = self.feature_extractor.extract_features(processed_audio)
            self.logger.debug("特征提取完成")
            
            self.logger.debug(f"目标音频特征提取完成: {audio_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"目标音频加载失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    def extract_features_from_audio(self, 
                                  audio_data: Union[np.ndarray, torch.Tensor],
                                  sample_rate: int) -> Dict[str, torch.Tensor]:
        """
        从音频数据提取PANNs特征
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            特征字典
        """
        try:
            # 预处理
            processed_audio = self.preprocessor.preprocess(audio_data, sample_rate)
            processed_audio = processed_audio.to(self.device)
            
            # 提取特征
            features = self.feature_extractor.extract_features(processed_audio)
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            return {}
    
    def calculate_similarity_with_target(self, 
                                       audio_data: Union[np.ndarray, torch.Tensor],
                                       sample_rate: int) -> Tuple[float, Dict[str, float]]:
        """
        计算当前音频与目标音频的相似度
        
        Args:
            audio_data: 当前音频数据
            sample_rate: 采样率
            
        Returns:
            (总相似度, 详细相似度字典)
        """
        if self.target_features is None:
            raise ValueError("请先加载目标音频")
        
        try:
            # 提取当前音频特征
            current_features = self.extract_features_from_audio(audio_data, sample_rate)
            
            if not current_features:
                return 0.0, {'error': '特征提取失败'}
            
            # 计算相似度
            similarity, details = self.similarity_calculator.calculate_similarity(
                self.target_features, current_features
            )
            
            # 如果使用未训练模型，降低相似度阈值
            if not self.is_model_trained and similarity > 0.1:
                similarity *= 0.1  # 大幅降低相似度
                details['adjusted_for_untrained_model'] = True
            
            return similarity, details
            
        except Exception as e:
            self.logger.error(f"相似度计算失败: {e}")
            return 0.0, {'error': str(e)}
    
    def get_performance_stats(self) -> Dict[str, any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        stats = {
            'device': str(self.device),
            'config': self.config,
            'target_loaded': self.target_features is not None
        }
        
        # 添加特征提取器统计
        extractor_stats = self.feature_extractor.get_performance_stats()
        stats.update(extractor_stats)
        
        return stats
    
    def clear_cache(self):
        """清空所有缓存"""
        self.feature_extractor.clear_cache()
        self.target_features = None
        self.logger.debug("缓存已清空")


# 兼容性函数，保持与原有代码的接口一致
def create_panns_feature_engine(config: Dict) -> PANNsFeatureEngine:
    """
    创建PANNs特征引擎实例
    
    Args:
        config: 配置字典
        
    Returns:
        PANNs特征引擎实例
    """
    return PANNsFeatureEngine(config=config)


def extract_panns_features(audio_data: np.ndarray, 
                          sample_rate: int,
                          engine: PANNsFeatureEngine) -> Dict[str, np.ndarray]:
    """
    提取PANNs特征的便捷函数
    
    Args:
        audio_data: 音频数据
        sample_rate: 采样率
        engine: PANNs特征引擎
        
    Returns:
        特征字典（numpy格式）
    """
    features = engine.extract_features_from_audio(audio_data, sample_rate)
    
    # 转换为numpy格式
    numpy_features = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            numpy_features[key] = value.cpu().numpy()
        else:
            numpy_features[key] = value
    
    return numpy_features