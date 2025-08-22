#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANNs在声音事件检测（SED）中的专门应用实现
提供完整的SED训练、评估和推理功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from dataclasses import dataclass

from voice5.panns_implementation import PANNsModel, LogMelSpectrogram


@dataclass
class SEDConfig:
    """声音事件检测配置"""
    sr: int = 32000
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 64
    fmin: int = 50
    fmax: int = 14000
    
    # 训练参数
    max_epoch: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # 模型参数
    backbone_arch: str = "cnn14"
    use_attention: bool = True
    dropout_rate: float = 0.1
    
    # 阈值设置
    detection_threshold: float = 0.5
    smoothing_alpha: float = 0.1  # 标签平滑


class SEDDataset(torch.utils.data.Dataset):
    """声音事件检测数据集"""
    
    def __init__(self, 
                 csv_file: str,
                 audio_dir: str,
                 config: SEDConfig,
                 transform=None,
                 duration: float = 10.0):
        
        self.df = pd.read_csv(csv_file)
        self.audio_dir = Path(audio_dir)
        self.config = config
        self.transform = transform
        self.duration = duration
        
        # 特征提取器
        self.mel_extractor = LogMelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.fmin,
            f_max=config.fmax
        )
        
        # 获取标签列
        self.labels = [col for col in self.df.columns if col.startswith('label_')]
        self.num_classes = len(self.labels)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载音频
        audio_path = self.audio_dir / str(row['filename'])
        waveform, sr = torchaudio.load(audio_path)
        
        # 重采样
        if sr != self.config.sr:
            resampler = torchaudio.transforms.Resample(sr, self.config.sr)
            waveform = resampler(waveform)
        
        # 取单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 裁剪或填充到固定长度
        target_length = int(self.duration * self.config.sr)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        else:
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
        
        # 归一化
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # 获取标签
        labels = torch.tensor(row[self.labels].values.astype(float), dtype=torch.float32)
        
        # 标签平滑
        if self.config.smoothing_alpha > 0:
            labels = (1 - self.config.smoothing_alpha) * labels + \
                     self.config.smoothing_alpha * 0.5
        
        return {
            'waveform': waveform.squeeze(),
            'labels': labels,
            'filename': str(row['filename'])
        }


class FrameLevelSED(nn.Module):
    """帧级别的声音事件检测模型"""
    
    def __init__(self, config: SEDConfig):
        super().__init__()
        
        self.config = config
        
        # PANNs主干网络
        backbone_config = {
            'backbone_arch': config.backbone_arch,
            'use_attention': config.use_attention,
            'embedding_dim': 512
        }
        
        self.backbone = PANNsModel(**backbone_config)
        
        # 帧级别解码器
        self.decode = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Conv1d(128, config.num_classes if hasattr(config, 'num_classes') else 527, 
                     kernel_size=1)
        )
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch_size, num_samples]
        
        Returns:
            frame_predictions: [batch_size, num_classes, num_frames]
        """
        
        batch_size = waveform.shape[0]
        
        # 提取骨干网络特征
        features = self.backbone.get_sound_event_detection_features(
            waveform, 
            hop_length=0.1
        )['frame_embeddings']
        
        # reshape features for decoder
        if len(features.shape) > 3:
            features = features.squeeze(1)  # [batch, time, features]
        
        # 转置维度适配Conv1d: [batch, features, time]
        features = features.transpose(1, 2)
        
        # 时间维度上的卷积
        frame_predictions = self.decode(features)  # [batch, classes, time_frames]
        
        return torch.sigmoid(frame_predictions)


class SEDMoodelWithCRNN(nn.Module):
    """CRNN结构的声音事件检测模型"""
    
    def __init__(self, config: SEDConfig, num_classes: int = 10):
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))  # 只在频率维度池化
        )
        
        # RNN序列建模
        self.gru = nn.GRU(
            input_size=256 * 8,  # 经过卷积后的特征维度
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512来自双向GRU
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: [batch_size, 1, freq_bins, time_frames]
        
        Returns:
            frame_predictions: [batch_size, num_classes, time_frames]  
        """
        batch_size, _, freq_bins, time_frames = spectrogram.shape
        
        # CNN特征提取
        features = self.cnn(spectrogram)  # [batch, channels, reduced_freq, time]
        
        # 重塑以适应RNN: [batch, time, features]
        features = features.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        features = features.reshape(batch_size, time_frames, -1)  # [batch, time, -1]
        
        # RNN序列建模
        hidden_states, _ = self.gru(features)  # [batch, time, 512]
        
        # 逐帧分类
        logits = self.classifier(hidden_states)  # [batch, time, num_classes]
        
        # 转置维度匹配PyTorch格式: [batch, time, classes] -> [batch, classes, time]
        logits = logits.permute(0, 2, 1)
        
        return torch.sigmoid(logits)


class SEDMetrics:
    """声音事件检测指标计算"""
    
    def __init__(self, num_classes: int, threshold: float = 0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        
    def compute_metrics(self, 
                       predictions: np.ndarray, 
                       targets: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: [batch_size, num_classes, num_frames]
            targets: [batch_size, num_classes, num_frames]
        
        Returns:
            dict: 包含precision, recall, f1, 和auc等指标
        """
        
        # 展平数据
        preds_flat = predictions.reshape(-1, predictions.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])
        
        # 二值化预测
        preds_binary = (preds_flat > self.threshold).astype(int)
        
        # 计算各帧指标
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            targets_flat.flatten(), 
            preds_binary.flatten(),
            average='weighted',
            zero_division=0
        )
        
        # 计算每个类别的AUC
        auc_scores = []
        for i in range(self.num_classes):
            try:
                auc = roc_auc_score(
                    targets_flat[:, i], 
                    preds_flat[:, i],
                    average=None
                )
                auc_scores.append(auc)
            except:
                auc_scores.append(0.0)
        
        return {
            'precision': precisions,
            'recall': recalls,
            'f1_score': f1s,
            'mean_auc': np.mean(auc_scores),
            'auc_scores': auc_scores
        }


class SEDPipeline:
    """完整的SED训练和评估流程"""
    
    def __init__(self, config: SEDConfig, model_name: str = "CRNN"):
        self.config = config
        self.model_name = model_name
        
        if model_name == "CRNN":
            self.model = SEDMoodelWithCRNN(config, 10)  # 10个声音事件类别
        else:
            self.model = FrameLevelSED(config)
        
        self.metrics = SEDMetrics(num_classes=10)
        
    def train(self, 
              train_dataset: SEDDataset, 
              val_dataset: SEDDataset,
              save_path: str,
              device: torch.device = torch.device('cpu')):
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            save_path: 模型保存路径
            device: 计算设备
        """
        self.model.to(device)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        criterion = nn.BCELoss()
        
        best_val_f1 = 0.0
        
        for epoch in range(self.config.max_epoch):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                waveform = batch['waveform'].to(device)
                labels = batch['labels'].to(device)
                
                # 提取梅尔频谱图
                mel_spec = SEDDataset('dummy.csv', 'dummy', self.config).mel_extractor(waveform)
                mel_spec = mel_spec.unsqueeze(1)  # [batch, 1, mel_bins, time]
                
                # 前向传播
                outputs = self.model(mel_spec)
                
                # 确保维度匹配
                if outputs.shape[-1] > labels.shape[-1]:
                    outputs = outputs[:, :, :labels.shape[-1]]
                elif outputs.shape[-1] < labels.shape[-1]:
                    pad_size = labels.shape[-1] - outputs.shape[-1]
                    outputs = F.pad(outputs, (0, pad_size))
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            val_metrics = self.evaluate(val_loader, device)
            
            print(f"Epoch {epoch+1}/{self.config.max_epoch}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            
            # 保存最佳模型
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'val_f1': best_val_f1
                }, save_path)
        
        print(f"Training completed. Best F1: {best_val_f1:.4f}")
        
    def evaluate(self, 
                val_loader: torch.utils.data.DataLoader,
                device: torch.device) -> Dict[str, float]:
        """评估模型"""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch['waveform'].to(device)
                labels = batch['labels'].to(device)
                
                # 提取频谱图
                mel_extractor = val_loader.dataset.mel_extractor
                mel_spec = mel_extractor(waveform).unsqueeze(1)
                
                # 预测
                predictions = self.model(mel_spec)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        # 合并所有batch的数据
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        return self.metrics.compute_metrics(all_predictions, all_targets)


def detect_sound_events(audio_path: str, 
                       model_path: str,
                       config: SEDConfig,
                       frame_hop: float = 0.1) -> pd.DataFrame:
    """
    对单个音频文件进行声音事件检测
    
    Args:
        audio_path: 音频文件路径
        model_path: 模型权重路径
        config: 配置参数
        frame_hop: 帧移（秒）
    
    Returns:
        DataFrame: 包含事件开始时间、结束时间、类別和置信度的结果
    """
    
    # 加载模型
    model = SEDMoodelWithCRNN(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载音频
    waveform, sr = torchaudio.load(audio_path)
    
    # 重采样和预处理
    if sr != config.sr:
        resampler = torchaudio.transforms.Resample(sr, config.sr)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 提取梅尔频谱图
    mel_extractor = LogMelSpectrogram(
        sample_rate=config.sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels
    )
    
    with torch.no_grad():
        mel_spec = mel_extractor(waveform).unsqueeze(1)
        predictions = model(mel_spec).squeeze()
        
        # 转换为numpy
        predictions = predictions.numpy()
        
    # 事件后处理
    events = []
    hop_samples = config.hop_length
    frame_time = hop_samples / config.sr
    
    for class_id in range(predictions.shape[0]):
        class_predictions = predictions[class_id]
        
        # 平滑处理
        smoothed = np.convolve(class_predictions, [0.2, 0.6, 0.2], mode='same')
        
        # 阈值检测
        detection_threshold = 0.4
        above_threshold = (smoothed > detection_threshold).astype(int)
        
        # 寻找连续事件
        diffs = np.diff(above_threshold)
        starts = np.where(diffs > 0)[0] + 1
        ends = np.where(diffs < 0)[0] + 1
        
        for start, end in zip(starts, ends):
            if end - start >= 3:  # 最小事件长度3帧
                max_prob = np.max(smoothed[start:end])
                events.append({
                    'start_time': start * frame_time,
                    'end_time': end * frame_time,
                    'class_id': class_id,
                    'confidence': float(max_prob)
                })
    
    return pd.DataFrame(events)


if __name__ == "__main__":
    # 配置和示例用法
    config = SEDConfig(
        sr=32000,
        max_epoch=50,
        batch_size=16,
        learning_rate=1e-3
    )
    
    # 创建测试数据（实际使用需要真实数据）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 示例：训练流程
    # pipeline = SEDPipeline(config, "CRNN")
    # pipeline.train(train_dataset, val_dataset, "best_model.pt", device)
    
    # 示例：推理流程
    # events = detect_sound_events("test.wav", "best_model.pt", config)
    # print("检测到的事件:")
    # print(events)