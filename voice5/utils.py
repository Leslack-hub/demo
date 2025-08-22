#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANNs音频检测器工具函数模块
提供音频处理、设备管理、性能监控等实用功能
"""

import os
import time
import psutil
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from functools import wraps
import json
from datetime import datetime


def timer(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.3f}秒")
        return result
    return wrapper


def log_performance(func):
    """性能日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"{func.__name__} 执行失败: {e}")
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            logger.info(
                f"{func.__name__} - "
                f"时间: {end_time - start_time:.3f}s, "
                f"内存: {end_memory - start_memory:+.1f}MB, "
                f"状态: {'成功' if success else '失败'}"
            )
        
        return result
    return wrapper


class AudioUtils:
    """音频处理工具类"""
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 32000) -> Tuple[np.ndarray, int]:
        """加载音频文件
        
        Args:
            file_path: 音频文件路径
            target_sr: 目标采样率
            
        Returns:
            (音频数据, 采样率)
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        try:
            # 优先使用torchaudio
            waveform, sr = torchaudio.load(file_path)
            audio = waveform.numpy().squeeze()
            
            # 转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # 重采样
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            return audio, sr
            
        except Exception as e:
            # 降级使用librosa
            try:
                audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
                return audio, sr
            except Exception as e2:
                raise RuntimeError(f"音频加载失败: {e}, {e2}")
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sr: int = 32000):
        """保存音频文件"""
        try:
            # 确保音频数据格式正确
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 归一化到[-1, 1]
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # 转换为torch张量并保存
            waveform = torch.from_numpy(audio).unsqueeze(0)
            torchaudio.save(file_path, waveform, sr)
            
        except Exception as e:
            raise RuntimeError(f"音频保存失败: {e}")
    
    @staticmethod
    def get_audio_info(file_path: str) -> Dict[str, any]:
        """获取音频文件信息"""
        try:
            info = torchaudio.info(file_path)
            return {
                'sample_rate': info.sample_rate,
                'num_frames': info.num_frames,
                'num_channels': info.num_channels,
                'duration': info.num_frames / info.sample_rate,
                'encoding': info.encoding,
                'bits_per_sample': info.bits_per_sample
            }
        except Exception as e:
            raise RuntimeError(f"获取音频信息失败: {e}")
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """裁剪静音部分"""
        try:
            # 计算音频能量
            energy = np.abs(audio)
            
            # 找到非静音部分
            non_silent = energy > threshold
            
            if not np.any(non_silent):
                # 如果全是静音，返回中间部分
                return audio[len(audio)//4:3*len(audio)//4]
            
            # 找到开始和结束位置
            start_idx = np.argmax(non_silent)
            end_idx = len(non_silent) - np.argmax(non_silent[::-1]) - 1
            
            return audio[start_idx:end_idx+1]
            
        except Exception as e:
            logging.warning(f"静音裁剪失败: {e}，返回原音频")
            return audio
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
        """音频归一化
        
        Args:
            audio: 音频数据
            method: 归一化方法 ('peak', 'rms')
        """
        if method == 'peak':
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
        elif method == 'rms':
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                return audio / rms * 0.1  # 目标RMS为0.1
        
        return audio


class DeviceUtils:
    """设备管理工具类"""
    
    @staticmethod
    def list_audio_devices() -> List[Dict[str, any]]:
        """列出所有音频设备"""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'is_default': i == sd.default.device[0]
                    })
        except Exception as e:
            logging.error(f"获取音频设备列表失败: {e}")
        
        return devices
    
    @staticmethod
    def get_optimal_device() -> torch.device:
        """获取最优计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logging.info("使用MPS设备")
        else:
            device = torch.device('cpu')
            logging.info("使用CPU设备")
        
        return device
    
    @staticmethod
    def get_device_info() -> Dict[str, any]:
        """获取设备信息"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
            'memory_available': psutil.virtual_memory().available / 1024**3,  # GB
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_available': True,
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(),
                'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
            })
        else:
            info['cuda_available'] = False
        
        if hasattr(torch.backends, 'mps'):
            info['mps_available'] = torch.backends.mps.is_available()
        
        return info


class PerformanceMonitor:
    """性能监控工具类"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = []
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.metrics = []
    
    def record(self, name: str, value: float, unit: str = ""):
        """记录性能指标"""
        if self.start_time is None:
            self.start()
        
        self.metrics.append({
            'timestamp': time.time() - self.start_time,
            'name': name,
            'value': value,
            'unit': unit
        })
    
    def get_summary(self) -> Dict[str, any]:
        """获取性能摘要"""
        if not self.metrics:
            return {}
        
        summary = {
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'metric_count': len(self.metrics),
            'metrics': {}
        }
        
        # 按名称分组统计
        for metric in self.metrics:
            name = metric['name']
            if name not in summary['metrics']:
                summary['metrics'][name] = {
                    'values': [],
                    'unit': metric['unit']
                }
            summary['metrics'][name]['values'].append(metric['value'])
        
        # 计算统计信息
        for name, data in summary['metrics'].items():
            values = data['values']
            data.update({
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            })
        
        return summary
    
    def save_report(self, file_path: str):
        """保存性能报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'device_info': DeviceUtils.get_device_info(),
            'performance_summary': self.get_summary(),
            'raw_metrics': self.metrics
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)


class FileUtils:
    """文件处理工具类"""
    
    @staticmethod
    def ensure_dir(dir_path: str):
        """确保目录存在"""
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """获取文件大小(字节)"""
        return Path(file_path).stat().st_size
    
    @staticmethod
    def clean_old_files(dir_path: str, max_age_days: int = 7):
        """清理旧文件"""
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for file_path in Path(dir_path).glob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    logging.info(f"删除旧文件: {file_path}")


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """设置日志配置"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        FileUtils.ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_audio_file(file_path: str) -> bool:
    """验证音频文件是否有效"""
    try:
        info = AudioUtils.get_audio_info(file_path)
        return info['duration'] > 0 and info['sample_rate'] > 0
    except:
        return False


def format_duration(seconds: float) -> str:
    """格式化时长显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}时{minutes}分{secs:.1f}秒"


def format_size(bytes_size: int) -> str:
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"


if __name__ == "__main__":
    # 工具函数使用示例
    print("=== PANNs工具函数示例 ===")
    
    # 设备信息
    device_info = DeviceUtils.get_device_info()
    print(f"设备信息: {device_info}")
    
    # 音频设备列表
    audio_devices = DeviceUtils.list_audio_devices()
    print(f"音频设备数量: {len(audio_devices)}")
    
    # 性能监控示例
    monitor = PerformanceMonitor()
    monitor.start()
    
    # 模拟一些操作
    time.sleep(0.1)
    monitor.record("test_metric", 1.23, "ms")
    
    summary = monitor.get_summary()
    print(f"性能摘要: {summary}")
    
    print("工具函数测试完成")