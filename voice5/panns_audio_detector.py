#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANNs音频检测器 - 基于PANNs技术的实时音频检测系统
支持目标音频检测、实时音频流处理和可视化界面
"""

import os
import time
import numpy as np
import torch
import torchaudio
import librosa
from pathlib import Path
import logging
import sounddevice as sd
from collections import deque
import threading
import argparse
from typing import Dict, Tuple, Optional, Callable
import functools

# 可视化相关导入
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PANNs相关导入
from panns_implementation import PANNsModel, LogMelSpectrogram
from panns_features import PANNsFeatureEngine, PANNsFeatureExtractor, PANNsSimilarityCalculator, AudioPreprocessor

# 设置matplotlib字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.1
plt.rcParams['agg.path.chunksize'] = 10000


def performance_monitor(func):
    """性能监控装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if hasattr(args[0], 'logger'):
            args[0].logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


class PANNsAudioDetector:
    """基于PANNs的音频检测器"""
    
    def __init__(self, 
                 target_audio_path: str,
                 model_arch: str = "cnn14",
                 debug_mode: bool = False,
                 input_device: Optional[int] = None,
                 detection_callback: Optional[Callable] = None):
        """
        初始化PANNs音频检测器
        
        Args:
            target_audio_path: 目标音频文件路径
            model_arch: PANNs模型架构 (cnn6, cnn10, cnn14)
            debug_mode: 是否启用调试模式
            input_device: 音频输入设备ID
            detection_callback: 检测回调函数
        """
        self.target_audio_path = target_audio_path
        self.model_arch = model_arch
        self.debug_mode = debug_mode
        self.input_device = input_device
        self.detection_callback = detection_callback
        
        # 音频参数配置
        self.sample_rate = 32000  # PANNs标准采样率
        self.chunk_size = 1024
        self.detection_threshold = 0.7  # PANNs检测阈值
        
        # 状态变量
        self.is_running = False
        self.detection_count = 0
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # 检测冷却时间（秒）
        self.is_currently_detecting = False
        
        # 缓冲区和线程锁
        self.audio_buffer = deque(maxlen=1024)
        self.buffer_lock = threading.Lock()
        
        # 可视化相关
        self.fig = None
        self.ax = None
        self.line = None
        self.detection_text = None
        self.waveform_data = deque(maxlen=int(0.2 * self.sample_rate))
        self._last_flush = 0
        self._last_plot_update = 0
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PANNsAudioDetector')
        
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.debug(f"使用设备: {self.device}")
        
        # 初始化PANNs特征引擎
        self._init_panns_feature_engine()
        
        # 加载目标音频
        self._load_target_audio()
        
        # 计算缓冲区参数
        self._calculate_buffer_params()
    
    def _init_panns_feature_engine(self):
        """初始化PANNs特征引擎"""
        try:
            self.logger.debug(f"初始化PANNs特征引擎: {self.model_arch}")
            
            # 初始化PANNs特征引擎
            self.feature_engine = PANNsFeatureEngine(
                device=self.device,
                config={
                    'model_arch': self.model_arch,
                    'sample_rate': 32000,
                    'normalize': True,
                    'trim_silence': True,
                    'embedding_weight': 0.5,
                    'logits_weight': 0.3,
                    'topk_weight': 0.2,
                    'top_k': 10,
                    'cache_size': 100
                }
            )
            
            self.logger.debug("PANNs特征引擎初始化完成")
            
        except Exception as e:
            self.logger.error(f"PANNs特征引擎初始化失败: {e}")
            raise
    
    @performance_monitor
    def _load_target_audio(self):
        """加载目标音频文件"""
        try:
            self.logger.debug(f"加载目标音频: {self.target_audio_path}")
            
            # 加载音频文件
            audio_data, sr = librosa.load(
                self.target_audio_path,
                sr=self.sample_rate
            )
            
            self.target_duration = len(audio_data) / sr
            self.logger.debug(f"目标音频时长: {self.target_duration:.2f}秒")
            
            # 使用特征引擎加载目标音频
            success = self.feature_engine.load_target_audio(self.target_audio_path)
            
            if not success:
                raise ValueError("目标音频加载失败")
            
            self.logger.debug("目标音频PANNs特征提取完成")
            
        except Exception as e:
            self.logger.error(f"加载目标音频失败: {e}")
            raise
    
    def _calculate_buffer_params(self):
        """计算缓冲区参数"""
        # 根据目标音频时长调整窗口大小
        self.window_duration = max(min(self.target_duration * 1.5, 3), 1)
        self.window_samples = int(self.window_duration * self.sample_rate)
        
        # 缓冲区时长
        self.buffer_duration = max(min(self.target_duration * 2 + 1, 10), 3)
        self.buffer_samples = int(self.buffer_duration * self.sample_rate)
        
        # 更新音频缓冲区大小
        self.audio_buffer = deque(maxlen=self.buffer_samples)
        
        self.logger.debug(f"缓冲区参数 - 窗口: {self.window_duration}s, 缓冲区: {self.buffer_duration}s")
    
    def _audio_callback(self, indata, frames, time, status):
        """音频输入回调函数"""
        if status:
            self.logger.warning(f"音频输入状态: {status}")
            return
        
        try:
            with self.buffer_lock:
                # 转换为单声道
                audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
                
                # 检查音频数据有效性
                if len(audio_chunk) > 0 and not np.all(audio_chunk == 0):
                    self.audio_buffer.extend(audio_chunk)
                    
                    # 更新可视化数据
                    if self.debug_mode:
                        self.waveform_data.extend(audio_chunk)
                        
        except Exception as e:
            self.logger.error(f"音频回调处理错误: {e}")
    
    def _get_audio_window(self):
        """从缓冲区获取活动窗口的音频数据"""
        with self.buffer_lock:
            if len(self.audio_buffer) < self.window_samples:
                return None
            
            # 获取最新的窗口数据
            window_data = np.array(list(self.audio_buffer)[-self.window_samples:])
            return window_data
    
    def _start_audio_stream(self):
        """启动音频流"""
        try:
            # 检查设备可用性
            if self.input_device is not None:
                devices = sd.query_devices()
                if self.input_device >= len(devices):
                    self.logger.error(f"设备ID {self.input_device} 不存在")
                    return False
                
                device_info = devices[self.input_device]
                if device_info['max_input_channels'] == 0:
                    self.logger.error(f"设备 {device_info['name']} 不支持音频输入")
                    return False
            
            # 构建音频流参数
            stream_params = {
                'samplerate': self.sample_rate,
                'channels': 1,
                'dtype': 'float32',
                'blocksize': self.chunk_size,
                'callback': self._audio_callback,
                'latency': 'low'
            }
            
            if self.input_device is not None:
                stream_params['device'] = (self.input_device, None)
            
            self.audio_stream = sd.InputStream(**stream_params)
            self.audio_stream.start()
            
            # 显示启动信息
            if self.input_device is not None:
                device_info = sd.query_devices()[self.input_device]
                self.logger.debug(f"音频流已启动 - 使用设备: {device_info['name']}")
            else:
                self.logger.debug("音频流已启动 - 使用默认设备")
            
            return True
            
        except Exception as e:
            self.logger.error(f"启动音频流失败: {e}")
            return False
    
    def _stop_audio_stream(self):
        """停止音频流"""
        try:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.logger.debug("音频流已停止")
        except Exception as e:
            self.logger.error(f"停止音频流失败: {e}")
    
    @performance_monitor
    def _detect_in_audio(self, audio_data: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """在音频数据中检测目标声音"""
        if audio_data is None or len(audio_data) == 0:
            return False, 0.0, {}
        
        # 计算音频电平
        audio_level = np.sqrt(np.mean(audio_data ** 2))
        
        # 提高音频电平阈值，避免检测静音或极低音量的音频
        if audio_level < 0.01:  # 从0.001提高到0.01
            if self.debug_mode:
                print(f"[DEBUG] 音频电平过低，跳过检测: {audio_level:.6f}")
            return False, 0.0, {'audio_level': audio_level, 'reason': 'low_level'}
        
        try:
            # 使用特征引擎计算相似度
            confidence, similarities = self.feature_engine.calculate_similarity_with_target(
                audio_data, self.sample_rate
            )
            
            # 添加调试信息
            if self.debug_mode:
                print(f"[DEBUG] 音频电平: {audio_level:.6f}, 置信度: {confidence:.3f}, 阈值: {self.detection_threshold}")
                if 'embedding' in similarities:
                    print(f"[DEBUG] 嵌入相似度: {similarities['embedding']:.3f}")
                if 'kl_divergence' in similarities:
                    print(f"[DEBUG] KL散度相似度: {similarities['kl_divergence']:.3f}")
                if 'topk_overlap' in similarities:
                    print(f"[DEBUG] TopK重叠度: {similarities['topk_overlap']:.3f}")
            
            # 判断是否检测到
            detected = confidence >= self.detection_threshold
            
            similarities['audio_level'] = audio_level
            
            return detected, confidence, similarities
            
        except Exception as e:
            self.logger.error(f"检测过程出错: {e}")
            return False, 0.0, {'error': str(e)}
    
    def _init_waveform_plot(self):
        """初始化波形图"""
        if self.debug_mode:
            self.fig, self.ax = plt.subplots(figsize=(12, 5))
            
            # 创建波形线
            self.line, = self.ax.plot([], [], 'b-', linewidth=1.0, alpha=0.8)
            
            # 设置坐标轴
            display_samples = min(3000, len(self.waveform_data) if self.waveform_data else 1000)
            self.ax.set_xlim(0, display_samples)
            self.ax.set_ylim(-0.1, 0.1)
            self.ax.set_xlabel("Samples")
            self.ax.set_ylabel("Amplitude")
            self.ax.set_title("PANNs Real-time Audio Detection")
            self.ax.grid(True, alpha=0.3)
            
            # 添加状态文本
            self.activity_text = self.ax.text(
                0.02, 0.95, 'Audio: Idle', transform=self.ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                zorder=10, clip_on=False
            )
            
            self.detection_text = self.ax.text(
                0.02, 0.85, '', transform=self.ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                zorder=10, clip_on=False
            )
            
            plt.ion()
            plt.show(block=False)
            
            self._last_flush = 0
            self._last_plot_update = 0
    
    def _update_waveform_plot(self):
        """更新波形图"""
        if self.debug_mode and self.line and self.fig:
            current_time = time.time()
            if hasattr(self, '_last_plot_update') and (current_time - self._last_plot_update) < 0.1:
                return
            self._last_plot_update = current_time
            
            with self.buffer_lock:
                if len(self.waveform_data) > 0:
                    max_display_samples = min(2000, len(self.waveform_data))
                    if len(self.waveform_data) > max_display_samples:
                        data_list = list(self.waveform_data)
                        data = np.array(data_list[-max_display_samples:])
                    else:
                        data = np.array(list(self.waveform_data))
                    
                    # 更新波形数据
                    x = np.arange(len(data))
                    self.line.set_data(x, data)
                    
                    # 动态调整Y轴范围
                    if len(data) > 0:
                        max_val = np.max(np.abs(data))
                        y_range = max(max_val * 1.2, 0.1)
                        self.ax.set_ylim(-y_range, y_range)
                    
                    # 更新X轴范围
                    self.ax.set_xlim(0, len(data))
            
            # 刷新图形
            try:
                self.fig.canvas.draw_idle()
                if hasattr(self, '_last_flush') and (time.time() - self._last_flush) > 0.05:
                    self.fig.canvas.flush_events()
                    self._last_flush = time.time()
            except:
                pass
    
    def start_detection(self):
        """开始检测"""
        self.is_running = True
        print("PANNs音频检测已启动")
        
        print("\n=== PANNs音频检测器已启动 ===")
        print("使用PANNs深度学习模型进行实时音频检测...")
        print(f"模型架构: {self.model_arch}")
        print(f"设备: {self.device}")
        
        # 显示设备信息
        if self.input_device is not None:
            device_info = sd.query_devices()[self.input_device]
            print(f"输入设备: {device_info['name']} (设备ID: {self.input_device})")
        else:
            default_device = sd.query_devices()[sd.default.device[0]]
            print(f"输入设备: {default_device['name']} (默认设备)")
        
        print(f"检测阈值: {self.detection_threshold}")
        print(f"缓冲区大小: {self.buffer_duration}秒")
        print(f"活动窗口: {self.window_duration}秒")
        print("按 Ctrl+C 停止检测\n")
        
        # 初始化可视化
        if self.debug_mode:
            self._init_waveform_plot()
        
        # 启动音频流
        if not self._start_audio_stream():
            print("启动音频流失败")
            return
        
        window_count = 0
        
        try:
            # 等待缓冲区填充
            print("正在填充音频缓冲区...")
            while len(self.audio_buffer) < self.window_samples and self.is_running:
                time.sleep(0.5)
            
            print("开始PANNs检测...\n")
            
            while self.is_running:
                window_count += 1
                
                # 获取活动窗口音频数据
                audio_data = self._get_audio_window()
                
                if audio_data is not None:
                    # PANNs检测
                    detected, confidence, similarities = self._detect_in_audio(audio_data)
                    
                    # 显示检测状态
                    if self.debug_mode and window_count % 3 == 0:
                        audio_level = similarities.get('audio_level', 0.0)
                        
                        # 更新活动状态文本
                        if hasattr(self, 'activity_text'):
                            if audio_level > 0.01:
                                self.activity_text.set_text('Audio: Active')
                                self.activity_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
                            else:
                                self.activity_text.set_text('Audio: Idle')
                                self.activity_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        # 更新检测状态文本
                        current_time = time.time()
                        
                        # 检查冷却期是否结束
                        cooldown_expired = (current_time - self.last_detection_time) > self.detection_cooldown
                        
                        if detected:
                            # 只有在冷却期结束后才报告新的检测
                            if not self.is_currently_detecting or cooldown_expired:
                                # 报告新的检测
                                self.detection_count += 1
                                detection_msg = f'检测到目标音频! (#{self.detection_count})'
                                print(f"\r{detection_msg} - 置信度: {confidence:.3f}", end="", flush=True)
                                # 记录检测成功日志
                                self.logger.warning(f"检测成功 #{self.detection_count} - 置信度: {confidence:.3f}")
                                
                                # 调用回调函数
                                if self.detection_callback:
                                    self.detection_callback(confidence, similarities)
                                
                                # 重置检测状态和时间
                                self.is_currently_detecting = True
                                self.last_detection_time = current_time
                            
                            if hasattr(self, 'detection_text'):
                                self.detection_text.set_text(f'检测: {confidence:.3f}')
                                self.detection_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8))
                        else:
                            # 如果没有检测到音频，重置检测状态
                            if self.is_currently_detecting:
                                self.is_currently_detecting = False
                            
                            if hasattr(self, 'detection_text'):
                                self.detection_text.set_text('')
                        
                        # 控制台状态显示
                        if not detected:
                            status_msg = f"检测中... 置信度: {confidence:.3f} | 音频电平: {audio_level:.4f}"
                            print(f"\r{status_msg}", end="", flush=True)
                    
                    # 更新可视化
                    if self.debug_mode:
                        self._update_waveform_plot()
                
                # 控制检测频率
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n检测被用户中断")
        except Exception as e:
            self.logger.error(f"检测过程中发生错误: {e}")
            print(f"\n检测错误: {e}")
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        self._stop_audio_stream()
        
        if self.debug_mode and self.fig:
            plt.close(self.fig)
        
        print(f"\n检测已停止。总共检测到 {self.detection_count} 次目标音频。")
        self.logger.debug("PANNs音频检测已停止")


def list_audio_devices():
    """列出可用的音频设备"""
    print("\n可用的音频设备:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (输入通道: {device['max_input_channels']})")


def parse_input_device(input_arg: str) -> Optional[int]:
    """解析输入设备参数"""
    if input_arg is None:
        return None
    
    try:
        device_id = int(input_arg)
        devices = sd.query_devices()
        if 0 <= device_id < len(devices):
            return device_id
        else:
            print(f"错误: 设备ID {device_id} 超出范围 (0-{len(devices)-1})")
            return None
    except ValueError:
        # 按名称搜索设备
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if input_arg.lower() in device['name'].lower() and device['max_input_channels'] > 0:
                print(f"找到匹配设备: {i}: {device['name']}")
                return i
        print(f"错误: 未找到名称包含 '{input_arg}' 的音频设备")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PANNs音频检测器')
    parser.add_argument('target_audio', help='目标音频文件路径')
    parser.add_argument('--model', '-m', default='cnn14', choices=['cnn6', 'cnn10', 'cnn14'],
                       help='PANNs模型架构 (默认: cnn14)')
    parser.add_argument('--device', '-d', help='音频输入设备ID或名称')
    parser.add_argument('--threshold', '-t', type=float, default=0.1,
                       help='检测阈值 (默认: 0.1)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式和可视化')
    parser.add_argument('--list-devices', action='store_true', help='列出可用音频设备')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # 检查目标音频文件
    if not Path(args.target_audio).exists():
        print(f"错误: 目标音频文件不存在: {args.target_audio}")
        return
    
    # 解析输入设备
    input_device = parse_input_device(args.device)
    
    try:
        # 创建检测器
        detector = PANNsAudioDetector(
            target_audio_path=args.target_audio,
            model_arch=args.model,
            debug_mode=args.debug,
            input_device=input_device
        )
        
        # 设置检测阈值
        detector.detection_threshold = args.threshold
        
        # 开始检测
        detector.start_detection()
        
    except Exception as e:
        print(f"启动检测器失败: {e}")
        logging.error(f"启动检测器失败: {e}")


if __name__ == "__main__":
    main()