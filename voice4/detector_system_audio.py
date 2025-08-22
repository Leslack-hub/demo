#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import librosa
from pathlib import Path
import logging
import sounddevice as sd
from collections import deque
import threading
import argparse
from audio_features import extract_audio_features
from config import get_config

# 添加matplotlib导入用于波形图可视化
import matplotlib
import matplotlib.pyplot as plt
# 设置字体避免乱码
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 设置字体大小和样式以提高可读性
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
# 设置更好的字体渲染以避免乱码
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
# 优化matplotlib性能
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.1
plt.rcParams['agg.path.chunksize'] = 10000
import matplotlib.animation as animation


class SystemAudioDetector:
    """使用buffer和活动窗口的音频检测器"""

    def __init__(self, target_audio_path, debug_mode=False, input_device=None, detection_callback=None):
        self.config = get_config()
        self.target_audio_path = target_audio_path
        self.is_running = False
        self.detection_count = 0
        self.input_device = input_device
        self.detection_callback = detection_callback  # 添加回调函数支持

        # 如果通过命令行参数启用了调试模式，则覆盖配置
        if debug_mode:
            self.config['debug']['enable_debug'] = True

        # 音频参数
        self.sample_rate = self.config['audio']['sample_rate']
        self.chunk_size = self.config['audio']['chunk_size']
        
        # 先初始化一个临时的音频缓冲区
        self.audio_buffer = deque(maxlen=1024)

        # 线程锁
        self.buffer_lock = threading.Lock()

        # 波形图相关属性
        self.fig = None
        self.ax = None
        self.line = None
        self.detection_text = None
        self.waveform_data = None

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SystemAudioDetector')

        # 加载目标音频
        self._load_target_audio()

        # 根据目标音频时长调整窗口大小，但至少1秒，最多3秒
        self.window_duration = max(min(self.target_duration * 1.5, 3), 1)
        # 活动窗口样本数
        self.window_samples = int(self.window_duration * self.sample_rate)

        # 根据目标音频时长计算缓冲区时长（目标音频时长的2倍）
        self.buffer_duration = self.target_duration * 2 + 1
        # 确保缓冲区时长至少为3秒，但不超过10秒
        self.buffer_duration = max(min(self.buffer_duration, 10), 3)
        # 计算buffer样本数
        self.buffer_samples = int(self.buffer_duration * self.sample_rate)

        # 更新音频缓冲区大小
        self.audio_buffer = deque(maxlen=self.buffer_samples)

        # 初始化波形数据缓冲区（用于可视化）
        self.waveform_buffer_size = int(0.3 * self.sample_rate)  # 0.3秒的波形数据，更适合实时显示
        self.waveform_data = deque(maxlen=self.waveform_buffer_size)
        
        # 初始化图形刷新时间戳
        self._last_flush = 0

    def _load_target_audio(self):
        """加载目标音频文件"""
        try:
            # 加载音频文件
            audio_data, sr = librosa.load(
                self.target_audio_path,
                sr=self.config['audio']['sample_rate']
            )

            self.target_duration = len(audio_data) / sr
            self.logger.info(f"加载目标音频: {self.target_audio_path}, 时长: {self.target_duration:.2f}秒")

            # 提取特征
            self.target_features = extract_audio_features(audio_data, sr, self.config['feature'])
            self.logger.info("目标音频特征提取完成")

        except Exception as e:
            self.logger.error(f"加载目标音频失败: {e}")
            raise

    def _audio_callback(self, indata, frames, time, status):
        """音频输入回调函数"""
        if status:
            self.logger.warning(f"音频输入状态: {status}")
            return

        try:
            # 将新的音频数据添加到缓冲区
            with self.buffer_lock:
                # 将二维数组转换为一维数组（单声道）
                audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
                
                # 检查音频数据是否有效
                if len(audio_chunk) > 0 and not np.all(audio_chunk == 0):
                    self.audio_buffer.extend(audio_chunk)
                    
                    # 在调试模式下，也更新波形数据缓冲区
                    if self.config['debug']['enable_debug']:
                        self.waveform_data.extend(audio_chunk)
                        # 限制波形数据长度以提高性能
                        if len(self.waveform_data) > 2000:
                            # 保留最新的数据以提高显示性能
                            while len(self.waveform_data) > 2000:
                                self.waveform_data.popleft()
                        # 由于使用了maxlen，deque会自动处理大小限制，无需手动截取
                    
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
            # 检查设备是否可用
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
                'latency': 'low'  # 降低延迟
            }
            
            # 如果指定了输入设备，添加设备参数
            if self.input_device is not None:
                stream_params['device'] = (self.input_device, None)  # (input, output)
                
            self.audio_stream = sd.InputStream(**stream_params)
            self.audio_stream.start()
            
            # 显示启动信息
            if self.input_device is not None:
                device_info = sd.query_devices()[self.input_device]
                self.logger.info(f"音频流已启动 - 使用设备: {device_info['name']}")
            else:
                self.logger.info("音频流已启动 - 使用默认设备")
            return True
        except Exception as e:
            self.logger.error(f"启动音频流失败: {e}")
            print(f"错误详情: {e}")
            print("建议检查:")
            print("1. 音频设备是否正常工作")
            print("2. 是否有其他程序占用音频设备")
            print("3. 系统音频权限是否已授予")
            return False

    def _stop_audio_stream(self):
        """停止音频流"""
        try:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.logger.info("音频流已停止")
        except Exception as e:
            self.logger.error(f"停止音频流失败: {e}")

    def _calculate_similarity(self, features1, features2):
        """计算两个特征向量的相似度"""
        try:
            # 获取权重配置
            weights = self.config['detection']['confidence_weight']
            
            # 处理MFCC相似度 - 使用统计特征而不是直接比较
            mfcc1_mean = np.mean(features1['mfcc'], axis=1)
            mfcc2_mean = np.mean(features2['mfcc'], axis=1)
            mfcc_sim = np.corrcoef(mfcc1_mean, mfcc2_mean)[0, 1]
            if np.isnan(mfcc_sim):
                mfcc_sim = 0.0

            # 检查MFCC相似度是否达到配置阈值，如果不达标直接忽略
            mfcc_threshold = self.config['detection']['mfcc_threshold']
            if mfcc_sim < mfcc_threshold:
                similarities = {
                    'mfcc': mfcc_sim,
                    'spectral': 0.0,
                    'chroma': 0.0,
                    'mel': 0.0
                }
                # 即使MFCC不达标，也返回实际的相似度值而不是0，以便调试
                confidence = (
                        weights['mfcc'] * max(0, mfcc_sim) +
                        weights['spectral'] * 0.0 +
                        weights['chroma'] * 0.0 +
                        weights['mel'] * 0.0
                )
                return confidence, similarities

            # 频谱质心相似度
            spectral1 = np.mean(features1['spectral_centroid'])
            spectral2 = np.mean(features2['spectral_centroid'])
            spectral_sim = 1.0 - abs(spectral1 - spectral2) / max(spectral1, spectral2, 1.0)

            # 色度特征相似度 - 使用统计特征
            chroma1_mean = np.mean(features1['chroma'], axis=1)
            chroma2_mean = np.mean(features2['chroma'], axis=1)
            chroma_sim = np.corrcoef(chroma1_mean, chroma2_mean)[0, 1]
            if np.isnan(chroma_sim):
                chroma_sim = 0.0

            # Mel频谱相似度 - 使用统计特征
            mel1_mean = np.mean(features1['mel_spectrogram'], axis=1)
            mel2_mean = np.mean(features2['mel_spectrogram'], axis=1)
            mel_sim = np.corrcoef(mel1_mean, mel2_mean)[0, 1]
            if np.isnan(mel_sim):
                mel_sim = 0.0

            # 计算加权平均相似度 - 只有所有特征都为正值时才计算
            weights = self.config['detection']['confidence_weight']

            # 计算加权平均相似度
            confidence = (
                    weights['mfcc'] * max(0, mfcc_sim) +
                    weights['spectral'] * max(0, spectral_sim) +
                    weights['chroma'] * max(0, chroma_sim) +
                    weights['mel'] * max(0, mel_sim)
            )
            
            # 检查是否主要特征达到基本要求（根据目标音频特征调整）
            # 如果关键特征太低，降低置信度
            if (mfcc_sim < 0.3 or spectral_sim < 0.1 or mel_sim < 0.3):
                confidence *= 0.5  # 降低置信度而不是直接设为0

            similarities = {
                'mfcc': mfcc_sim,
                'spectral': spectral_sim,
                'chroma': chroma_sim,
                'mel': mel_sim
            }

            return confidence, similarities

        except Exception as e:
            self.logger.error(f"相似度计算错误: {e}")
            return 0.0, {}

    def _detect_in_audio(self, audio_data):
        """在音频数据中检测目标声音"""
        if audio_data is None or len(audio_data) == 0:
            return False, 0.0, {}

        # 计算音频电平
        audio_level = np.sqrt(np.mean(audio_data ** 2))

        # 如果音频电平太低，跳过检测
        if audio_level < 0.001:
            return False, 0.0, {'audio_level': audio_level}

        try:
            # 提取当前音频特征
            current_features = extract_audio_features(audio_data, self.config['audio']['sample_rate'],
                                                      self.config['feature'])

            # 计算相似度
            confidence, similarities = self._calculate_similarity(self.target_features, current_features)

            # 判断是否检测到
            detected = confidence >= self.config['detection']['min_confidence']

            similarities['audio_level'] = audio_level

            return detected, confidence, similarities

        except Exception as e:
            self.logger.error(f"检测过程出错: {e}")
            return False, 0.0, {'error': str(e)}

    def _init_waveform_plot(self):
        """初始化波形图"""
        # 初始化图形变量
        self.fig = None
        self.ax = None
        self.line = None
        self.activity_text = None
        self.detection_text = None
        
        if self.config['debug']['enable_debug']:
            # 创建图形和轴
            self.fig, self.ax = plt.subplots(figsize=(12, 5))
            
            # 创建波形线
            self.line, = self.ax.plot([], [], 'b-', linewidth=1.0, alpha=0.8)
            
            # 设置坐标轴
            display_samples = min(3000, self.waveform_buffer_size)  # 限制显示的样本数
            self.ax.set_xlim(0, display_samples)
            self.ax.set_ylim(-0.1, 0.1)  # 初始范围较小
            self.ax.set_xlabel("Samples")
            self.ax.set_ylabel("Amplitude")
            self.ax.set_title("Real-time Audio Waveform")
            self.ax.grid(True, alpha=0.3)
            
            # 添加活动指示文本 (左上角)
            self.activity_text = self.ax.text(0.02, 0.95, 'Audio: Idle', transform=self.ax.transAxes, 
                                            fontsize=10, verticalalignment='top', horizontalalignment='left',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                            zorder=10, clip_on=False)
            # 添加检测指示文本 (左上角，稍低位置)
            self.detection_text = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes,
                                             fontsize=10, verticalalignment='top', horizontalalignment='left',
                                             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                                             zorder=10, clip_on=False)
            
            # 启用交互模式
            plt.ion()
            plt.show(block=False)
            
            # 初始化图形刷新时间戳
            self._last_flush = 0
            self._last_plot_update = 0
    
    def _clear_console_line(self):
        """清除控制台当前行以避免文本重叠"""
        if self.config['debug']['enable_debug']:
            # 输出足够多的空格来覆盖之前的文本
            print("\r" + " " * 120, end="\r", flush=True)
    
    def _update_waveform_plot(self):
        """更新波形图"""
        if self.config['debug']['enable_debug'] and self.line and self.fig:
            # 限制更新频率以提高性能 (提高到20 FPS)
            import time
            current_time = time.time()
            if hasattr(self, '_last_plot_update') and (current_time - self._last_plot_update) < 0.05:  # 20 FPS
                return
            self._last_plot_update = current_time
            
            with self.buffer_lock:
                if len(self.waveform_data) > 0:
                    # 获取最新的固定数量样本以提高性能
                    max_display_samples = min(3000, len(self.waveform_data))  # 减少样本数提高性能
                    if len(self.waveform_data) > max_display_samples:
                        # 取最新的样本
                        data_list = list(self.waveform_data)
                        data = np.array(data_list[-max_display_samples:])
                    else:
                        data = np.array(list(self.waveform_data))
                    
                    # 更新波形数据
                    x = np.arange(len(data))
                    self.line.set_data(x, data)
                    
                    # 动态调整Y轴范围以提高可视化效果
                    if len(data) > 0:
                        max_val = np.max(np.abs(data))
                        y_range = max(max_val * 1.2, 0.1)  # 设置最小范围
                        self.ax.set_ylim(-y_range, y_range)
                    
                    # 更新X轴范围
                    self.ax.set_xlim(0, len(data))
            
            # 注意：活动状态更新已移至检测循环中处理，避免重复更新
            
            # 刷新图形
            try:
                # 使用更高效的绘图方法
                self.fig.canvas.draw_idle()
                # 减少flush_events调用频率以提高性能
                if hasattr(self, '_last_flush') and (time.time() - self._last_flush) > 0.05:
                    self.fig.canvas.flush_events()
                    self._last_flush = time.time()
            except:
                pass  # 忽略绘图错误

    def start_detection(self):
        """开始检测"""
        self.is_running = True
        self.logger.info("音频检测已启动")

        print("\n=== 音频检测器已启动 ===")
        print("使用实时音频流和活动窗口检测指定声音...")
        
        # 显示当前使用的音频设备
        if self.input_device is not None:
            device_info = sd.query_devices()[self.input_device]
            print(f"输入设备: {device_info['name']} (设备ID: {self.input_device})")
        else:
            default_device = sd.query_devices()[sd.default.device[0]]
            print(f"输入设备: {default_device['name']} (默认设备)")
            
        print("请确保相关音频权限已开启")
        print(f"检测阈值: {self.config['detection']['min_confidence']}")
        print(f"缓冲区大小: {self.buffer_duration}秒")
        print(f"活动窗口: {self.window_duration}秒")
        print("按 Ctrl+C 停止检测\n")

        # 初始化波形图
        if self.config['debug']['enable_debug']:
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
                time.sleep(0.3)

            print("开始检测...\n")

            while self.is_running:
                window_count += 1

                # 获取活动窗口音频数据
                audio_data = self._get_audio_window()

                if audio_data is not None:
                    # 检测
                    detected, confidence, similarities = self._detect_in_audio(audio_data)

                    # 显示状态（仅在调试模式下）
                    if self.config['debug']['enable_debug']:
                        audio_level = similarities.get('audio_level', 0.0)
                        buffer_fill = len(self.audio_buffer) / self.buffer_samples * 100
                        # 使用完整的行宽并清除行尾以避免文本重叠
                        status_text = f"缓冲区: {buffer_fill:.1f}% | 音频电平: {audio_level:.4f} | 置信度: {confidence:.3f} | 窗口: {window_count}"
                        # 确保文本不会重叠，使用固定宽度并清除多余字符
                        print(f"\r{status_text:<120}", end="", flush=True)
                        
                        # 更新检测指示文本
                        if self.detection_text:
                            if detected:
                                self.detection_text.set_text(f'Hit! confidence level: {confidence:.3f}')
                                # 保持黄色背景不变
                            elif confidence > 0.15:
                                self.detection_text.set_text(f'progress...confidence level: {confidence:.3f}')
                                # 保持黄色背景不变
                            else:
                                self.detection_text.set_text('')
                        
                        # 更新活动状态文本
                        if self.activity_text:
                            audio_level = similarities.get('audio_level', 0.0)
                            if audio_level > 0.01:  # 活动阈值
                                self.activity_text.set_text("Audio: Active")
                                # 设置红色背景表示活动状态
                                self.activity_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                            else:
                                self.activity_text.set_text("Audio: Idle")
                                # 设置白色背景表示空闲状态
                                self.activity_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        # 更新波形图（降低更新频率以提高性能）
                        if window_count % 3 == 0:  # 每3个窗口更新一次波形图
                            self._update_waveform_plot()

                    # 处理检测结果（无论是否在调试模式下都应该处理）
                    if detected:
                        self.detection_count += 1
                        # 清除当前行以避免文本重叠
                        self._clear_console_line()
                        print(f"\n🎯 检测到指定声音! (第{self.detection_count}次)")
                        print(f"置信度: {confidence:.3f}")

                        if self.config['debug']['enable_debug']:
                            print(
                                f"详细相似度: {', '.join([f'{k}:{v:.3f}' for k, v in similarities.items() if k != 'audio_level'])}")

                        # 调用回调函数（如果已设置）
                        if self.detection_callback:
                            try:
                                self.detection_callback()
                            except Exception as e:
                                self.logger.error(f"回调函数执行错误: {e}")

                        # 清空缓冲区以避免重复触发
                        with self.buffer_lock:
                            self.audio_buffer.clear()

                        while len(self.audio_buffer) < self.window_samples and self.is_running:
                            time.sleep(0.5)

                        print("\n继续监听..")

                    # 如果启用调试模式且置信度较高，显示详细信息
                    elif self.config['debug']['enable_debug'] and confidence > 0.1:
                        # 清除当前行以避免文本重叠
                        self._clear_console_line()
                        print(f"\n调试信息 - 置信度: {confidence:.3f}")
                        print(
                            f"详细相似度: {', '.join([f'{k}:{v:.3f}' for k, v in similarities.items() if k != 'audio_level'])}")

                else:
                    # 使用完整的行宽并清除行尾以避免文本重叠
                    status_text = f"等待音频数据... 窗口: {window_count}"
                    # 确保文本不会重叠，使用固定宽度并清除多余字符
                    print(f"\r{status_text:<120}", end="", flush=True)

                time.sleep(0.3)  # 活动窗口更新间隔

        except KeyboardInterrupt:
            print("\n\n收到停止信号...")
        except Exception as e:
            self.logger.error(f"检测过程异常: {e}")
        finally:
            self.stop_detection()

    def stop_detection(self):
        """停止检测"""
        self.is_running = False

        # 停止音频流
        self._stop_audio_stream()

        # 清空缓冲区
        with self.buffer_lock:
            self.audio_buffer.clear()

        # 关闭图形窗口（如果存在）
        if self.config['debug']['enable_debug'] and self.fig:
            try:
                plt.close(self.fig)
            except:
                pass  # 忽略关闭错误

        self.logger.info("音频检测已停止")
        # 清除当前行以避免文本重叠
        self._clear_console_line()
        print("\n音频检测已停止")
        if self.detection_count > 0:
            print(f"总共检测到 {self.detection_count} 次指定声音")


def list_audio_devices():
    """列出所有可用的音频设备"""
    devices = sd.query_devices()
    print("\n可用的音频设备:")
    print("-" * 60)
    
    input_devices = []
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append(f"{device['max_input_channels']}输入")
            input_devices.append((i, device['name']))
        if device['max_output_channels'] > 0:
            device_type.append(f"{device['max_output_channels']}输出")
            
        marker = "*" if i == sd.default.device[0] else " "
        print(f"{marker}{i:2d}: {device['name']:<30} ({', '.join(device_type)}")
    
    print("-" * 60)
    print("* 表示默认设备")
    print("\n可用的输入设备 (可用于 --input 参数):")
    for device_id, name in input_devices:
        print(f"  {device_id}: {name}")
    print()
    
    return input_devices


def parse_input_device(input_arg):
    """解析输入设备参数"""
    if input_arg is None:
        return None
        
    # 尝试解析为数字
    try:
        device_id = int(input_arg)
        devices = sd.query_devices()
        if 0 <= device_id < len(devices):
            device = devices[device_id]
            if device['max_input_channels'] > 0:
                return device_id
            else:
                print(f"错误: 设备 {device_id} ({device['name']}) 不支持音频输入")
                return None
        else:
            print(f"错误: 设备 ID {device_id} 不存在")
            return None
    except ValueError:
        # 尝试按名称匹配
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0 and input_arg.lower() in device['name'].lower():
                return i
        print(f"错误: 未找到匹配的输入设备: {input_arg}")
        return None


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='音频检测器')
    parser.add_argument('target', nargs='?', type=str, help='目标音频文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--input', type=str, help='指定输入设备 (ID 或名称部分匹配)')
    parser.add_argument('--list-devices', action='store_true', help='列出所有可用的音频设备')
    args = parser.parse_args()
    
    # 如果只是要列出设备
    if args.list_devices:
        try:
            list_audio_devices()
            return
        except Exception as e:
            print(f"查询设备失败: {e}")
            return
    
    # 检查是否提供了目标文件
    if not args.target:
        parser.error("必须提供目标音频文件路径，或使用 --list-devices 查看设备列表")

    target_audio_path = args.target
    if not os.path.exists(target_audio_path):
        print(f"错误: 目标音频文件不存在: {target_audio_path}")
        return
    
    # 解析输入设备
    input_device = None
    if args.input:
        input_device = parse_input_device(args.input)
        if input_device is None:
            print("使用 --list-devices 查看可用设备")
            return
        else:
            device_info = sd.query_devices()[input_device]
            print(f"使用输入设备: {input_device} - {device_info['name']}")

    # 检查sounddevice是否可用
    try:
        # 检查可用的音频设备
        devices = sd.query_devices()
        if not devices:
            print("错误: 未找到可用的音频设备")
            return
            
        # 检查是否有可用的输入设备
        input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
        if not input_devices:
            print("错误: 未找到可用的音频输入设备")
            print("请检查麦克风是否正确连接")
            return
            
        # 如果没有指定输入设备，建议使用默认输入设备
        if input_device is None:
            default_input = sd.default.device[0]
            if default_input in input_devices:
                print(f"提示: 将使用默认输入设备 {default_input} - {devices[default_input]['name']}")
            else:
                print("警告: 默认设备不支持音频输入，建议使用 --input 参数指定输入设备")
                print("可用的输入设备:")
                for dev_id in input_devices:
                    print(f"  {dev_id}: {devices[dev_id]['name']}")
                    
    except ImportError:
        print("错误: 需要安装sounddevice库")
        print("请运行: pip install sounddevice")
        return
    except Exception as e:
        print(f"音频设备检查失败: {e}")
        return

    try:
        detector = SystemAudioDetector(target_audio_path, debug_mode=args.debug, input_device=input_device)
        detector.start_detection()
    except Exception as e:
        print(f"检测器启动失败: {e}")


if __name__ == "__main__":
    main()
