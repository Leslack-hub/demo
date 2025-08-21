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


class SystemAudioDetector:
    """使用buffer和活动窗口的音频检测器"""

    def __init__(self, target_audio_path, debug_mode=False, input_source=None):
        self.config = get_config()
        self.target_audio_path = target_audio_path
        self.is_running = False
        self.detection_count = 0
        self.input_source = input_source  # 添加input_source参数

        # 如果通过命令行参数启用了调试模式，则覆盖配置
        if debug_mode:
            self.config['debug']['enable_debug'] = True

        # 音频参数
        self.sample_rate = self.config['audio']['sample_rate']
        self.chunk_size = self.config['audio']['chunk_size']
        self.window_duration = 2

        # 活动窗口样本数
        self.window_samples = int(self.window_duration * self.sample_rate)

        # 先初始化一个临时的音频缓冲区
        self.audio_buffer = deque(maxlen=1024)

        # 线程锁
        self.buffer_lock = threading.Lock()

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SystemAudioDetector')

        # 加载目标音频
        self._load_target_audio()

        # 根据目标音频时长计算缓冲区时长（目标音频时长的2倍）
        self.buffer_duration = self.target_duration * 2 + 1
        # 确保缓冲区时长至少为5秒
        self.buffer_duration = max(self.buffer_duration, 3)
        # 计算buffer样本数
        self.buffer_samples = int(self.buffer_duration * self.sample_rate)

        # 更新音频缓冲区大小
        self.audio_buffer = deque(maxlen=self.buffer_samples)

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

        # 将新的音频数据添加到缓冲区
        with self.buffer_lock:
            # 将二维数组转换为一维数组（单声道）
            audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
            self.audio_buffer.extend(audio_chunk)

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
            # 查询可用的音频设备
            devices = sd.query_devices()
            
            # 根据input_source参数选择设备
            input_device_idx = None
            if self.input_source is not None:
                # 如果指定了输入源，直接使用指定的设备
                try:
                    input_device_idx = int(self.input_source)
                    if input_device_idx >= len(devices):
                        self.logger.warning(f"指定的音频设备不存在，使用默认设备")
                        input_device_idx = sd.default.device[0]
                except ValueError:
                    # 如果input_source不是数字，按名称查找设备
                    for i, device in enumerate(devices):
                        if self.input_source in device['name']:
                            input_device_idx = i
                            break
                    if input_device_idx is None:
                        self.logger.warning(f"未找到名为'{self.input_source}'的音频设备，使用默认设备")
                        input_device_idx = sd.default.device[0]
            else:
                # 默认行为：查找合适的输出设备（有输入通道的设备）
                for i, device in enumerate(devices):
                    # 优先选择 BlackHole（虚拟音频设备，常用于音频监听）
                    if 'BlackHole' in device['name'] and device['max_input_channels'] > 0:
                        input_device_idx = i
                        break
                
                # 如果没有找到 BlackHole，查找其他有输入通道的设备
                if input_device_idx is None:
                    for i, device in enumerate(devices):
                        if device['max_input_channels'] > 0:
                            input_device_idx = i
                            break
                
                # 如果仍然没有找到合适的设备，使用默认设备
                if input_device_idx is None:
                    self.logger.warning("未找到合适的音频输入设备，使用默认设备")
                    input_device_idx = sd.default.device[0]  # 使用默认输入设备
            
            self.logger.info(f"使用音频设备: {devices[input_device_idx]['name']} (ID: {input_device_idx})")
            
            # 创建音频流，使用找到的设备
            self.audio_stream = sd.InputStream(
                device=input_device_idx,
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            self.audio_stream.start()
            self.logger.info("音频流已启动")
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
                self.logger.info("音频流已停止")
        except Exception as e:
            self.logger.error(f"停止音频流失败: {e}")

    def _calculate_similarity(self, features1, features2):
        """计算两个特征向量的相似度"""
        try:
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
                return 0.0, similarities

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

            # 检查是否主要特征达到基本要求（根据目标音频特征调整）
            if (mfcc_sim < 0.5 or spectral_sim < 0.2 or mel_sim < 0.5):
                confidence = 0.0
            else:
                confidence = (
                        weights['mfcc'] * max(0, mfcc_sim) +
                        weights['spectral'] * max(0, spectral_sim) +
                        weights['chroma'] * max(0, chroma_sim) +
                        weights['mel'] * max(0, mel_sim)
                )

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

    def start_detection(self):
        """开始检测"""
        self.is_running = True
        self.logger.info("音频检测已启动")

        print("\n=== 音频检测器已启动 ===")
        if self.input_source is not None:
            if self.input_source == "1":
                print("监听麦克风输入...")
            elif self.input_source == "2":
                print("监听系统音频输出...")
            else:
                print(f"监听音频设备: {self.input_source}")
        else:
            print("监听系统音频输出...")
            print("请确保已安装并配置了音频监听设备（如BlackHole）")
        print(f"检测阈值: {self.config['detection']['min_confidence']}")
        print(f"缓冲区大小: {self.buffer_duration}秒")
        print(f"活动窗口: {self.window_duration}秒")
        print("按 Ctrl+C 停止检测\n")

        # 启动音频流
        if not self._start_audio_stream():
            print("启动音频流失败")
            return

        window_count = 0

        try:
            # 等待缓冲区填充
            print("正在填充音频缓冲区...")
            while len(self.audio_buffer) < self.window_samples and self.is_running:
                time.sleep(0.1)

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
                        print(
                            f"\r缓冲区: {buffer_fill:.1f}% | 音频电平: {audio_level:.4f} | 置信度: {confidence:.3f} | 窗口: {window_count}",
                            end="", flush=True)
                        if confidence > 0.15:
                            print(f"\n⚡ 检测中... 置信度: {confidence:.3f}")

                    if detected:
                        self.detection_count += 1
                        print(f"🎯 检测到指定声音! (第{self.detection_count}次)")
                        print(f"置信度: {confidence:.3f}")

                        if self.config['debug']['enable_debug']:
                            print(
                                f"详细相似度: {', '.join([f'{k}:{v:.3f}' for k, v in similarities.items() if k != 'audio_level'])}")

                        # 清空缓冲区以避免重复触发
                        with self.buffer_lock:
                            self.audio_buffer.clear()

                        while len(self.audio_buffer) < self.window_samples and self.is_running:
                            time.sleep(0.5)

                        print("\n继续监听..")

                    # 如果启用调试模式且置信度较高，显示详细信息
                    elif self.config['debug']['enable_debug'] and confidence > 0.1:
                        print(f"\n调试信息 - 置信度: {confidence:.3f}")
                        print(
                            f"详细相似度: {', '.join([f'{k}:{v:.3f}' for k, v in similarities.items() if k != 'audio_level'])}")

                else:
                    print(f"\r等待音频数据... 窗口: {window_count}", end="", flush=True)

                time.sleep(0.1)  # 活动窗口更新间隔

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

        self.logger.info("音频检测已停止")
        print("\n音频检测已停止")
        if self.detection_count > 0:
            print(f"总共检测到 {self.detection_count} 次指定声音")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='音频检测器')
    parser.add_argument('target', type=str, help='目标音频文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--input', type=str, help='音频输入源: 设备ID或设备名称关键字 (如: "麦克风", "BlackHole")')
    args = parser.parse_args()

    target_audio_path = args.target

    if not os.path.exists(target_audio_path):
        print(f"错误: 目标音频文件不存在: {target_audio_path}")
        return

    # 检查sounddevice是否可用
    try:
        import sounddevice as sd
        # 检查可用的音频设备
        devices = sd.query_devices()
        if not devices:
            print("错误: 未找到可用的音频设备")
            return
    except ImportError:
        print("错误: 需要安装sounddevice库")
        print("请运行: pip install sounddevice")
        return
    except Exception as e:
        print(f"音频设备检查失败: {e}")
        return

    try:
        detector = SystemAudioDetector(target_audio_path, debug_mode=args.debug, input_source=args.input)
        detector.start_detection()
    except Exception as e:
        print(f"检测器启动失败: {e}")


if __name__ == "__main__":
    main()
