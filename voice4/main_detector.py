#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版音频检测系统主程序
实现实时音频流监听，检测指定声音模式
"""

import numpy as np
import librosa
import pyaudio
import threading
import time
import logging
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from audio_features import AudioMatcher, extract_audio_features
from config import get_config, validate_config

class OptimizedAudioDetector:
    """优化版音频检测器"""
    
    def __init__(self, config_path=None):
        """初始化检测器"""
        self.config = get_config()
        validate_config()
        
        # 设置日志
        self._setup_logging()
        
        # 音频参数
        self.sample_rate = self.config['audio']['sample_rate']
        self.chunk_size = self.config['audio']['chunk_size']
        self.channels = self.config['audio']['channels']
        
        # 检测参数
        self.detection_config = self.config['detection']
        self.last_detection_time = 0
        self.min_detection_interval = self.detection_config['min_detection_interval']
        
        # 音频流和缓冲
        self.audio = None
        self.stream = None
        self.audio_buffer = deque(maxlen=int(self.config['performance']['buffer_size'] * self.sample_rate / self.chunk_size))
        
        # 检测状态
        self.is_running = False
        self.detection_thread = None
        
        # 加载目标音频模板
        self.matcher = None
        self.target_features = None
        self._load_target_audio()
        
        self.logger.info("音频检测器初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.config['log']
        
        # 创建logger
        self.logger = logging.getLogger('AudioDetector')
        self.logger.setLevel(getattr(logging, log_config['level']))
        
        # 清除现有handlers
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(log_config['format'])
        
        # 控制台输出
        if log_config['enable_console_log']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件输出
        if log_config['enable_file_log']:
            file_handler = logging.FileHandler(self.config['file']['log_file'], encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _load_target_audio(self):
        """加载目标音频文件并提取特征"""
        target_path = self.config['file']['target_audio_path']
        
        try:
            # 加载音频文件
            audio_data, sr = librosa.load(target_path, sr=self.sample_rate)
            self.logger.info(f"加载目标音频: {target_path}, 时长: {len(audio_data)/sr:.2f}秒")
            
            # 提取特征
            self.target_features = extract_audio_features(audio_data, sr, self.config['feature'])
            
            # 创建匹配器
            self.matcher = AudioMatcher()
            
            self.logger.info("目标音频特征提取完成")
            
        except Exception as e:
            self.logger.error(f"加载目标音频失败: {e}")
            raise
    
    def _init_audio_stream(self):
        """初始化音频流"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # 获取默认输入设备信息
            device_info = self.audio.get_default_input_device_info()
            self.logger.info(f"使用音频设备: {device_info['name']}")
            
            # 创建音频流
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.logger.info("音频流初始化成功")
            
        except Exception as e:
            self.logger.error(f"音频流初始化失败: {e}")
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        if status:
            self.logger.warning(f"音频流状态: {status}")
        
        # 转换音频数据
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # 添加到缓冲区
        self.audio_buffer.append(audio_chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_buffer(self):
        """处理音频缓冲区中的数据"""
        if len(self.audio_buffer) < 2:
            return
        
        # 获取最近的音频数据
        recent_audio = np.concatenate(list(self.audio_buffer)[-10:])
        
        # 检查音频长度
        if len(recent_audio) < self.sample_rate * 0.5:  # 至少0.5秒
            return
        
        try:
            # 提取特征
            current_features = extract_audio_features(recent_audio, self.sample_rate, self.config['feature'])
            
            # 计算相似度
            similarities = self._calculate_similarities(current_features)
            
            # 计算综合置信度
            confidence = self._calculate_confidence(similarities)
            
            # 检测判断
            if confidence >= self.detection_config['min_confidence']:
                current_time = time.time()
                if current_time - self.last_detection_time >= self.min_detection_interval:
                    self._trigger_detection(confidence, similarities)
                    self.last_detection_time = current_time
            
            # 调试信息
            if self.config['debug']['verbose'] and hasattr(self, '_debug_counter'):
                self._debug_counter += 1
                if self._debug_counter % 50 == 0:  # 每50次处理输出一次
                    self.logger.debug(f"置信度: {confidence:.3f}, 相似度: {similarities}")
        
        except Exception as e:
            self.logger.error(f"音频处理错误: {e}")
    
    def _calculate_similarities(self, current_features):
        """计算各种相似度指标"""
        similarities = {}
        
        try:
            # MFCC相似度
            if 'mfcc' in current_features and 'mfcc' in self.target_features:
                similarities['mfcc'] = self.matcher.cosine_similarity(
                    current_features['mfcc'].flatten(),
                    self.target_features['mfcc'].flatten()
                )
            
            # 频谱质心相似度
            if 'spectral_centroid' in current_features and 'spectral_centroid' in self.target_features:
                similarities['spectral'] = self.matcher.cosine_similarity(
                    current_features['spectral_centroid'],
                    self.target_features['spectral_centroid']
                )
            
            # 色度特征相似度
            if 'chroma' in current_features and 'chroma' in self.target_features:
                similarities['chroma'] = self.matcher.cosine_similarity(
                    current_features['chroma'].flatten(),
                    self.target_features['chroma'].flatten()
                )
            
            # Mel频谱相似度
            if 'mel_spectrogram' in current_features and 'mel_spectrogram' in self.target_features:
                similarities['mel'] = self.matcher.cosine_similarity(
                    current_features['mel_spectrogram'].flatten(),
                    self.target_features['mel_spectrogram'].flatten()
                )
            
            # 互相关
            if 'mfcc' in current_features and 'mfcc' in self.target_features:
                similarities['correlation'] = self.matcher.cross_correlation(
                    current_features['mfcc'].flatten(),
                    self.target_features['mfcc'].flatten()
                )
        
        except Exception as e:
            self.logger.error(f"相似度计算错误: {e}")
            similarities = {}
        
        return similarities
    
    def _calculate_confidence(self, similarities):
        """计算综合置信度"""
        if not similarities:
            return 0.0
        
        weights = self.detection_config['confidence_weight']
        confidence = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in similarities:
                confidence += similarities[feature] * weight
                total_weight += weight
        
        if total_weight > 0:
            confidence /= total_weight
        
        return confidence
    
    def _trigger_detection(self, confidence, similarities):
        """触发检测事件"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 输出检测结果
        print(f"\n[{timestamp}] 出现了指定声音! (置信度: {confidence:.3f})")
        
        # 详细日志
        self.logger.info(f"检测到目标声音 - 置信度: {confidence:.3f}")
        self.logger.info(f"相似度详情: {similarities}")
        
        # 调试信息
        if self.config['debug']['enable_debug']:
            for feature, score in similarities.items():
                print(f"  {feature}: {score:.3f}")
    
    def start_detection(self):
        """开始检测"""
        if self.is_running:
            self.logger.warning("检测已在运行中")
            return
        
        try:
            # 初始化音频流
            self._init_audio_stream()
            
            # 启动音频流
            self.stream.start_stream()
            
            # 设置运行状态
            self.is_running = True
            self._debug_counter = 0
            
            # 启动检测线程
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.logger.info("音频检测已启动")
            print("\n=== 音频检测系统已启动 ===")
            print("正在监听音频流，检测指定声音...")
            print("按 Ctrl+C 停止检测\n")
            
        except Exception as e:
            self.logger.error(f"启动检测失败: {e}")
            self.stop_detection()
            raise
    
    def _detection_loop(self):
        """检测循环"""
        while self.is_running:
            try:
                self._process_audio_buffer()
                time.sleep(0.1)  # 100ms间隔
            except Exception as e:
                self.logger.error(f"检测循环错误: {e}")
                time.sleep(0.5)
    
    def stop_detection(self):
        """停止检测"""
        self.logger.info("正在停止音频检测...")
        
        # 设置停止标志
        self.is_running = False
        
        # 等待检测线程结束
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # 停止音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.logger.error(f"停止音频流错误: {e}")
        
        # 释放音频资源
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                self.logger.error(f"释放音频资源错误: {e}")
        
        self.logger.info("音频检测已停止")
        print("\n音频检测已停止")

def main():
    """主函数"""
    detector = None
    
    try:
        # 创建检测器
        detector = OptimizedAudioDetector()
        
        # 开始检测
        detector.start_detection()
        
        # 保持运行
        while detector.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n收到停止信号...")
    except Exception as e:
        print(f"\n检测系统错误: {e}")
        logging.error(f"系统错误: {e}")
    finally:
        if detector:
            detector.stop_detection()

if __name__ == "__main__":
    main()