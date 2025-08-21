#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时音频检测系统
监听音频流，检测指定的声音模式
"""

import numpy as np
import pyaudio
import librosa
import threading
import time
from collections import deque
import queue
import warnings
from audio_features import AudioFeatureExtractor, AudioMatcher
warnings.filterwarnings('ignore')

class AudioDetector:
    """
    实时音频检测器
    """
    
    def __init__(self, target_audio_path, 
                 sample_rate=22050, 
                 chunk_size=1024,
                 buffer_duration=3.0,
                 detection_threshold=0.7,
                 min_detection_interval=2.0):
        """
        初始化音频检测器
        
        Args:
            target_audio_path: 目标音频文件路径
            sample_rate: 采样率
            chunk_size: 音频块大小
            buffer_duration: 缓冲区时长（秒）
            detection_threshold: 检测阈值
            min_detection_interval: 最小检测间隔（秒）
        """
        self.target_audio_path = target_audio_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_duration = buffer_duration
        self.detection_threshold = detection_threshold
        self.min_detection_interval = min_detection_interval
        
        # 音频处理组件
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        self.matcher = AudioMatcher(self.feature_extractor)
        
        # 缓冲区
        self.buffer_size = int(sample_rate * buffer_duration)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # 控制变量
        self.is_listening = False
        self.last_detection_time = 0
        
        # 线程安全队列
        self.audio_queue = queue.Queue()
        
        # PyAudio实例
        self.audio = None
        self.stream = None
        
        # 加载目标音频模板
        self.load_target_template()
        
        print(f"音频检测器初始化完成")
        print(f"采样率: {sample_rate} Hz")
        print(f"缓冲区时长: {buffer_duration} 秒")
        print(f"检测阈值: {detection_threshold}")
    
    def load_target_template(self):
        """
        加载目标音频模板
        """
        try:
            print(f"正在加载目标音频模板: {self.target_audio_path}")
            
            # 加载目标音频
            target_audio, sr = librosa.load(self.target_audio_path, sr=self.sample_rate)
            
            # 提取目标音频特征
            self.target_features = self.feature_extractor.extract_all_features(target_audio)
            
            # 标准化特征
            self.target_features = self.feature_extractor.normalize_features(self.target_features)
            
            # 存储原始音频用于互相关
            self.target_audio = target_audio
            
            print(f"目标音频模板加载完成，时长: {len(target_audio)/sr:.2f} 秒")
            
        except Exception as e:
            print(f"加载目标音频模板失败: {e}")
            raise
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        音频回调函数
        """
        if status:
            print(f"音频流状态: {status}")
        
        # 将音频数据放入队列
        self.audio_queue.put(in_data)
        
        return (None, pyaudio.paContinue)
    
    def process_audio_chunk(self, audio_data):
        """
        处理音频块
        
        Args:
            audio_data: 音频数据
        """
        try:
            # 转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # 添加到缓冲区
            self.audio_buffer.extend(audio_array)
            
            # 如果缓冲区满了，进行检测
            if len(self.audio_buffer) >= self.buffer_size:
                self.detect_pattern()
        
        except Exception as e:
            print(f"处理音频块时出错: {e}")
    
    def detect_pattern(self):
        """
        检测音频模式
        """
        try:
            current_time = time.time()
            
            # 检查最小检测间隔
            if current_time - self.last_detection_time < self.min_detection_interval:
                return
            
            # 获取当前缓冲区音频
            current_audio = np.array(list(self.audio_buffer))
            
            # 如果音频太短，跳过
            if len(current_audio) < len(self.target_audio):
                return
            
            # 使用滑动窗口进行检测
            target_length = len(self.target_audio)
            max_similarity = 0
            best_method = ""
            
            # 在缓冲区中滑动检测
            step_size = self.chunk_size
            for start_idx in range(0, len(current_audio) - target_length, step_size):
                end_idx = start_idx + target_length
                audio_segment = current_audio[start_idx:end_idx]
                
                # 多种检测方法
                similarities = self.compute_similarities(audio_segment)
                
                # 取最高相似度
                for method, similarity in similarities.items():
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_method = method
            
            # 检查是否超过阈值
            if max_similarity > self.detection_threshold:
                self.on_pattern_detected(max_similarity, best_method)
                self.last_detection_time = current_time
        
        except Exception as e:
            print(f"模式检测时出错: {e}")
    
    def compute_similarities(self, audio_segment):
        """
        计算多种相似度指标
        
        Args:
            audio_segment: 音频片段
            
        Returns:
            相似度字典
        """
        similarities = {}
        
        try:
            # 提取当前音频特征
            current_features = self.feature_extractor.extract_all_features(audio_segment)
            current_features = self.feature_extractor.normalize_features(current_features)
            
            # 1. MFCC余弦相似度
            if 'mfcc' in current_features and 'mfcc' in self.target_features:
                mfcc_similarity = self.matcher.cosine_similarity(
                    current_features['mfcc'], 
                    self.target_features['mfcc']
                )
                similarities['mfcc_cosine'] = max(0, mfcc_similarity)
            
            # 2. 互相关
            correlation, _ = self.matcher.cross_correlation(audio_segment, self.target_audio)
            similarities['cross_correlation'] = max(0, correlation)
            
            # 3. 频谱质心相似度
            if ('spectral_centroid' in current_features and 
                'spectral_centroid' in self.target_features):
                
                centroid_similarity = self.matcher.cosine_similarity(
                    current_features['spectral_centroid'].reshape(-1, 1),
                    self.target_features['spectral_centroid'].reshape(-1, 1)
                )
                similarities['spectral_centroid'] = max(0, centroid_similarity)
            
            # 4. 色度特征相似度
            if 'chroma' in current_features and 'chroma' in self.target_features:
                chroma_similarity = self.matcher.cosine_similarity(
                    current_features['chroma'],
                    self.target_features['chroma']
                )
                similarities['chroma'] = max(0, chroma_similarity)
            
            # 5. 梅尔频谱相似度
            if ('mel_spectrogram' in current_features and 
                'mel_spectrogram' in self.target_features):
                
                mel_similarity = self.matcher.cosine_similarity(
                    current_features['mel_spectrogram'],
                    self.target_features['mel_spectrogram']
                )
                similarities['mel_spectrogram'] = max(0, mel_similarity)
            
        except Exception as e:
            print(f"计算相似度时出错: {e}")
        
        return similarities
    
    def on_pattern_detected(self, similarity, method):
        """
        检测到模式时的回调函数
        
        Args:
            similarity: 相似度值
            method: 检测方法
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n🎵 [{timestamp}] 出现了指定声音!")
        print(f"   相似度: {similarity:.4f}")
        print(f"   检测方法: {method}")
        print(f"   阈值: {self.detection_threshold}")
        print("-" * 50)
    
    def start_listening(self):
        """
        开始监听音频
        """
        try:
            print("正在初始化音频设备...")
            
            # 初始化PyAudio
            self.audio = pyaudio.PyAudio()
            
            # 打开音频流
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            print("音频设备初始化完成")
            print(f"开始监听音频流... (按 Ctrl+C 停止)")
            print(f"检测阈值: {self.detection_threshold}")
            print("-" * 50)
            
            self.is_listening = True
            self.stream.start_stream()
            
            # 处理音频数据的线程
            processing_thread = threading.Thread(target=self.audio_processing_loop)
            processing_thread.daemon = True
            processing_thread.start()
            
            # 主循环
            try:
                while self.is_listening:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n收到停止信号...")
                self.stop_listening()
        
        except Exception as e:
            print(f"启动音频监听失败: {e}")
            self.stop_listening()
    
    def audio_processing_loop(self):
        """
        音频处理循环
        """
        while self.is_listening:
            try:
                # 从队列获取音频数据
                audio_data = self.audio_queue.get(timeout=1.0)
                self.process_audio_chunk(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"音频处理循环错误: {e}")
    
    def stop_listening(self):
        """
        停止监听音频
        """
        print("正在停止音频监听...")
        
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("音频监听已停止")
    
    def set_detection_threshold(self, threshold):
        """
        设置检测阈值
        
        Args:
            threshold: 新的检测阈值
        """
        self.detection_threshold = threshold
        print(f"检测阈值已更新为: {threshold}")
    
    def get_status(self):
        """
        获取检测器状态
        
        Returns:
            状态字典
        """
        return {
            'is_listening': self.is_listening,
            'sample_rate': self.sample_rate,
            'buffer_size': len(self.audio_buffer),
            'detection_threshold': self.detection_threshold,
            'last_detection_time': self.last_detection_time
        }

def main():
    """
    主函数
    """
    target_audio_path = "/Users/leslack/lsc/300_study/python/demo/voice4/target.wav"
    
    # 创建检测器
    detector = AudioDetector(
        target_audio_path=target_audio_path,
        sample_rate=22050,
        chunk_size=1024,
        buffer_duration=3.0,
        detection_threshold=0.6,  # 可以调整这个值来平衡检测率和误触率
        min_detection_interval=2.0
    )
    
    try:
        # 开始监听
        detector.start_listening()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        detector.stop_listening()

if __name__ == "__main__":
    main()