#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带实时波形显示的音频检测系统
实现实时音频流监听、波形显示和声音检测
"""

import numpy as np
import librosa
import sounddevice as sd
import threading
import time
import logging
from collections import deque
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
warnings.filterwarnings('ignore')

from audio_features import AudioMatcher, extract_audio_features
from config import get_config, validate_config

class WaveformAudioDetector:
    """带波形显示的音频检测器"""
    
    def __init__(self):
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
        
        # 音频缓冲
        buffer_frames = int(self.config['performance']['buffer_size'] * self.sample_rate / self.chunk_size)
        self.audio_buffer = deque(maxlen=buffer_frames)
        
        # 波形显示缓冲
        self.waveform_buffer_size = int(self.sample_rate * 2)  # 2秒的数据
        self.waveform_buffer = deque(maxlen=self.waveform_buffer_size)
        
        # 检测状态
        self.is_running = False
        self.detection_thread = None
        self.stream = None
        
        # GUI相关
        self.root = None
        self.fig = None
        self.ax = None
        self.line = None
        self.canvas = None
        self.status_label = None
        self.confidence_label = None
        
        # 加载目标音频模板
        self.matcher = None
        self.target_features = None
        self._load_target_audio()
        
        self.logger.info("带波形显示的音频检测器初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.config['log']
        
        # 创建logger
        self.logger = logging.getLogger('WaveformDetector')
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
    
    def _setup_gui(self):
        """设置GUI界面"""
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("实时音频检测系统 - 波形显示")
        self.root.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 状态信息框架
        status_frame = ttk.LabelFrame(main_frame, text="检测状态", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="状态: 未启动", font=('Arial', 12))
        self.status_label.pack(anchor=tk.W)
        
        self.confidence_label = ttk.Label(status_frame, text="置信度: 0.000", font=('Arial', 12))
        self.confidence_label.pack(anchor=tk.W)
        
        # 波形显示框架
        waveform_frame = ttk.LabelFrame(main_frame, text="实时音频波形", padding=10)
        waveform_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_title('实时音频波形')
        self.ax.set_xlabel('时间 (秒)')
        self.ax.set_ylabel('幅度')
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True, alpha=0.3)
        
        # 初始化波形线
        x_data = np.linspace(0, 2, self.waveform_buffer_size)
        y_data = np.zeros(self.waveform_buffer_size)
        self.line, = self.ax.plot(x_data, y_data, 'b-', linewidth=1)
        
        # 将matplotlib嵌入tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, waveform_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(button_frame, text="开始检测", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            self.logger.warning(f"音频流状态: {status}")
        
        # 转换为单声道
        if indata.shape[1] > 1:
            audio_chunk = np.mean(indata, axis=1)
        else:
            audio_chunk = indata[:, 0]
        
        # 添加到检测缓冲区
        self.audio_buffer.append(audio_chunk.copy())
        
        # 添加到波形显示缓冲区
        for sample in audio_chunk:
            self.waveform_buffer.append(sample)
    
    def _update_waveform(self):
        """更新波形显示"""
        if len(self.waveform_buffer) > 0:
            # 获取波形数据
            waveform_data = np.array(list(self.waveform_buffer))
            
            # 如果数据不足，用零填充
            if len(waveform_data) < self.waveform_buffer_size:
                padding = np.zeros(self.waveform_buffer_size - len(waveform_data))
                waveform_data = np.concatenate([padding, waveform_data])
            
            # 更新波形线
            self.line.set_ydata(waveform_data)
            
            # 更新画布
            self.canvas.draw_idle()
    
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
            
            # 更新GUI显示
            if self.root:
                self.root.after(0, self._update_gui_status, confidence)
            
            # 检测判断
            if confidence >= self.detection_config['min_confidence']:
                current_time = time.time()
                if current_time - self.last_detection_time >= self.min_detection_interval:
                    self._trigger_detection(confidence, similarities)
                    self.last_detection_time = current_time
        
        except Exception as e:
            self.logger.error(f"音频处理错误: {e}")
    
    def _update_gui_status(self, confidence):
        """更新GUI状态显示"""
        if self.confidence_label:
            self.confidence_label.config(text=f"置信度: {confidence:.3f}")
        
        if self.status_label:
            if confidence >= self.detection_config['min_confidence']:
                self.status_label.config(text="状态: 检测到目标声音!", foreground="red")
            else:
                self.status_label.config(text="状态: 正在监听...", foreground="green")
    
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
            # 显示可用音频设备
            print("\n可用音频设备:")
            print(sd.query_devices())
            
            # 设置默认设备
            sd.default.samplerate = self.sample_rate
            sd.default.channels = 1  # 输入单声道
            
            # 启动音频流
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                blocksize=self.chunk_size,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            
            self.stream.start()
            
            # 设置运行状态
            self.is_running = True
            
            # 启动检测线程
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            # 启动波形更新线程
            self.waveform_thread = threading.Thread(target=self._waveform_loop, daemon=True)
            self.waveform_thread.start()
            
            # 更新GUI按钮状态
            if self.start_button:
                self.start_button.config(state=tk.DISABLED)
            if self.stop_button:
                self.stop_button.config(state=tk.NORMAL)
            
            self.logger.info("音频检测已启动")
            print("\n=== 音频检测系统已启动 ===")
            print("正在监听音频流，检测指定声音...")
            
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
    
    def _waveform_loop(self):
        """波形更新循环"""
        while self.is_running:
            try:
                self._update_waveform()
                time.sleep(0.05)  # 50ms间隔，20fps
            except Exception as e:
                self.logger.error(f"波形更新错误: {e}")
                time.sleep(0.1)
    
    def stop_detection(self):
        """停止检测"""
        self.logger.info("正在停止音频检测...")
        
        # 设置停止标志
        self.is_running = False
        
        # 等待检测线程结束
        if hasattr(self, 'detection_thread') and self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        if hasattr(self, 'waveform_thread') and self.waveform_thread and self.waveform_thread.is_alive():
            self.waveform_thread.join(timeout=2.0)
        
        # 停止音频流
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.logger.error(f"停止音频流错误: {e}")
        
        # 更新GUI按钮状态
        if self.start_button:
            self.start_button.config(state=tk.NORMAL)
        if self.stop_button:
            self.stop_button.config(state=tk.DISABLED)
        
        self.logger.info("音频检测已停止")
        print("\n音频检测已停止")
    
    def on_closing(self):
        """窗口关闭事件"""
        self.stop_detection()
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """运行检测器"""
        # 设置GUI
        self._setup_gui()
        
        # 运行GUI主循环
        self.root.mainloop()

def main():
    """主函数"""
    try:
        # 创建检测器
        detector = WaveformAudioDetector()
        
        # 运行检测器
        detector.run()
        
    except KeyboardInterrupt:
        print("\n收到停止信号...")
    except Exception as e:
        print(f"\n检测系统错误: {e}")
        logging.error(f"系统错误: {e}")

if __name__ == "__main__":
    main()