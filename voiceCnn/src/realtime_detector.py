# -*- coding: utf-8 -*-
"""
实时音频捕获和识别系统
"""

import json
import logging
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
import tensorflow as tf

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from config.config import AUDIO_CONFIG, DETECTION_CONFIG


class RealTimeAudioDetector:
    """
    实时音频检测器
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = DETECTION_CONFIG['confidence_threshold'],
                 window_size: float = AUDIO_CONFIG['window_size'],
                 overlap: float = AUDIO_CONFIG['overlap'],
                 device_index: Optional[int] = DETECTION_CONFIG['device_index']):
        """
        初始化实时音频检测器
        
        Args:
            model_path: 模型文件路径
            confidence_threshold: 置信度阈值
            window_size: 检测窗口大小(秒)
            overlap: 窗口重叠(秒)
            device_index: 音频设备索引
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.overlap = overlap
        self.device_index = device_index
        
        # 音频参数
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        self.channels = DETECTION_CONFIG['channels']
        self.buffer_size = DETECTION_CONFIG['buffer_size']
        
        # 计算窗口参数
        self.window_samples = int(self.window_size * self.sample_rate)
        self.overlap_samples = int(self.overlap * self.sample_rate)
        self.hop_samples = self.window_samples - self.overlap_samples
        
        # 初始化组件
        self.audio_processor = AudioProcessor()
        self.model = None
        self.is_tflite = False
        
        # 音频缓冲区
        self.audio_buffer = deque(maxlen=self.window_samples * 2)
        self.audio_queue = queue.Queue()
        
        # 控制变量
        self.is_running = False
        self.detection_thread = None
        self.audio_thread = None
        
        # 检测结果
        self.detection_history = deque(maxlen=100)
        self.last_detection_time = 0
        self.detection_callback = None
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'processing_times': deque(maxlen=100)
        }
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 加载模型
        self._load_model()
    
    def _setup_logger(self) -> logging.Logger:
        """
        设置日志器
        
        Returns:
            日志器实例
        """
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self):
        """
        加载模型
        """
        try:
            model_path = Path(self.model_path)
            
            if model_path.suffix == '.tflite':
                # 加载TensorFlow Lite模型
                self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
                self.interpreter.allocate_tensors()
                
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                self.is_tflite = True
                self.logger.info(f"已加载TensorFlow Lite模型: {model_path}")
                
            else:
                # 加载Keras模型
                self.model = tf.keras.models.load_model(model_path)
                self.is_tflite = False
                self.logger.info(f"已加载Keras模型: {model_path}")
                
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def _predict(self, features: np.ndarray) -> float:
        """
        使用模型进行预测
        
        Args:
            features: 输入特征
            
        Returns:
            预测概率
        """
        try:
            if self.is_tflite:
                # TensorFlow Lite推理
                self.interpreter.set_tensor(self.input_details[0]['index'], features)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]['index'])
                return float(output[0][0]) if output.shape[-1] == 1 else float(output[0][1])
            else:
                # Keras模型推理
                prediction = self.model.predict(features, verbose=0)
                return float(prediction[0][0]) if prediction.shape[-1] == 1 else float(prediction[0][1])
                
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            return 0.0
    
    def _audio_callback(self, indata, frames, time, status):
        """
        音频回调函数
        
        Args:
            indata: 输入音频数据
            frames: 帧数
            time: 时间信息
            status: 状态信息
        """
        if status:
            self.logger.warning(f"音频回调状态: {status}")
        
        # 将音频数据放入队列
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        self.audio_queue.put(audio_data.copy())
    
    def _audio_processing_thread(self):
        """
        音频处理线程
        """
        while self.is_running:
            try:
                # 从队列获取音频数据
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # 添加到缓冲区
                self.audio_buffer.extend(audio_chunk)
                
                # 检查是否有足够的数据进行检测
                if len(self.audio_buffer) >= self.window_samples:
                    # 提取窗口数据
                    window_data = np.array(list(self.audio_buffer)[-self.window_samples:])
                    
                    # 进行检测
                    self._detect_in_window(window_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"音频处理线程错误: {e}")
    
    def _detect_in_window(self, audio_window: np.ndarray):
        """
        在音频窗口中进行检测
        
        Args:
            audio_window: 音频窗口数据
        """
        start_time = time.time()
        
        try:
            # 预处理音频
            features = self.audio_processor.preprocess_for_realtime(audio_window)
            
            # 添加批次和通道维度
            features = np.expand_dims(features, axis=0)  # 批次维度
            features = np.expand_dims(features, axis=-1)  # 通道维度
            
            # 归一化
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            # 预测
            confidence = self._predict(features)
            
            # 记录处理时间
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
            # 检测结果
            current_time = time.time()
            detection_result = {
                'timestamp': current_time,
                'confidence': confidence,
                'detected': confidence > self.confidence_threshold,
                'processing_time': processing_time
            }
            
            self.detection_history.append(detection_result)
            
            # 如果检测到目标声音
            if detection_result['detected']:
                self.stats['total_detections'] += 1
                self.last_detection_time = current_time
                
                self.logger.info(
                    f"检测到目标声音! 置信度: {confidence:.3f}, "
                    f"处理时间: {processing_time*1000:.1f}ms"
                )
                
                # 调用回调函数
                if self.detection_callback:
                    try:
                        self.detection_callback(detection_result)
                    except Exception as e:
                        self.logger.error(f"回调函数执行失败: {e}")
            
        except Exception as e:
            self.logger.error(f"检测过程中出现错误: {e}")
    
    def set_detection_callback(self, callback: Callable):
        """
        设置检测回调函数
        
        Args:
            callback: 回调函数，接收检测结果字典作为参数
        """
        self.detection_callback = callback
    
    def start_detection(self):
        """
        开始实时检测
        """
        if self.is_running:
            self.logger.warning("检测已在运行中")
            return
        
        self.logger.info("开始实时音频检测...")
        self.is_running = True
        
        try:
            # 启动音频处理线程
            self.audio_thread = threading.Thread(target=self._audio_processing_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # 启动音频流
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                dtype=np.float32
            )
            
            self.stream.start()
            self.logger.info("音频检测已启动")
            
        except Exception as e:
            self.logger.error(f"启动检测失败: {e}")
            self.is_running = False
            raise
    
    def stop_detection(self):
        """
        停止实时检测
        """
        if not self.is_running:
            self.logger.warning("检测未在运行")
            return
        
        self.logger.info("停止实时音频检测...")
        self.is_running = False
        
        try:
            # 停止音频流
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
            # 等待线程结束
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2.0)
            
            self.logger.info("音频检测已停止")
            
        except Exception as e:
            self.logger.error(f"停止检测时出现错误: {e}")
    
    def get_statistics(self) -> dict:
        """
        获取检测统计信息
        
        Returns:
            统计信息字典
        """
        processing_times = list(self.stats['processing_times'])
        
        return {
            'total_detections': self.stats['total_detections'],
            'detection_rate': len([d for d in self.detection_history if d['detected']]) / max(len(self.detection_history), 1),
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'max_processing_time': np.max(processing_times) if processing_times else 0,
            'min_processing_time': np.min(processing_times) if processing_times else 0,
            'last_detection_time': self.last_detection_time,
            'buffer_size': len(self.audio_buffer),
            'queue_size': self.audio_queue.qsize()
        }
    
    def get_recent_detections(self, count: int = 10) -> list:
        """
        获取最近的检测结果
        
        Args:
            count: 返回的检测结果数量
            
        Returns:
            最近的检测结果列表
        """
        return list(self.detection_history)[-count:]
    
    def save_detection_log(self, filepath: str):
        """
        保存检测日志
        
        Args:
            filepath: 保存路径
        """
        log_data = {
            'detection_history': list(self.detection_history),
            'statistics': self.get_statistics(),
            'config': {
                'confidence_threshold': self.confidence_threshold,
                'window_size': self.window_size,
                'overlap': self.overlap,
                'sample_rate': self.sample_rate
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"检测日志已保存到: {filepath}")
    
    def __enter__(self):
        """
        上下文管理器入口
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口
        """
        self.stop_detection()


def list_audio_devices():
    """
    列出可用的音频设备
    """
    print("可用的音频设备:")
    print(sd.query_devices())


def main():
    """
    主函数 - 演示实时检测
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='实时音频检测')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--threshold', type=float, default=0.95, help='置信度阈值')
    parser.add_argument('--device', type=int, default=None, help='音频设备索引')
    parser.add_argument('--list_devices', action='store_true', help='列出音频设备')
    parser.add_argument('--duration', type=int, default=60, help='检测持续时间(秒)')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # 检测回调函数
    def detection_callback(result):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"检测到目标声音! 置信度: {result['confidence']:.3f}")
    
    try:
        # 创建检测器
        detector = RealTimeAudioDetector(
            model_path=args.model,
            confidence_threshold=args.threshold,
            device_index=args.device
        )
        
        # 设置回调
        detector.set_detection_callback(detection_callback)
        
        # 开始检测
        detector.start_detection()
        
        print(f"开始检测，持续 {args.duration} 秒...")
        print("按 Ctrl+C 停止检测")
        
        # 运行指定时间
        start_time = time.time()
        try:
            while time.time() - start_time < args.duration:
                time.sleep(1)
                
                # 每10秒打印统计信息
                if int(time.time() - start_time) % 10 == 0:
                    stats = detector.get_statistics()
                    print(f"统计: 总检测次数={stats['total_detections']}, "
                          f"平均处理时间={stats['avg_processing_time']*1000:.1f}ms")
        
        except KeyboardInterrupt:
            print("\n用户中断检测")
        
        # 停止检测
        detector.stop_detection()
        
        # 保存日志
        log_path = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        detector.save_detection_log(log_path)
        
        # 打印最终统计
        final_stats = detector.get_statistics()
        print("\n最终统计:")
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main()