#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import tempfile
import os
import time
import numpy as np
import librosa
from pathlib import Path
import logging
from audio_features import extract_audio_features
from config import get_config

class SystemAudioDetector:
    """使用系统音频录制的检测器"""
    
    def __init__(self, target_audio_path):
        self.config = get_config()
        self.target_audio_path = target_audio_path
        self.is_running = False
        self.detection_count = 0
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SystemAudioDetector')
        
        # 加载目标音频
        self._load_target_audio()
        
    def _load_target_audio(self):
        """加载目标音频文件"""
        try:
            # 加载音频文件
            audio_data, sr = librosa.load(
                self.target_audio_path, 
                sr=self.config['audio']['sample_rate']
            )
            
            duration = len(audio_data) / sr
            self.logger.info(f"加载目标音频: {self.target_audio_path}, 时长: {duration:.2f}秒")
            
            # 提取特征
            self.target_features = extract_audio_features(audio_data, sr, self.config['feature'])
            self.logger.info("目标音频特征提取完成")
            
        except Exception as e:
            self.logger.error(f"加载目标音频失败: {e}")
            raise
    
    def _record_audio_chunk(self, duration=2.0):
        """录制一段音频"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # 使用sox录制音频
            cmd = [
                'sox', '-d', '-r', str(self.config['audio']['sample_rate']),
                '-c', '1', '-b', '16', temp_path, 'trim', '0', str(duration)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 2)
            
            if result.returncode == 0:
                # 加载录制的音频
                audio_data, sr = librosa.load(temp_path, sr=self.config['audio']['sample_rate'])
                return audio_data
            else:
                self.logger.warning(f"录音失败: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.warning("录音超时")
            return None
        except Exception as e:
            self.logger.warning(f"录音异常: {e}")
            return None
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _calculate_similarity(self, features1, features2):
        """计算两个特征向量的相似度"""
        try:
            # 处理MFCC相似度 - 使用统计特征而不是直接比较
            mfcc1_mean = np.mean(features1['mfcc'], axis=1)
            mfcc2_mean = np.mean(features2['mfcc'], axis=1)
            mfcc_sim = np.corrcoef(mfcc1_mean, mfcc2_mean)[0, 1]
            if np.isnan(mfcc_sim):
                mfcc_sim = 0.0
            
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
        audio_level = np.sqrt(np.mean(audio_data**2))
        
        # 如果音频电平太低，跳过检测
        if audio_level < 0.001:
            return False, 0.0, {'audio_level': audio_level}
        
        try:
            # 提取当前音频特征
            current_features = extract_audio_features(audio_data, self.config['audio']['sample_rate'], self.config['feature'])
            
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
        self.logger.info("系统音频检测已启动")
        
        print("\n=== 系统音频检测器已启动 ===")
        print("使用系统录音功能检测指定声音...")
        print("请确保麦克风权限已开启")
        print(f"检测阈值: {self.config['detection']['min_confidence']}")
        print("按 Ctrl+C 停止检测\n")
        
        chunk_count = 0
        
        try:
            while self.is_running:
                chunk_count += 1
                
                # 录制音频片段
                print(f"\r正在录制音频片段 {chunk_count}...", end="", flush=True)
                audio_data = self._record_audio_chunk(duration=1.5)
                
                if audio_data is not None:
                    # 检测
                    detected, confidence, similarities = self._detect_in_audio(audio_data)
                    
                    # 显示状态
                    audio_level = similarities.get('audio_level', 0.0)
                    print(f"\r音频电平: {audio_level:.4f} | 置信度: {confidence:.3f} | 片段: {chunk_count}", end="", flush=True)
                    
                    if confidence > 0.15:
                        print(f"\n⚡ 检测中... 置信度: {confidence:.3f}")
                    
                    if detected:
                        self.detection_count += 1
                        print(f"\n\n🎯 检测到指定声音! (第{self.detection_count}次)")
                        print(f"置信度: {confidence:.3f}")
                        
                        if self.config['debug']['enable_debug']:
                            print(f"详细相似度: {', '.join([f'{k}:{v:.3f}' for k, v in similarities.items() if k != 'audio_level'])}")
                        
                        print("继续监听...\n")
                    
                    # 如果启用调试模式且置信度较高，显示详细信息
                    elif self.config['debug']['enable_debug'] and confidence > 0.1:
                        print(f"\n调试信息 - 置信度: {confidence:.3f}")
                        print(f"详细相似度: {', '.join([f'{k}:{v:.3f}' for k, v in similarities.items() if k != 'audio_level'])}")
                
                else:
                    print(f"\r录音失败，重试中... 片段: {chunk_count}", end="", flush=True)
                
                time.sleep(0.1)  # 短暂暂停
                
        except KeyboardInterrupt:
            print("\n\n收到停止信号...")
        except Exception as e:
            self.logger.error(f"检测过程异常: {e}")
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        self.logger.info("系统音频检测已停止")
        print("\n系统音频检测已停止")
        if self.detection_count > 0:
            print(f"总共检测到 {self.detection_count} 次指定声音")

def main():
    """主函数"""
    target_audio_path = "/Users/leslack/lsc/300_study/python/demo/voice4/target.wav"
    
    if not os.path.exists(target_audio_path):
        print(f"错误: 目标音频文件不存在: {target_audio_path}")
        return
    
    # 检查sox是否可用
    try:
        subprocess.run(['sox', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 需要安装sox音频工具")
        print("请运行: brew install sox")
        return
    
    try:
        detector = SystemAudioDetector(target_audio_path)
        detector.start_detection()
    except Exception as e:
        print(f"检测器启动失败: {e}")

if __name__ == "__main__":
    main()