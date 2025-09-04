#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本使用示例

展示如何使用声音识别系统的各个组件
"""

import sys
from pathlib import Path

import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.data_augmentation import AudioAugmentor
from src.model import create_model
from src.utils import setup_project_logging
from config.config import AUDIO_CONFIG, MODEL_CONFIG


def example_audio_processing():
    """
    示例：音频处理
    """
    print("=== 音频处理示例 ===")
    
    # 创建音频处理器
    processor = AudioProcessor()
    
    # 假设有一个音频文件
    audio_file = Path("data/positive_samples/example.wav")
    
    if audio_file.exists():
        print(f"处理音频文件: {audio_file}")
        
        # 加载和预处理音频
        features = processor.extract_features(audio_file)
        print(f"提取的特征形状: {features.shape}")
        
        # 提取不同类型的特征
        mel_spec = processor.extract_mel_spectrogram(audio_file)
        mfcc = processor.extract_mfcc(audio_file)
        spectral = processor.extract_spectral_features(audio_file)
        
        print(f"梅尔频谱图形状: {mel_spec.shape}")
        print(f"MFCC特征形状: {mfcc.shape}")
        print(f"频谱特征形状: {spectral.shape}")
    else:
        print(f"音频文件不存在: {audio_file}")
        print("请先准备训练数据")


def example_data_augmentation():
    """
    示例：数据增强
    """
    print("\n=== 数据增强示例 ===")
    
    # 创建数据增强器
    augmentor = AudioAugmentor()
    
    # 生成示例音频数据
    sample_rate = AUDIO_CONFIG['sample_rate']
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成正弦波作为示例
    frequency = 440  # A4音符
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    print(f"原始音频长度: {len(audio)} 样本")
    
    # 应用不同的增强技术
    augmented_audio = augmentor.time_stretch(audio, rate=1.2)
    print(f"时间拉伸后长度: {len(augmented_audio)} 样本")
    
    pitch_shifted = augmentor.pitch_shift(audio, n_steps=2)
    print(f"音调变换后长度: {len(pitch_shifted)} 样本")
    
    # 添加噪音
    noise_level = 0.01
    noisy_audio = augmentor.add_noise(audio, noise_level=noise_level)
    print(f"添加噪音后音频范围: [{np.min(noisy_audio):.3f}, {np.max(noisy_audio):.3f}]")
    
    # 音量调节
    volume_factor = 0.8
    volume_adjusted = augmentor.adjust_volume(audio, factor=volume_factor)
    print(f"音量调节后最大值: {np.max(np.abs(volume_adjusted)):.3f}")


def example_model_creation():
    """
    示例：模型创建
    """
    print("\n=== 模型创建示例 ===")
    
    # 定义输入形状（基于梅尔频谱图）
    input_shape = (
        AUDIO_CONFIG['n_mels'],
        int(AUDIO_CONFIG['sample_rate'] * AUDIO_CONFIG['duration'] / AUDIO_CONFIG['hop_length']) + 1,
        1
    )
    
    print(f"输入形状: {input_shape}")
    
    # 创建不同类型的模型
    models = ['cnn', 'mobilenet', 'resnet']
    
    for model_type in models:
        print(f"\n创建 {model_type.upper()} 模型...")
        
        try:
            model = create_model(model_type, input_shape=input_shape)
            
            # 显示模型信息
            total_params = model.count_params()
            trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            
            print(f"  总参数数量: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  模型层数: {len(model.layers)}")
            
            # 测试模型预测
            dummy_input = np.random.random((1,) + input_shape)
            prediction = model.predict(dummy_input, verbose=0)
            print(f"  预测输出形状: {prediction.shape}")
            print(f"  预测值: {prediction[0][0]:.4f}")
            
        except Exception as e:
            print(f"  创建 {model_type} 模型时出错: {e}")


def example_feature_extraction_pipeline():
    """
    示例：完整的特征提取流水线
    """
    print("\n=== 特征提取流水线示例 ===")
    
    # 创建处理器
    processor = AudioProcessor()
    
    # 模拟批量处理
    data_dir = Path("data/positive_samples")
    
    if data_dir.exists():
        audio_files = list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))
        
        if audio_files:
            print(f"找到 {len(audio_files)} 个音频文件")
            
            # 处理前几个文件作为示例
            sample_files = audio_files[:min(3, len(audio_files))]
            
            features_list = []
            labels_list = []
            
            for audio_file in sample_files:
                try:
                    print(f"处理: {audio_file.name}")
                    
                    # 提取特征
                    features = processor.extract_features(audio_file)
                    features_list.append(features)
                    labels_list.append(1)  # 正样本标签
                    
                    print(f"  特征形状: {features.shape}")
                    print(f"  特征范围: [{np.min(features):.3f}, {np.max(features):.3f}]")
                    
                except Exception as e:
                    print(f"  处理文件 {audio_file.name} 时出错: {e}")
            
            if features_list:
                # 合并特征
                all_features = np.array(features_list)
                all_labels = np.array(labels_list)
                
                print(f"\n批量特征形状: {all_features.shape}")
                print(f"标签形状: {all_labels.shape}")
                
                # 计算统计信息
                print(f"特征均值: {np.mean(all_features):.4f}")
                print(f"特征标准差: {np.std(all_features):.4f}")
        else:
            print("未找到音频文件")
    else:
        print(f"数据目录不存在: {data_dir}")


def example_configuration():
    """
    示例：配置使用
    """
    print("\n=== 配置示例 ===")
    
    print("音频配置:")
    for key, value in AUDIO_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n模型配置:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    # 展示如何修改配置
    print("\n修改配置示例:")
    print("# 临时修改采样率")
    print("AUDIO_CONFIG['sample_rate'] = 16000")
    print("# 临时修改批次大小")
    print("MODEL_CONFIG['batch_size'] = 64")


def example_logging():
    """
    示例：日志使用
    """
    print("\n=== 日志示例 ===")
    
    # 设置日志
    logger = setup_project_logging('INFO')
    
    # 使用日志
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.debug("这是一条调试日志（可能不会显示）")
    
    print("日志已配置完成")


def main():
    """
    主函数 - 运行所有示例
    """
    print("🎵 游戏声音识别系统 - 基本使用示例")
    print("=" * 50)
    
    examples = [
        example_configuration,
        example_logging,
        example_audio_processing,
        example_data_augmentation,
        example_model_creation,
        example_feature_extraction_pipeline
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"运行示例时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("✅ 所有示例运行完成")
    print("\n💡 提示:")
    print("- 准备好训练数据后，可以运行 python main.py train")
    print("- 查看更多选项: python main.py --help")
    print("- 快速检查环境: python quick_start.py")


if __name__ == '__main__':
    main()