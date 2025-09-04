# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config import AUDIO_CONFIG, MODEL_CONFIG, DATA_CONFIG


def setup_project_logging(log_level: str = 'INFO', 
                         log_file: Optional[str] = None) -> logging.Logger:
    """
    设置项目日志
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        
    Returns:
        日志器实例
    """
    logger = logging.getLogger('voice_detection')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_audio_file(file_path: Path) -> bool:
    """
    验证音频文件是否有效
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        是否有效
    """
    try:
        info = sf.info(file_path)
        return info.duration > 0 and info.samplerate > 0
    except Exception:
        return False


def scan_audio_directory(directory: Path, 
                        extensions: List[str] = None) -> List[Path]:
    """
    扫描目录中的音频文件
    
    Args:
        directory: 目录路径
        extensions: 支持的文件扩展名
        
    Returns:
        音频文件路径列表
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(directory.rglob(f'*{ext}'))
        audio_files.extend(directory.rglob(f'*{ext.upper()}'))
    
    # 验证文件
    valid_files = [f for f in audio_files if validate_audio_file(f)]
    
    return sorted(valid_files)


def create_data_summary(data_dir: Path) -> Dict[str, Any]:
    """
    创建数据集摘要
    
    Args:
        data_dir: 数据目录
        
    Returns:
        数据摘要字典
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'directories': {},
        'total_files': 0,
        'total_duration': 0.0
    }
    
    for subdir in ['positive_samples', 'negative_samples', 'background_noise']:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            audio_files = scan_audio_directory(subdir_path)
            
            total_duration = 0.0
            file_info = []
            
            for audio_file in audio_files:
                try:
                    info = sf.info(audio_file)
                    total_duration += info.duration
                    file_info.append({
                        'file': str(audio_file.relative_to(data_dir)),
                        'duration': info.duration,
                        'sample_rate': info.samplerate,
                        'channels': info.channels
                    })
                except Exception as e:
                    print(f"处理文件 {audio_file} 时出错: {e}")
            
            summary['directories'][subdir] = {
                'count': len(audio_files),
                'duration': total_duration,
                'files': file_info
            }
            
            summary['total_files'] += len(audio_files)
            summary['total_duration'] += total_duration
    
    return summary


def save_data_summary(summary: Dict[str, Any], output_path: Path):
    """
    保存数据摘要
    
    Args:
        summary: 数据摘要
        output_path: 输出路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def visualize_audio_waveform(audio_file: Path, 
                            output_path: Optional[Path] = None,
                            duration: Optional[float] = None):
    """
    可视化音频波形
    
    Args:
        audio_file: 音频文件路径
        output_path: 输出图片路径
        duration: 显示时长(秒)
    """
    # 加载音频
    audio, sr = librosa.load(audio_file, sr=AUDIO_CONFIG['sample_rate'])
    
    if duration:
        max_samples = int(duration * sr)
        audio = audio[:max_samples]
    
    # 创建时间轴
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    
    # 绘制波形
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, audio)
    plt.title(f'音频波形 - {audio_file.name}')
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')
    plt.grid(True, alpha=0.3)
    
    # 绘制频谱图
    plt.subplot(2, 1, 2)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, 
        n_mels=AUDIO_CONFIG['n_mels'],
        n_fft=AUDIO_CONFIG['n_fft'],
        hop_length=AUDIO_CONFIG['hop_length']
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    librosa.display.specshow(
        mel_spec_db, sr=sr, 
        hop_length=AUDIO_CONFIG['hop_length'],
        x_axis='time', y_axis='mel'
    )
    plt.title('梅尔频谱图')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_model_performance(history: Dict[str, List[float]], 
                              output_dir: Path):
    """
    可视化模型性能
    
    Args:
        history: 训练历史
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置样式
    plt.style.use('seaborn-v0_8')
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['loss'], label='训练损失', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='验证损失', linewidth=2)
    axes[0, 0].set_title('模型损失', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(history['accuracy'], label='训练准确率', linewidth=2)
    axes[0, 1].plot(history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[0, 1].set_title('模型准确率', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 精确率曲线
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='训练精确率', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='验证精确率', linewidth=2)
        axes[1, 0].set_title('模型精确率', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('精确率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 召回率曲线
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='训练召回率', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='验证召回率', linewidth=2)
        axes[1, 1].set_title('模型召回率', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('召回率')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_confusion_matrix_plot(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                class_names: List[str],
                                output_path: Path):
    """
    创建混淆矩阵图
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        output_path: 输出路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    
    plt.title('混淆矩阵', fontsize=16, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    
    # 添加准确率信息
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.02, 0.02, f'总体准确率: {accuracy:.3f}', fontsize=10)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_roc_curve_plot(y_true: np.ndarray, 
                         y_scores: np.ndarray,
                         output_path: Path):
    """
    创建ROC曲线图
    
    Args:
        y_true: 真实标签
        y_scores: 预测分数
        output_path: 输出路径
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='随机分类器')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)', fontsize=12)
    plt.ylabel('真正率 (TPR)', fontsize=12)
    plt.title('ROC曲线', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def backup_model(model_path: Path, backup_dir: Path):
    """
    备份模型文件
    
    Args:
        model_path: 模型文件路径
        backup_dir: 备份目录
    """
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"{model_path.stem}_{timestamp}{model_path.suffix}"
    backup_path = backup_dir / backup_name
    
    shutil.copy2(model_path, backup_path)
    print(f"模型已备份到: {backup_path}")


def convert_audio_format(input_path: Path, 
                        output_path: Path,
                        target_sr: int = AUDIO_CONFIG['sample_rate'],
                        target_channels: int = 1):
    """
    转换音频格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        target_sr: 目标采样率
        target_channels: 目标声道数
    """
    # 加载音频
    audio, sr = librosa.load(input_path, sr=target_sr, mono=(target_channels == 1))
    
    # 保存音频
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, target_sr)


def batch_convert_audio(input_dir: Path, 
                       output_dir: Path,
                       target_format: str = 'wav',
                       target_sr: int = AUDIO_CONFIG['sample_rate']):
    """
    批量转换音频格式
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        target_format: 目标格式
        target_sr: 目标采样率
    """
    audio_files = scan_audio_directory(input_dir)
    
    for audio_file in audio_files:
        try:
            # 构建输出路径
            relative_path = audio_file.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(f'.{target_format}')
            
            # 转换音频
            convert_audio_format(audio_file, output_path, target_sr)
            
            print(f"已转换: {audio_file} -> {output_path}")
            
        except Exception as e:
            print(f"转换文件 {audio_file} 时出错: {e}")


def calculate_dataset_statistics(features: np.ndarray, 
                               labels: np.ndarray) -> Dict[str, Any]:
    """
    计算数据集统计信息
    
    Args:
        features: 特征数组
        labels: 标签数组
        
    Returns:
        统计信息字典
    """
    stats = {
        'total_samples': len(features),
        'positive_samples': int(np.sum(labels == 1)),
        'negative_samples': int(np.sum(labels == 0)),
        'feature_shape': features.shape,
        'feature_stats': {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features))
        },
        'class_distribution': {
            'positive_ratio': float(np.mean(labels == 1)),
            'negative_ratio': float(np.mean(labels == 0))
        }
    }
    
    return stats


def create_project_report(project_dir: Path, 
                         output_path: Path):
    """
    创建项目报告
    
    Args:
        project_dir: 项目目录
        output_path: 输出路径
    """
    report = {
        'project_info': {
            'name': 'Voice Detection System',
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            'description': '基于深度学习的游戏声音事件检测系统'
        },
        'directory_structure': {},
        'data_summary': {},
        'model_info': {},
        'configuration': {
            'audio_config': AUDIO_CONFIG,
            'model_config': dict(MODEL_CONFIG),
            'data_config': dict(DATA_CONFIG)
        }
    }
    
    # 扫描目录结构
    for item in project_dir.rglob('*'):
        if item.is_file():
            relative_path = str(item.relative_to(project_dir))
            report['directory_structure'][relative_path] = {
                'size': item.stat().st_size,
                'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
            }
    
    # 数据摘要
    data_dir = project_dir / 'data'
    if data_dir.exists():
        report['data_summary'] = create_data_summary(data_dir)
    
    # 模型信息
    models_dir = project_dir / 'models'
    if models_dir.exists():
        model_files = list(models_dir.glob('*.h5')) + list(models_dir.glob('*.tflite'))
        report['model_info'] = {
            'available_models': [str(f.name) for f in model_files],
            'model_count': len(model_files)
        }
    
    # 保存报告
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"项目报告已保存到: {output_path}")


if __name__ == '__main__':
    # 示例用法
    project_root = Path(__file__).parent.parent
    
    # 创建数据摘要
    data_summary = create_data_summary(project_root / 'data')
    save_data_summary(data_summary, project_root / 'output' / 'data_summary.json')
    
    # 创建项目报告
    create_project_report(project_root, project_root / 'output' / 'project_report.json')
    
    print("工具函数演示完成")