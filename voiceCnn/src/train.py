# -*- coding: utf-8 -*-
"""
模型训练脚本
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor, create_dataset_from_directory
from src.data_augmentation import AudioAugmentor, load_background_audios
from src.model import create_model
from config.config import (
    DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG,
    LOGGING_CONFIG
)


def setup_logging():
    """
    设置日志
    """
    log_dir = LOGGING_CONFIG['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_and_preprocess_data(logger):
    """
    加载和预处理数据
    
    Returns:
        处理后的特征和标签
    """
    logger.info("开始加载和预处理数据...")
    
    # 初始化音频处理器
    processor = AudioProcessor()
    
    # 加载正样本
    logger.info("加载正样本...")
    positive_dir = DATA_CONFIG['positive_samples']
    if not positive_dir.exists() or not any(positive_dir.iterdir()):
        raise ValueError(f"正样本目录为空: {positive_dir}")
    
    positive_features, positive_labels = create_dataset_from_directory(
        positive_dir, processor, 'mel_spectrogram'
    )
    logger.info(f"加载了 {len(positive_features)} 个正样本")
    
    # 加载负样本
    logger.info("加载负样本...")
    negative_dir = DATA_CONFIG['negative_samples']
    if not negative_dir.exists() or not any(negative_dir.iterdir()):
        raise ValueError(f"负样本目录为空: {negative_dir}")
    
    negative_features, negative_labels = create_dataset_from_directory(
        negative_dir, processor, 'mel_spectrogram'
    )
    logger.info(f"加载了 {len(negative_features)} 个负样本")
    
    # 合并数据
    all_features = np.concatenate([positive_features, negative_features], axis=0)
    all_labels = np.concatenate([positive_labels, negative_labels], axis=0)
    
    logger.info(f"总共加载了 {len(all_features)} 个样本")
    logger.info(f"正样本: {np.sum(all_labels == 1)}, 负样本: {np.sum(all_labels == 0)}")
    
    return all_features, all_labels, processor


def apply_data_augmentation(features, labels, processor, logger):
    """
    应用数据增强
    
    Args:
        features: 原始特征
        labels: 原始标签
        processor: 音频处理器
        logger: 日志器
        
    Returns:
        增强后的特征和标签
    """
    logger.info("开始数据增强...")
    
    # 初始化数据增强器
    augmentor = AudioAugmentor()
    
    # 加载背景音频
    background_dir = DATA_CONFIG['background_noise']
    background_audios = []
    if background_dir.exists() and any(background_dir.iterdir()):
        background_audios = load_background_audios(background_dir)
        logger.info(f"加载了 {len(background_audios)} 个背景音频")
    else:
        logger.warning("未找到背景音频，将跳过混音增强")
    
    # 分离正负样本
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    
    positive_features = features[positive_indices]
    negative_features = features[negative_indices]
    
    # 将梅尔频谱图转换回音频进行增强
    # 注意：这里简化处理，实际应用中可能需要更复杂的逆变换
    augmented_features = []
    augmented_labels = []
    
    # 添加原始样本
    for feature, label in zip(features, labels):
        augmented_features.append(feature)
        augmented_labels.append(label)
    
    # 对正样本进行更多增强
    augmentation_factor = 3
    for i, feature in enumerate(positive_features):
        for _ in range(augmentation_factor):
            # 简单的增强：添加噪声和随机变换
            augmented_feature = feature + np.random.normal(0, 0.01, feature.shape)
            augmented_features.append(augmented_feature)
            augmented_labels.append(1)
    
    # 对负样本进行适量增强
    for i, feature in enumerate(negative_features[:len(positive_features)]):
        augmented_feature = feature + np.random.normal(0, 0.005, feature.shape)
        augmented_features.append(augmented_feature)
        augmented_labels.append(0)
    
    augmented_features = np.array(augmented_features)
    augmented_labels = np.array(augmented_labels)
    
    logger.info(f"数据增强完成，总样本数: {len(augmented_features)}")
    logger.info(f"正样本: {np.sum(augmented_labels == 1)}, 负样本: {np.sum(augmented_labels == 0)}")
    
    return augmented_features, augmented_labels


def prepare_data_for_training(features, labels, logger):
    """
    准备训练数据
    
    Args:
        features: 特征数组
        labels: 标签数组
        logger: 日志器
        
    Returns:
        训练和验证数据
    """
    logger.info("准备训练数据...")
    
    # 添加通道维度
    if len(features.shape) == 3:
        features = np.expand_dims(features, axis=-1)
    
    # 归一化特征
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    # 分割数据集
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, 
        test_size=TRAINING_CONFIG['validation_split'],
        random_state=42,
        stratify=labels
    )
    
    logger.info(f"训练集大小: {len(X_train)}")
    logger.info(f"验证集大小: {len(X_val)}")
    logger.info(f"训练集正样本比例: {np.mean(y_train):.3f}")
    logger.info(f"验证集正样本比例: {np.mean(y_val):.3f}")
    
    return X_train, X_val, y_train, y_val


def train_model(X_train, X_val, y_train, y_val, model_type, logger):
    """
    训练模型
    
    Args:
        X_train, X_val, y_train, y_val: 训练和验证数据
        model_type: 模型类型
        logger: 日志器
        
    Returns:
        训练好的模型和训练历史
    """
    logger.info(f"开始训练 {model_type} 模型...")
    
    # 创建模型
    model_builder = create_model(
        model_type=model_type,
        input_shape=X_train.shape[1:],
        num_classes=2
    )
    
    logger.info("模型架构:")
    model_builder.get_model_summary()
    
    # 设置回调函数
    checkpoint_path = MODEL_CONFIG['checkpoint_dir'] / f"{model_type}_best.h5"
    callbacks = model_builder.get_callbacks(str(checkpoint_path))
    
    # 训练模型
    history = model_builder.model.fit(
        X_train, y_train,
        batch_size=TRAINING_CONFIG['batch_size'],
        epochs=TRAINING_CONFIG['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 如果是迁移学习模型，进行微调
    if model_type in ['mobilenet', 'resnet']:
        logger.info("开始微调模型...")
        model_builder.fine_tune_model()
        
        # 继续训练
        fine_tune_history = model_builder.model.fit(
            X_train, y_train,
            batch_size=TRAINING_CONFIG['batch_size'],
            epochs=TRAINING_CONFIG['epochs'] // 2,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 合并训练历史
        for key in history.history.keys():
            history.history[key].extend(fine_tune_history.history[key])
    
    return model_builder, history


def evaluate_model(model_builder, X_val, y_val, logger):
    """
    评估模型
    
    Args:
        model_builder: 模型构建器
        X_val, y_val: 验证数据
        logger: 日志器
    """
    logger.info("评估模型性能...")
    
    # 预测
    y_pred_proba = model_builder.model.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # 分类报告
    report = classification_report(y_val, y_pred, target_names=['负样本', '正样本'])
    logger.info(f"分类报告:\n{report}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['负样本', '正样本'],
                yticklabels=['负样本', '正样本'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存图片
    output_dir = MODEL_CONFIG['save_dir'] / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return report, cm


def plot_training_history(history, model_type):
    """
    绘制训练历史
    
    Args:
        history: 训练历史
        model_type: 模型类型
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失
    axes[0, 0].plot(history.history['loss'], label='训练损失')
    axes[0, 0].plot(history.history['val_loss'], label='验证损失')
    axes[0, 0].set_title('模型损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    
    # 准确率
    axes[0, 1].plot(history.history['accuracy'], label='训练准确率')
    axes[0, 1].plot(history.history['val_accuracy'], label='验证准确率')
    axes[0, 1].set_title('模型准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    
    # 精确率
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='训练精确率')
        axes[1, 0].plot(history.history['val_precision'], label='验证精确率')
        axes[1, 0].set_title('模型精确率')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('精确率')
        axes[1, 0].legend()
    
    # 召回率
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='训练召回率')
        axes[1, 1].plot(history.history['val_recall'], label='验证召回率')
        axes[1, 1].set_title('模型召回率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('召回率')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = MODEL_CONFIG['save_dir'] / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{model_type}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_training_results(model_builder, history, report, model_type, logger):
    """
    保存训练结果
    
    Args:
        model_builder: 模型构建器
        history: 训练历史
        report: 评估报告
        model_type: 模型类型
        logger: 日志器
    """
    logger.info("保存训练结果...")
    
    # 保存最佳模型
    best_model_path = MODEL_CONFIG['save_dir'] / f'{model_type}_best_model.h5'
    model_builder.save_model(str(best_model_path))
    
    # 转换为TensorFlow Lite
    tflite_path = MODEL_CONFIG['save_dir'] / f'{model_type}_model.tflite'
    model_builder.convert_to_tflite(str(tflite_path))
    
    # 保存训练历史
    history_path = MODEL_CONFIG['save_dir'] / f'{model_type}_training_history.json'
    with open(history_path, 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        json.dump(history_dict, f, indent=2)
    
    # 保存评估报告
    report_path = MODEL_CONFIG['save_dir'] / f'{model_type}_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"训练结果已保存到: {MODEL_CONFIG['save_dir']}")


def main():
    """
    主训练函数
    """
    parser = argparse.ArgumentParser(description='训练音频分类模型')
    parser.add_argument('--model_type', type=str, default='cnn',
                       choices=['cnn', 'mobilenet', 'resnet'],
                       help='模型类型')
    parser.add_argument('--augment', action='store_true',
                       help='是否使用数据增强')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info(f"开始训练 {args.model_type} 模型")
    
    try:
        # 加载数据
        features, labels, processor = load_and_preprocess_data(logger)
        
        # 数据增强
        if args.augment:
            features, labels = apply_data_augmentation(features, labels, processor, logger)
        
        # 准备训练数据
        X_train, X_val, y_train, y_val = prepare_data_for_training(features, labels, logger)
        
        # 更新训练配置
        if args.epochs:
            TRAINING_CONFIG['epochs'] = args.epochs
        
        # 训练模型
        model_builder, history = train_model(X_train, X_val, y_train, y_val, args.model_type, logger)
        
        # 评估模型
        report, cm = evaluate_model(model_builder, X_val, y_val, logger)
        
        # 绘制训练历史
        plot_training_history(history, args.model_type)
        
        # 保存结果
        save_training_results(model_builder, history, report, args.model_type, logger)
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main()