# -*- coding: utf-8 -*-
"""
深度学习模型架构
"""

import os
from typing import Tuple

import tensorflow as tf
from config.config import AUDIO_CONFIG, TRAINING_CONFIG, MODEL_CONFIG
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50


class AudioClassificationModel:
    """
    音频分类模型类
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = None,
                 num_classes: int = 2,
                 model_type: str = 'cnn'):
        """
        初始化模型
        
        Args:
            input_shape: 输入形状 (height, width, channels)
            num_classes: 分类数量
            model_type: 模型类型 ('cnn', 'mobilenet', 'resnet')
        """
        if input_shape is None:
            # 默认梅尔频谱图形状
            time_steps = int(AUDIO_CONFIG['sample_rate'] * AUDIO_CONFIG['duration'] / AUDIO_CONFIG['hop_length']) + 1
            input_shape = (AUDIO_CONFIG['n_mels'], time_steps, 1)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        
    def create_cnn_model(self) -> keras.Model:
        """
        创建自定义CNN模型
        
        Returns:
            Keras模型
        """
        model = models.Sequential([
            # 输入层
            layers.Input(shape=self.input_shape),
            
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二个卷积块
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三个卷积块
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第四个卷积块
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 全局平均池化
            layers.GlobalAveragePooling2D(),
            
            # 全连接层
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # 输出层
            layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def create_mobilenet_model(self) -> keras.Model:
        """
        创建基于MobileNetV2的迁移学习模型
        
        Returns:
            Keras模型
        """
        # 调整输入形状以适应MobileNetV2
        if self.input_shape[2] == 1:
            # 如果是单通道，需要转换为3通道
            inputs = layers.Input(shape=self.input_shape)
            x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        else:
            inputs = layers.Input(shape=self.input_shape)
            x = inputs
        
        # 调整尺寸以满足MobileNetV2的最小输入要求
        x = layers.Lambda(lambda img: tf.image.resize(img, (96, 96)))(x)
        
        # 加载预训练的MobileNetV2
        base_model = MobileNetV2(
            input_shape=(96, 96, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # 冻结基础模型的权重
        base_model.trainable = False
        
        # 添加自定义分类头
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model
    
    def create_resnet_model(self) -> keras.Model:
        """
        创建基于ResNet50的迁移学习模型
        
        Returns:
            Keras模型
        """
        # 调整输入形状以适应ResNet50
        if self.input_shape[2] == 1:
            inputs = layers.Input(shape=self.input_shape)
            x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        else:
            inputs = layers.Input(shape=self.input_shape)
            x = inputs
        
        # 调整尺寸以满足ResNet50的最小输入要求
        x = layers.Lambda(lambda img: tf.image.resize(img, (224, 224)))(x)
        
        # 加载预训练的ResNet50
        base_model = ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # 冻结基础模型的权重
        base_model.trainable = False
        
        # 添加自定义分类头
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model
    
    def build_model(self) -> keras.Model:
        """
        构建模型
        
        Returns:
            构建好的模型
        """
        if self.model_type == 'cnn':
            self.model = self.create_cnn_model()
        elif self.model_type == 'mobilenet':
            self.model = self.create_mobilenet_model()
        elif self.model_type == 'resnet':
            self.model = self.create_resnet_model()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return self.model
    
    def compile_model(self, 
                     learning_rate: float = TRAINING_CONFIG['learning_rate'],
                     metrics: list = None):
        """
        编译模型
        
        Args:
            learning_rate: 学习率
            metrics: 评估指标
        """
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用 build_model()")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall']
        
        # 选择损失函数
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        
        # 编译模型
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
    
    def get_callbacks(self, 
                     checkpoint_path: str = None,
                     early_stopping_patience: int = TRAINING_CONFIG['early_stopping_patience'],
                     reduce_lr_patience: int = TRAINING_CONFIG['reduce_lr_patience']) -> list:
        """
        获取训练回调函数
        
        Args:
            checkpoint_path: 模型检查点保存路径
            early_stopping_patience: 早停耐心值
            reduce_lr_patience: 学习率衰减耐心值
            
        Returns:
            回调函数列表
        """
        callback_list = []
        
        # 早停
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # 学习率衰减
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=TRAINING_CONFIG['reduce_lr_factor'],
            patience=reduce_lr_patience,
            min_lr=TRAINING_CONFIG['min_lr'],
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # 模型检查点
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            checkpoint = callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callback_list.append(checkpoint)
        
        # TensorBoard日志
        log_dir = MODEL_CONFIG['save_dir'] / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard = callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callback_list.append(tensorboard)
        
        return callback_list
    
    def fine_tune_model(self, unfreeze_layers: int = 50):
        """
        微调预训练模型
        
        Args:
            unfreeze_layers: 解冻的层数
        """
        if self.model_type in ['mobilenet', 'resnet']:
            # 解冻部分层进行微调
            base_model = self.model.layers[2] if self.model_type == 'mobilenet' else self.model.layers[2]
            base_model.trainable = True
            
            # 冻结前面的层
            for layer in base_model.layers[:-unfreeze_layers]:
                layer.trainable = False
            
            # 使用较小的学习率重新编译
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=TRAINING_CONFIG['learning_rate'] / 10),
                loss='binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未构建")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        self.model = keras.models.load_model(filepath)
        print(f"模型已从 {filepath} 加载")
    
    def convert_to_tflite(self, output_path: str, quantize: bool = True):
        """
        转换为TensorFlow Lite模型
        
        Args:
            output_path: 输出路径
            quantize: 是否量化
        """
        if self.model is None:
            raise ValueError("模型尚未构建")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite模型已保存到: {output_path}")
    
    def get_model_summary(self):
        """
        获取模型摘要
        """
        if self.model is None:
            raise ValueError("模型尚未构建")
        
        return self.model.summary()


def create_model(model_type: str = 'cnn', 
                input_shape: Tuple[int, int, int] = None,
                num_classes: int = 2) -> AudioClassificationModel:
    """
    创建音频分类模型的便捷函数
    
    Args:
        model_type: 模型类型
        input_shape: 输入形状
        num_classes: 分类数量
        
    Returns:
        音频分类模型实例
    """
    model_builder = AudioClassificationModel(
        input_shape=input_shape,
        num_classes=num_classes,
        model_type=model_type
    )
    
    model_builder.build_model()
    model_builder.compile_model()
    
    return model_builder