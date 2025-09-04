# 游戏声音识别系统

基于深度学习的游戏角色技能声音识别系统，能够实时检测和识别游戏中的特定声音事件。

## 项目概述

本项目实现了一个完整的声音识别系统，包括：
- 音频数据预处理和特征提取
- 数据增强技术
- 深度学习模型训练（CNN、MobileNet、ResNet）
- 实时音频捕获和识别
- 模型评估和可视化工具

## 项目结构

```
voice6/
├── config/
│   └── config.py              # 项目配置文件
├── data/
│   ├── positive_samples/      # 正样本（目标声音）
│   ├── negative_samples/      # 负样本（非目标声音）
│   ├── background_noise/      # 背景噪音
│   └── processed/             # 处理后的数据
├── models/
│   ├── saved_models/          # 保存的模型
│   ├── checkpoints/           # 训练检查点
│   └── tflite/                # TensorFlow Lite模型
├── src/
│   ├── audio_processor.py     # 音频处理模块
│   ├── data_augmentation.py   # 数据增强模块
│   ├── model.py               # 模型定义
│   ├── train.py               # 训练脚本
│   ├── realtime_detector.py   # 实时检测器
│   └── utils.py               # 工具函数
├── logs/                      # 日志文件
├── output/                    # 输出结果
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明
```

## 环境要求

- Python 3.8+
- TensorFlow 2.x
- 其他依赖见 `requirements.txt`

## 安装步骤

1. 克隆项目到本地
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

## 数据准备

### 1. 训练数据组织

将训练素材按以下结构组织：

```
data/
├── positive_samples/          # 正样本文件夹
│   ├── skill_sound_1.wav
│   ├── skill_sound_2.wav
│   └── ...
├── negative_samples/          # 负样本文件夹
│   ├── other_sound_1.wav
│   ├── other_sound_2.wav
│   └── ...
└── background_noise/          # 背景噪音文件夹
    ├── ambient_1.wav
    ├── ambient_2.wav
    └── ...
```

### 2. 数据要求

- **音频格式**：支持 WAV、MP3、FLAC、M4A、OGG
- **采样率**：建议 22050 Hz（系统会自动重采样）
- **时长**：建议每个样本 1-5 秒
- **数量**：
  - 正样本：至少 100 个
  - 负样本：至少 200 个
  - 背景噪音：至少 20 个

### 3. 数据质量建议

- **正样本**：包含目标技能声音的清晰录音
- **负样本**：包含其他游戏声音、环境音等
- **背景噪音**：游戏环境音、白噪音等
- **音量平衡**：各类样本音量应相对均衡

## 使用方法

### 1. 模型训练

```bash
# 基本训练（使用CNN模型）
python src/train.py

# 使用MobileNet模型
python src/train.py --model_type mobilenet

# 使用ResNet模型
python src/train.py --model_type resnet

# 启用数据增强
python src/train.py --augment

# 自定义训练轮数
python src/train.py --epochs 100
```

### 2. 实时检测

```bash
# 启动实时检测
python src/realtime_detector.py

# 指定模型文件
python src/realtime_detector.py --model_path models/saved_models/best_model.h5

# 使用TensorFlow Lite模型
python src/realtime_detector.py --model_path models/tflite/model.tflite

# 设置检测时长（秒）
python src/realtime_detector.py --duration 60
```

### 3. 音频设备配置

```python
# 列出可用音频设备
import sounddevice as sd
print(sd.query_devices())

# 在realtime_detector.py中设置设备ID
detector = RealTimeAudioDetector(
    model_path='models/saved_models/best_model.h5',
    device_id=1  # 使用设备ID 1
)
```

## 配置说明

主要配置在 `config/config.py` 中：

### 音频配置
- `sample_rate`: 采样率（默认22050）
- `duration`: 音频片段时长（默认2秒）
- `n_mels`: 梅尔频谱维度（默认128）
- `n_fft`: FFT窗口大小（默认2048）

### 模型配置
- `batch_size`: 批次大小（默认32）
- `epochs`: 训练轮数（默认50）
- `learning_rate`: 学习率（默认0.001）

### 检测配置
- `confidence_threshold`: 置信度阈值（默认0.7）
- `buffer_size`: 音频缓冲区大小（默认4096）

## 模型类型

### 1. CNN模型
- 轻量级卷积神经网络
- 适合快速训练和部署
- 推荐用于初始实验

### 2. MobileNet模型
- 基于MobileNetV2的迁移学习
- 平衡了性能和效率
- 适合移动端部署

### 3. ResNet模型
- 基于ResNet50的迁移学习
- 更高的识别精度
- 需要更多计算资源

## 性能优化

### 1. 数据增强
- 时间拉伸：改变音频播放速度
- 音调变换：改变音频音调
- 噪音添加：增加背景噪音
- 音量调节：随机调整音量

### 2. 模型优化
- 使用TensorFlow Lite进行模型量化
- 支持GPU加速训练
- 实现模型剪枝和压缩

### 3. 实时检测优化
- 音频缓冲区管理
- 多线程处理
- 内存使用优化

## 评估指标

训练完成后，系统会生成以下评估报告：

- **准确率（Accuracy）**：整体分类准确率
- **精确率（Precision）**：正样本预测的准确性
- **召回率（Recall）**：正样本的检出率
- **F1分数**：精确率和召回率的调和平均
- **混淆矩阵**：详细的分类结果
- **ROC曲线**：模型性能曲线

## 输出文件

训练完成后，会在 `output/` 目录生成：

- `best_model.h5`: 最佳Keras模型
- `model.tflite`: TensorFlow Lite模型
- `training_history.json`: 训练历史
- `evaluation_report.json`: 评估报告
- `confusion_matrix.png`: 混淆矩阵图
- `training_curves.png`: 训练曲线图
- `roc_curve.png`: ROC曲线图

## 故障排除

### 常见问题

1. **音频设备问题**
   ```
   错误：No audio devices found
   解决：检查音频设备连接，运行 sd.query_devices() 查看可用设备
   ```

2. **内存不足**
   ```
   错误：OOM when allocating tensor
   解决：减少batch_size或使用更小的模型
   ```

3. **模型加载失败**
   ```
   错误：Unable to load model
   解决：检查模型文件路径和格式
   ```

4. **音频格式不支持**
   ```
   错误：Unsupported audio format
   解决：转换为WAV格式或安装额外的音频解码器
   ```

### 性能调优

1. **提高检测精度**
   - 增加训练数据量
   - 使用数据增强
   - 调整模型架构
   - 优化超参数

2. **降低延迟**
   - 使用TensorFlow Lite模型
   - 减少音频缓冲区大小
   - 优化特征提取

3. **减少误报**
   - 调整置信度阈值
   - 增加负样本多样性
   - 使用时间平滑

## 扩展功能

### 1. 多类别检测
修改模型输出层支持多个技能声音的同时检测。

### 2. 在线学习
实现模型的增量学习，支持新样本的在线更新。

### 3. 云端部署
将模型部署到云端，提供API接口服务。

### 4. 移动端应用
使用TensorFlow Lite在移动设备上部署模型。

## 技术支持

如遇到问题，请检查：
1. Python版本和依赖包是否正确安装
2. 音频设备是否正常工作
3. 训练数据是否按要求组织
4. 配置参数是否合理

## 许可证

本项目仅供学习和研究使用。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持CNN、MobileNet、ResNet模型
- 实现实时音频检测
- 提供完整的训练和评估流程