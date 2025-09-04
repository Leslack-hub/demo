#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始脚本

帮助用户快速验证项目设置和依赖
"""

import importlib
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def check_python_version():
    """
    检查Python版本
    """
    print("检查Python版本...")
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python版本过低，需要Python 3.8或更高版本")
        return False
    else:
        print("✅ Python版本符合要求")
        return True

def check_dependencies():
    """
    检查依赖包
    """
    print("\n检查依赖包...")
    
    required_packages = [
        'tensorflow',
        'librosa',
        'soundfile',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'sounddevice',
        'pydub',
        'tqdm',
        'pyyaml',
        'click'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有依赖包已安装")
        return True

def check_audio_devices():
    """
    检查音频设备
    """
    print("\n检查音频设备...")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if input_devices:
            print(f"✅ 找到 {len(input_devices)} 个音频输入设备")
            print("可用输入设备:")
            for i, device in enumerate(input_devices):
                print(f"  {i}: {device['name']}")
            return True
        else:
            print("❌ 未找到音频输入设备")
            return False
            
    except Exception as e:
        print(f"❌ 检查音频设备时出错: {e}")
        return False

def check_project_structure():
    """
    检查项目结构
    """
    print("\n检查项目结构...")
    
    project_root = Path(__file__).parent
    required_dirs = [
        'config',
        'data',
        'data/positive_samples',
        'data/negative_samples', 
        'data/background_noise',
        'data/processed',
        'models',
        'models/saved_models',
        'models/checkpoints',
        'models/tflite',
        'src',
        'logs',
        'output'
    ]
    
    required_files = [
        'config/config.py',
        'src/audio_processor.py',
        'src/data_augmentation.py',
        'src/model.py',
        'src/train.py',
        'src/realtime_detector.py',
        'src/utils.py',
        'requirements.txt',
        'README.md',
        'main.py'
    ]
    
    missing_items = []
    
    # 检查目录
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - 目录不存在")
            missing_items.append(dir_path)
    
    # 检查文件
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            missing_items.append(file_path)
    
    if missing_items:
        print(f"\n缺少以下项目: {', '.join(missing_items)}")
        return False
    else:
        print("\n✅ 项目结构完整")
        return True

def check_data_availability():
    """
    检查训练数据
    """
    print("\n检查训练数据...")
    
    project_root = Path(__file__).parent
    data_dir = project_root / 'data'
    
    data_status = {}
    
    for subdir in ['positive_samples', 'negative_samples', 'background_noise']:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            # 统计音频文件
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(list(subdir_path.glob(f'*{ext}')))
                audio_files.extend(list(subdir_path.glob(f'*{ext.upper()}')))
            
            count = len(audio_files)
            data_status[subdir] = count
            
            if count > 0:
                print(f"✅ {subdir}: {count} 个文件")
            else:
                print(f"⚠️  {subdir}: 无音频文件")
        else:
            print(f"❌ {subdir}: 目录不存在")
            data_status[subdir] = 0
    
    # 评估数据充足性
    if data_status.get('positive_samples', 0) >= 10 and data_status.get('negative_samples', 0) >= 10:
        print("\n✅ 训练数据基本充足")
        return True
    else:
        print("\n⚠️  训练数据不足，建议:")
        print("   - 正样本至少10个文件")
        print("   - 负样本至少10个文件")
        print("   - 背景噪音文件可选")
        return False

def test_basic_functionality():
    """
    测试基本功能
    """
    print("\n测试基本功能...")
    
    try:
        # 测试配置加载
        from config.config import AUDIO_CONFIG, MODEL_CONFIG
        print("✅ 配置文件加载成功")
        
        # 测试音频处理
        from src.audio_processor import AudioProcessor
        processor = AudioProcessor()
        print("✅ 音频处理器初始化成功")
        
        # 测试模型创建
        from src.model import create_model
        model = create_model('cnn', input_shape=(128, 87, 1))
        print("✅ 模型创建成功")
        
        # 测试数据增强
        from src.data_augmentation import AudioAugmentor
        augmentor = AudioAugmentor()
        print("✅ 数据增强器初始化成功")
        
        print("\n✅ 所有基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_next_steps():
    """
    显示后续步骤
    """
    print("\n" + "="*50)
    print("🎉 项目设置检查完成!")
    print("="*50)
    
    print("\n📋 后续步骤:")
    print("\n1. 准备训练数据:")
    print("   - 将正样本音频文件放入 data/positive_samples/")
    print("   - 将负样本音频文件放入 data/negative_samples/")
    print("   - 将背景噪音文件放入 data/background_noise/ (可选)")
    
    print("\n2. 验证数据:")
    print("   python main.py data validate")
    
    print("\n3. 开始训练:")
    print("   python main.py train --model-type cnn --epochs 50")
    
    print("\n4. 实时检测:")
    print("   python main.py detect --model-path models/saved_models/best_model.h5")
    
    print("\n5. 查看更多选项:")
    print("   python main.py --help")
    
    print("\n📖 详细说明请参考 README.md")

def main():
    """
    主函数
    """
    print("🚀 游戏声音识别系统 - 快速开始检查")
    print("="*50)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_audio_devices,
        check_project_structure,
        check_data_availability,
        test_basic_functionality
    ]
    
    results = []
    
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ 检查过程中出错: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "="*50)
    print("📊 检查结果总结")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total} 项检查")
    
    if passed == total:
        print("\n🎉 所有检查都通过了！项目已准备就绪。")
        show_next_steps()
    elif passed >= total * 0.8:
        print("\n⚠️  大部分检查通过，但仍有一些问题需要解决。")
        print("请根据上述错误信息进行修复。")
    else:
        print("\n❌ 多项检查失败，请先解决基础环境问题。")
        print("建议:")
        print("1. 确保Python版本 >= 3.8")
        print("2. 安装所需依赖: pip install -r requirements.txt")
        print("3. 检查音频设备连接")

if __name__ == '__main__':
    main()