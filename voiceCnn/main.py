#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏声音识别系统 - 主程序入口

提供统一的命令行接口，支持训练、检测、数据处理等功能
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.train import main as train_main
from src.realtime_detector import main as detect_main
from src.utils import (
    setup_project_logging, 
    create_data_summary, 
    save_data_summary,
    create_project_report,
    batch_convert_audio,
    scan_audio_directory
)
from config.config import PROJECT_ROOT


def setup_data_command(args):
    """
    数据设置命令
    """
    logger = setup_project_logging('INFO')
    
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / 'data'
    
    if args.action == 'summary':
        # 创建数据摘要
        logger.info("创建数据摘要...")
        summary = create_data_summary(data_dir)
        
        output_path = PROJECT_ROOT / 'output' / 'data_summary.json'
        save_data_summary(summary, output_path)
        
        print(f"数据摘要已保存到: {output_path}")
        print(f"总文件数: {summary['total_files']}")
        print(f"总时长: {summary['total_duration']:.2f} 秒")
        
        for dir_name, info in summary['directories'].items():
            print(f"{dir_name}: {info['count']} 文件, {info['duration']:.2f} 秒")
    
    elif args.action == 'convert':
        # 批量转换音频格式
        if not args.input_dir or not args.output_dir:
            print("错误: 转换操作需要指定 --input-dir 和 --output-dir")
            return
        
        logger.info(f"批量转换音频: {args.input_dir} -> {args.output_dir}")
        batch_convert_audio(
            Path(args.input_dir),
            Path(args.output_dir),
            args.format,
            args.sample_rate
        )
        print("音频转换完成")
    
    elif args.action == 'validate':
        # 验证数据集
        logger.info("验证数据集...")
        
        issues = []
        
        for subdir in ['positive_samples', 'negative_samples']:
            subdir_path = data_dir / subdir
            if not subdir_path.exists():
                issues.append(f"缺少目录: {subdir}")
                continue
            
            audio_files = scan_audio_directory(subdir_path)
            if len(audio_files) == 0:
                issues.append(f"目录 {subdir} 中没有有效的音频文件")
            elif len(audio_files) < 10:
                issues.append(f"目录 {subdir} 中音频文件过少 ({len(audio_files)} 个)")
        
        if issues:
            print("数据集验证发现问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("数据集验证通过")


def train_command(args):
    """
    训练命令
    """
    # 构建训练参数
    train_args = argparse.Namespace(
        model_type=args.model_type,
        augment=args.augment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split
    )
    
    # 调用训练主函数
    train_main(train_args)


def detect_command(args):
    """
    检测命令
    """
    # 构建检测参数
    detect_args = argparse.Namespace(
        model_path=args.model_path,
        device_id=args.device_id,
        duration=args.duration,
        threshold=args.threshold,
        save_detections=args.save_detections
    )
    
    # 调用检测主函数
    detect_main(detect_args)


def report_command(args):
    """
    报告命令
    """
    logger = setup_project_logging('INFO')
    
    logger.info("生成项目报告...")
    
    output_path = PROJECT_ROOT / 'output' / 'project_report.json'
    create_project_report(PROJECT_ROOT, output_path)
    
    print(f"项目报告已保存到: {output_path}")
    
    # 显示简要信息
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print("\n项目概览:")
        print(f"  项目名称: {report['project_info']['name']}")
        print(f"  版本: {report['project_info']['version']}")
        print(f"  创建时间: {report['project_info']['created']}")
        
        if 'data_summary' in report and report['data_summary']:
            print(f"  数据文件: {report['data_summary']['total_files']} 个")
            print(f"  总时长: {report['data_summary']['total_duration']:.2f} 秒")
        
        if 'model_info' in report and report['model_info']:
            print(f"  可用模型: {report['model_info']['model_count']} 个")


def list_devices_command(args):
    """
    列出音频设备命令
    """
    try:
        import sounddevice as sd
        
        print("可用音频设备:")
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append('输入')
            if device['max_output_channels'] > 0:
                device_type.append('输出')
            
            print(f"  {i}: {device['name']} ({', '.join(device_type)})")
            print(f"      采样率: {device['default_samplerate']} Hz")
            print(f"      输入通道: {device['max_input_channels']}")
            print(f"      输出通道: {device['max_output_channels']}")
            print()
        
        # 显示默认设备
        try:
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            print(f"默认输入设备: {default_input}")
            print(f"默认输出设备: {default_output}")
        except:
            pass
            
    except ImportError:
        print("错误: 需要安装 sounddevice 库")
        print("运行: pip install sounddevice")
    except Exception as e:
        print(f"获取音频设备信息时出错: {e}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description='游戏声音识别系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练模型
  python main.py train --model-type cnn --epochs 50
  
  # 实时检测
  python main.py detect --model-path models/saved_models/best_model.h5
  
  # 数据处理
  python main.py data summary
  
  # 生成报告
  python main.py report
  
  # 列出音频设备
  python main.py devices
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 数据处理命令
    data_parser = subparsers.add_parser('data', help='数据处理操作')
    data_parser.add_argument('action', choices=['summary', 'convert', 'validate'],
                           help='数据操作类型')
    data_parser.add_argument('--data-dir', type=str,
                           help='数据目录路径')
    data_parser.add_argument('--input-dir', type=str,
                           help='输入目录（用于转换）')
    data_parser.add_argument('--output-dir', type=str,
                           help='输出目录（用于转换）')
    data_parser.add_argument('--format', type=str, default='wav',
                           help='目标音频格式')
    data_parser.add_argument('--sample-rate', type=int, default=22050,
                           help='目标采样率')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--model-type', type=str, default='cnn',
                            choices=['cnn', 'mobilenet', 'resnet'],
                            help='模型类型')
    train_parser.add_argument('--augment', action='store_true',
                            help='启用数据增强')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='训练轮数')
    train_parser.add_argument('--batch-size', type=int, default=32,
                            help='批次大小')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                            help='学习率')
    train_parser.add_argument('--validation-split', type=float, default=0.2,
                            help='验证集比例')
    
    # 检测命令
    detect_parser = subparsers.add_parser('detect', help='实时检测')
    detect_parser.add_argument('--model-path', type=str,
                             help='模型文件路径')
    detect_parser.add_argument('--device-id', type=int,
                             help='音频设备ID')
    detect_parser.add_argument('--duration', type=int, default=60,
                             help='检测时长（秒）')
    detect_parser.add_argument('--threshold', type=float, default=0.7,
                             help='置信度阈值')
    detect_parser.add_argument('--save-detections', action='store_true',
                             help='保存检测结果')
    
    # 报告命令
    report_parser = subparsers.add_parser('report', help='生成项目报告')
    
    # 设备列表命令
    devices_parser = subparsers.add_parser('devices', help='列出音频设备')
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行对应命令
    try:
        if args.command == 'data':
            setup_data_command(args)
        elif args.command == 'train':
            train_command(args)
        elif args.command == 'detect':
            detect_command(args)
        elif args.command == 'report':
            report_command(args)
        elif args.command == 'devices':
            list_devices_command(args)
        else:
            print(f"未知命令: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n操作被用户中断")
    except Exception as e:
        print(f"执行命令时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()