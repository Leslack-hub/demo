#!/usr/bin/env python3
import json
# import sys
# import argparse
from collections import defaultdict


def process_keybindings(input_file, output_file):
    commands = defaultdict(int)
    with open(input_file, 'r') as file:
        keybindings = json.load(file)  # 返回列表
    processed = []

    for item in keybindings:
        if not isinstance(item, dict):
            continue

        command = item.get('command', '')
        if command not in commands:
            processed.append(item)
            commands[command] = len(processed) - 1
        else:
            print(f"重复的command {command}")
            index = commands.get(command)
            processed[index] = item
    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=2)

    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == '__main__':
    process_keybindings("keyboard.json", "keyboard_processed.json")
