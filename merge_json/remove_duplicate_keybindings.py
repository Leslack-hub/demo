#!/usr/bin/env python3
from typing import List, Dict, Any
import json
from collections import defaultdict


def process_keybindings(input_file: str, output_file: str) -> None:
    commands: defaultdict[str, int] = defaultdict(int)
    with open(input_file, 'r') as file:
        keybindings: List[Dict[str, Any]] = json.load(file)
    processed: List[Dict[str, Any]] = []

    for item in keybindings:
        if not isinstance(item, dict):
            continue

        command: str = item.get('command', '')
        if command not in commands:
            processed.append(item)
            commands[command] = len(processed) - 1
        else:
            print(f"重复的command {command}")
            index: int = commands.get(command)
            processed[index] = item

    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=2)

    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == '__main__':
    process_keybindings("keyboard.json", "keyboard_processed.json")
