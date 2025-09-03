# SSE Client 可执行文件使用说明

本项目包含使用PyInstaller打包的可执行文件，可以在不同平台上运行而无需安装Python环境。

## 可用的可执行文件

### macOS
- `httpx_request_macos`: 适用于macOS的可执行文件

### Windows
- `httpx_request_windows.exe`: 适用于Windows的可执行文件
- `httpx_request_windows_amd64.exe`: 适用于Windows AMD64架构的可执行文件（注意：在macOS上构建时，实际生成的是macOS可执行文件，不是Windows可执行文件）

## 使用方法

### macOS
```bash
./httpx_request_macos
```

或者带初始消息启动：
```bash
./httpx_request_macos "你好，世界！"
```

### Windows
```cmd
httpx_request_windows.exe
```

或者带初始消息启动：
```cmd
httpx_request_windows.exe "你好，世界！"
```

## 功能特性

- 支持多轮对话
- 自动保存对话历史到chat目录
- 支持特殊命令：
  - `quit` 或 `exit` 或 `退出`: 退出程序
  - `clear` 或 `清空`: 清空对话历史

## 重新打包

如果需要重新打包可执行文件，可以使用以下命令：

```bash
# 为当前平台打包
python build_executables.py

# 仅为macOS打包
python build_executables.py macos

# 仅为Windows打包
python build_executables.py windows

# 仅为Windows AMD64打包
python build_executables.py windows-amd64

# 为所有平台打包
python build_executables.py all
```

## 交叉编译说明

注意：PyInstaller 不支持真正的跨平台编译。在 macOS 上构建的可执行文件只能在 macOS 上运行，即使指定了不同的架构。要生成真正的 Windows 可执行文件，需要在 Windows 机器上进行构建。