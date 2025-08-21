import argparse
import librosa
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import sounddevice as sd


class RealTimeAudioVisualizer:
    """实时音频波形图可视化类"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, window_size=44100, debug=False,
                 sound_detected_callback=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.debug = debug
        self.sound_detected_callback = sound_detected_callback
        self.audio_data = np.zeros(window_size)

        if self.debug:
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [])
            self.activity_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=12)
            self.sound_detected_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes, fontsize=12)

            # 设置图形属性
            self.ax.set_xlim(0, window_size)
            self.ax.set_ylim(-1, 1)
            self.ax.set_title("Real-time Audio Waveform")
            self.ax.set_xlabel("Samples")
            self.ax.set_ylabel("Amplitude")
        else:
            self.fig, self.ax = None, None
            self.line = None
            self.activity_text = None
            self.sound_detected_text = None

        # 音频活动检测阈值
        self.activity_threshold = 0.01

        # 声音检测相关属性
        self.target_sound = None
        self.sound_detection_threshold = 0.05  # 稍微降低阈值以提高敏感度
        self.audio_buffer = np.array([])
        self.buffer_size = sample_rate * 5  # 5秒的音频缓冲区
        
    def load_target_sound(self, file_path):
        """加载目标声音文件"""
        try:
            self.target_sound, _ = librosa.load(file_path, sr=self.sample_rate)
            print(f"Target sound loaded. Duration: {len(self.target_sound) / self.sample_rate:.2f} seconds")
            return True
        except Exception as e:
            print(f"Error loading target sound: {e}")
            return False
    
    def detect_sound(self, audio_chunk):
        """检测指定声音"""
        if self.target_sound is None:
            return False
            
        # 将新的音频块添加到缓冲区
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # 保持缓冲区大小
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        
        # 如果缓冲区太小，无法进行有效检测
        if len(self.audio_buffer) < len(self.target_sound):
            return False
            
        # 使用交叉相关性检测匹配
        try:
            correlation = scipy.signal.correlate(self.audio_buffer, self.target_sound, mode='valid')
            if len(correlation) == 0:
                return False
                
            normalized_correlation = correlation / (np.linalg.norm(self.audio_buffer) * np.linalg.norm(self.target_sound) + 1e-10)
            
            # 检查是否有超过阈值的相关性
            max_correlation = np.max(normalized_correlation)
            if max_correlation > self.sound_detection_threshold:
                # 重置缓冲区以避免重复检测
                self.audio_buffer = np.array([])
                # 仅在debug模式下打印调试信息
                if self.debug:
                    print(f"Sound detected! Correlation: {max_correlation:.4f}")
                return True
        except Exception as e:
            print(f"Error in sound detection: {e}")
            
        return False
    
    def audio_callback(self, indata, frames, time, status):
        """音频回调函数"""
        # 获取新音频数据
        new_data = indata[:, 0]
        
        # 更新音频数据缓冲区
        self.audio_data = np.roll(self.audio_data, -len(new_data))
        self.audio_data[-len(new_data):] = new_data
        
        # 检测音频活动
        rms = np.sqrt(np.mean(new_data ** 2))
        is_active = rms > self.activity_threshold
        
        # 更新活动状态显示（仅在debug模式下）
        if self.debug and self.activity_text:
            if is_active:
                self.activity_text.set_text("Audio Active")
                self.activity_text.set_color('red')
            else:
                self.activity_text.set_text("No Audio")
                self.activity_text.set_color('black')
        
        # 检测指定声音
        if self.detect_sound(new_data):
            # 仅在debug模式下更新文本显示
            if self.debug and self.sound_detected_text:
                self.sound_detected_text.set_text("出现了指定声音")
                self.sound_detected_text.set_color('blue')
            # 调用回调函数（如果已设置）
            if self.sound_detected_callback:
                self.sound_detected_callback()
        else:
            # 仅在debug模式下更新文本显示
            if self.debug and self.sound_detected_text:
                self.sound_detected_text.set_text("")
    
    def update_plot(self, frame):
        """更新波形图"""
        if self.debug and self.line:
            self.line.set_data(np.arange(len(self.audio_data)), self.audio_data)
            return self.line, self.activity_text
        return None
    
    def start_visualization(self):
        """开始实时可视化"""
        if self.debug:
            # 设置动画
            ani = animation.FuncAnimation(
                self.fig, 
                self.update_plot, 
                blit=True, 
                interval=30,  # 30ms更新一次
                cache_frame_data=False
            )
            
            # 开始音频流
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size
            ):
                plt.show()
        else:
            # 仅在后台进行音频检测，不显示波形图
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size
            ):
                # 保持程序运行，直到用户中断
                try:
                    while True:
                        sd.sleep(1000)  # 每秒检查一次
                except KeyboardInterrupt:
                    print("程序已停止")


def sound_detected_callback():
    print("回调函数：检测到指定声音！")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="实时音频波形可视化和声音检测")
    parser.add_argument("--debug", action="store_true", help="启用调试模式以显示波形图")
    parser.add_argument("audio_file", nargs="?", default="target_audio.wav", help="目标音频文件路径")
    args = parser.parse_args()
    
    if args.debug:
        print("Starting real-time audio waveform visualization...")
        print("Close the window to stop the program")
    else:
        print("Starting audio detection in background...")
        print("Press Ctrl+C to stop the program")
    
    # 创建可视化器，传入回调函数
    visualizer = RealTimeAudioVisualizer(debug=args.debug, sound_detected_callback=sound_detected_callback)
    
    # 加载目标声音文件
    if visualizer.load_target_sound(args.audio_file):
        print(f"Sound detection enabled with file: {args.audio_file}")
    else:
        print(f"Sound detection disabled - could not load target sound: {args.audio_file}")
    
    # 开始可视化
    visualizer.start_visualization()


if __name__ == "__main__":
    main()