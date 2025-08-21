import librosa
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import sounddevice as sd


class RealTimeAudioVisualizer:
    """实时音频波形图可视化类"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, window_size=44100):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.audio_data = np.zeros(window_size)
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
        
        # 音频活动检测阈值
        self.activity_threshold = 0.01
        
        # 声音检测相关属性
        self.target_sound = None
        self.sound_detection_threshold = 0.1
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
            if np.max(normalized_correlation) > self.sound_detection_threshold:
                # 重置缓冲区以避免重复检测
                self.audio_buffer = np.array([])
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
        rms = np.sqrt(np.mean(new_data**2))
        is_active = rms > self.activity_threshold
        
        # 更新活动状态显示
        if is_active:
            self.activity_text.set_text("Audio Active")
            self.activity_text.set_color('red')
        else:
            self.activity_text.set_text("No Audio")
            self.activity_text.set_color('black')
            
        # 检测指定声音
        if self.detect_sound(new_data):
            self.sound_detected_text.set_text("出现了指定声音")
            self.sound_detected_text.set_color('blue')
            print("出现了指定声音")
        else:
            self.sound_detected_text.set_text("")
    
    def update_plot(self, frame):
        """更新波形图"""
        self.line.set_data(np.arange(len(self.audio_data)), self.audio_data)
        return self.line, self.activity_text
    
    def start_visualization(self):
        """开始实时可视化"""
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


def main():
    """主函数"""
    print("Starting real-time audio waveform visualization...")
    print("Close the window to stop the program")
    
    # 创建可视化器
    visualizer = RealTimeAudioVisualizer()
    
    # 加载目标声音文件
    if visualizer.load_target_sound("src/target_audio.wav"):
        print("Sound detection enabled")
    else:
        print("Sound detection disabled - could not load target sound")
    
    # 开始可视化
    visualizer.start_visualization()


if __name__ == "__main__":
    main()