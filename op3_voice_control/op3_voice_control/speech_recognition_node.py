#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import vosk
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from scipy import signal
import noisereduce as nr
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognition_node')
        
        # Publishers
        self.command_publisher = self.create_publisher(String, '/voice_command', 10)
        self.status_publisher = self.create_publisher(String, '/voice_status', 10)
        
        descriptor = ParameterDescriptor(
            name='device_id',
            type=ParameterType.PARAMETER_STRING,
            description='Audio device ID for speech recognition (default: c920_mic)',
        )

        # Parameters
        self.declare_parameter('model_path', '~/op3_voice_tutorial/models/vosk-model-small-en-us')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('device_id', 'c920_mic', descriptor)
        self.declare_parameter('noise_reduction', True)
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        device_id = self.get_parameter('device_id').get_parameter_value().string_value
        self.noise_reduction = self.get_parameter('noise_reduction').get_parameter_value().bool_value

        if device_id == 'c920_mic' or device_id == 'default':
            device_id = None

        # Load Vosk model
        try:
            model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(model, self.sample_rate)
            self.get_logger().info(f"Vosk model loaded from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load Vosk model: {e}")
            return
        
        # Audio queue
        self.audio_queue = queue.Queue()
        
        # Audio stream configuration
        try:
            self.stream = sd.InputStream(
                device=device_id,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                callback=self.audio_callback,
                blocksize=8000
            )
            self.get_logger().info("Audio stream initialized")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize audio stream: {e}")
            return
        
        # Noise filter setup
        self.setup_noise_filter()
        
        # Start speech recognition thread
        self.recognition_thread = threading.Thread(target=self.recognition_loop)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
        # Start audio stream
        self.stream.start()
        self.get_logger().info("Speech recognition node started")
        
    def setup_noise_filter(self):
        """Setup for motor noise filtering"""
        # Motor noise frequency band (typically 50-1000Hz)
        self.motor_noise_freqs = [50, 100, 200, 500, 1000]  # Hz
        
        # Notch filter coefficients
        self.notch_filters = []
        for freq in self.motor_noise_freqs:
            # Design notch filter (Q=30 for narrow band)
            b, a = signal.iirnotch(freq, 30, fs=self.sample_rate)
            self.notch_filters.append((b, a))
    
    def apply_noise_filtering(self, audio_data):
        """Apply noise filtering"""
        if not self.noise_reduction:
            return audio_data
            
        # Type conversion
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Apply notch filter (motor noise removal)
        filtered_audio = audio_float
        for b, a in self.notch_filters:
            filtered_audio = signal.filtfilt(b, a, filtered_audio)
        
        # Apply advanced noise reduction
        try:
            # Use first 0.5 seconds as noise profile
            if len(filtered_audio) > self.sample_rate // 2:
                noise_sample = filtered_audio[:self.sample_rate // 2]
                filtered_audio = nr.reduce_noise(
                    y=filtered_audio, 
                    sr=self.sample_rate,
                    y_noise=noise_sample,
                    prop_decrease=0.8
                )
        except Exception as e:
            self.get_logger().warn(f"Advanced noise reduction failed: {e}")
        
        # Convert back to int16
        return (filtered_audio * 32768.0).astype(np.int16)
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback function"""
        if status:
            self.get_logger().warn(f"Audio callback status: {status}")
        
        # Apply noise filtering
        filtered_data = self.apply_noise_filtering(indata[:, 0])
        self.audio_queue.put(filtered_data.copy())
    
    def recognition_loop(self):
        """Speech recognition main loop"""
        while rclpy.ok():
            try:
                # Get audio data
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # Speech recognition with Vosk
                    if self.recognizer.AcceptWaveform(audio_data.tobytes()):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '').strip()
                        
                        if text:
                            self.get_logger().info(f"Recognized: {text}")
                            
                            # Publish command
                            msg = String()
                            msg.data = text
                            self.command_publisher.publish(msg)
                            
                            # Publish status
                            status_msg = String()
                            status_msg.data = f"recognized: {text}"
                            self.status_publisher.publish(status_msg)
                    
                time.sleep(0.01)  # CPU usage control
                
            except Exception as e:
                self.get_logger().error(f"Recognition error: {e}")
                time.sleep(1.0)
    
    def destroy_node(self):
        """Cleanup on node shutdown"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()