import numpy as np
from scipy import signal
import librosa

class AdaptiveNoiseFilter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_profile = None
        self.noise_gate_threshold = 0.01  # Noise gate threshold
        
    def learn_noise_profile(self, noise_audio, duration=2.0):
        """Learn noise profile"""
        noise_length = int(duration * self.sample_rate)
        if len(noise_audio) < noise_length:
            noise_audio = np.tile(noise_audio, noise_length // len(noise_audio) + 1)
        
        noise_sample = noise_audio[:noise_length]
        
        # Transform to frequency domain
        freqs, psd = signal.welch(noise_sample, self.sample_rate, nperseg=1024)
        self.noise_profile = {'freqs': freqs, 'psd': psd}
        
        return True
    
    def apply_spectral_subtraction(self, audio_data, alpha=2.0):
        """Apply spectral subtraction"""
        if self.noise_profile is None:
            return audio_data
        
        # STFT
        D = librosa.stft(audio_data.astype(np.float32))
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Noise estimation
        noise_magnitude = np.sqrt(np.interp(
            librosa.fft_frequencies(sr=self.sample_rate), 
            self.noise_profile['freqs'], 
            self.noise_profile['psd']
        ))
        
        # Spectral subtraction
        enhanced_magnitude = magnitude - alpha * noise_magnitude[:, np.newaxis]
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
        
        # Inverse transform
        enhanced_D = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_D)
        
        return (enhanced_audio * 32767).astype(np.int16)
    
    def apply_noise_gate(self, audio_data):
        """Apply noise gate"""
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        
        if rms < self.noise_gate_threshold:
            return np.zeros_like(audio_data)
        else:
            return audio_data