"""
AI model integration for enhanced bark detection using modern deep learning approaches.
This is a fallback version that works without PyTorch for development environments.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import pickle
from datetime import datetime
import warnings

# Check for available libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)


class FallbackAIDetector:
    """Enhanced fallback detector for systems without AI libraries."""
    
    def __init__(self, sample_rate: int = 44100, model_type: str = 'enhanced'):
        self.sample_rate = sample_rate
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using enhanced fallback AI detector (model_type: {model_type})")
        
        # Enhanced detection parameters
        self.energy_threshold = 0.02
        self.frequency_bands = {
            'low': (50, 300),      # Very low frequency
            'bark_low': (300, 800),   # Typical bark range start
            'bark_mid': (800, 2000),  # Main bark frequency range
            'bark_high': (2000, 4000), # Higher bark harmonics
            'high': (4000, 8000)      # Very high frequency
        }
    
    def predict_bark(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Enhanced heuristic-based detection with multiple features."""
        if len(audio_data) == 0:
            return False, 0.0
        
        try:
            # Convert to float if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Feature 1: RMS Energy
            rms = np.sqrt(np.mean(audio_float ** 2))
            energy_score = min(rms / self.energy_threshold, 1.0)
            
            # Feature 2: Zero crossing rate (indicates frequency content)
            zero_crossings = np.sum(np.diff(np.signbit(audio_float)))
            zcr_rate = zero_crossings / len(audio_float)
            zcr_score = min(zcr_rate * 1000, 1.0)  # Normalize
            
            # Feature 3: Spectral features using FFT
            if LIBROSA_AVAILABLE:
                spectral_score = self._analyze_spectrum_librosa(audio_float)
            else:
                spectral_score = self._analyze_spectrum_numpy(audio_float)
            
            # Feature 4: Temporal pattern analysis
            temporal_score = self._analyze_temporal_pattern(audio_float)
            
            # Feature 5: Peak analysis
            peak_score = self._analyze_peaks(audio_float)
            
            # Combine features with weights
            weights = {
                'energy': 0.25,
                'zcr': 0.15,
                'spectral': 0.30,
                'temporal': 0.20,
                'peaks': 0.10
            }
            
            combined_score = (
                weights['energy'] * energy_score +
                weights['zcr'] * zcr_score +
                weights['spectral'] * spectral_score +
                weights['temporal'] * temporal_score +
                weights['peaks'] * peak_score
            )
            
            # Determine if it's a bark (adjustable threshold)
            threshold = 0.4  # Can be tuned
            is_bark = combined_score > threshold
            
            return is_bark, combined_score
            
        except Exception as e:
            self.logger.debug(f"Error in enhanced detection: {e}")
            return False, 0.0
    
    def _analyze_spectrum_librosa(self, audio_data: np.ndarray) -> float:
        """Analyze frequency spectrum using librosa."""
        try:
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=40,
                fmax=8000
            )
            
            # Focus on bark-relevant frequency bands
            mel_mean = np.mean(mel_spec, axis=1)
            
            # Weight different frequency bands
            bark_bands = mel_mean[10:25]  # Approximate bark frequency range in mel scale
            bark_energy = np.mean(bark_bands)
            
            # Normalize
            return min(bark_energy * 1000, 1.0)
            
        except Exception:
            return self._analyze_spectrum_numpy(audio_data)
    
    def _analyze_spectrum_numpy(self, audio_data: np.ndarray) -> float:
        """Analyze frequency spectrum using numpy FFT."""
        try:
            # Compute FFT
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            
            # Analyze energy in bark frequency bands
            bark_score = 0.0
            total_weight = 0.0
            
            for band_name, (f_low, f_high) in self.frequency_bands.items():
                if 'bark' in band_name:
                    mask = (freqs >= f_low) & (freqs <= f_high)
                    band_energy = np.mean(magnitude[mask]) if np.any(mask) else 0
                    
                    # Weight mid-range frequencies more heavily
                    weight = 2.0 if band_name == 'bark_mid' else 1.0
                    bark_score += band_energy * weight
                    total_weight += weight
            
            if total_weight > 0:
                bark_score /= total_weight
            
            # Normalize to 0-1 range
            return min(bark_score / 1000, 1.0)
            
        except Exception:
            return 0.0
    
    def _analyze_temporal_pattern(self, audio_data: np.ndarray) -> float:
        """Analyze temporal patterns characteristic of barks."""
        try:
            # Barks typically have rapid onset and decay
            # Compute envelope
            envelope = np.abs(audio_data)
            
            # Smooth envelope
            window_size = max(len(envelope) // 50, 10)
            smoothed = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            
            # Find peak and analyze onset/decay
            peak_idx = np.argmax(smoothed)
            peak_value = smoothed[peak_idx]
            
            if peak_value < 0.01:  # Too quiet
                return 0.0
            
            # Analyze onset (rapid rise characteristic of barks)
            onset_portion = smoothed[:peak_idx] if peak_idx > 0 else []
            if len(onset_portion) > 0:
                onset_slope = (peak_value - onset_portion[0]) / len(onset_portion)
            else:
                onset_slope = 0
            
            # Analyze decay
            decay_portion = smoothed[peak_idx:] if peak_idx < len(smoothed) else []
            if len(decay_portion) > 1:
                decay_slope = abs((decay_portion[-1] - peak_value) / len(decay_portion))
            else:
                decay_slope = 0
            
            # Bark-like pattern: rapid onset, moderate decay
            onset_score = min(onset_slope * 10, 1.0)
            decay_score = min(decay_slope * 5, 1.0)
            
            return (onset_score + decay_score) / 2
            
        except Exception:
            return 0.0
    
    def _analyze_peaks(self, audio_data: np.ndarray) -> float:
        """Analyze peak characteristics."""
        try:
            # Find peaks in the audio signal
            envelope = np.abs(audio_data)
            
            # Simple peak detection
            peaks = []
            for i in range(1, len(envelope) - 1):
                if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]:
                    if envelope[i] > 0.05:  # Minimum amplitude threshold
                        peaks.append(i)
            
            if not peaks:
                return 0.0
            
            # Barks typically have 1-3 main peaks
            num_peaks = len(peaks)
            peak_score = 1.0 if 1 <= num_peaks <= 3 else max(0.5 - abs(num_peaks - 2) * 0.1, 0.0)
            
            # Analyze peak amplitudes
            peak_amplitudes = [envelope[p] for p in peaks]
            if peak_amplitudes:
                max_amplitude = max(peak_amplitudes)
                amplitude_score = min(max_amplitude * 5, 1.0)
            else:
                amplitude_score = 0.0
            
            return (peak_score + amplitude_score) / 2
            
        except Exception:
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_type': 'enhanced_fallback',
            'description': 'Multi-feature heuristic detector with spectral analysis',
            'features': ['energy', 'zero_crossing_rate', 'spectral_analysis', 'temporal_pattern', 'peak_analysis'],
            'librosa_available': LIBROSA_AVAILABLE,
            'sample_rate': self.sample_rate
        }


def create_ai_detector(model_type: str = 'auto', **kwargs) -> FallbackAIDetector:
    """
    Factory function to create AI detector.
    
    Args:
        model_type: Type of model (currently only 'auto' and 'fallback' supported)
        **kwargs: Additional arguments for detector
        
    Returns:
        AI detector instance
    """
    logging.getLogger(__name__).info(
        "Using enhanced fallback detector (PyTorch/Transformers not available)"
    )
    return FallbackAIDetector(model_type='enhanced', **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing enhanced AI bark detection...")
    
    # Test model creation
    detector = create_ai_detector(model_type='auto')
    
    # Model info
    info = detector.get_model_info()
    print(f"Model info: {info}")
    
    # Generate test audio
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulated bark (short burst with decay)
    bark_audio = (0.3 * np.sin(2 * np.pi * 800 * t) * 
                  np.exp(-t * 3) * (np.sin(2 * np.pi * 10 * t) > 0))
    bark_audio = (bark_audio * 32767).astype(np.int16)
    
    # Test prediction
    is_bark, confidence = detector.predict_bark(bark_audio)
    print(f"Bark prediction: is_bark={is_bark}, confidence={confidence:.3f}")
    
    # Test with noise
    noise_audio = np.random.normal(0, 0.1, len(bark_audio)).astype(np.int16)
    is_noise, noise_conf = detector.predict_bark(noise_audio)
    print(f"Noise prediction: is_bark={is_noise}, confidence={noise_conf:.3f}")
    
    # Test with silence
    silence_audio = np.zeros(len(bark_audio), dtype=np.int16)
    is_silence, silence_conf = detector.predict_bark(silence_audio)
    print(f"Silence prediction: is_bark={is_silence}, confidence={silence_conf:.3f}")
    
    print("\nEnhanced AI model testing completed.")