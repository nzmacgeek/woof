"""
Audio capture module for continuous microphone monitoring.
Handles real-time audio stream capture and buffer management.
"""

import pyaudio
import numpy as np
import threading
import queue
import time
import logging
from typing import Callable, Optional
from collections import deque


class AudioCapture:
    """Handles real-time audio capture from microphone."""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 chunk_size: int = 4096,
                 channels: int = 1,
                 format_type: int = pyaudio.paInt16,
                 buffer_duration: float = 10.0):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            chunk_size: Number of samples per audio chunk
            channels: Number of audio channels (1 for mono)
            format_type: Audio format (16-bit integer)
            buffer_duration: Duration of audio buffer in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format_type = format_type
        self.buffer_duration = buffer_duration
        
        # Calculate buffer size in chunks
        self.buffer_size = int((sample_rate * buffer_duration) / chunk_size)
        
        # Audio buffer - circular buffer for continuous recording
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Threading and queue management
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        
        # PyAudio instance
        self.audio = None
        self.stream = None
        
        # Callback for real-time audio processing
        self.audio_callback: Optional[Callable] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def list_audio_devices(self):
        """List available audio input devices."""
        audio = pyaudio.PyAudio()
        devices = []
        
        try:
            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
        finally:
            audio.terminate()
            
        return devices
    
    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for real-time audio processing."""
        self.audio_callback = callback
    
    def start_recording(self, device_index: Optional[int] = None):
        """Start audio recording in a separate thread."""
        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return
        
        self.audio = pyaudio.PyAudio()
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index,
                stream_callback=self._audio_stream_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            # Start background thread for audio processing
            self.recording_thread = threading.Thread(
                target=self._audio_processing_thread,
                daemon=True
            )
            self.recording_thread.start()
            
            self.logger.info(f"Started audio recording at {self.sample_rate}Hz")
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.stop_recording()
            raise
    
    def stop_recording(self):
        """Stop audio recording and cleanup resources."""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        
        self.logger.info("Stopped audio recording")
    
    def _audio_stream_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream - runs in separate thread."""
        if status:
            self.logger.warning(f"Audio stream status: {status}")
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to queue for processing
        self.audio_queue.put(audio_data)
        
        return (None, pyaudio.paContinue)
    
    def _audio_processing_thread(self):
        """Background thread for processing audio data."""
        while self.is_recording:
            try:
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Add to circular buffer
                self.audio_buffer.append(audio_data)
                
                # Call user callback if set
                if self.audio_callback:
                    self.audio_callback(audio_data)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in audio processing thread: {e}")
    
    def get_recent_audio(self, duration_seconds: float) -> np.ndarray:
        """
        Get recent audio data from buffer.
        
        Args:
            duration_seconds: Duration of audio to retrieve in seconds
            
        Returns:
            numpy array of audio samples
        """
        if not self.audio_buffer:
            return np.array([])
        
        # Calculate number of chunks needed
        chunks_needed = int((self.sample_rate * duration_seconds) / self.chunk_size)
        chunks_needed = min(chunks_needed, len(self.audio_buffer))
        
        if chunks_needed == 0:
            return np.array([])
        
        # Get recent chunks and concatenate
        recent_chunks = list(self.audio_buffer)[-chunks_needed:]
        return np.concatenate(recent_chunks)
    
    def get_audio_level(self) -> float:
        """Get current audio level (RMS) as percentage."""
        if not self.audio_buffer:
            return 0.0
        
        # Get most recent chunk
        recent_audio = self.audio_buffer[-1] if self.audio_buffer else np.array([])
        
        if len(recent_audio) == 0:
            return 0.0
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(recent_audio.astype(np.float32) ** 2))
        
        # Convert to percentage (normalize to 16-bit range)
        level_percent = (rms / 32767.0) * 100.0
        
        return min(level_percent, 100.0)
    
    def is_sound_detected(self, threshold: float = 1.0) -> bool:
        """
        Check if sound is detected above threshold.
        
        Args:
            threshold: Sound level threshold as percentage
            
        Returns:
            True if sound level exceeds threshold
        """
        return self.get_audio_level() > threshold
    
    def save_recent_audio(self, filepath: str, duration_seconds: float = 5.0):
        """
        Save recent audio to WAV file.
        
        Args:
            filepath: Path to save audio file
            duration_seconds: Duration of audio to save
        """
        import soundfile as sf
        
        audio_data = self.get_recent_audio(duration_seconds)
        
        if len(audio_data) > 0:
            # Normalize to float32 for soundfile
            audio_normalized = audio_data.astype(np.float32) / 32767.0
            sf.write(filepath, audio_normalized, self.sample_rate)
            self.logger.info(f"Saved {duration_seconds}s of audio to {filepath}")
        else:
            self.logger.warning("No audio data available to save")


# Example usage and testing
if __name__ == "__main__":
    import os
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create audio capture instance
    capture = AudioCapture(sample_rate=44100, chunk_size=4096)
    
    # List available devices
    print("Available audio devices:")
    devices = capture.list_audio_devices()
    for device in devices:
        print(f"  {device['index']}: {device['name']} "
              f"({device['channels']} ch, {device['sample_rate']} Hz)")
    
    if not devices:
        print("No audio input devices found!")
        sys.exit(1)
    
    # Set up callback to monitor audio levels
    def audio_callback(audio_data):
        level = capture.get_audio_level()
        if level > 5.0:  # Only print when there's significant audio
            print(f"Audio level: {level:.1f}%")
    
    capture.set_audio_callback(audio_callback)
    
    try:
        # Start recording
        print(f"\nStarting audio capture (Ctrl+C to stop)...")
        capture.start_recording()
        
        # Monitor for 30 seconds or until interrupted
        start_time = time.time()
        while time.time() - start_time < 30:
            time.sleep(0.1)
            
            # Check for sound detection
            if capture.is_sound_detected(threshold=10.0):
                print("Sound detected!")
                
                # Save recent audio when sound is detected
                timestamp = int(time.time())
                filepath = f"/tmp/test_audio_{timestamp}.wav"
                capture.save_recent_audio(filepath, duration_seconds=3.0)
    
    except KeyboardInterrupt:
        print("\nStopping audio capture...")
    
    finally:
        capture.stop_recording()
        print("Audio capture stopped")