"""
Audio utilities for MP3 encoding, format conversion, and file management.
Provides efficient audio storage and cross-platform compatibility.
"""

import os
import logging
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any
import numpy as np
from datetime import datetime
import shutil

# Try importing audio libraries with fallbacks
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import wave
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False


class AudioEncoder:
    """Audio encoding and format conversion utilities."""
    
    def __init__(self, default_quality: str = 'medium'):
        """
        Initialize audio encoder.
        
        Args:
            default_quality: Default encoding quality ('low', 'medium', 'high')
        """
        self.default_quality = default_quality
        self.logger = logging.getLogger(__name__)
        
        # Quality presets for MP3 encoding
        self.quality_presets = {
            'low': {'bitrate': '96k', 'sample_rate': 22050},
            'medium': {'bitrate': '128k', 'sample_rate': 44100},
            'high': {'bitrate': '192k', 'sample_rate': 44100},
            'highest': {'bitrate': '320k', 'sample_rate': 44100}
        }
        
        # Check available encoders
        self.available_encoders = self._check_available_encoders()
        self.logger.info(f"Available encoders: {list(self.available_encoders.keys())}")
    
    def _check_available_encoders(self) -> Dict[str, bool]:
        """Check which audio encoders are available."""
        encoders = {
            'soundfile': SOUNDFILE_AVAILABLE,
            'pydub': PYDUB_AVAILABLE,
            'wave': WAVE_AVAILABLE,
            'ffmpeg': False,
            'lame': False
        }
        
        # Check for external executables
        if PYDUB_AVAILABLE:
            encoders['ffmpeg'] = which("ffmpeg") is not None
            encoders['lame'] = which("lame") is not None
        else:
            # Manual check for FFmpeg and LAME
            encoders['ffmpeg'] = shutil.which("ffmpeg") is not None
            encoders['lame'] = shutil.which("lame") is not None
        
        return encoders
    
    def save_audio_as_mp3(self, 
                         audio_data: np.ndarray,
                         filepath: str,
                         sample_rate: int = 44100,
                         quality: Optional[str] = None) -> bool:
        """
        Save audio data as MP3 file.
        
        Args:
            audio_data: Raw audio samples
            filepath: Output MP3 file path
            sample_rate: Audio sample rate
            quality: Encoding quality ('low', 'medium', 'high', 'highest')
            
        Returns:
            True if successful, False otherwise
        """
        if quality is None:
            quality = self.default_quality
        
        if quality not in self.quality_presets:
            self.logger.warning(f"Invalid quality '{quality}', using 'medium'")
            quality = 'medium'
        
        preset = self.quality_presets[quality]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Try different encoding methods in order of preference
        if self._encode_with_pydub(audio_data, filepath, sample_rate, preset):
            return True
        elif self._encode_with_ffmpeg(audio_data, filepath, sample_rate, preset):
            return True
        elif self._encode_with_lame(audio_data, filepath, sample_rate, preset):
            return True
        else:
            # Fallback to WAV if MP3 encoding fails
            wav_filepath = filepath.replace('.mp3', '.wav')
            return self.save_audio_as_wav(audio_data, wav_filepath, sample_rate)
    
    def _encode_with_pydub(self, 
                          audio_data: np.ndarray,
                          filepath: str,
                          sample_rate: int,
                          preset: Dict[str, Any]) -> bool:
        """Encode using pydub library."""
        if not self.available_encoders['pydub']:
            return False
        
        try:
            # Convert numpy array to pydub AudioSegment
            if audio_data.dtype == np.int16:
                audio_segment = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,  # 16-bit
                    channels=1
                )
            else:
                # Convert to int16 first
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )
            
            # Resample if needed
            if sample_rate != preset['sample_rate']:
                audio_segment = audio_segment.set_frame_rate(preset['sample_rate'])
            
            # Export as MP3
            audio_segment.export(
                filepath,
                format="mp3",
                bitrate=preset['bitrate'],
                parameters=["-q:a", "2"]  # Good quality/speed balance
            )
            
            self.logger.debug(f"MP3 encoded with pydub: {filepath}")
            return True
            
        except Exception as e:
            self.logger.debug(f"Pydub encoding failed: {e}")
            return False
    
    def _encode_with_ffmpeg(self, 
                           audio_data: np.ndarray,
                           filepath: str,
                           sample_rate: int,
                           preset: Dict[str, Any]) -> bool:
        """Encode using FFmpeg directly."""
        if not self.available_encoders['ffmpeg']:
            return False
        
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Save as WAV first
            if not self.save_audio_as_wav(audio_data, temp_wav_path, sample_rate):
                return False
            
            # Convert to MP3 using FFmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', temp_wav_path,
                '-codec:a', 'libmp3lame',
                '-b:a', preset['bitrate'],
                '-ar', str(preset['sample_rate']),
                '-ac', '1',  # Mono
                '-q:a', '2',  # Good quality
                filepath
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up temporary file
            os.unlink(temp_wav_path)
            
            if result.returncode == 0:
                self.logger.debug(f"MP3 encoded with FFmpeg: {filepath}")
                return True
            else:
                self.logger.debug(f"FFmpeg encoding failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.debug(f"FFmpeg encoding failed: {e}")
            # Clean up if temporary file exists
            try:
                os.unlink(temp_wav_path)
            except:
                pass
            return False
    
    def _encode_with_lame(self, 
                         audio_data: np.ndarray,
                         filepath: str,
                         sample_rate: int,
                         preset: Dict[str, Any]) -> bool:
        """Encode using LAME directly."""
        if not self.available_encoders['lame']:
            return False
        
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Save as WAV first
            if not self.save_audio_as_wav(audio_data, temp_wav_path, sample_rate):
                return False
            
            # Convert to MP3 using LAME
            lame_cmd = [
                'lame',
                '-b', preset['bitrate'].rstrip('k'),  # Remove 'k' suffix
                '--resample', str(preset['sample_rate'] / 1000),  # Convert to kHz
                '-m', 'm',  # Mono
                '-q', '2',  # Good quality
                temp_wav_path,
                filepath
            ]
            
            result = subprocess.run(
                lame_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up temporary file
            os.unlink(temp_wav_path)
            
            if result.returncode == 0:
                self.logger.debug(f"MP3 encoded with LAME: {filepath}")
                return True
            else:
                self.logger.debug(f"LAME encoding failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.debug(f"LAME encoding failed: {e}")
            # Clean up if temporary file exists
            try:
                os.unlink(temp_wav_path)
            except:
                pass
            return False
    
    def save_audio_as_wav(self, 
                         audio_data: np.ndarray,
                         filepath: str,
                         sample_rate: int = 44100) -> bool:
        """
        Save audio data as WAV file.
        
        Args:
            audio_data: Raw audio samples
            filepath: Output WAV file path
            sample_rate: Audio sample rate
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Try different saving methods
        if self._save_wav_with_soundfile(audio_data, filepath, sample_rate):
            return True
        elif self._save_wav_with_wave(audio_data, filepath, sample_rate):
            return True
        else:
            self.logger.error(f"Failed to save WAV file: {filepath}")
            return False
    
    def _save_wav_with_soundfile(self, 
                                audio_data: np.ndarray,
                                filepath: str,
                                sample_rate: int) -> bool:
        """Save WAV using soundfile library."""
        if not self.available_encoders['soundfile']:
            return False
        
        try:
            # Ensure audio data is in correct format
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            sf.write(filepath, audio_float, sample_rate)
            self.logger.debug(f"WAV saved with soundfile: {filepath}")
            return True
            
        except Exception as e:
            self.logger.debug(f"Soundfile WAV saving failed: {e}")
            return False
    
    def _save_wav_with_wave(self, 
                           audio_data: np.ndarray,
                           filepath: str,
                           sample_rate: int) -> bool:
        """Save WAV using standard wave library."""
        if not self.available_encoders['wave']:
            return False
        
        try:
            # Convert to int16 if needed
            if audio_data.dtype == np.int16:
                audio_int16 = audio_data
            else:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            self.logger.debug(f"WAV saved with wave: {filepath}")
            return True
            
        except Exception as e:
            self.logger.debug(f"Wave library WAV saving failed: {e}")
            return False
    
    def get_file_size_reduction(self, wav_path: str, mp3_path: str) -> Optional[Tuple[int, int, float]]:
        """
        Calculate file size reduction from WAV to MP3.
        
        Args:
            wav_path: Path to WAV file
            mp3_path: Path to MP3 file
            
        Returns:
            (wav_size, mp3_size, reduction_percentage) or None if files don't exist
        """
        try:
            if not (os.path.exists(wav_path) and os.path.exists(mp3_path)):
                return None
            
            wav_size = os.path.getsize(wav_path)
            mp3_size = os.path.getsize(mp3_path)
            reduction = ((wav_size - mp3_size) / wav_size) * 100
            
            return wav_size, mp3_size, reduction
            
        except Exception as e:
            self.logger.error(f"Error calculating file size reduction: {e}")
            return None
    
    def batch_convert_to_mp3(self, 
                            input_dir: str,
                            output_dir: str,
                            quality: str = 'medium',
                            remove_originals: bool = False) -> Dict[str, Any]:
        """
        Batch convert WAV files to MP3.
        
        Args:
            input_dir: Directory containing WAV files
            output_dir: Directory for MP3 output
            quality: Encoding quality
            remove_originals: Whether to delete original WAV files
            
        Returns:
            Conversion results
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_space_saved': 0,
            'files': []
        }
        
        # Find all WAV files
        wav_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        results['total_files'] = len(wav_files)
        
        for wav_path in wav_files:
            try:
                # Read WAV file
                if SOUNDFILE_AVAILABLE:
                    audio_data, sample_rate = sf.read(wav_path, dtype=np.float32)
                else:
                    # Fallback to wave library
                    with wave.open(wav_path, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                
                # Generate output path
                rel_path = os.path.relpath(wav_path, input_dir)
                mp3_path = os.path.join(output_dir, rel_path.replace('.wav', '.mp3'))
                
                # Ensure output subdirectory exists
                os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
                
                # Convert to MP3
                if self.save_audio_as_mp3(audio_data, mp3_path, sample_rate, quality):
                    results['successful'] += 1
                    
                    # Calculate space savings
                    size_info = self.get_file_size_reduction(wav_path, mp3_path)
                    if size_info:
                        wav_size, mp3_size, reduction = size_info
                        results['total_space_saved'] += (wav_size - mp3_size)
                        
                        results['files'].append({
                            'wav_path': wav_path,
                            'mp3_path': mp3_path,
                            'wav_size': wav_size,
                            'mp3_size': mp3_size,
                            'reduction_percent': reduction,
                            'status': 'success'
                        })
                        
                        # Remove original if requested
                        if remove_originals:
                            os.remove(wav_path)
                    
                else:
                    results['failed'] += 1
                    results['files'].append({
                        'wav_path': wav_path,
                        'status': 'failed'
                    })
                
            except Exception as e:
                results['failed'] += 1
                results['files'].append({
                    'wav_path': wav_path,
                    'status': 'error',
                    'error': str(e)
                })
                self.logger.error(f"Error converting {wav_path}: {e}")
        
        self.logger.info(f"Batch conversion complete: {results['successful']}/{results['total_files']} successful, "
                        f"{results['total_space_saved']/1024/1024:.1f} MB saved")
        
        return results


def create_audio_filename(prefix: str = 'bark',
                         timestamp: Optional[datetime] = None,
                         dog_id: Optional[str] = None,
                         extension: str = 'mp3') -> str:
    """
    Generate a standardized audio filename.
    
    Args:
        prefix: Filename prefix
        timestamp: Event timestamp (uses current time if None)
        dog_id: Dog identifier
        extension: File extension
        
    Returns:
        Generated filename
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Format: bark_20240101_123456_dog001.mp3
    time_str = timestamp.strftime('%Y%m%d_%H%M%S')
    
    if dog_id:
        filename = f"{prefix}_{time_str}_{dog_id}.{extension}"
    else:
        filename = f"{prefix}_{time_str}.{extension}"
    
    return filename


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing audio encoding utilities...")
    
    # Create test audio encoder
    encoder = AudioEncoder(default_quality='medium')
    
    # Generate test audio
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulated bark audio
    test_audio = (0.3 * np.sin(2 * np.pi * 800 * t) * 
                  np.exp(-t * 3) * (np.sin(2 * np.pi * 10 * t) > 0))
    test_audio = (test_audio * 32767).astype(np.int16)
    
    # Test filename generation
    filename = create_audio_filename(dog_id='test_dog', extension='mp3')
    print(f"Generated filename: {filename}")
    
    # Test WAV saving
    wav_path = f"test_output/test_audio.wav"
    if encoder.save_audio_as_wav(test_audio, wav_path, sample_rate):
        print(f"WAV saved successfully: {wav_path}")
        
        # Test MP3 conversion
        mp3_path = f"test_output/test_audio.mp3"
        if encoder.save_audio_as_mp3(test_audio, mp3_path, sample_rate, 'medium'):
            print(f"MP3 saved successfully: {mp3_path}")
            
            # Check file sizes
            size_info = encoder.get_file_size_reduction(wav_path, mp3_path)
            if size_info:
                wav_size, mp3_size, reduction = size_info
                print(f"File size reduction: {wav_size} -> {mp3_size} bytes ({reduction:.1f}% reduction)")
        
        # Clean up test files
        try:
            os.remove(wav_path)
            os.remove(mp3_path)
            os.rmdir("test_output")
        except:
            pass
    
    print("Audio encoding testing completed.")