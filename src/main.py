"""
Main application for dog bark detection and monitoring.
Coordinates audio capture, bark detection, dog identification, and database logging.
"""

import os
import sys
import time
import signal
import threading
import logging
import queue
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

# Import our modules
from .audio_capture import AudioCapture
from .bark_detector import BarkDetector
from .dog_identifier import DogIdentifier
from .database import BarkEventDatabase
from .bark_pattern_analyzer import AdvancedBarkAnalyzer
from .ai_models import create_ai_detector
from .audio_utils import AudioEncoder, create_audio_filename


class BarkMonitor:
    """Main application class for bark monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bark monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_running = False
        self.is_paused = False
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.audio_capture = None
        self.bark_detector = None
        self.ai_bark_detector = None
        self.dog_identifier = None
        self.database = None
        self.pattern_analyzer = None
        
        # Event processing
        self.event_queue = queue.Queue()
        self.processing_thread = None
        
        # Statistics
        self.session_stats = {
            'start_time': None,
            'total_barks_detected': 0,
            'dogs_identified': set(),
            'audio_files_saved': 0,
            'false_positives': 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Bark monitor initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        log_file = self.config.get('log_file')
        
        # Create logs directory if needed
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
    
    def initialize_components(self):
        """Initialize all monitoring components."""
        try:
            # Initialize audio capture
            self.audio_capture = AudioCapture(
                sample_rate=self.config.get('sample_rate', 44100),
                chunk_size=self.config.get('chunk_size', 4096),
                buffer_duration=self.config.get('buffer_duration', 10.0)
            )
            
            # Set audio callback
            self.audio_capture.set_audio_callback(self._audio_callback)
            
            # Initialize bark detector
            bark_model_path = self.config.get('bark_model_path')
            self.bark_detector = BarkDetector(
                sample_rate=self.config.get('sample_rate', 44100),
                model_path=bark_model_path if bark_model_path and os.path.exists(bark_model_path) else None
            )
            
            # Initialize AI bark detector if enabled
            if self.config.get('enable_ai', True):
                try:
                    self.ai_bark_detector = create_ai_detector(
                        model_type=self.config.get('ai_model_type', 'auto'),
                        sample_rate=self.config.get('sample_rate', 44100)
                    )
                    ai_info = self.ai_bark_detector.get_model_info()
                    self.logger.info(f"AI detection enabled: {ai_info['model_type']} model with {ai_info.get('total_parameters', 'unknown')} parameters")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize AI detector: {e}")
                    self.ai_bark_detector = None
            else:
                self.ai_bark_detector = None
            
            # Initialize dog identifier
            dog_model_path = self.config.get('dog_model_path')
            self.dog_identifier = DogIdentifier(
                sample_rate=self.config.get('sample_rate', 44100),
                max_dogs=self.config.get('max_dogs', 10)
            )
            
            if dog_model_path and os.path.exists(dog_model_path):
                self.dog_identifier.load_model(dog_model_path)
            
            # Initialize database
            db_path = self.config.get('database_path', os.path.join('data', 'bark_events.db'))
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.database = BarkEventDatabase(db_path)
            
            # Initialize pattern analyzer
            self.pattern_analyzer = AdvancedBarkAnalyzer()
            
            # Initialize audio encoder
            audio_quality = self.config.get('audio_quality', 'medium')
            self.audio_encoder = AudioEncoder(default_quality=audio_quality)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _audio_callback(self, audio_data: np.ndarray):
        """Callback for real-time audio processing."""
        if self.is_paused:
            return
        
        try:
            # Quick bark detection
            is_bark_like, confidence = self.bark_detector.is_bark_like(audio_data)
            
            if is_bark_like and confidence > self.config.get('detection_threshold', 0.5):
                # Add to processing queue for detailed analysis
                timestamp = datetime.now()
                self.event_queue.put({
                    'timestamp': timestamp,
                    'audio_data': audio_data.copy(),
                    'preliminary_confidence': confidence
                })
                
                self.logger.debug(f"Potential bark detected (confidence: {confidence:.3f})")
        
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
    
    def _process_events(self):
        """Background thread for processing detected events."""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1.0)
                
                if event is None:  # Shutdown signal
                    break
                
                self._analyze_bark_event(event)
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
    
    def _analyze_bark_event(self, event: Dict[str, Any]):
        """Analyze a potential bark event in detail."""
        try:
            timestamp = event['timestamp']
            audio_data = event['audio_data']
            
            # Detailed bark detection with traditional method
            is_bark_traditional, bark_probability_traditional = self.bark_detector.predict_bark(audio_data)
            
            # AI-enhanced bark detection if available
            is_bark_ai = False
            bark_probability_ai = 0.0
            if self.ai_bark_detector:
                try:
                    is_bark_ai, bark_probability_ai = self.ai_bark_detector.predict_bark(audio_data)
                except Exception as e:
                    self.logger.debug(f"AI detection failed: {e}")
            
            # Combine traditional and AI predictions
            if self.ai_bark_detector:
                # Weighted average of traditional and AI predictions
                combined_probability = (bark_probability_traditional * 0.4) + (bark_probability_ai * 0.6)
                is_bark = is_bark_traditional or is_bark_ai
                detection_method = "ai_enhanced"
            else:
                combined_probability = bark_probability_traditional
                is_bark = is_bark_traditional
                detection_method = "ml_classifier"
            
            if not is_bark or combined_probability < self.config.get('bark_threshold', 0.7):
                self.session_stats['false_positives'] += 1
                self.logger.debug(f"False positive filtered out (traditional: {bark_probability_traditional:.3f}, AI: {bark_probability_ai:.3f}, combined: {combined_probability:.3f})")
                return
            
            # Dog identification
            dog_id, identification_confidence = self.dog_identifier.identify_dog(audio_data)
            
            # Calculate audio characteristics
            duration = len(audio_data) / self.config.get('sample_rate', 44100)
            rms_energy = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            bark_intensity = min(rms_energy * 100, 100.0)  # Convert to percentage
            
            # Save audio file if configured
            audio_file_path = None
            if self.config.get('save_audio', True):
                audio_file_path = self._save_audio_file(audio_data, timestamp, dog_id)
            
            # Log to database
            event_id = self.database.log_bark_event(
                timestamp=timestamp,
                duration=duration,
                dog_id=dog_id if identification_confidence > 0.5 else None,
                confidence=identification_confidence,
                audio_file_path=audio_file_path,
                detection_method="ml_classifier",
                bark_intensity=bark_intensity,
                background_noise_level=self.audio_capture.get_audio_level(),
                notes=f"Auto-detected (trad: {bark_probability_traditional:.3f}, AI: {bark_probability_ai:.3f}, combined: {combined_probability:.3f})"
            )
            
            # Update session statistics
            self.session_stats['total_barks_detected'] += 1
            if dog_id and identification_confidence > 0.5:
                self.session_stats['dogs_identified'].add(dog_id)
            if audio_file_path:
                self.session_stats['audio_files_saved'] += 1
            
            # Add bark sample to dog identifier for learning
            if is_bark and dog_id and identification_confidence > 0.7:
                self.dog_identifier.add_bark_sample(audio_data, dog_id)
            
            self.logger.info(
                f"Bark detected (ID: {event_id}) - "
                f"Dog: {dog_id or 'unknown'} "
                f"(conf: {identification_confidence:.3f}), "
                f"Duration: {duration:.1f}s, "
                f"Intensity: {bark_intensity:.1f}%"
            )
            
            # Trigger alert if configured
            if self.config.get('enable_alerts', False):
                self._trigger_alert(event_id, dog_id, duration, bark_intensity)
        
        except Exception as e:
            self.logger.error(f"Error analyzing bark event: {e}")
    
    def _save_audio_file(self, audio_data: np.ndarray, timestamp: datetime, dog_id: str) -> str:
        """Save audio data to file."""
        try:
            # Create recordings directory
            recordings_dir = self.config.get('recordings_dir', 'recordings')
            os.makedirs(recordings_dir, exist_ok=True)
            
            # Determine audio format from config
            audio_format = self.config.get('audio_format', 'mp3').lower()
            
            # Generate filename using utility function
            filename = create_audio_filename(
                prefix='bark',
                timestamp=timestamp,
                dog_id=dog_id,
                extension=audio_format
            )
            filepath = os.path.join(recordings_dir, filename)
            
            # Save audio in specified format
            sample_rate = self.config.get('sample_rate', 44100)
            
            if audio_format == 'mp3':
                # Use MP3 encoding for space efficiency
                audio_quality = self.config.get('audio_quality', 'medium')
                success = self.audio_encoder.save_audio_as_mp3(
                    audio_data, filepath, sample_rate, audio_quality
                )
            else:
                # Fallback to WAV
                success = self.audio_encoder.save_audio_as_wav(
                    audio_data, filepath, sample_rate
                )
            
            if success:
                return filepath
            else:
                self.logger.error(f"Failed to save audio in {audio_format} format")
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to save audio file: {e}")
            return None
    
    def _trigger_alert(self, event_id: int, dog_id: str, duration: float, intensity: float):
        """Trigger alert for bark event."""
        # This could be extended to send notifications, emails, etc.
        alert_threshold = self.config.get('alert_intensity_threshold', 80.0)
        
        if intensity >= alert_threshold:
            self.logger.warning(
                f"HIGH INTENSITY BARK ALERT - "
                f"Event {event_id}, Dog: {dog_id or 'unknown'}, "
                f"Duration: {duration:.1f}s, Intensity: {intensity:.1f}%"
            )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self, device_index: Optional[int] = None):
        """Start bark monitoring."""
        if self.is_running:
            self.logger.warning("Monitor is already running")
            return
        
        try:
            # Initialize components if not already done
            if self.audio_capture is None:
                self.initialize_components()
            
            # Check if audio devices are available
            devices = self.audio_capture.list_audio_devices()
            if not devices:
                self.logger.error("No audio input devices found")
                # Run in simulation mode for testing
                self._run_simulation_mode()
                return
            
            self.logger.info("Available audio devices:")
            for device in devices:
                self.logger.info(f"  {device['index']}: {device['name']}")
            
            # Start components
            self.is_running = True
            self.session_stats['start_time'] = datetime.now()
            
            # Start event processing thread
            self.processing_thread = threading.Thread(
                target=self._process_events,
                daemon=True
            )
            self.processing_thread.start()
            
            # Start audio capture
            self.audio_capture.start_recording(device_index)
            
            self.logger.info("Bark monitoring started")
            
            # Main monitoring loop
            self._monitoring_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.stop()
            raise
    
    def _run_simulation_mode(self):
        """Run in simulation mode when no audio devices are available."""
        self.logger.info("Running in simulation mode (no audio devices)")
        
        # Initialize components first
        if self.bark_detector is None:
            self.initialize_components()
        
        self.is_running = True
        self.session_stats['start_time'] = datetime.now()
        
        # Start event processing thread
        self.processing_thread = threading.Thread(
            target=self._process_events,
            daemon=True
        )
        self.processing_thread.start()
        
        # Simulate bark events periodically
        try:
            while self.is_running:
                time.sleep(10)  # Simulate a bark every 10 seconds
                
                if not self.is_running:
                    break
                
                # Generate simulated bark
                sample_rate = self.config.get('sample_rate', 44100)
                duration = 1.5
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                # Simulated bark audio
                bark_audio = (0.3 * np.sin(2 * np.pi * 800 * t) * 
                             np.exp(-t * 3) * (np.sin(2 * np.pi * 10 * t) > 0))
                bark_audio = (bark_audio * 32767).astype(np.int16)
                
                # Add to processing queue
                self.event_queue.put({
                    'timestamp': datetime.now(),
                    'audio_data': bark_audio,
                    'preliminary_confidence': 0.8
                })
                
                self.logger.info("Simulated bark event generated")
        
        except KeyboardInterrupt:
            pass
        
        self.stop()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            # Print status every minute
            last_status_time = time.time()
            
            while self.is_running:
                time.sleep(1)
                
                # Print periodic status
                current_time = time.time()
                if current_time - last_status_time >= 60:  # Every minute
                    self._print_status()
                    last_status_time = current_time
        
        except KeyboardInterrupt:
            self.logger.info("Monitoring interrupted by user")
        
        finally:
            self.stop()
    
    def _print_status(self):
        """Print current monitoring status."""
        if self.session_stats['start_time']:
            runtime = datetime.now() - self.session_stats['start_time']
            runtime_str = str(runtime).split('.')[0]  # Remove microseconds
            
            self.logger.info(
                f"Status - Runtime: {runtime_str}, "
                f"Barks: {self.session_stats['total_barks_detected']}, "
                f"Dogs: {len(self.session_stats['dogs_identified'])}, "
                f"Audio files: {self.session_stats['audio_files_saved']}, "
                f"False positives: {self.session_stats['false_positives']}"
            )
    
    def pause(self):
        """Pause monitoring."""
        self.is_paused = True
        self.logger.info("Monitoring paused")
    
    def resume(self):
        """Resume monitoring."""
        self.is_paused = False
        self.logger.info("Monitoring resumed")
    
    def stop(self):
        """Stop bark monitoring."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping bark monitoring...")
        self.is_running = False
        
        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop_recording()
        
        # Signal processing thread to stop
        if self.processing_thread:
            self.event_queue.put(None)  # Shutdown signal
            self.processing_thread.join(timeout=2.0)
        
        # Save models if they were updated
        if self.dog_identifier and len(self.dog_identifier.bark_fingerprints) > 0:
            dog_model_path = self.config.get('dog_model_path', os.path.join('data', 'dog_identification_model.pkl'))
            os.makedirs(os.path.dirname(dog_model_path), exist_ok=True)
            self.dog_identifier.save_model(dog_model_path)
        
        # Print final statistics
        self._print_final_stats()
        
        self.logger.info("Bark monitoring stopped")
    
    def _print_final_stats(self):
        """Print final session statistics."""
        if not self.session_stats['start_time']:
            return
        
        runtime = datetime.now() - self.session_stats['start_time']
        
        self.logger.info("=== SESSION STATISTICS ===")
        self.logger.info(f"Total runtime: {runtime}")
        self.logger.info(f"Total barks detected: {self.session_stats['total_barks_detected']}")
        self.logger.info(f"Unique dogs identified: {len(self.session_stats['dogs_identified'])}")
        self.logger.info(f"Audio files saved: {self.session_stats['audio_files_saved']}")
        self.logger.info(f"False positives filtered: {self.session_stats['false_positives']}")
        
        if self.session_stats['dogs_identified']:
            self.logger.info(f"Dogs detected: {', '.join(sorted(self.session_stats['dogs_identified']))}")


# Default configuration
DEFAULT_CONFIG = {
    'sample_rate': 44100,
    'chunk_size': 4096,
    'buffer_duration': 10.0,
    'detection_threshold': 0.5,
    'bark_threshold': 0.7,
    'save_audio': True,
    'enable_alerts': True,
    'alert_intensity_threshold': 80.0,
    'max_dogs': 10,
    'log_level': 'INFO',
    'log_file': os.path.join('logs', 'bark_monitor.log'),
    'database_path': os.path.join('data', 'bark_events.db'),
    'recordings_dir': 'recordings',
    'bark_model_path': os.path.join('data', 'bark_detection_model.pkl'),
    'dog_model_path': os.path.join('data', 'dog_identification_model.pkl')
}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dog Bark Detection and Monitoring System')
    parser.add_argument('--device', type=int, help='Audio device index to use')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode')
    parser.add_argument('--list-devices', action='store_true', help='List available audio devices')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)
    
    # Create monitor
    monitor = BarkMonitor(config)
    
    # List devices if requested
    if args.list_devices:
        try:
            monitor.initialize_components()
            devices = monitor.audio_capture.list_audio_devices()
            print("Available audio devices:")
            for device in devices:
                print(f"  {device['index']}: {device['name']} "
                      f"({device['channels']} ch, {device['sample_rate']} Hz)")
        except Exception as e:
            print(f"Error listing devices: {e}")
        return
    
    # Start monitoring
    try:
        if args.simulate:
            monitor._run_simulation_mode()
        else:
            monitor.start(device_index=args.device)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()