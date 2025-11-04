"""
Advanced bark pattern analyzer for detecting cadences, sequences, and annoyance factors.
Uses temporal analysis, rhythm detection, and behavioral pattern recognition.
"""

import numpy as np
import librosa
from scipy import signal
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json


@dataclass
class BarkEvent:
    """Single bark event with detailed characteristics."""
    timestamp: datetime
    duration: float
    intensity: float
    frequency_peak: float
    frequency_range: Tuple[float, float]
    spectral_centroid: float
    zero_crossing_rate: float
    pitch_stability: float
    onset_sharpness: float


@dataclass
class BarkSequence:
    """Sequence of bark events forming a pattern."""
    events: List[BarkEvent]
    start_time: datetime
    end_time: datetime
    total_duration: float
    inter_bark_intervals: List[float]
    rhythm_regularity: float
    cadence_type: str  # 'single', 'burst', 'continuous', 'rhythmic'
    annoyance_score: float
    persistence_score: float


class AdvancedBarkAnalyzer:
    """Advanced analysis of bark patterns, cadences, and annoyance factors."""
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize advanced bark analyzer."""
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection parameters
        self.min_sequence_duration = 2.0  # seconds
        self.max_inter_bark_gap = 5.0     # seconds to consider part of sequence
        self.rhythm_tolerance = 0.3        # tolerance for rhythm detection
        
        # Annoyance scoring weights
        self.annoyance_weights = {
            'duration': 0.25,
            'intensity': 0.20,
            'frequency': 0.15,
            'persistence': 0.20,
            'time_of_day': 0.20
        }
        
        # Time-of-day multipliers for annoyance
        self.time_multipliers = {
            'night': 3.0,      # 22:00 - 06:00
            'early_morning': 2.5,  # 06:00 - 08:00
            'evening': 1.8,    # 17:00 - 22:00
            'daytime': 1.0     # 08:00 - 17:00
        }
    
    def analyze_bark_audio(self, audio_data: np.ndarray, timestamp: datetime) -> BarkEvent:
        """
        Analyze a single bark audio sample to extract detailed characteristics.
        
        Args:
            audio_data: Raw audio samples
            timestamp: When the bark occurred
            
        Returns:
            BarkEvent with detailed characteristics
        """
        try:
            # Convert to float if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            duration = len(audio_float) / self.sample_rate
            
            # Basic intensity
            rms_energy = np.sqrt(np.mean(audio_float ** 2))
            intensity = min(rms_energy * 100, 100.0)
            
            # Spectral analysis
            stft = librosa.stft(audio_float, hop_length=512)
            magnitude = np.abs(stft)
            
            # Frequency characteristics
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_float, sr=self.sample_rate
            )[0]
            spectral_centroid = np.mean(spectral_centroids)
            
            # Find dominant frequency peak
            freqs = librosa.fft_frequencies(sr=self.sample_rate)
            magnitude_mean = np.mean(magnitude, axis=1)
            peak_freq_idx = np.argmax(magnitude_mean)
            frequency_peak = freqs[peak_freq_idx]
            
            # Frequency range (90% energy bounds)
            cumulative_energy = np.cumsum(magnitude_mean)
            total_energy = cumulative_energy[-1]
            freq_low_idx = np.where(cumulative_energy >= 0.05 * total_energy)[0][0]
            freq_high_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0][0]
            frequency_range = (freqs[freq_low_idx], freqs[freq_high_idx])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_float)[0]
            zero_crossing_rate = np.mean(zcr)
            
            # Pitch stability (variation in fundamental frequency)
            pitches, magnitudes = librosa.piptrack(
                y=audio_float, sr=self.sample_rate, threshold=0.1
            )
            
            # Extract pitch track
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_track.append(pitch)
            
            if len(pitch_track) > 1:
                pitch_stability = 1.0 - (np.std(pitch_track) / (np.mean(pitch_track) + 1e-7))
                pitch_stability = max(0.0, min(1.0, pitch_stability))
            else:
                pitch_stability = 0.0
            
            # Onset sharpness (how quickly the sound starts)
            onset_frames = librosa.onset.onset_detect(
                y=audio_float, sr=self.sample_rate, units='frames'
            )
            
            if len(onset_frames) > 0:
                # Analyze the energy buildup around first onset
                onset_frame = onset_frames[0]
                pre_onset = max(0, onset_frame - 10)
                post_onset = min(len(spectral_centroids), onset_frame + 10)
                
                if post_onset > pre_onset:
                    energy_before = np.mean(spectral_centroids[pre_onset:onset_frame])
                    energy_after = np.mean(spectral_centroids[onset_frame:post_onset])
                    onset_sharpness = (energy_after - energy_before) / (energy_after + 1e-7)
                    onset_sharpness = max(0.0, min(1.0, onset_sharpness))
                else:
                    onset_sharpness = 0.5
            else:
                onset_sharpness = 0.5
            
            return BarkEvent(
                timestamp=timestamp,
                duration=duration,
                intensity=intensity,
                frequency_peak=frequency_peak,
                frequency_range=frequency_range,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zero_crossing_rate,
                pitch_stability=pitch_stability,
                onset_sharpness=onset_sharpness
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing bark audio: {e}")
            # Return minimal bark event
            return BarkEvent(
                timestamp=timestamp,
                duration=len(audio_data) / self.sample_rate,
                intensity=50.0,
                frequency_peak=1000.0,
                frequency_range=(500.0, 2000.0),
                spectral_centroid=1000.0,
                zero_crossing_rate=0.1,
                pitch_stability=0.5,
                onset_sharpness=0.5
            )
    
    def detect_bark_sequences(self, bark_events: List[BarkEvent]) -> List[BarkSequence]:
        """
        Group individual bark events into sequences and analyze patterns.
        
        Args:
            bark_events: List of individual bark events
            
        Returns:
            List of bark sequences with pattern analysis
        """
        if not bark_events:
            return []
        
        sequences = []
        current_sequence = [bark_events[0]]
        
        for i in range(1, len(bark_events)):
            prev_event = bark_events[i-1]
            curr_event = bark_events[i]
            
            # Calculate time gap between events
            time_gap = (curr_event.timestamp - prev_event.timestamp).total_seconds()
            
            if time_gap <= self.max_inter_bark_gap:
                # Continue current sequence
                current_sequence.append(curr_event)
            else:
                # End current sequence and start new one
                if len(current_sequence) >= 1:
                    sequence = self._analyze_sequence(current_sequence)
                    if sequence.total_duration >= self.min_sequence_duration:
                        sequences.append(sequence)
                
                current_sequence = [curr_event]
        
        # Don't forget the last sequence
        if len(current_sequence) >= 1:
            sequence = self._analyze_sequence(current_sequence)
            if sequence.total_duration >= self.min_sequence_duration:
                sequences.append(sequence)
        
        return sequences
    
    def _analyze_sequence(self, events: List[BarkEvent]) -> BarkSequence:
        """Analyze a sequence of bark events for patterns and characteristics."""
        if not events:
            raise ValueError("Cannot analyze empty sequence")
        
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        total_duration = (end_time - start_time).total_seconds() + events[-1].duration
        
        # Calculate inter-bark intervals
        inter_bark_intervals = []
        for i in range(1, len(events)):
            interval = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            inter_bark_intervals.append(interval)
        
        # Analyze rhythm regularity
        rhythm_regularity = self._calculate_rhythm_regularity(inter_bark_intervals)
        
        # Determine cadence type
        cadence_type = self._classify_cadence(events, inter_bark_intervals, total_duration)
        
        # Calculate persistence score
        persistence_score = self._calculate_persistence_score(events, total_duration)
        
        # Calculate annoyance score
        annoyance_score = self._calculate_annoyance_score(events, total_duration, start_time)
        
        return BarkSequence(
            events=events,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            inter_bark_intervals=inter_bark_intervals,
            rhythm_regularity=rhythm_regularity,
            cadence_type=cadence_type,
            annoyance_score=annoyance_score,
            persistence_score=persistence_score
        )
    
    def _calculate_rhythm_regularity(self, intervals: List[float]) -> float:
        """Calculate how regular the rhythm is (0=irregular, 1=perfectly regular)."""
        if len(intervals) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 0.0
        
        cv = std_interval / mean_interval
        
        # Convert to regularity score (lower CV = higher regularity)
        regularity = np.exp(-cv)
        
        return min(1.0, max(0.0, regularity))
    
    def _classify_cadence(self, events: List[BarkEvent], intervals: List[float], 
                         total_duration: float) -> str:
        """Classify the type of barking cadence."""
        num_barks = len(events)
        
        if num_barks == 1:
            return 'single'
        
        if total_duration <= 3.0 and num_barks >= 3:
            return 'burst'
        
        if len(intervals) > 0:
            avg_interval = np.mean(intervals)
            rhythm_regularity = self._calculate_rhythm_regularity(intervals)
            
            if rhythm_regularity > 0.7 and avg_interval < 2.0:
                return 'rhythmic'
            
            if total_duration > 30.0 and avg_interval < 10.0:
                return 'continuous'
        
        return 'irregular'
    
    def _calculate_persistence_score(self, events: List[BarkEvent], 
                                   total_duration: float) -> float:
        """Calculate how persistent/sustained the barking is (0-1)."""
        num_barks = len(events)
        
        # Base score from number of barks
        bark_score = min(1.0, num_barks / 20.0)  # Normalize to 20 barks
        
        # Duration score
        duration_score = min(1.0, total_duration / 60.0)  # Normalize to 1 minute
        
        # Intensity consistency score
        intensities = [event.intensity for event in events]
        intensity_cv = np.std(intensities) / (np.mean(intensities) + 1e-7)
        consistency_score = np.exp(-intensity_cv)
        
        # Combined persistence score
        persistence = (bark_score + duration_score + consistency_score) / 3.0
        
        return min(1.0, max(0.0, persistence))
    
    def _calculate_annoyance_score(self, events: List[BarkEvent], 
                                 total_duration: float, start_time: datetime) -> float:
        """Calculate annoyance score based on multiple factors (0-100)."""
        
        # Duration factor (longer = more annoying)
        duration_factor = min(1.0, total_duration / 120.0)  # Normalize to 2 minutes
        
        # Intensity factor (louder = more annoying)
        avg_intensity = np.mean([event.intensity for event in events])
        intensity_factor = avg_intensity / 100.0
        
        # Frequency factor (certain frequencies more annoying)
        avg_frequency = np.mean([event.frequency_peak for event in events])
        # Frequencies around 2-4 kHz are typically most annoying to humans
        if 2000 <= avg_frequency <= 4000:
            frequency_factor = 1.0
        elif 1000 <= avg_frequency <= 6000:
            frequency_factor = 0.8
        else:
            frequency_factor = 0.6
        
        # Persistence factor
        persistence_factor = self._calculate_persistence_score(events, total_duration)
        
        # Time of day factor
        hour = start_time.hour
        if 22 <= hour or hour <= 6:  # Night
            time_factor = self.time_multipliers['night']
        elif 6 <= hour <= 8:  # Early morning
            time_factor = self.time_multipliers['early_morning']
        elif 17 <= hour <= 22:  # Evening
            time_factor = self.time_multipliers['evening']
        else:  # Daytime
            time_factor = self.time_multipliers['daytime']
        
        # Weighted combination
        weights = self.annoyance_weights
        annoyance_score = (
            weights['duration'] * duration_factor +
            weights['intensity'] * intensity_factor +
            weights['frequency'] * frequency_factor +
            weights['persistence'] * persistence_factor
        ) * time_factor * weights['time_of_day']
        
        # Scale to 0-100
        annoyance_score = min(100.0, annoyance_score * 100.0)
        
        return annoyance_score
    
    def analyze_daily_patterns(self, sequences: List[BarkSequence]) -> Dict[str, Any]:
        """Analyze daily patterns and trends in barking behavior."""
        if not sequences:
            return {}
        
        # Group by hour of day
        hourly_counts = {}
        hourly_intensity = {}
        hourly_annoyance = {}
        
        for seq in sequences:
            hour = seq.start_time.hour
            
            hourly_counts[hour] = hourly_counts.get(hour, 0) + len(seq.events)
            
            avg_intensity = np.mean([event.intensity for event in seq.events])
            if hour not in hourly_intensity:
                hourly_intensity[hour] = []
            hourly_intensity[hour].append(avg_intensity)
            
            if hour not in hourly_annoyance:
                hourly_annoyance[hour] = []
            hourly_annoyance[hour].append(seq.annoyance_score)
        
        # Calculate averages
        for hour in hourly_intensity:
            hourly_intensity[hour] = np.mean(hourly_intensity[hour])
        
        for hour in hourly_annoyance:
            hourly_annoyance[hour] = np.mean(hourly_annoyance[hour])
        
        # Identify peak hours
        if hourly_counts:
            peak_hour = max(hourly_counts.keys(), key=lambda h: hourly_counts[h])
            peak_count = hourly_counts[peak_hour]
        else:
            peak_hour = None
            peak_count = 0
        
        # Calculate overall statistics
        total_sequences = len(sequences)
        total_barks = sum(len(seq.events) for seq in sequences)
        avg_sequence_duration = np.mean([seq.total_duration for seq in sequences])
        avg_annoyance = np.mean([seq.annoyance_score for seq in sequences])
        
        # Cadence type distribution
        cadence_counts = {}
        for seq in sequences:
            cadence_counts[seq.cadence_type] = cadence_counts.get(seq.cadence_type, 0) + 1
        
        return {
            'total_sequences': total_sequences,
            'total_barks': total_barks,
            'avg_sequence_duration': avg_sequence_duration,
            'avg_annoyance_score': avg_annoyance,
            'peak_hour': peak_hour,
            'peak_hour_count': peak_count,
            'hourly_distribution': hourly_counts,
            'hourly_intensity': hourly_intensity,
            'hourly_annoyance': hourly_annoyance,
            'cadence_distribution': cadence_counts,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_annoyance_assessment(self, sequences: List[BarkSequence]) -> Dict[str, Any]:
        """Generate a comprehensive annoyance assessment."""
        if not sequences:
            return {'assessment': 'No barking detected', 'score': 0}
        
        # Calculate overall metrics
        total_duration = sum(seq.total_duration for seq in sequences)
        avg_annoyance = np.mean([seq.annoyance_score for seq in sequences])
        max_annoyance = max(seq.annoyance_score for seq in sequences)
        
        # Count high-annoyance events
        high_annoyance_count = sum(1 for seq in sequences if seq.annoyance_score > 70)
        
        # Time distribution analysis
        night_sequences = [seq for seq in sequences 
                          if 22 <= seq.start_time.hour or seq.start_time.hour <= 6]
        night_annoyance = np.mean([seq.annoyance_score for seq in night_sequences]) if night_sequences else 0
        
        # Generate assessment
        if avg_annoyance >= 80:
            level = "Severe"
            description = "Extremely disruptive barking pattern with high intensity and persistence."
        elif avg_annoyance >= 60:
            level = "High"
            description = "Significantly annoying barking that likely constitutes a nuisance."
        elif avg_annoyance >= 40:
            level = "Moderate"
            description = "Noticeable barking that may be bothersome to neighbors."
        elif avg_annoyance >= 20:
            level = "Low"
            description = "Occasional barking with minimal disruption."
        else:
            level = "Minimal"
            description = "Very limited barking activity."
        
        return {
            'assessment_level': level,
            'description': description,
            'overall_score': avg_annoyance,
            'peak_score': max_annoyance,
            'total_duration_minutes': total_duration / 60.0,
            'high_annoyance_incidents': high_annoyance_count,
            'night_disturbance_score': night_annoyance,
            'sequence_count': len(sequences),
            'recommendation': self._get_recommendation(avg_annoyance, night_annoyance, high_annoyance_count)
        }
    
    def _get_recommendation(self, avg_annoyance: float, night_annoyance: float, 
                          high_incidents: int) -> str:
        """Generate recommendation based on annoyance analysis."""
        if avg_annoyance >= 70 or night_annoyance >= 60 or high_incidents >= 5:
            return ("Strong evidence of nuisance barking. Documentation is sufficient "
                   "for formal complaint to authorities.")
        elif avg_annoyance >= 50 or night_annoyance >= 40 or high_incidents >= 3:
            return ("Moderate nuisance barking detected. Continue monitoring and "
                   "consider informal resolution before formal complaint.")
        elif avg_annoyance >= 30:
            return ("Some disruptive barking noted. Monitor for patterns and "
                   "escalation before taking action.")
        else:
            return ("Limited barking activity. Normal pet behavior within "
                   "acceptable limits.")


# Example usage and testing
if __name__ == "__main__":
    import soundfile as sf
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    analyzer = AdvancedBarkAnalyzer()
    
    print("Testing advanced bark pattern analysis...")
    
    # Generate test bark events
    def generate_test_bark_events():
        """Generate realistic test bark events."""
        events = []
        base_time = datetime.now()
        
        # Simulate different types of barking patterns
        
        # 1. Single isolated bark
        events.append(BarkEvent(
            timestamp=base_time,
            duration=1.2,
            intensity=75.0,
            frequency_peak=1200.0,
            frequency_range=(800.0, 2000.0),
            spectral_centroid=1100.0,
            zero_crossing_rate=0.15,
            pitch_stability=0.8,
            onset_sharpness=0.9
        ))
        
        # 2. Burst of barks (rapid succession)
        burst_start = base_time + timedelta(minutes=5)
        for i in range(4):
            events.append(BarkEvent(
                timestamp=burst_start + timedelta(seconds=i*0.8),
                duration=0.8,
                intensity=85.0 + i*2,
                frequency_peak=1400.0 + i*50,
                frequency_range=(900.0, 2200.0),
                spectral_centroid=1300.0,
                zero_crossing_rate=0.18,
                pitch_stability=0.7,
                onset_sharpness=0.85
            ))
        
        # 3. Rhythmic barking (regular intervals)
        rhythmic_start = base_time + timedelta(minutes=15)
        for i in range(8):
            events.append(BarkEvent(
                timestamp=rhythmic_start + timedelta(seconds=i*2.5),
                duration=1.0,
                intensity=70.0,
                frequency_peak=1000.0,
                frequency_range=(700.0, 1800.0),
                spectral_centroid=950.0,
                zero_crossing_rate=0.12,
                pitch_stability=0.9,
                onset_sharpness=0.7
            ))
        
        # 4. Night barking (high annoyance)
        night_time = base_time.replace(hour=23, minute=30)
        for i in range(3):
            events.append(BarkEvent(
                timestamp=night_time + timedelta(seconds=i*10),
                duration=1.5,
                intensity=80.0,
                frequency_peak=1600.0,
                frequency_range=(1000.0, 2500.0),
                spectral_centroid=1400.0,
                zero_crossing_rate=0.16,
                pitch_stability=0.75,
                onset_sharpness=0.8
            ))
        
        return events
    
    # Test sequence detection and analysis
    test_events = generate_test_bark_events()
    sequences = analyzer.detect_bark_sequences(test_events)
    
    print(f"Detected {len(sequences)} bark sequences from {len(test_events)} individual barks")
    
    for i, seq in enumerate(sequences):
        print(f"\nSequence {i+1}:")
        print(f"  Type: {seq.cadence_type}")
        print(f"  Duration: {seq.total_duration:.1f} seconds")
        print(f"  Barks: {len(seq.events)}")
        print(f"  Rhythm regularity: {seq.rhythm_regularity:.3f}")
        print(f"  Persistence score: {seq.persistence_score:.3f}")
        print(f"  Annoyance score: {seq.annoyance_score:.1f}")
        print(f"  Time: {seq.start_time.strftime('%H:%M:%S')}")
    
    # Daily pattern analysis
    daily_patterns = analyzer.analyze_daily_patterns(sequences)
    print(f"\nDaily Pattern Analysis:")
    print(f"  Total sequences: {daily_patterns['total_sequences']}")
    print(f"  Total barks: {daily_patterns['total_barks']}")
    print(f"  Average sequence duration: {daily_patterns['avg_sequence_duration']:.1f}s")
    print(f"  Average annoyance score: {daily_patterns['avg_annoyance_score']:.1f}")
    
    # Annoyance assessment
    assessment = analyzer.get_annoyance_assessment(sequences)
    print(f"\nAnnoyance Assessment:")
    print(f"  Level: {assessment['assessment_level']}")
    print(f"  Description: {assessment['description']}")
    print(f"  Overall score: {assessment['overall_score']:.1f}")
    print(f"  Recommendation: {assessment['recommendation']}")
    
    print("\nAdvanced bark pattern analysis test completed.")