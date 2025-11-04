"""
Bark detection module using audio feature extraction and machine learning.
Analyzes audio patterns to identify dog barking events.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class BarkDetector:
    """Machine learning-based bark detection system."""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 model_path: Optional[str] = None):
        """
        Initialize bark detector.
        
        Args:
            sample_rate: Audio sampling rate
            frame_length: Frame size for spectral analysis
            hop_length: Hop size for spectral analysis
            model_path: Path to pre-trained model file
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # Machine learning components
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_chroma = 12
        self.n_spectral_features = 7
        
        # Bark detection thresholds
        self.bark_probability_threshold = 0.7
        self.min_bark_duration = 0.1  # seconds
        self.max_bark_duration = 3.0  # seconds
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract audio features for bark detection.
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            Feature vector as numpy array
        """
        if len(audio_data) == 0:
            return np.array([])
        
        features = []
        
        try:
            # Convert to float if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 1. MFCC features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(
                y=audio_float,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            
            # Statistical features from MFCCs
            features.extend([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),
                np.min(mfccs, axis=1)
            ])
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_float, sr=self.sample_rate
            )[0]
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_float, sr=self.sample_rate
            )[0]
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_float, sr=self.sample_rate
            )[0]
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                audio_float
            )[0]
            
            # Add spectral feature statistics
            for feature_array in [spectral_centroids, spectral_rolloff, 
                                spectral_bandwidth, zero_crossing_rate]:
                features.extend([
                    np.mean(feature_array),
                    np.std(feature_array),
                    np.max(feature_array),
                    np.min(feature_array)
                ])
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_float, sr=self.sample_rate
            )
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1)
            ])
            
            # 4. Temporal features
            # RMS energy
            rms = librosa.feature.rms(y=audio_float)[0]
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.max(rms)
            ])
            
            # Duration and tempo-related features
            duration = len(audio_float) / self.sample_rate
            features.append(duration)
            
            # 5. Frequency domain features
            # Fundamental frequency estimation
            pitches, magnitudes = librosa.piptrack(
                y=audio_float, sr=self.sample_rate
            )
            
            # Extract dominant frequencies
            dominant_freqs = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    dominant_freqs.append(pitch)
            
            if dominant_freqs:
                features.extend([
                    np.mean(dominant_freqs),
                    np.std(dominant_freqs),
                    np.max(dominant_freqs),
                    np.min(dominant_freqs)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Flatten all features
            feature_vector = np.concatenate([
                np.array(f).flatten() for f in features
            ])
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.array([])
    
    def is_bark_like(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Quick heuristic check for bark-like characteristics.
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            (is_bark_like, confidence_score)
        """
        if len(audio_data) == 0:
            return False, 0.0
        
        try:
            # Convert to float
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Check duration
            duration = len(audio_float) / self.sample_rate
            if duration < self.min_bark_duration or duration > self.max_bark_duration:
                return False, 0.0
            
            # Check energy level
            rms = np.sqrt(np.mean(audio_float ** 2))
            if rms < 0.01:  # Too quiet
                return False, 0.0
            
            # Check for impulsive nature (high peak-to-average ratio)
            peak_to_avg = np.max(np.abs(audio_float)) / (rms + 1e-7)
            if peak_to_avg < 2.0:  # Not impulsive enough
                return False, 0.5
            
            # Check spectral characteristics
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_float, sr=self.sample_rate
            )[0]
            
            avg_centroid = np.mean(spectral_centroid)
            
            # Dog barks typically have energy in mid-high frequencies (500-4000 Hz)
            if avg_centroid < 500 or avg_centroid > 8000:
                return False, 0.3
            
            # Check for harmonic content (barks usually have harmonics)
            zero_crossings = librosa.feature.zero_crossing_rate(audio_float)[0]
            avg_zcr = np.mean(zero_crossings)
            
            # Reasonable zero crossing rate for barks
            if avg_zcr < 0.01 or avg_zcr > 0.3:
                return False, 0.4
            
            # Calculate confidence based on how well it matches bark characteristics
            confidence = 0.0
            
            # Duration score
            if 0.2 <= duration <= 1.5:
                confidence += 0.3
            
            # Energy score
            if 0.05 <= rms <= 0.8:
                confidence += 0.2
            
            # Spectral centroid score
            if 1000 <= avg_centroid <= 4000:
                confidence += 0.3
            
            # Peak-to-average score
            if 3.0 <= peak_to_avg <= 8.0:
                confidence += 0.2
            
            return confidence > 0.6, confidence
            
        except Exception as e:
            self.logger.error(f"Error in bark heuristic check: {e}")
            return False, 0.0
    
    def predict_bark(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if audio contains barking using trained model.
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            (is_bark, probability)
        """
        if not self.is_trained:
            # Fall back to heuristic method
            return self.is_bark_like(audio_data)
        
        features = self.extract_features(audio_data)
        
        if len(features) == 0:
            return False, 0.0
        
        try:
            # Reshape for prediction
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction probability
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            bark_probability = probabilities[1] if len(probabilities) > 1 else 0.0
            
            is_bark = bark_probability >= self.bark_probability_threshold
            
            return is_bark, bark_probability
            
        except Exception as e:
            self.logger.error(f"Error in bark prediction: {e}")
            return False, 0.0
    
    def train_model(self, 
                   bark_samples: List[np.ndarray],
                   non_bark_samples: List[np.ndarray],
                   save_path: Optional[str] = None) -> Dict:
        """
        Train the bark detection model.
        
        Args:
            bark_samples: List of audio samples containing barks
            non_bark_samples: List of audio samples without barks
            save_path: Path to save trained model
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Starting model training...")
        
        # Extract features from all samples
        bark_features = []
        non_bark_features = []
        
        for sample in bark_samples:
            features = self.extract_features(sample)
            if len(features) > 0:
                bark_features.append(features)
        
        for sample in non_bark_samples:
            features = self.extract_features(sample)
            if len(features) > 0:
                non_bark_features.append(features)
        
        if len(bark_features) == 0 or len(non_bark_features) == 0:
            raise ValueError("Need both bark and non-bark samples for training")
        
        # Ensure all feature vectors have the same length
        min_length = min([len(f) for f in bark_features + non_bark_features])
        bark_features = [f[:min_length] for f in bark_features]
        non_bark_features = [f[:min_length] for f in non_bark_features]
        
        # Combine features and labels
        X = np.array(bark_features + non_bark_features)
        y = np.array([1] * len(bark_features) + [0] * len(non_bark_features))
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        # Predictions for detailed evaluation
        y_pred = self.classifier.predict(X_test_scaled)
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'n_bark_samples': len(bark_features),
            'n_non_bark_samples': len(non_bark_features),
            'feature_dimension': X.shape[1]
        }
        
        self.is_trained = True
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        self.logger.info(f"Model training completed. Test accuracy: {test_score:.3f}")
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'sample_rate': self.sample_rate,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'bark_probability_threshold': self.bark_probability_threshold,
            'min_bark_duration': self.min_bark_duration,
            'max_bark_duration': self.max_bark_duration,
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            
            # Update parameters if they exist in saved model
            for param in ['sample_rate', 'frame_length', 'hop_length',
                         'bark_probability_threshold', 'min_bark_duration',
                         'max_bark_duration']:
                if param in model_data:
                    setattr(self, param, model_data[param])
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {filepath}: {e}")
            raise
    
    def analyze_audio_file(self, filepath: str) -> Dict:
        """
        Analyze an audio file for bark detection.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(filepath, sr=self.sample_rate)
            
            # Predict if it contains barking
            is_bark, probability = self.predict_bark(audio_data)
            
            # Extract additional analysis
            duration = len(audio_data) / self.sample_rate
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Get spectral features for analysis
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio_data, sr=sr
            ))
            
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(
                audio_data
            ))
            
            results = {
                'filepath': filepath,
                'is_bark': is_bark,
                'bark_probability': probability,
                'duration': duration,
                'rms_energy': rms_energy,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zero_crossing_rate,
                'sample_rate': sr,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio file {filepath}: {e}")
            return {
                'filepath': filepath,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    import soundfile as sf
    import tempfile
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bark detector
    detector = BarkDetector()
    
    # Generate some test audio samples
    sample_rate = 44100
    duration = 1.0  # 1 second
    
    # Simulate a bark-like sound (short bursts at different frequencies)
    def generate_bark_sample():
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Multiple frequency components typical of dog barks
        bark = (0.3 * np.sin(2 * np.pi * 800 * t) * 
                np.exp(-t * 3) * (np.sin(2 * np.pi * 10 * t) > 0) +
                0.2 * np.sin(2 * np.pi * 1200 * t) * 
                np.exp(-t * 4) * (np.sin(2 * np.pi * 8 * t) > 0))
        
        # Add some noise
        bark += 0.05 * np.random.normal(0, 1, len(bark))
        
        return bark
    
    # Simulate non-bark sounds
    def generate_non_bark_sample():
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Random choice of different non-bark sounds
        choice = np.random.randint(0, 3)
        
        if choice == 0:
            # Low-frequency rumble
            sound = 0.1 * np.sin(2 * np.pi * 60 * t)
        elif choice == 1:
            # High-frequency whistle
            sound = 0.1 * np.sin(2 * np.pi * 5000 * t)
        else:
            # White noise
            sound = 0.05 * np.random.normal(0, 1, len(t))
        
        return sound
    
    print("Testing bark detection...")
    
    # Test heuristic detection
    bark_sample = generate_bark_sample()
    non_bark_sample = generate_non_bark_sample()
    
    is_bark1, conf1 = detector.is_bark_like(bark_sample)
    is_bark2, conf2 = detector.is_bark_like(non_bark_sample)
    
    print(f"Bark sample: is_bark={is_bark1}, confidence={conf1:.3f}")
    print(f"Non-bark sample: is_bark={is_bark2}, confidence={conf2:.3f}")
    
    # Test feature extraction
    features = detector.extract_features(bark_sample)
    print(f"Extracted {len(features)} features from bark sample")
    
    # Create training data
    print("\nGenerating training data...")
    bark_samples = [generate_bark_sample() for _ in range(50)]
    non_bark_samples = [generate_non_bark_sample() for _ in range(50)]
    
    # Train model
    print("Training model...")
    try:
        results = detector.train_model(bark_samples, non_bark_samples)
        print(f"Training completed!")
        print(f"Train accuracy: {results['train_accuracy']:.3f}")
        print(f"Test accuracy: {results['test_accuracy']:.3f}")
        
        # Test trained model
        is_bark3, prob3 = detector.predict_bark(bark_sample)
        is_bark4, prob4 = detector.predict_bark(non_bark_sample)
        
        print(f"\nTrained model results:")
        print(f"Bark sample: is_bark={is_bark3}, probability={prob3:.3f}")
        print(f"Non-bark sample: is_bark={is_bark4}, probability={prob4:.3f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
    
    print("\nBark detection test completed.")