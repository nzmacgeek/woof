"""
Dog identification module using audio fingerprinting and clustering.
Distinguishes between different dogs based on their barking patterns.
"""

import numpy as np
import librosa
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import hashlib


class DogIdentifier:
    """Identifies individual dogs based on their barking patterns."""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_mfcc: int = 13,
                 max_dogs: int = 10,
                 clustering_method: str = 'kmeans'):
        """
        Initialize dog identifier.
        
        Args:
            sample_rate: Audio sampling rate
            n_mfcc: Number of MFCC coefficients for fingerprinting
            max_dogs: Maximum number of dogs to identify
            clustering_method: 'kmeans', 'dbscan', or 'gmm'
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_dogs = max_dogs
        self.clustering_method = clustering_method
        
        # Clustering model
        self.clusterer = None
        self.scaler = StandardScaler()
        
        # Dog database
        self.dog_profiles = {}  # dog_id -> profile data
        self.bark_fingerprints = []  # List of all fingerprints
        self.bark_labels = []  # Corresponding dog IDs
        
        # Model state
        self.is_trained = False
        self.n_clusters = 0
        
        self.logger = logging.getLogger(__name__)
    
    def extract_fingerprint(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract audio fingerprint for dog identification.
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            Audio fingerprint as feature vector
        """
        if len(audio_data) == 0:
            return np.array([])
        
        try:
            # Convert to float if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            features = []
            
            # 1. MFCC features - primary identifier
            mfccs = librosa.feature.mfcc(
                y=audio_float,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
            # Statistical features from MFCCs
            features.extend([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.median(mfccs, axis=1),
                np.percentile(mfccs, 25, axis=1),
                np.percentile(mfccs, 75, axis=1)
            ])
            
            # 2. Pitch and formant characteristics
            # Fundamental frequency
            pitches, magnitudes = librosa.piptrack(
                y=audio_float, sr=self.sample_rate
            )
            
            # Extract pitch statistics
            valid_pitches = pitches[pitches > 0]
            if len(valid_pitches) > 0:
                pitch_features = [
                    np.mean(valid_pitches),
                    np.std(valid_pitches),
                    np.median(valid_pitches),
                    np.min(valid_pitches),
                    np.max(valid_pitches)
                ]
            else:
                pitch_features = [0.0] * 5
            
            features.extend(pitch_features)
            
            # 3. Spectral shape features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_float, sr=self.sample_rate
            )[0]
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_float, sr=self.sample_rate
            )[0]
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_float, sr=self.sample_rate
            )[0]
            
            # Add spectral statistics
            for spec_feature in [spectral_centroids, spectral_bandwidth, spectral_rolloff]:
                features.extend([
                    np.mean(spec_feature),
                    np.std(spec_feature),
                    np.median(spec_feature)
                ])
            
            # 4. Harmonic characteristics
            harmonic, percussive = librosa.effects.hpss(audio_float)
            
            # Harmonic-to-percussive ratio
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            
            if percussive_energy > 0:
                hpr = harmonic_energy / percussive_energy
            else:
                hpr = 0.0
            
            features.append(hpr)
            
            # 5. Temporal characteristics
            # Zero crossing rate variations
            zcr = librosa.feature.zero_crossing_rate(audio_float)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.var(zcr)
            ])
            
            # Onset characteristics
            onset_frames = librosa.onset.onset_detect(
                y=audio_float, sr=self.sample_rate
            )
            
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
                onset_intervals = np.diff(onset_times)
                
                features.extend([
                    len(onset_frames),  # Number of onsets
                    np.mean(onset_intervals) if len(onset_intervals) > 0 else 0,
                    np.std(onset_intervals) if len(onset_intervals) > 0 else 0
                ])
            else:
                features.extend([0, 0, 0])
            
            # 6. Vocal tract characteristics (formants approximation)
            # Use LPC (Linear Predictive Coding) coefficients
            try:
                # Compute LPC coefficients
                lpc_order = 10
                windowed = audio_float * np.hamming(len(audio_float))
                
                # Simple autocorrelation-based LPC
                autocorr = np.correlate(windowed, windowed, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                if len(autocorr) > lpc_order:
                    # Extract first few autocorrelation coefficients as features
                    features.extend(autocorr[1:lpc_order+1] / (autocorr[0] + 1e-7))
                else:
                    features.extend([0.0] * lpc_order)
                    
            except Exception:
                features.extend([0.0] * 10)
            
            # Flatten and return fingerprint
            fingerprint = np.concatenate([
                np.array(f).flatten() for f in features
            ])
            
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Error extracting fingerprint: {e}")
            return np.array([])
    
    def add_bark_sample(self, audio_data: np.ndarray, dog_id: Optional[str] = None) -> str:
        """
        Add a bark sample to the database.
        
        Args:
            audio_data: Raw audio samples
            dog_id: Known dog ID, or None for unknown dog
            
        Returns:
            Assigned dog ID
        """
        fingerprint = self.extract_fingerprint(audio_data)
        
        if len(fingerprint) == 0:
            raise ValueError("Could not extract fingerprint from audio")
        
        # Generate automatic ID if not provided
        if dog_id is None:
            # Create ID based on fingerprint hash
            fingerprint_hash = hashlib.md5(fingerprint.tobytes()).hexdigest()[:8]
            dog_id = f"dog_{fingerprint_hash}"
        
        # Add to database
        self.bark_fingerprints.append(fingerprint)
        self.bark_labels.append(dog_id)
        
        # Update dog profile
        if dog_id not in self.dog_profiles:
            self.dog_profiles[dog_id] = {
                'id': dog_id,
                'first_heard': datetime.now().isoformat(),
                'bark_count': 0,
                'avg_fingerprint': fingerprint.copy(),
                'fingerprint_variance': np.zeros_like(fingerprint)
            }
        
        profile = self.dog_profiles[dog_id]
        profile['bark_count'] += 1
        profile['last_heard'] = datetime.now().isoformat()
        
        # Update running average of fingerprint
        n = profile['bark_count']
        old_avg = profile['avg_fingerprint']
        new_avg = old_avg + (fingerprint - old_avg) / n
        
        # Update variance estimate
        if n > 1:
            profile['fingerprint_variance'] = (
                (n - 2) * profile['fingerprint_variance'] + 
                (fingerprint - old_avg) * (fingerprint - new_avg)
            ) / (n - 1)
        
        profile['avg_fingerprint'] = new_avg
        
        self.logger.info(f"Added bark sample for {dog_id} (total: {n} samples)")
        
        return dog_id
    
    def train_clustering(self) -> Dict[str, Any]:
        """Train clustering model on collected fingerprints."""
        if len(self.bark_fingerprints) < 2:
            raise ValueError("Need at least 2 bark samples for clustering")
        
        # Convert to numpy array
        X = np.array(self.bark_fingerprints)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        if self.clustering_method == 'kmeans':
            n_clusters = min(len(set(self.bark_labels)), self.max_dogs, len(X) // 2)
            n_clusters = max(2, n_clusters)
            
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            
        elif self.clustering_method == 'dbscan':
            # Use DBSCAN for automatic cluster detection
            self.clusterer = DBSCAN(
                eps=0.5,
                min_samples=2
            )
            
        elif self.clustering_method == 'gmm':
            n_clusters = min(len(set(self.bark_labels)), self.max_dogs, len(X) // 2)
            n_clusters = max(2, n_clusters)
            
            self.clusterer = GaussianMixture(
                n_components=n_clusters,
                random_state=42
            )
        
        # Fit clustering model
        cluster_labels = self.clusterer.fit_predict(X_scaled)
        
        # Calculate clustering quality
        if len(set(cluster_labels)) > 1:
            silhouette = silhouette_score(X_scaled, cluster_labels)
        else:
            silhouette = 0.0
        
        self.n_clusters = len(set(cluster_labels))
        self.is_trained = True
        
        # Update dog profiles with cluster information
        unique_labels = set(cluster_labels)
        for i, label in enumerate(cluster_labels):
            dog_id = self.bark_labels[i]
            if dog_id in self.dog_profiles:
                self.dog_profiles[dog_id]['cluster_id'] = int(label)
        
        results = {
            'n_samples': len(X),
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette,
            'clustering_method': self.clustering_method,
            'unique_dogs': len(set(self.bark_labels))
        }
        
        self.logger.info(f"Clustering completed: {self.n_clusters} clusters, "
                        f"silhouette score: {silhouette:.3f}")
        
        return results
    
    def identify_dog(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Identify which dog is barking.
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            (dog_id, confidence_score)
        """
        fingerprint = self.extract_fingerprint(audio_data)
        
        if len(fingerprint) == 0:
            return "unknown", 0.0
        
        if not self.is_trained or len(self.dog_profiles) == 0:
            # No training data available
            return "unknown", 0.0
        
        # Scale fingerprint
        fingerprint_scaled = self.scaler.transform(fingerprint.reshape(1, -1))[0]
        
        # Find closest match using profile averages
        best_match = "unknown"
        best_distance = float('inf')
        best_confidence = 0.0
        
        for dog_id, profile in self.dog_profiles.items():
            if 'avg_fingerprint' not in profile:
                continue
            
            # Scale the profile's average fingerprint
            try:
                avg_fp_scaled = self.scaler.transform(
                    profile['avg_fingerprint'].reshape(1, -1)
                )[0]
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(fingerprint_scaled - avg_fp_scaled)
                
                # Calculate confidence based on distance and variance
                if 'fingerprint_variance' in profile:
                    variance = np.mean(profile['fingerprint_variance'])
                    # Confidence decreases with distance, increases with lower variance
                    confidence = np.exp(-distance) / (1 + variance)
                else:
                    confidence = np.exp(-distance)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = dog_id
                    best_confidence = confidence
                    
            except Exception as e:
                self.logger.warning(f"Error comparing with {dog_id}: {e}")
                continue
        
        # Normalize confidence to 0-1 range
        best_confidence = min(1.0, max(0.0, best_confidence))
        
        # If confidence is too low, return unknown
        if best_confidence < 0.3:
            return "unknown", best_confidence
        
        return best_match, best_confidence
    
    def get_dog_statistics(self) -> Dict[str, Any]:
        """Get statistics about identified dogs."""
        stats = {
            'total_dogs': len(self.dog_profiles),
            'total_barks': len(self.bark_fingerprints),
            'is_trained': self.is_trained,
            'dogs': {}
        }
        
        for dog_id, profile in self.dog_profiles.items():
            stats['dogs'][dog_id] = {
                'bark_count': profile['bark_count'],
                'first_heard': profile.get('first_heard'),
                'last_heard': profile.get('last_heard'),
                'cluster_id': profile.get('cluster_id', -1)
            }
        
        return stats
    
    def save_model(self, filepath: str):
        """Save dog identification model and profiles."""
        model_data = {
            'clusterer': self.clusterer,
            'scaler': self.scaler,
            'dog_profiles': self.dog_profiles,
            'bark_fingerprints': self.bark_fingerprints,
            'bark_labels': self.bark_labels,
            'is_trained': self.is_trained,
            'n_clusters': self.n_clusters,
            'sample_rate': self.sample_rate,
            'n_mfcc': self.n_mfcc,
            'max_dogs': self.max_dogs,
            'clustering_method': self.clustering_method,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Dog identification model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load dog identification model and profiles."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.clusterer = model_data.get('clusterer')
            self.scaler = model_data.get('scaler', StandardScaler())
            self.dog_profiles = model_data.get('dog_profiles', {})
            self.bark_fingerprints = model_data.get('bark_fingerprints', [])
            self.bark_labels = model_data.get('bark_labels', [])
            self.is_trained = model_data.get('is_trained', False)
            self.n_clusters = model_data.get('n_clusters', 0)
            
            # Update parameters if available
            for param in ['sample_rate', 'n_mfcc', 'max_dogs', 'clustering_method']:
                if param in model_data:
                    setattr(self, param, model_data[param])
            
            self.logger.info(f"Dog identification model loaded from {filepath}")
            self.logger.info(f"Loaded {len(self.dog_profiles)} dog profiles, "
                           f"{len(self.bark_fingerprints)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {filepath}: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    import soundfile as sf
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create dog identifier
    identifier = DogIdentifier(clustering_method='kmeans')
    
    print("Testing dog identification...")
    
    # Generate different types of bark samples
    def generate_dog_bark(dog_type: str, duration: float = 1.0):
        """Generate synthetic bark for different dog types."""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if dog_type == "small_dog":
            # Higher pitch, shorter duration
            bark = (0.4 * np.sin(2 * np.pi * 1200 * t) * 
                   np.exp(-t * 8) * (np.sin(2 * np.pi * 15 * t) > 0) +
                   0.2 * np.sin(2 * np.pi * 2000 * t) * 
                   np.exp(-t * 10) * (np.sin(2 * np.pi * 12 * t) > 0))
                   
        elif dog_type == "large_dog":
            # Lower pitch, longer duration
            bark = (0.5 * np.sin(2 * np.pi * 400 * t) * 
                   np.exp(-t * 3) * (np.sin(2 * np.pi * 6 * t) > 0) +
                   0.3 * np.sin(2 * np.pi * 600 * t) * 
                   np.exp(-t * 4) * (np.sin(2 * np.pi * 5 * t) > 0))
                   
        else:  # medium_dog
            # Medium pitch
            bark = (0.4 * np.sin(2 * np.pi * 800 * t) * 
                   np.exp(-t * 5) * (np.sin(2 * np.pi * 10 * t) > 0) +
                   0.2 * np.sin(2 * np.pi * 1200 * t) * 
                   np.exp(-t * 6) * (np.sin(2 * np.pi * 8 * t) > 0))
        
        # Add some random variation
        bark += 0.05 * np.random.normal(0, 1, len(bark))
        
        return bark
    
    # Generate training samples
    print("Generating training samples...")
    
    # Add samples for different dogs
    for i in range(5):
        # Small dog samples
        bark = generate_dog_bark("small_dog")
        identifier.add_bark_sample(bark, "small_dog_1")
        
        # Large dog samples  
        bark = generate_dog_bark("large_dog")
        identifier.add_bark_sample(bark, "large_dog_1")
        
        # Medium dog samples
        bark = generate_dog_bark("medium_dog")
        identifier.add_bark_sample(bark, "medium_dog_1")
    
    # Add more samples for a second small dog
    for i in range(3):
        bark = generate_dog_bark("small_dog")
        # Slightly modify the bark characteristics
        bark = bark * 0.8 + 0.1 * np.random.normal(0, 1, len(bark))
        identifier.add_bark_sample(bark, "small_dog_2")
    
    print(f"Added samples for {len(identifier.dog_profiles)} dogs")
    
    # Train clustering
    print("Training clustering model...")
    try:
        results = identifier.train_clustering()
        print(f"Training completed!")
        print(f"Clusters: {results['n_clusters']}")
        print(f"Silhouette score: {results['silhouette_score']:.3f}")
        
        # Test identification
        print("\nTesting identification...")
        
        # Test with known dog types
        test_bark_small = generate_dog_bark("small_dog")
        test_bark_large = generate_dog_bark("large_dog")
        
        dog_id1, conf1 = identifier.identify_dog(test_bark_small)
        dog_id2, conf2 = identifier.identify_dog(test_bark_large)
        
        print(f"Small dog test: identified as {dog_id1} (confidence: {conf1:.3f})")
        print(f"Large dog test: identified as {dog_id2} (confidence: {conf2:.3f})")
        
        # Show statistics
        stats = identifier.get_dog_statistics()
        print(f"\nDog statistics:")
        for dog_id, dog_stats in stats['dogs'].items():
            print(f"  {dog_id}: {dog_stats['bark_count']} barks, cluster {dog_stats['cluster_id']}")
        
    except Exception as e:
        print(f"Training failed: {e}")
    
    print("\nDog identification test completed.")