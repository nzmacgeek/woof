"""
Database module for storing bark events and evidence.
Manages SQLite database for logging bark incidents with timestamps, 
dog identification, and audio file references.
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib
from pathlib import Path


class BarkEventDatabase:
    """Manages database operations for bark event logging."""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Bark events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS bark_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        duration REAL NOT NULL,
                        dog_id TEXT,
                        confidence REAL,
                        audio_file_path TEXT,
                        audio_file_hash TEXT,
                        detection_method TEXT,
                        bark_intensity REAL,
                        background_noise_level REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        notes TEXT
                    )
                ''')
                
                # Dog profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS dog_profiles (
                        dog_id TEXT PRIMARY KEY,
                        first_detected TEXT NOT NULL,
                        last_detected TEXT,
                        total_bark_count INTEGER DEFAULT 0,
                        avg_bark_duration REAL,
                        typical_bark_times TEXT,  -- JSON array of typical times
                        estimated_size TEXT,      -- small/medium/large
                        bark_characteristics TEXT, -- JSON object
                        owner_address TEXT,
                        notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Daily summaries table for reports
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_summaries (
                        date TEXT PRIMARY KEY,
                        total_barks INTEGER,
                        total_duration REAL,
                        dogs_detected TEXT,  -- JSON array of dog IDs
                        peak_hours TEXT,     -- JSON array of hour:count pairs
                        longest_incident_duration REAL,
                        disturbance_score REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Evidence exports table for legal documentation
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS evidence_exports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        export_date TEXT NOT NULL,
                        date_range_start TEXT NOT NULL,
                        date_range_end TEXT NOT NULL,
                        export_type TEXT NOT NULL,  -- summary/detailed/audio
                        file_path TEXT NOT NULL,
                        exported_events_count INTEGER,
                        export_hash TEXT,
                        purpose TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_bark_events_timestamp 
                    ON bark_events(timestamp)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_bark_events_dog_id 
                    ON bark_events(dog_id)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_bark_events_date 
                    ON bark_events(date(timestamp))
                ''')
                
                conn.commit()
                self.logger.info(f"Database initialized at {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def log_bark_event(self, 
                      timestamp: datetime,
                      duration: float,
                      dog_id: Optional[str] = None,
                      confidence: Optional[float] = None,
                      audio_file_path: Optional[str] = None,
                      detection_method: str = "unknown",
                      bark_intensity: Optional[float] = None,
                      background_noise_level: Optional[float] = None,
                      notes: Optional[str] = None) -> int:
        """
        Log a bark event to the database.
        
        Args:
            timestamp: When the bark occurred
            duration: Duration in seconds
            dog_id: Identified dog (if any)
            confidence: Identification confidence (0-1)
            audio_file_path: Path to recorded audio file
            detection_method: How the bark was detected
            bark_intensity: Relative intensity of the bark
            background_noise_level: Background noise level
            notes: Additional notes
            
        Returns:
            Event ID from database
        """
        try:
            # Calculate audio file hash if file exists
            audio_file_hash = None
            if audio_file_path and os.path.exists(audio_file_path):
                audio_file_hash = self._calculate_file_hash(audio_file_path)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO bark_events (
                        timestamp, duration, dog_id, confidence, 
                        audio_file_path, audio_file_hash, detection_method,
                        bark_intensity, background_noise_level, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp.isoformat(),
                    duration,
                    dog_id,
                    confidence,
                    audio_file_path,
                    audio_file_hash,
                    detection_method,
                    bark_intensity,
                    background_noise_level,
                    notes
                ))
                
                event_id = cursor.lastrowid
                conn.commit()
                
                # Update dog profile
                if dog_id:
                    self._update_dog_profile(dog_id, timestamp, duration)
                
                self.logger.info(f"Logged bark event {event_id} for dog {dog_id} "
                               f"at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                return event_id
                
        except Exception as e:
            self.logger.error(f"Failed to log bark event: {e}")
            raise
    
    def _update_dog_profile(self, dog_id: str, timestamp: datetime, duration: float):
        """Update dog profile with new bark event."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if profile exists
                cursor.execute('SELECT * FROM dog_profiles WHERE dog_id = ?', (dog_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing profile
                    cursor.execute('''
                        UPDATE dog_profiles 
                        SET last_detected = ?,
                            total_bark_count = total_bark_count + 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE dog_id = ?
                    ''', (timestamp.isoformat(), dog_id))
                    
                    # Update average duration
                    cursor.execute('''
                        SELECT AVG(duration) FROM bark_events WHERE dog_id = ?
                    ''', (dog_id,))
                    avg_duration = cursor.fetchone()[0]
                    
                    cursor.execute('''
                        UPDATE dog_profiles 
                        SET avg_bark_duration = ?
                        WHERE dog_id = ?
                    ''', (avg_duration, dog_id))
                    
                else:
                    # Create new profile
                    cursor.execute('''
                        INSERT INTO dog_profiles (
                            dog_id, first_detected, last_detected, 
                            total_bark_count, avg_bark_duration
                        ) VALUES (?, ?, ?, 1, ?)
                    ''', (dog_id, timestamp.isoformat(), timestamp.isoformat(), duration))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update dog profile for {dog_id}: {e}")
    
    def get_events_by_date_range(self, 
                                start_date: datetime, 
                                end_date: datetime,
                                dog_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get bark events within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            dog_id: Filter by specific dog (optional)
            
        Returns:
            List of event dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM bark_events 
                    WHERE timestamp >= ? AND timestamp <= ?
                '''
                params = [start_date.isoformat(), end_date.isoformat()]
                
                if dog_id:
                    query += ' AND dog_id = ?'
                    params.append(dog_id)
                
                query += ' ORDER BY timestamp'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = dict(row)
                    # Convert timestamp back to datetime
                    event['timestamp'] = datetime.fromisoformat(event['timestamp'])
                    events.append(event)
                
                return events
                
        except Exception as e:
            self.logger.error(f"Failed to get events by date range: {e}")
            return []
    
    def get_daily_summary(self, date: datetime) -> Dict[str, Any]:
        """
        Get summary of bark events for a specific day.
        
        Args:
            date: Date to summarize
            
        Returns:
            Summary dictionary
        """
        try:
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            events = self.get_events_by_date_range(start_of_day, end_of_day)
            
            if not events:
                return {
                    'date': date.date().isoformat(),
                    'total_barks': 0,
                    'total_duration': 0,
                    'dogs_detected': [],
                    'peak_hours': {},
                    'longest_incident_duration': 0,
                    'disturbance_score': 0
                }
            
            # Calculate summary statistics
            total_barks = len(events)
            total_duration = sum(event['duration'] for event in events)
            dogs_detected = list(set(event['dog_id'] for event in events if event['dog_id']))
            longest_duration = max(event['duration'] for event in events)
            
            # Calculate hourly distribution
            hourly_counts = {}
            for event in events:
                hour = event['timestamp'].hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            
            # Calculate disturbance score (0-100)
            # Based on frequency, duration, and time of day
            disturbance_score = 0
            for event in events:
                hour = event['timestamp'].hour
                duration_factor = min(event['duration'] / 60.0, 1.0)  # Max 1 for 1+ minute
                
                # Weight by time of day (night hours have higher weight)
                if 22 <= hour or hour <= 6:  # 10 PM to 6 AM
                    time_weight = 2.0
                elif 7 <= hour <= 8 or 17 <= hour <= 21:  # Early morning/evening
                    time_weight = 1.5
                else:  # Daytime
                    time_weight = 1.0
                
                disturbance_score += duration_factor * time_weight
            
            # Normalize to 0-100 scale
            disturbance_score = min(disturbance_score * 5, 100)
            
            summary = {
                'date': date.date().isoformat(),
                'total_barks': total_barks,
                'total_duration': total_duration,
                'dogs_detected': dogs_detected,
                'peak_hours': hourly_counts,
                'longest_incident_duration': longest_duration,
                'disturbance_score': disturbance_score
            }
            
            # Cache summary in database
            self._cache_daily_summary(summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get daily summary: {e}")
            return {}
    
    def _cache_daily_summary(self, summary: Dict[str, Any]):
        """Cache daily summary in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_summaries (
                        date, total_barks, total_duration, dogs_detected,
                        peak_hours, longest_incident_duration, disturbance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    summary['date'],
                    summary['total_barks'],
                    summary['total_duration'],
                    json.dumps(summary['dogs_detected']),
                    json.dumps(summary['peak_hours']),
                    summary['longest_incident_duration'],
                    summary['disturbance_score']
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to cache daily summary: {e}")
    
    def generate_evidence_report(self, 
                                start_date: datetime,
                                end_date: datetime,
                                output_path: str,
                                report_type: str = "detailed",
                                purpose: str = "legal_evidence") -> str:
        """
        Generate evidence report for legal purposes.
        
        Args:
            start_date: Start of reporting period
            end_date: End of reporting period
            output_path: Where to save the report
            report_type: Type of report (summary/detailed/audio)
            purpose: Purpose of the report
            
        Returns:
            Path to generated report file
        """
        try:
            events = self.get_events_by_date_range(start_date, end_date)
            
            if report_type == "summary":
                report_content = self._generate_summary_report(events, start_date, end_date)
            elif report_type == "detailed":
                report_content = self._generate_detailed_report(events, start_date, end_date)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Write report to file
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(output_path)
            
            # Log export in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO evidence_exports (
                        export_date, date_range_start, date_range_end,
                        export_type, file_path, exported_events_count,
                        export_hash, purpose
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    start_date.isoformat(),
                    end_date.isoformat(),
                    report_type,
                    output_path,
                    len(events),
                    file_hash,
                    purpose
                ))
                conn.commit()
            
            self.logger.info(f"Generated {report_type} evidence report: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate evidence report: {e}")
            raise
    
    def _generate_summary_report(self, events: List[Dict], start_date: datetime, end_date: datetime) -> str:
        """Generate summary evidence report."""
        total_events = len(events)
        total_duration = sum(event['duration'] for event in events)
        dogs_involved = set(event['dog_id'] for event in events if event['dog_id'])
        
        # Calculate daily breakdown
        daily_counts = {}
        for event in events:
            date_key = event['timestamp'].date().isoformat()
            daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
        
        # Generate report
        report = f"""
BARK DISTURBANCE EVIDENCE REPORT - SUMMARY
==========================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Reporting Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}

SUMMARY STATISTICS
-----------------
Total Bark Events: {total_events}
Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)
Dogs Identified: {len(dogs_involved)}
Average Events per Day: {total_events / max(1, (end_date - start_date).days):.1f}

DAILY BREAKDOWN
--------------
"""
        
        for date, count in sorted(daily_counts.items()):
            report += f"{date}: {count} events\n"
        
        if dogs_involved:
            report += f"\nDOGS IDENTIFIED\n--------------\n"
            for dog_id in sorted(dogs_involved):
                dog_events = [e for e in events if e['dog_id'] == dog_id]
                report += f"{dog_id}: {len(dog_events)} events\n"
        
        report += f"\n\nThis report contains evidence of noise disturbance caused by dog barking.\n"
        report += f"All timestamps are in local time zone.\n"
        report += f"Audio recordings may be available for verification.\n"
        
        return report
    
    def _generate_detailed_report(self, events: List[Dict], start_date: datetime, end_date: datetime) -> str:
        """Generate detailed evidence report."""
        report = f"""
BARK DISTURBANCE EVIDENCE REPORT - DETAILED
===========================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Reporting Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}

DETAILED EVENT LOG
-----------------
"""
        
        for i, event in enumerate(events, 1):
            report += f"\nEvent #{i}\n"
            report += f"Date/Time: {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"Duration: {event['duration']:.1f} seconds\n"
            
            if event['dog_id']:
                report += f"Dog ID: {event['dog_id']}\n"
                if event['confidence']:
                    report += f"Identification Confidence: {event['confidence']:.2f}\n"
            
            if event['bark_intensity']:
                report += f"Bark Intensity: {event['bark_intensity']:.2f}\n"
            
            if event['audio_file_path']:
                report += f"Audio File: {event['audio_file_path']}\n"
                if event['audio_file_hash']:
                    report += f"File Hash (MD5): {event['audio_file_hash']}\n"
            
            if event['notes']:
                report += f"Notes: {event['notes']}\n"
            
            report += "-" * 40 + "\n"
        
        return report
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of a file for integrity verification."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total events
                cursor.execute('SELECT COUNT(*) FROM bark_events')
                total_events = cursor.fetchone()[0]
                
                # Total dogs
                cursor.execute('SELECT COUNT(*) FROM dog_profiles')
                total_dogs = cursor.fetchone()[0]
                
                # Date range
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM bark_events')
                date_range = cursor.fetchone()
                
                # Recent activity (last 7 days)
                seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute('SELECT COUNT(*) FROM bark_events WHERE timestamp >= ?', (seven_days_ago,))
                recent_events = cursor.fetchone()[0]
                
                return {
                    'total_events': total_events,
                    'total_dogs': total_dogs,
                    'earliest_event': date_range[0],
                    'latest_event': date_range[1],
                    'recent_events_7_days': recent_events,
                    'database_file_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    print(f"Testing database operations with {db_path}")
    
    try:
        # Create database instance
        db = BarkEventDatabase(db_path)
        
        # Log some test events
        now = datetime.now()
        
        # Log several bark events
        for i in range(5):
            event_time = now - timedelta(hours=i)
            event_id = db.log_bark_event(
                timestamp=event_time,
                duration=1.5 + i * 0.5,
                dog_id=f"dog_{i % 2}",  # Two different dogs
                confidence=0.8 + i * 0.05,
                detection_method="ml_classifier",
                bark_intensity=0.7 + i * 0.1,
                notes=f"Test event {i}"
            )
            print(f"Logged event {event_id}")
        
        # Get events from last day
        yesterday = now - timedelta(days=1)
        events = db.get_events_by_date_range(yesterday, now)
        print(f"Found {len(events)} events in last day")
        
        # Generate daily summary
        summary = db.get_daily_summary(now)
        print(f"Daily summary: {summary['total_barks']} barks, "
              f"disturbance score: {summary['disturbance_score']:.1f}")
        
        # Generate evidence report
        report_path = tempfile.mktemp(suffix='.txt')
        db.generate_evidence_report(
            start_date=yesterday,
            end_date=now,
            output_path=report_path,
            report_type="summary"
        )
        print(f"Generated evidence report: {report_path}")
        
        # Show database stats
        stats = db.get_database_stats()
        print(f"Database stats: {stats}")
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        if 'report_path' in locals() and os.path.exists(report_path):
            os.unlink(report_path)
    
    print("Database testing completed.")