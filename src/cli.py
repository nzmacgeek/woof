"""
Command-line interface and utilities for the bark monitoring system.
Provides commands for training, analysis, reporting, and system management.
"""

import click
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import BarkMonitor, DEFAULT_CONFIG
from database import BarkEventDatabase
from bark_detector import BarkDetector
from dog_identifier import DogIdentifier


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Dog Bark Detection and Monitoring System."""
    ctx.ensure_object(dict)
    
    # Load configuration
    ctx.obj['config'] = DEFAULT_CONFIG.copy()
    
    if config and os.path.exists(config):
        with open(config, 'r') as f:
            user_config = json.load(f)
        ctx.obj['config'].update(user_config)
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--device', '-d', type=int, help='Audio device index to use')
@click.option('--simulate', is_flag=True, help='Run in simulation mode')
@click.pass_context
def monitor(ctx, device, simulate):
    """Start bark monitoring."""
    config = ctx.obj['config']
    
    click.echo("Starting Dog Bark Monitor...")
    click.echo(f"Configuration: {config.get('log_level')} logging, "
               f"saving to {config.get('database_path')}")
    
    monitor = BarkMonitor(config)
    
    try:
        if simulate:
            click.echo("Running in simulation mode...")
            monitor._run_simulation_mode()
        else:
            monitor.start(device_index=device)
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped by user")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list_devices(ctx):
    """List available audio input devices."""
    config = ctx.obj['config']
    
    try:
        from audio_capture import AudioCapture
        capture = AudioCapture()
        devices = capture.list_audio_devices()
        
        if not devices:
            click.echo("No audio input devices found.")
            return
        
        click.echo("Available audio input devices:")
        for device in devices:
            click.echo(f"  {device['index']}: {device['name']} "
                      f"({device['channels']} channels, {device['sample_rate']} Hz)")
    
    except Exception as e:
        click.echo(f"Error listing devices: {e}", err=True)


@cli.group()
def train():
    """Training commands for bark detection and dog identification."""
    pass


@train.command()
@click.argument('bark_samples_dir', type=click.Path(exists=True))
@click.argument('non_bark_samples_dir', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output model file path')
@click.pass_context
def bark_detector(ctx, bark_samples_dir, non_bark_samples_dir, output):
    """Train bark detection model."""
    config = ctx.obj['config']
    
    if not output:
        output = config.get('bark_model_path', os.path.join('data', 'bark_detection_model.pkl'))
    
    click.echo(f"Training bark detection model...")
    click.echo(f"Bark samples: {bark_samples_dir}")
    click.echo(f"Non-bark samples: {non_bark_samples_dir}")
    
    try:
        import librosa
        import numpy as np
        
        # Load audio samples
        bark_samples = []
        non_bark_samples = []
        
        # Load bark samples
        bark_dir = Path(bark_samples_dir)
        for audio_file in bark_dir.glob('*.wav'):
            try:
                audio_data, sr = librosa.load(str(audio_file), sr=config['sample_rate'])
                bark_samples.append(audio_data)
                click.echo(f"Loaded bark sample: {audio_file.name}")
            except Exception as e:
                click.echo(f"Error loading {audio_file}: {e}", err=True)
        
        # Load non-bark samples
        non_bark_dir = Path(non_bark_samples_dir)
        for audio_file in non_bark_dir.glob('*.wav'):
            try:
                audio_data, sr = librosa.load(str(audio_file), sr=config['sample_rate'])
                non_bark_samples.append(audio_data)
                click.echo(f"Loaded non-bark sample: {audio_file.name}")
            except Exception as e:
                click.echo(f"Error loading {audio_file}: {e}", err=True)
        
        if len(bark_samples) == 0 or len(non_bark_samples) == 0:
            click.echo("Need both bark and non-bark samples for training", err=True)
            return
        
        # Train model
        detector = BarkDetector(sample_rate=config['sample_rate'])
        
        os.makedirs(os.path.dirname(output), exist_ok=True)
        results = detector.train_model(bark_samples, non_bark_samples, output)
        
        click.echo(f"Training completed!")
        click.echo(f"Training accuracy: {results['train_accuracy']:.3f}")
        click.echo(f"Test accuracy: {results['test_accuracy']:.3f}")
        click.echo(f"Model saved to: {output}")
    
    except ImportError:
        click.echo("Required packages not installed. Please install librosa and scikit-learn.", err=True)
    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)


@cli.group()
def report():
    """Generate reports and statistics."""
    pass


@report.command()
@click.option('--days', '-d', default=7, help='Number of days to include in report')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', 'report_format', default='summary', 
              type=click.Choice(['summary', 'detailed']), help='Report format')
@click.pass_context
def generate(ctx, days, output, report_format):
    """Generate evidence report."""
    config = ctx.obj['config']
    
    try:
        db = BarkEventDatabase(config['database_path'])
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if not output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output = f"evidence_report_{timestamp}.txt"
        
        click.echo(f"Generating {report_format} report for {days} days...")
        click.echo(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        report_path = db.generate_evidence_report(
            start_date=start_date,
            end_date=end_date,
            output_path=output,
            report_type=report_format,
            purpose="legal_evidence"
        )
        
        click.echo(f"Report generated: {report_path}")
        
        # Show preview
        with open(report_path, 'r') as f:
            preview = f.read(500)
            click.echo("\\nReport preview:")
            click.echo("-" * 40)
            click.echo(preview)
            if len(preview) == 500:
                click.echo("... (truncated)")
    
    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)


@report.command()
@click.option('--days', '-d', default=30, help='Number of days to analyze')
@click.pass_context
def stats(ctx, days):
    """Show database statistics."""
    config = ctx.obj['config']
    
    try:
        db = BarkEventDatabase(config['database_path'])
        
        # Overall statistics
        stats = db.get_database_stats()
        
        click.echo("=== DATABASE STATISTICS ===")
        click.echo(f"Total events: {stats.get('total_events', 0)}")
        click.echo(f"Total dogs: {stats.get('total_dogs', 0)}")
        click.echo(f"Recent events (7 days): {stats.get('recent_events_7_days', 0)}")
        
        if stats.get('earliest_event'):
            click.echo(f"Earliest event: {stats['earliest_event']}")
        if stats.get('latest_event'):
            click.echo(f"Latest event: {stats['latest_event']}")
        
        # Daily summaries for recent days
        click.echo(f"\\n=== DAILY SUMMARIES (Last {days} days) ===")
        
        end_date = datetime.now()
        for i in range(days):
            date = end_date - timedelta(days=i)
            summary = db.get_daily_summary(date)
            
            if summary['total_barks'] > 0:
                click.echo(f"{summary['date']}: {summary['total_barks']} barks, "
                          f"{summary['total_duration']:.1f}s total, "
                          f"disturbance score: {summary['disturbance_score']:.1f}")
    
    except Exception as e:
        click.echo(f"Error getting statistics: {e}", err=True)


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.pass_context
def analyze(ctx, audio_file):
    """Analyze an audio file for bark detection."""
    config = ctx.obj['config']
    
    try:
        # Initialize components
        bark_detector = BarkDetector(sample_rate=config['sample_rate'])
        dog_identifier = DogIdentifier(sample_rate=config['sample_rate'])
        
        # Load models if available
        bark_model_path = config.get('bark_model_path')
        if bark_model_path and os.path.exists(bark_model_path):
            bark_detector.load_model(bark_model_path)
            click.echo("Loaded bark detection model")
        
        dog_model_path = config.get('dog_model_path')
        if dog_model_path and os.path.exists(dog_model_path):
            dog_identifier.load_model(dog_model_path)
            click.echo("Loaded dog identification model")
        
        # Analyze file
        click.echo(f"Analyzing: {audio_file}")
        
        import librosa
        audio_data, sr = librosa.load(audio_file, sr=config['sample_rate'])
        
        # Bark detection
        is_bark, bark_probability = bark_detector.predict_bark(audio_data)
        click.echo(f"Bark detection: {'YES' if is_bark else 'NO'} "
                  f"(probability: {bark_probability:.3f})")
        
        # Dog identification
        if is_bark:
            dog_id, confidence = dog_identifier.identify_dog(audio_data)
            click.echo(f"Dog identification: {dog_id} (confidence: {confidence:.3f})")
        
        # Audio characteristics
        duration = len(audio_data) / sr
        rms_energy = (audio_data ** 2).mean() ** 0.5
        
        click.echo(f"Duration: {duration:.2f} seconds")
        click.echo(f"RMS Energy: {rms_energy:.4f}")
    
    except ImportError:
        click.echo("Required packages not installed. Please install librosa.", err=True)
    except Exception as e:
        click.echo(f"Analysis failed: {e}", err=True)


@cli.command()
@click.option('--output', '-o', default='config/config.json', help='Output configuration file')
def init_config(output):
    """Initialize configuration file."""
    try:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        
        with open(output, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        
        click.echo(f"Configuration file created: {output}")
        click.echo("Edit this file to customize the bark monitor settings.")
    
    except Exception as e:
        click.echo(f"Error creating configuration file: {e}", err=True)


@cli.command()
@click.pass_context
def test(ctx):
    """Run system self-test."""
    config = ctx.obj['config']
    
    click.echo("Running system self-test...")
    
    # Test imports
    try:
        import numpy as np
        import librosa
        import sklearn
        import pyaudio
        import soundfile
        click.echo("✓ All required packages are installed")
    except ImportError as e:
        click.echo(f"✗ Missing package: {e}", err=True)
        return
    
    # Test audio devices
    try:
        from audio_capture import AudioCapture
        capture = AudioCapture()
        devices = capture.list_audio_devices()
        if devices:
            click.echo(f"✓ Found {len(devices)} audio input device(s)")
        else:
            click.echo("⚠ No audio input devices found (will run in simulation mode)")
    except Exception as e:
        click.echo(f"✗ Audio system error: {e}", err=True)
    
    # Test database
    try:
        db_path = config['database_path']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db = BarkEventDatabase(db_path)
        click.echo("✓ Database system working")
    except Exception as e:
        click.echo(f"✗ Database error: {e}", err=True)
    
    # Test model loading
    bark_model_path = config.get('bark_model_path')
    if bark_model_path and os.path.exists(bark_model_path):
        click.echo("✓ Bark detection model found")
    else:
        click.echo("⚠ No bark detection model found (will use heuristic detection)")
    
    dog_model_path = config.get('dog_model_path')
    if dog_model_path and os.path.exists(dog_model_path):
        click.echo("✓ Dog identification model found")
    else:
        click.echo("⚠ No dog identification model found (will create new one)")
    
    click.echo("\\nSystem test completed!")


if __name__ == '__main__':
    cli()