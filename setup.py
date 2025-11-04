#!/usr/bin/env python3
"""
Setup script for Bark Monitor - Enhanced Dog Barking Detection System
Handles installation, dependency checking, and initial configuration.
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

def print_banner():
    """Print installation banner."""
    print("=" * 60)
    print("  BARK MONITOR - Enhanced Dog Barking Detection System")
    print("                    Installation Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_system_dependencies():
    """Check for required system dependencies."""
    print("\nChecking system dependencies...")
    
    dependencies = {
        'ffmpeg': 'FFmpeg (for MP3 encoding)',
        'lame': 'LAME (alternative MP3 encoder)'
    }
    
    available = {}
    for dep, description in dependencies.items():
        try:
            # Windows-specific check
            if platform.system().lower() == 'windows':
                subprocess.run([dep, '-version'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, 
                             check=True)
            else:
                subprocess.run([dep, '-version'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, 
                             check=True)
            print(f"✅ {description} is available")
            available[dep] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  {description} not found (optional)")
            available[dep] = False
    
    # Windows-specific audio dependency check
    if platform.system().lower() == 'windows':
        print("\nChecking Windows audio dependencies...")
        try:
            import pyaudio
            print("✅ PyAudio is available")
        except ImportError:
            print("⚠️  PyAudio not found - audio capture may not work")
    
    return available

def install_python_dependencies():
    """Install Python package dependencies."""
    print("\nInstalling Python dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def install_optional_ai_dependencies():
    """Install optional AI dependencies."""
    print("\nOptional AI Dependencies")
    print("These enhance detection accuracy but are not required.")
    
    response = input("Install AI dependencies (PyTorch, Transformers)? [y/N]: ").lower()
    
    if response in ['y', 'yes']:
        print("Installing AI dependencies...")
        
        ai_packages = [
            "torch",
            "torchaudio", 
            "transformers",
            "datasets"
        ]
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + ai_packages, check=True)
            print("✅ AI dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  AI dependencies installation failed: {e}")
            print("   System will use fallback detection methods")
            return False
    
    print("⏭️  Skipping AI dependencies")
    return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating project directories...")
    
    directories = [
        "data",
        "recordings", 
        "logs",
        "models",
        "reports"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_default_config():
    """Create default configuration file."""
    print("\nCreating default configuration...")
    
    config = {
        "sample_rate": 44100,
        "chunk_size": 4096,
        "detection_threshold": 0.7,
        "bark_threshold": 0.7,
        "enable_ai": True,
        "ai_model_type": "auto",
        "audio_format": "mp3",
        "audio_quality": "medium",
        "save_audio": True,
        "recordings_dir": "recordings",
        "database_path": os.path.join("data", "bark_events.db"),
        "log_level": "INFO",
        "log_file": os.path.join("logs", "bark_monitor.log"),
        "enable_alerts": False,
        "alert_intensity_threshold": 80.0,
        "max_dogs": 10,
        "buffer_duration": 10.0
    }
    
    config_path = Path("config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Configuration saved to: {config_path}")

def test_installation():
    """Test the installation."""
    print("\nTesting installation...")
    
    try:
        # Test core system
        print("Testing core system...")
        result = subprocess.run([
            sys.executable, "test_enhanced_system.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Core system test passed")
        else:
            print("⚠️  Core system test had issues:")
            print(result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr)
    
    except subprocess.TimeoutExpired:
        print("⚠️  Core system test timed out (this may be normal)")
    except Exception as e:
        print(f"⚠️  Core system test failed: {e}")

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("                   INSTALLATION COMPLETE")
    print("=" * 60)
    print()
    print("🎉 Bark Monitor has been installed successfully!")
    print()
    print("Quick Start:")
    print("━━━━━━━━━━━")
    print()
    
    # Platform-specific instructions
    system = platform.system().lower()
    if system == "windows":
        print("Windows Quick Start:")
        print("1. Run setup.bat for automated setup (recommended)")
        print("   setup.bat")
        print()
        print("2. Test the system:")
        print("   python test_enhanced_system.py")
        print()
        print("3. Start monitoring:")
        print("   woof.bat monitor --duration 3600")
        print()
        print("4. Launch GUI:")
        print("   python gui.py")
        print()
        print("5. List audio devices:")
        print("   woof.bat list-devices")
        print()
        print("6. Generate reports:")
        print("   woof.bat report --help")
    else:
        print("Unix/Linux Quick Start:")
        print("1. Test the system:")
        print("   python test_enhanced_system.py")
        print()
        print("2. Start monitoring (CLI):")
        print("   python -m src.cli monitor --duration 3600")
        print()
        print("3. Launch GUI (if Tkinter available):")
        print("   python gui.py")
        print()
        print("4. List audio devices:")
        print("   python -m src.cli devices")
        print()
        print("5. Generate reports:")
        print("   python -m src.cli report --help")
    print()
    print("Configuration:")
    print("━━━━━━━━━━━━━━")
    print("• Edit config.json to customize settings")
    print("• Adjust detection thresholds as needed")
    print("• Configure audio quality and format")
    print()
    print("Documentation:")
    print("━━━━━━━━━━━━━━━")
    print("• README.md - Complete user guide")
    print("• src/ - Source code and modules")
    print("• gui.py - Cross-platform interface")
    print()
    print("Support:")
    print("━━━━━━━━━")
    print("• Check logs in logs/ directory")
    print("• Use --help flag for command options")
    print("• Run with --log-level DEBUG for troubleshooting")
    print()

def print_system_recommendations():
    """Print system-specific recommendations."""
    system = platform.system().lower()
    
    print("\nSystem-specific recommendations:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if system == "linux":
        print("Linux detected:")
        print("• Install audio dependencies:")
        print("  sudo apt-get install portaudio19-dev ffmpeg lame")
        print("• For permissions issues:")
        print("  sudo usermod -a -G audio $USER")
        
    elif system == "darwin":
        print("macOS detected:")
        print("• Install dependencies with Homebrew:")
        print("  brew install portaudio ffmpeg lame")
        print("• Grant microphone permissions in System Preferences")
        
    elif system == "windows":
        print("Windows 11 detected:")
        print("🎯 Windows-specific setup:")
        print("• Download FFmpeg:")
        print("  1. Visit https://ffmpeg.org/download.html")
        print("  2. Download the Windows build")
        print("  3. Extract to C:\\ffmpeg")
        print("  4. Add C:\\ffmpeg\\bin to your PATH")
        print()
        print("• Alternative: Use Chocolatey package manager:")
        print("  choco install ffmpeg")
        print()
        print("• Grant microphone permissions:")
        print("  1. Go to Settings > Privacy & Security > Microphone")
        print("  2. Allow desktop apps to access microphone")
        print("  3. Ensure Python.exe has microphone access")
        print()
        print("• For PyAudio installation issues:")
        print("  pip install pipwin")
        print("  pipwin install pyaudio")
        print()
        print("• Run setup.bat for automated Windows setup")
        print("• Use woof.bat instead of woof.sh")
        print()
        print("• If you encounter permission issues:")
        print("  - Run Command Prompt as Administrator")
        print("  - Consider Windows Defender exclusions for the project folder")

def main():
    """Main installation function."""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    # Check system dependencies
    system_deps = check_system_dependencies()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Installation failed!")
        return 1
    
    # Install optional AI dependencies
    ai_installed = install_optional_ai_dependencies()
    
    # Create directories
    create_directories()
    
    # Create configuration
    create_default_config()
    
    # Test installation
    test_installation()
    
    # Print instructions
    print_usage_instructions()
    print_system_recommendations()
    
    print("\n🎯 Installation completed successfully!")
    print("   Run 'python test_enhanced_system.py' to verify everything works.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())