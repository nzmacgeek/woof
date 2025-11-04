# Bark Monitor - Windows 11 Quick Start Guide

## Prerequisites

1. **Python 3.8 or higher**
   - Download from [python.org](https://python.org)
   - ⚠️ **Important**: Check "Add Python to PATH" during installation

2. **Microphone access**
   - Go to Settings > Privacy & Security > Microphone
   - Enable "Allow desktop apps to access your microphone"

## Automated Installation (Recommended)

1. **Download/Clone the project**
   ```powershell
   git clone <repository-url>
   cd bark-monitor
   ```

2. **Run the setup script**
   ```powershell
   setup.bat
   ```
   
   This will:
   - Create a virtual environment
   - Install all dependencies
   - Set up directories and configuration
   - Test the installation

## Manual Installation

If the automated setup doesn't work:

1. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Create directories**
   ```powershell
   mkdir logs, data, recordings, reports
   ```

3. **Copy configuration**
   ```powershell
   copy config\default.json config.json
   ```

## Quick Test

Run the Windows compatibility test:
```powershell
python test_windows_compatibility.py
```

## Usage

### Start Monitoring
```powershell
woof.bat monitor --duration 3600
```

### Launch GUI
```powershell
python gui.py
```

### List Audio Devices
```powershell
woof.bat list-devices
```

### Generate Reports
```powershell
woof.bat report --days 30
```

### Get Help
```powershell
woof.bat help
```

## Optional: Install FFmpeg

For MP3 encoding support:

### Option A: Manual Download
1. Visit [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Download Windows build
3. Extract to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your PATH

### Option B: Chocolatey (if installed)
```powershell
choco install ffmpeg
```

## Troubleshooting

### PyAudio Installation Issues
```powershell
pip install pipwin
pipwin install pyaudio
```

### Permission Issues
- Run Command Prompt as Administrator
- Check Windows Defender exclusions

### Microphone Not Working
- Check Windows Sound settings
- Verify microphone drivers
- Test with other audio applications

### Import Errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

## System Requirements

- **OS**: Windows 11 (also works on Windows 10)
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB
- **Storage**: 500MB for system + space for recordings
- **Audio**: Microphone or audio input device

## File Structure

```
bark-monitor/
├── setup.bat              # Windows setup script
├── woof.bat               # Windows launcher
├── requirements.txt       # Python dependencies
├── config.json           # Configuration file
├── gui.py                # GUI application
├── src/                  # Source code
├── logs/                 # Log files
├── data/                 # Database and models
├── recordings/           # Audio recordings
└── reports/              # Generated reports
```

## Support

If you encounter issues:

1. Run the compatibility test: `python test_windows_compatibility.py`
2. Check logs in the `logs/` directory
3. Try running with debug mode: `woof.bat monitor --log-level DEBUG`
4. Ensure all dependencies are installed: `pip list`

## Performance Notes

- **CPU Usage**: 5-15% during monitoring
- **Memory**: 150-300MB typical usage
- **Storage**: ~50KB per bark event (MP3 compressed)
- **Latency**: <100ms detection response time

The system is optimized for Windows 11 and should run smoothly on modern hardware.