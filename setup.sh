#!/bin/bash
# Dog Bark Monitor Setup Script

set -e  # Exit on any error

echo "Dog Bark Detection Tool - Setup Script"
echo "======================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python $PYTHON_VERSION"

# Check if we have the required Python version
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    echo "Error: Python 3.8 or higher is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Setting up in directory: $SCRIPT_DIR"
echo ""

# Check for system dependencies
echo "Checking system dependencies..."

# Check for venv module
if ! python3 -c "import venv" 2>/dev/null; then
    echo "Error: Python venv module not available."
    echo "On Ubuntu/Debian, install with: sudo apt install python3-venv"
    echo "On other systems, ensure Python was installed with venv support."
    exit 1
fi

# Check for audio development libraries (non-fatal)
if ! pkg-config --exists portaudio-2.0 2>/dev/null; then
    echo "Warning: PortAudio development libraries not found."
    echo "Audio recording may not work. Install with:"
    echo "  Ubuntu/Debian: sudo apt install portaudio19-dev"
    echo "  macOS: brew install portaudio"
    echo "  Windows: Usually included with Python"
    echo ""
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies installed successfully."
else
    echo "Error: requirements.txt not found."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs recordings config

# Initialize configuration if it doesn't exist
if [ ! -f "config/config.json" ]; then
    echo "Creating default configuration..."
    python src/cli.py init-config --output config/config.json
fi

# Make launcher script executable
if [ -f "woof.sh" ]; then
    chmod +x woof.sh
    echo "Launcher script made executable."
fi

# Run system test
echo ""
echo "Running system self-test..."
python src/cli.py test

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Test the system: ./woof.sh monitor-sim"
echo "2. List audio devices: ./woof.sh list-devices"
echo "3. Start monitoring: ./woof.sh monitor"
echo "4. View help: ./woof.sh help"
echo ""
echo "For detailed usage instructions, see README.md"