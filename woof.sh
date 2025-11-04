#!/bin/bash
# Dog Bark Monitor Launcher Script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Change to project directory
cd "$SCRIPT_DIR"

# Check if running with arguments
if [ $# -eq 0 ]; then
    echo "Dog Bark Detection and Monitoring System"
    echo "========================================"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Available commands:"
    echo "  monitor         Start bark monitoring"
    echo "  monitor-sim     Start in simulation mode"
    echo "  test            Run system self-test"
    echo "  stats           Show database statistics"
    echo "  report          Generate evidence report"
    echo "  list-devices    List audio devices"
    echo "  help            Show detailed help"
    echo ""
    echo "Examples:"
    echo "  $0 monitor"
    echo "  $0 monitor-sim"
    echo "  $0 stats"
    echo "  $0 report --days 30"
    echo ""
    exit 1
fi

# Handle special commands
case "$1" in
    "monitor-sim")
        shift
        python src/cli.py monitor --simulate "$@"
        ;;
    "help")
        python src/cli.py --help
        ;;
    *)
        python src/cli.py "$@"
        ;;
esac