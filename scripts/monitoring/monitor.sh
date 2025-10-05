#!/bin/bash
# System Monitor Tool for Qwen
# Usage: ./monitor.sh "your monitoring command"
# Improved version with proper process cleanup and timeout handling

# Function to cleanup background processes on exit
cleanup() {
    echo "🧹 Cleaning up any background processes..."
    # Kill any qwen processes started by this script
    pkill -f "qwen.*--allowed-mcp-server-names system-monitor" 2>/dev/null || true
    # Kill any child processes of this script
    if [ ! -z "$SCRIPT_PID" ]; then
        pkill -P $SCRIPT_PID 2>/dev/null || true
    fi
    echo "✅ Cleanup complete"
}

# Function to handle script termination
terminate() {
    echo "⚠️  Script interrupted, cleaning up..."
    cleanup
    exit 1
}

# Set up signal handlers
trap terminate SIGINT SIGTERM
trap cleanup EXIT

# Store script PID for cleanup
SCRIPT_PID=$$

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"your monitoring command\""
    echo "Examples:"
    echo "  $0 \"Monitor system resources\""
    echo "  $0 \"Check for orphaned processes\""
    echo "  $0 \"List top CPU consuming processes\""
    echo "  $0 \"Show system health status\""
    exit 1
fi

COMMAND="$*"
echo "🤖 Running system monitoring command: $COMMAND"
echo "=========================================="

# Run qwen with timeout and proper process handling
timeout 60 qwen --allowed-mcp-server-names system-monitor -p "$COMMAND" --yolo

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "⚠️  Command timed out after 1 minute"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Command failed with exit code: $EXIT_CODE"
else
    echo "✅ Command completed successfully"
fi

# Cleanup will be called automatically due to trap EXIT
echo "🏁 Monitoring script finished"
