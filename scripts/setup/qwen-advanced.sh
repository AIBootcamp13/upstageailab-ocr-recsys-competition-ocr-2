#!/bin/bash
# Advanced Qwen memory optimization script
# Implements streaming, chunked processing, and selective context loading

# Configuration
HEAP_SIZE=12288  # 12GB heap
MAX_FILES=50     # Limit files loaded at once
CHUNK_SIZE=1000  # Process in chunks

# Export environment variables to control behavior
export GEMINI_CLI_NO_RELAUNCH=1  # Prevent automatic relaunch
export NODE_OPTIONS="--max-old-space-size=$HEAP_SIZE --optimize-for-size"

# Function to run Qwen with memory monitoring
run_qwen_with_monitoring() {
    local prompt="$1"
    local max_files="$2"

    echo "Starting Qwen with heap limit: ${HEAP_SIZE}MB, max files: $max_files"

    # Monitor memory usage in background
    (
        while true; do
            ps aux | grep "qwen\|gemini" | grep -v grep | awk '{print $4,$6}' || break
            sleep 5
        done
    ) &
    local monitor_pid=$!

    # Run qwen with limited context
    echo "$prompt" | ./scripts/qwen-memfix.sh --yolo --all-files=false --prompt "Focus on specific files only"

    # Clean up monitor
    kill $monitor_pid 2>/dev/null
}

# Function for chunked processing
run_chunked_task() {
    local base_prompt="$1"
    local file_list="$2"

    echo "Processing in chunks to reduce memory usage..."

    # Split files into chunks
    echo "$file_list" | split -l $CHUNK_SIZE - /tmp/qwen_chunk_

    for chunk in /tmp/qwen_chunk_*; do
        if [ -f "$chunk" ]; then
            local files=$(cat "$chunk")
            echo "Processing chunk with $(echo "$files" | wc -l) files..."

            local chunk_prompt="$base_prompt

Process only these specific files:
$files"

            run_qwen_with_monitoring "$chunk_prompt" $MAX_FILES
        fi
    done

    # Cleanup
    rm -f /tmp/qwen_chunk_*
}

# Main execution
case "$1" in
    "monitor")
        run_qwen_with_monitoring "$2" "${3:-$MAX_FILES}"
        ;;
    "chunked")
        run_chunked_task "$2" "$3"
        ;;
    *)
        echo "Usage: $0 [monitor <prompt> [max_files]] | [chunked <base_prompt> <file_list>]"
        ;;
esac
