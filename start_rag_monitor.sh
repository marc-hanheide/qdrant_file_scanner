#!/bin/bash

# RAG Monitor Startup Script for macOS
# This script waits for Qdrant to be available before starting the RAG monitor

# Configuration - MODIFY THESE PATHS FOR YOUR SETUP
RAG_MONITOR_DIR="/Users/mhanheide/workspace/qdrant_file_scanner"
VENV_PATH="$RAG_MONITOR_DIR/.venv"
CONFIG_FILE="$RAG_MONITOR_DIR/config.yaml"
LOG_FILE="$HOME/Library/Logs/rag-monitor-startup.log"

# Qdrant connection settings (from your config.yaml)
QDRANT_HOST="localhost"
QDRANT_PORT="6333"
MAX_WAIT_TIME=300  # Maximum time to wait for Qdrant (5 minutes)

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check if Qdrant is available
check_qdrant() {
    curl -s -f "http://$QDRANT_HOST:$QDRANT_PORT/collections" > /dev/null 2>&1
    return $?
}

# Function to wait for Qdrant
wait_for_qdrant() {
    log_message "Waiting for Qdrant server at $QDRANT_HOST:$QDRANT_PORT..."
    
    local elapsed=0
    while [ $elapsed -lt $MAX_WAIT_TIME ]; do
        if check_qdrant; then
            log_message "Qdrant server is available!"
            return 0
        fi
        
        log_message "Qdrant not available yet, waiting... (${elapsed}s elapsed)"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    
    log_message "ERROR: Qdrant server not available after ${MAX_WAIT_TIME} seconds"
    return 1
}

# Main execution
log_message "Starting RAG Monitor startup script..."

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    log_message "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_message "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Wait for Qdrant to be available
if ! wait_for_qdrant; then
    log_message "ERROR: Cannot start RAG monitor - Qdrant unavailable"
    exit 1
fi

# Activate virtual environment and start RAG monitor
log_message "Starting RAG monitor..."
cd "$RAG_MONITOR_DIR"

# Source the virtual environment
source "$VENV_PATH/bin/activate"

# Optional: Send notification that RAG monitor is starting
# Uncomment the next line if you want desktop notifications
# osascript -e 'display notification "RAG Monitor started successfully" with title "RAG Monitor"'

# Start the RAG monitor with the specified config
# Options:
# - Default: Scan existing files then monitor for changes
# - --scan-only: Only scan existing files, don't monitor
# - --monitor-only: Only monitor for changes, skip initial scan
exec "$VENV_PATH/bin/rag-monitor" --monitor-only --config "$CONFIG_FILE" >> "$LOG_FILE" 2>&1
