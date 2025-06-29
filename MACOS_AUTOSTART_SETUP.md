# RAG Monitor macOS Autostart Setup

This guide explains how to set up the RAG Monitor to start automatically when you log into macOS, and ensure it waits for the Qdrant server to be available before starting.

## Prerequisites

1. RAG File Monitor is already installed and working
2. Qdrant server is configured to start automatically (or you start it manually)
3. You know the absolute path to your RAG File Monitor installation

## Step 1: Create a Startup Script

First, we'll create a startup script that handles waiting for Qdrant and starting the RAG monitor.

### 1.1: Create the startup script

Create a file called `start_rag_monitor.sh` in your RAG File Monitor directory:

```bash
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

# Start the RAG monitor with the specified config
exec "$VENV_PATH/bin/rag-monitor" --config "$CONFIG_FILE" >> "$LOG_FILE" 2>&1
```

### 1.2: Make the script executable

```bash
chmod +x start_rag_monitor.sh
```

### 1.3: Test the script

Before setting up autostart, test the script manually:

```bash
./start_rag_monitor.sh
```

Check the log file to see if it works properly:

```bash
tail -f ~/Library/Logs/rag-monitor-startup.log
```

## Step 2: Create a Launch Agent

macOS uses Launch Agents to automatically start programs when a user logs in.

### 2.1: Create the plist file

Create a file called `com.ragmonitor.startup.plist` in `~/Library/LaunchAgents/`:

```bash
mkdir -p ~/Library/LaunchAgents
```

Create the plist file with this content (adjust the path to your RAG Monitor directory):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ragmonitor.startup</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/Users/mhanheide/workspace/qdrant_file_scanner/start_rag_monitor.sh</string>
    </array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>/Users/mhanheide/Library/Logs/rag-monitor-launchd-out.log</string>
    
    <key>StandardErrorPath</key>
    <string>/Users/mhanheide/Library/Logs/rag-monitor-launchd-err.log</string>
    
    <key>WorkingDirectory</key>
    <string>/Users/mhanheide/workspace/qdrant_file_scanner</string>
    
    <key>ProcessType</key>
    <string>Background</string>
    
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
```

**Important**: Replace `/Users/mhanheide/workspace/qdrant_file_scanner` with the actual path to your RAG Monitor installation.

### 2.2: Load the Launch Agent

Load the launch agent so it starts automatically:

```bash
launchctl load ~/Library/LaunchAgents/com.ragmonitor.startup.plist
```

### 2.3: Verify it's loaded

Check if the launch agent is loaded:

```bash
launchctl list | grep ragmonitor
```

You should see an entry like:
```
12345   0   com.ragmonitor.startup
```

## Step 3: Configuration Options

### 3.1: Modify startup script configuration

Edit the `start_rag_monitor.sh` script to adjust these settings:

- `RAG_MONITOR_DIR`: Path to your RAG Monitor installation
- `QDRANT_HOST` and `QDRANT_PORT`: Your Qdrant server settings
- `MAX_WAIT_TIME`: How long to wait for Qdrant (in seconds)
- `LOG_FILE`: Where to write startup logs

### 3.2: Monitor-only mode

If you prefer to only monitor for changes (skip the initial scan), modify the last line in `start_rag_monitor.sh`:

```bash
exec "$VENV_PATH/bin/rag-monitor" --config "$CONFIG_FILE" --monitor-only >> "$LOG_FILE" 2>&1
```

## Step 4: Managing the Service

### Start the service manually
```bash
launchctl start com.ragmonitor.startup
```

### Stop the service
```bash
launchctl stop com.ragmonitor.startup
```

### Unload the service (disable autostart)
```bash
launchctl unload ~/Library/LaunchAgents/com.ragmonitor.startup.plist
```

### Reload the service (after making changes)
```bash
launchctl unload ~/Library/LaunchAgents/com.ragmonitor.startup.plist
launchctl load ~/Library/LaunchAgents/com.ragmonitor.startup.plist
```

## Step 5: Monitoring and Troubleshooting

### Check if the service is running
```bash
launchctl list | grep ragmonitor
ps aux | grep rag-monitor
```

### View logs
```bash
# Startup script logs
tail -f ~/Library/Logs/rag-monitor-startup.log

# Launch daemon stdout
tail -f ~/Library/Logs/rag-monitor-launchd-out.log

# Launch daemon stderr
tail -f ~/Library/Logs/rag-monitor-launchd-err.log

# RAG monitor application logs
tail -f /Users/mhanheide/workspace/qdrant_file_scanner/rag_monitor.log
```

### Common issues and solutions

1. **Service won't start**: Check the paths in both the plist file and startup script
2. **Qdrant timeout**: Increase `MAX_WAIT_TIME` in the startup script
3. **Permission issues**: Ensure the startup script is executable and paths are correct
4. **Virtual environment issues**: Verify the `.venv` path is correct

### Testing different scenarios

1. **Test without Qdrant running**: Start the script when Qdrant is off to verify the waiting mechanism
2. **Test with Qdrant running**: Verify it starts immediately when Qdrant is available
3. **Test restart behavior**: The service should restart if it crashes (due to `KeepAlive` setting)

## Step 6: Optional Enhancements

### 6.1: Add notification on start

Add this to the startup script to get notified when RAG monitor starts:

```bash
# Add after "Starting RAG monitor..." line
osascript -e 'display notification "RAG Monitor started successfully" with title "RAG Monitor"'
```

### 6.2: Add Qdrant dependency check

You can also set up Qdrant to start automatically using similar steps, or modify the script to start Qdrant if it's not running.

### 6.3: Health check endpoint

Consider adding a simple health check endpoint to the RAG monitor to verify it's working properly.

## Summary

After completing these steps:

1. RAG Monitor will automatically start when you log in
2. It will wait for Qdrant to be available before starting
3. It will restart automatically if it crashes
4. You can monitor its status through log files
5. You can easily enable/disable the autostart feature

The service will run in the background and begin monitoring your configured directories as soon as both the system starts and Qdrant becomes available.
