#!/bin/bash
# NiotoShield Face Security System Startup Script
# This script starts the face security system at boot time

# Process command line arguments
HEADLESS_MODE=true
DAEMON_MODE=false

if [ "$1" == "--gui" ]; then
  HEADLESS_MODE=false
fi

if [ "$1" == "--daemon" ]; then
  HEADLESS_MODE=true
  DAEMON_MODE=true
fi

# GPIO pin configuration for status LEDs
GREEN_LED_PIN=22  # GPIO22 for system running indicator
RED_LED_PIN=27    # GPIO27 for camera error indicator

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configure GPIO pins for LEDs if running as root or with proper permissions
setup_leds() {
  # Check if GPIO sysfs interface is available before using it
  if [ -d "/sys/class/gpio" ]; then
    echo "Setting up GPIO LEDs..."
    
    # Create GPIO paths if they don't exist
    if [ ! -d "/sys/class/gpio/gpio$GREEN_LED_PIN" ]; then
      echo "$GREEN_LED_PIN" > /sys/class/gpio/export 2>/dev/null || true
      sleep 0.1  # Give system time to create the GPIO node
    fi
    if [ ! -d "/sys/class/gpio/gpio$RED_LED_PIN" ]; then
      echo "$RED_LED_PIN" > /sys/class/gpio/export 2>/dev/null || true
      sleep 0.1  # Give system time to create the GPIO node
    fi
    
    # Set direction to output
    if [ -d "/sys/class/gpio/gpio$GREEN_LED_PIN" ]; then
      echo "out" > /sys/class/gpio/gpio$GREEN_LED_PIN/direction 2>/dev/null || true
      # Initialize LED off
      echo "0" > /sys/class/gpio/gpio$GREEN_LED_PIN/value 2>/dev/null || true
    else
      echo "WARNING: Could not set up green LED GPIO$GREEN_LED_PIN"
    fi
    
    if [ -d "/sys/class/gpio/gpio$RED_LED_PIN" ]; then
      echo "out" > /sys/class/gpio/gpio$RED_LED_PIN/direction 2>/dev/null || true
      # Initialize LED off
      echo "0" > /sys/class/gpio/gpio$RED_LED_PIN/value 2>/dev/null || true
    else
      echo "WARNING: Could not set up red LED GPIO$RED_LED_PIN"
    fi
  else
    echo "WARNING: GPIO sysfs interface not available, skipping LED setup"
  fi
}

# Turn on green LED to indicate system is running
green_led_on() {
  if [ -f "/sys/class/gpio/gpio$GREEN_LED_PIN/value" ]; then
    echo "1" > /sys/class/gpio/gpio$GREEN_LED_PIN/value 2>/dev/null || true
  fi
}

# Turn on red LED to indicate camera error
red_led_on() {
  if [ -f "/sys/class/gpio/gpio$RED_LED_PIN/value" ]; then
    echo "1" > /sys/class/gpio/gpio$RED_LED_PIN/value 2>/dev/null || true
  fi
}

# Turn off red LED when camera is fixed
red_led_off() {
  if [ -f "/sys/class/gpio/gpio$RED_LED_PIN/value" ]; then
    echo "0" > /sys/class/gpio/gpio$RED_LED_PIN/value 2>/dev/null || true
  fi
}

echo "----------------------------------------"
echo "    NiotoShield Security System"
echo "----------------------------------------"
echo "Starting system at $(date)"
echo "Working directory: $(pwd)"
if [ "$HEADLESS_MODE" = true ]; then
  echo "Running in headless mode (no GUI)"
else
  echo "Running in GUI mode"
fi
if [ "$DAEMON_MODE" = true ]; then
  echo "Running as daemon service"
fi

# Setup LEDs
setup_leds

# Make sure the Python script is executable
chmod +x face_security.py

# Check for USB camera
if [ ! -e "/dev/video0" ]; then
    echo "WARNING: USB camera (/dev/video0) not found!"
    echo "The security system requires a USB camera to function properly."
    # Turn on red LED to indicate error
    red_led_on
    
    # We'll continue anyway, as the camera might be connected later
    echo "Will attempt to use USB camera when connected."
fi

# Check if model file exists
if [ ! -f "EasyShield_V2.5.pt" ]; then
    echo "ERROR: Anti-spoofing model not found!"
    echo "Please make sure EasyShield_V2.5.pt exists in this directory."
    # Turn on red LED to indicate error
    red_led_on
    exit 1
fi

# Check if embeddings directory exists
if [ ! -d "embeddings" ]; then
    echo "WARNING: Embeddings directory not found!"
    echo "Face recognition will not work without embeddings."
fi

# Check if virtual environment exists and activate it
if [ -d "rasso" ]; then
    echo "Activating virtual environment..."
    source rasso/bin/activate
else
    echo "WARNING: Virtual environment 'rasso' not found!"
    echo "Using system Python environment."
fi

# Set display environment variables for GUI mode
if [ "$HEADLESS_MODE" = false ]; then
  export DISPLAY=:0
  export XAUTHORITY=/home/moh/.Xauthority
  echo "Display environment variables set for GUI mode"
fi

# Initialize counter and variables for restart mechanism
MAX_CRASHES=0  # 0 means infinite restarts
crash_count=0
start_time=$(date +%s)
last_crash_time=$start_time
crash_log_file="security_system_crashes.log"

# Function to log crash information
log_crash() {
    local exit_code=$1
    local current_time=$(date)
    local uptime=$(($(date +%s) - last_crash_time))
    
    echo "[$current_time] System crashed with exit code $exit_code (uptime: ${uptime}s)" >> "$crash_log_file"
    echo "System crashed with exit code $exit_code"
    echo "Uptime before crash: ${uptime} seconds"
}

# Turn on green LED to indicate system is starting
green_led_on

# Start the security system in a loop that restarts on crash
echo "Starting face security system with automatic restart..."
while true; do
    # Record the start time of this run
    last_crash_time=$(date +%s)
    
    # Start the system with the appropriate mode
    if [ "$HEADLESS_MODE" = true ]; then
        python face_security.py --headless --led_pins $GREEN_LED_PIN,$RED_LED_PIN
    else
        python face_security.py --led_pins $GREEN_LED_PIN,$RED_LED_PIN
    fi
    
    # Get the exit code
    exit_code=$?
    
    # If exit code is 0 (clean exit) or 130 (Ctrl+C), don't restart
    if [ $exit_code -eq 0 ]; then
        echo "Security system exited cleanly with code $exit_code"
        echo "Not restarting as this was a normal shutdown."
        break
    elif [ $exit_code -eq 130 ]; then
        echo "Security system was terminated by user (Ctrl+C)"
        echo "Not restarting as this was a manual interruption."
        break
    elif [ $exit_code -eq 5 ]; then
        # Special exit code for camera errors
        echo "Camera error detected (exit code 5)"
        red_led_on
    else
        # Log the crash
        log_crash $exit_code
        
        # Increment crash counter
        crash_count=$((crash_count + 1))
        
        # Check if we've hit the maximum number of allowed crashes (if MAX_CRASHES > 0)
        if [ $MAX_CRASHES -gt 0 ] && [ $crash_count -ge $MAX_CRASHES ]; then
            echo "Reached maximum number of restart attempts ($MAX_CRASHES)"
            echo "Please check the system for errors."
            break
        fi
        
        # Calculate time since the process started
        crash_time=$(date +%s)
        run_time=$((crash_time - last_crash_time))
        
        # If the process ran for less than 10 seconds before crashing,
        # wait longer before restarting to prevent rapid crash loops
        if [ $run_time -lt 10 ]; then
            echo "System crashed quickly after start. Waiting 30 seconds before restart..."
            sleep 30
        else
            echo "Waiting 5 seconds before restarting..."
            sleep 5
        fi
        
        echo "Restarting security system (attempt $crash_count)..."
        echo "----------------------------------------"
    fi
done

# Turn off LEDs when system exits
if [ -f "/sys/class/gpio/gpio$GREEN_LED_PIN/value" ]; then
    echo "0" > /sys/class/gpio/gpio$GREEN_LED_PIN/value 2>/dev/null || true
fi
if [ -f "/sys/class/gpio/gpio$RED_LED_PIN/value" ]; then
    echo "0" > /sys/class/gpio/gpio$RED_LED_PIN/value 2>/dev/null || true
fi

echo "Security system has stopped."
echo "Check $crash_log_file for crash history."

# Don't keep terminal open in daemon mode
if [ "$DAEMON_MODE" = false ]; then
read -p "Press Enter to exit..."
fi