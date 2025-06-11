#!/bin/bash
# Start Monitor Service
# This script starts the update monitor service
# Usage: 
#   ./start_monitor.sh        # Start in background mode
#   ./start_monitor.sh --gui  # Start in console/terminal mode

# Process command line arguments
CONSOLE_MODE=false
if [ "$1" == "--gui" ]; then
  CONSOLE_MODE=true
fi

# Navigate to the server profile directory
cd /home/moh/Desktop/server_profile

# Make sure the check_updates.py script is executable
chmod +x check_updates.py

# Activate the virtual environment
source rasso/bin/activate

if [ "$CONSOLE_MODE" = true ]; then
  # Run in console mode (visible terminal)
  echo "Starting NiotoShield Monitor in console mode..."
  
  # For console mode, start in a new terminal window
  if [ -x "$(command -v lxterminal)" ]; then
    # For Raspberry Pi OS (LXDE)
    lxterminal --title="NiotoShield Monitor" -e "bash -c 'cd $(pwd) && source rasso/bin/activate && python check_updates.py --console; exec bash'"
  elif [ -x "$(command -v xterm)" ]; then
    # Fallback to xterm if available
    xterm -title "NiotoShield Monitor" -e "cd $(pwd) && source rasso/bin/activate && python check_updates.py --console; bash"
  else
    # If no terminal emulator is available, run in the current terminal
    python check_updates.py --console
  fi
  
  echo "NiotoShield Monitor started in console mode"
else
  # Run in background mode
  echo "Starting NiotoShield Monitor in background mode..."
  
  # Start the monitor script in the background, logging to a file
  nohup python check_updates.py > monitor.log 2>&1 &
  
  # Save the PID to a file so we can stop it later if needed
  echo $! > monitor.pid
  
  echo "Monitor service started with PID $(cat monitor.pid)"
fi 