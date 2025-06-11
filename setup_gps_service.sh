#!/bin/bash

# NiaotoShield GPS Service Setup Script
# This script sets up the GPS server as a system service on Raspberry Pi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Set directory variables
INSTALL_DIR="/home/moh/Desktop/server_profile"
SERVICE_NAME="niatoshield-gps"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
VENV_PATH="${INSTALL_DIR}/rasso"

# Check if the directory and virtual environment exist
if [ ! -d "$INSTALL_DIR" ]; then
  echo "Error: Installation directory $INSTALL_DIR does not exist"
  exit 1
fi

if [ ! -d "$VENV_PATH" ]; then
  echo "Error: Virtual environment directory $VENV_PATH does not exist"
  exit 1
fi

if [ ! -f "${INSTALL_DIR}/gps_server.py" ]; then
  echo "Error: GPS server script not found in $INSTALL_DIR"
  exit 1
fi

# Make the script executable
chmod +x ${INSTALL_DIR}/gps_server.py

# Enable serial port
echo "Configuring serial port..."
if grep -q "enable_uart=1" /boot/config.txt; then
  echo "Serial port already enabled"
else
  echo "enable_uart=1" >> /boot/config.txt
  echo "Serial port enabled in /boot/config.txt"
fi

# Disable serial console
if grep -q "console=serial0" /boot/cmdline.txt; then
  sed -i 's/console=serial0,[0-9]\+ //' /boot/cmdline.txt
  echo "Serial console disabled in cmdline.txt"
fi

# Create systemd service
echo "Creating systemd service..."
cat > $SERVICE_FILE << EOF
[Unit]
Description=NiaotoShield GPS Service
After=network.target

[Service]
Type=simple
User=moh
WorkingDirectory=$INSTALL_DIR
ExecStart=${VENV_PATH}/bin/python ${INSTALL_DIR}/gps_server.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=$SERVICE_NAME
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
echo "Enabling and starting service..."
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME

# Check service status
echo "Service status:"
systemctl status $SERVICE_NAME --no-pager

echo ""
echo "NiaotoShield GPS Service has been installed and started."
echo "To check logs, run: journalctl -u $SERVICE_NAME"
echo "To restart service, run: sudo systemctl restart $SERVICE_NAME"
echo ""
echo "NOTE: A system reboot may be required for serial port changes to take effect."
echo "Run 'sudo reboot' if the service doesn't work correctly after installation." 