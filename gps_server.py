#!/usr/bin/env python3
import serial
import time
import json
import requests
import os
import signal
import sys
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
SERIAL_PORT = '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'  # Arduino via USB - change as needed
BAUD_RATE = 9600
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_KEY')  # Changed from SUPABASE_API_KEY to match .env
PRODUCT_KEY = os.getenv('PRODUCT_KEY')
USER_ID = os.getenv('USER_ID')  # Will be fetched later if not set

# Message markers (must match Arduino)
MSG_START = '#'
MSG_END = '$'

# Global variables
last_gps_data = {'lat': 0.0, 'lng': 0.0, 'timestamp': 0, 'status': 'INVALID'}
serial_port = None
running = True

# Set up colorful console output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log(message, level="INFO"):
    """Print a colorful log message with timestamp"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if level == "INFO":
        color = Colors.BLUE
    elif level == "SUCCESS":
        color = Colors.GREEN
    elif level == "WARNING":
        color = Colors.YELLOW
    elif level == "ERROR":
        color = Colors.RED
    elif level == "DEBUG":
        color = Colors.HEADER
    else:
        color = ""
        
    print(f"{color}[{timestamp}] [{level}] {message}{Colors.ENDC}")

def signal_handler(sig, frame):
    """Handle Ctrl+C signal to clean up resources"""
    global running
    log("Shutting down GPS server...", "WARNING")
    running = False
    if serial_port and serial_port.is_open:
        serial_port.close()
    sys.exit(0)

def fetch_user_id_from_product_key():
    """Fetch user_id from product_key in Supabase products table"""
    global USER_ID
    
    if USER_ID:
        return True
        
    if not SUPABASE_URL or not SUPABASE_API_KEY or not PRODUCT_KEY:
        log("Missing environment variables for Supabase connection", "ERROR")
        return False
    
    headers = {
        'apikey': SUPABASE_API_KEY,
        'Authorization': f'Bearer {SUPABASE_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    try:
        # Query the products table to get user_id from product_key
        query_url = f"{SUPABASE_URL}/rest/v1/products?product_key=eq.{PRODUCT_KEY}"
        
        response = requests.get(query_url, headers=headers)
        
        if response.status_code == 200 and len(response.json()) > 0:
            user_id = response.json()[0]['user_id']
            if user_id:
                USER_ID = user_id
                log(f"Found user_id: {USER_ID} for product_key: {PRODUCT_KEY}", "SUCCESS")
                return True
            else:
                log(f"Product found but no user_id associated with product_key: {PRODUCT_KEY}", "WARNING")
        else:
            log(f"Failed to find product with key: {PRODUCT_KEY}", "ERROR")
        
        return False
        
    except Exception as e:
        log(f"Error fetching user_id: {str(e)}", "ERROR")
        return False

def update_supabase_location(lat, lng, status):
    """Update the localisations table in Supabase with GPS coordinates"""
    global USER_ID
    
    if not SUPABASE_URL or not SUPABASE_API_KEY or not PRODUCT_KEY:
        log("Missing environment variables for Supabase connection", "ERROR")
        return False
    
    # Try to fetch user_id if not already set
    if not USER_ID and not fetch_user_id_from_product_key():
        log("Could not get user_id from product_key", "ERROR")
        return False
    
    log(f"Updating Supabase with: lat={lat}, lng={lng}, status={status}", "INFO")
    
    headers = {
        'apikey': SUPABASE_API_KEY,
        'Authorization': f'Bearer {SUPABASE_API_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal'
    }
    
    try:
        # Check if entry exists
        query_url = f"{SUPABASE_URL}/rest/v1/localisations?user_id=eq.{USER_ID}&product_key=eq.{PRODUCT_KEY}"
        
        response = requests.get(query_url, headers=headers)
        
        data = {
            'user_id': USER_ID,
            'product_key': PRODUCT_KEY,
            'lat': lat,
            'lng': lng,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        
        if response.status_code == 200 and len(response.json()) > 0:
            # Entry exists, update it
            record_id = response.json()[0]['id']
            update_url = f"{SUPABASE_URL}/rest/v1/localisations?id=eq.{record_id}"
            
            response = requests.patch(update_url, headers=headers, json=data)
            if response.status_code == 204:
                log(f"Updated location in Supabase: lat={lat}, lng={lng}, status={status}", "SUCCESS")
                return True
            else:
                log(f"Failed to update Supabase record: {response.status_code}", "ERROR")
        else:
            # Entry doesn't exist, create it
            insert_url = f"{SUPABASE_URL}/rest/v1/localisations"
            
            response = requests.post(insert_url, headers=headers, json=data)
            if response.status_code == 201:
                log(f"Created new location entry in Supabase: lat={lat}, lng={lng}, status={status}", "SUCCESS")
                return True
            else:
                log(f"Failed to create Supabase entry: {response.status_code}", "ERROR")
        
        return False
        
    except Exception as e:
        log(f"Error updating Supabase: {str(e)}", "ERROR")
        return False

def send_command(command):
    """Send a simple text command to the Arduino"""
    if not serial_port or not serial_port.is_open:
        log("Serial port not open", "ERROR")
        return False
    
    try:
        # Send command with newline
        full_command = f"{command}\n"
        serial_port.write(full_command.encode('utf-8'))
        serial_port.flush()
        log(f"Sent command: {command}", "INFO")
        return True
    except Exception as e:
        log(f"Error sending command: {str(e)}", "ERROR")
    
    return False

def read_response(timeout=1.0):
    """Read response from Arduino with timeout"""
    if not serial_port or not serial_port.is_open:
        return ""
    
    response = ""
    end_time = time.time() + timeout
    
    while time.time() < end_time:
        if serial_port.in_waiting > 0:
            try:
                line = serial_port.readline().decode('utf-8', errors='replace')
                response += line
                
                # Look for protocol-formatted GPS data
                if MSG_START in response and MSG_END in response:
                    start_idx = response.find(MSG_START)
                    end_idx = response.find(MSG_END, start_idx)
                    if start_idx != -1 and end_idx != -1:
                        protocol_msg = response[start_idx+1:end_idx].strip()
                        if protocol_msg.startswith("GPS:"):
                            process_gps_message(protocol_msg)
                
                # If we've received a complete response, return it
                if "GPS Data:" in response and "STORED" in response:
                    return response
                elif "turned ON" in response or "turned OFF" in response:
                    return response
            except Exception as e:
                log(f"Error reading response: {str(e)}", "ERROR")
                break
        
        time.sleep(0.01)  # Small delay
    
    return response

def process_gps_message(message):
    """Process GPS messages from Arduino"""
    global last_gps_data
    
    if message.startswith("GPS:"):
        try:
            # Format: GPS:lat,lng,status
            parts = message[4:].split(',')
            if len(parts) >= 2:
                lat = float(parts[0])
                lng = float(parts[1])
                status = parts[2] if len(parts) > 2 else "UNKNOWN"
                
                # Update the global variable
                last_gps_data = {
                    'lat': lat,
                    'lng': lng,
                    'status': status,
                    'timestamp': time.time()
                }
                
                log(f"Received GPS: lat={lat}, lng={lng}, status={status}", "SUCCESS")
        except Exception as e:
            log(f"Error parsing GPS data: {str(e)}", "ERROR")

def request_gps_data():
    """Request and process GPS data from Arduino"""
    global last_gps_data
    
    send_command("gps")
    response = read_response(2.0)  # Allow up to 2 seconds for response
    
    if not response:
        log("No response to GPS request", "WARNING")
        return False
    
    # Extract GPS data from human-readable format if protocol format failed
    if last_gps_data['timestamp'] == 0:
        try:
            # Try to extract latitude and longitude using regex
            lat_match = re.search(r"Latitude: ([\d.-]+)", response)
            lng_match = re.search(r"Longitude: ([\d.-]+)", response)
            status_match = re.search(r"Status: (\w+)", response)
            
            if lat_match and lng_match:
                lat = float(lat_match.group(1))
                lng = float(lng_match.group(1))
                status = status_match.group(1) if status_match else "UNKNOWN"
                
                last_gps_data = {
                    'lat': lat,
                    'lng': lng,
                    'status': status,
                    'timestamp': time.time()
                }
                
                log(f"Parsed GPS from text response: lat={lat}, lng={lng}, status={status}", "SUCCESS")
        except Exception as e:
            log(f"Error parsing text GPS response: {str(e)}", "ERROR")
    
    return True

def main():
    """Main function to run the GPS server"""
    global serial_port, running, last_gps_data
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    log("Starting NiatoShield GPS Server (USB Version)", "INFO")
    log(f"Using serial port: {SERIAL_PORT} at {BAUD_RATE} baud", "INFO")
    
    last_gps_request = 0
    last_supabase_update = 0
    
    try:
        # Initialize serial port
        serial_port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        log(f"Opened serial port {SERIAL_PORT} at {BAUD_RATE} baud", "SUCCESS")
        
        # Wait for Arduino to initialize
        time.sleep(3)
        
        # Read welcome message
        welcome = serial_port.read_all().decode('utf-8', errors='replace')
        if welcome:
            log(f"Arduino says: {welcome.strip()}", "INFO")
        
        # Main loop
        while running:
            current_time = time.time()
            
            # Request GPS data every 10 seconds
            if current_time - last_gps_request >= 10:
                log("Requesting GPS data...", "INFO")
                request_gps_data()
                last_gps_request = current_time
            
            # Update Supabase every 60 seconds
            if current_time - last_supabase_update >= 60 and last_gps_data['timestamp'] > 0:
                log("Performing scheduled Supabase update...", "INFO")
                if update_supabase_location(
                    last_gps_data['lat'],
                    last_gps_data['lng'],
                    last_gps_data['status']
                ):
                    last_supabase_update = current_time
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        log("Keyboard interrupt received, shutting down...", "WARNING")
    except Exception as e:
        log(f"Error in main loop: {str(e)}", "ERROR")
    finally:
        if serial_port and serial_port.is_open:
            serial_port.close()
            log("Serial port closed", "INFO")
        log("GPS server shutdown complete", "INFO")

if __name__ == "__main__":
    main() 