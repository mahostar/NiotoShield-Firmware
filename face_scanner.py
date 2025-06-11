#!/usr/bin/env python3
"""
NiotoShield Face Security System
--------------------------------
Combines anti-spoofing detection with face recognition for a complete security solution.
Uses GPIO button input to trigger the scanning process.

Features:
- Anti-spoofing detection using YOLOv8 model
- Face recognition using embeddings
- IP camera integration
- Database notifications
- GPIO button trigger
- Status LED indicators
- Audio feedback through audio jack
"""

import os
import sys
import cv2
import time
import json
import numpy as np
import RPi.GPIO as GPIO
import torch
from ultralytics import YOLO
import threading
import requests
from dotenv import load_dotenv
import insightface
from insightface.app import FaceAnalysis
import logging
from datetime import datetime
import glob
from statistics import mean
import argparse
import serial
import pygame
import gc  # Add explicit import for garbage collection
import uuid  # Add for generating unique filenames

# Initialize pygame mixer for audio
pygame.mixer.init()

# Audio file paths
SOUNDS = {
    'system_start': 'NiotoShield_Activated.wav',
    'scan_start': 'initiate_face_scanning_please_wait.wav',
    'access_granted': 'acesess_granted.wav',
    'spoof_detected': 'Face_spoofing_attack.wav',
    'unauthorized_face': 'unauthorized_face_detected_sending_alert_to_the_owner.wav',
    'no_face': 'no_face_detected.wav'
}

# Load sound files
loaded_sounds = {}
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
    pygame.mixer.set_num_channels(8)  # Increase number of channels for multiple sounds
    
    for key, file in SOUNDS.items():
        try:
            loaded_sounds[key] = pygame.mixer.Sound(file)
        except Exception as e:
            logger.error(f"Failed to load sound file {file}: {e}")
except Exception as e:
    logger.error(f"Error initializing sound system: {e}")

# Sound playback lock to prevent overlapping
sound_lock = threading.Lock()

def play_sound(sound_key):
    """Play a sound file with thread safety and wait for completion"""
    if sound_key in loaded_sounds:
        with sound_lock:
            try:
                # Stop any currently playing sounds
                pygame.mixer.stop()
                sound = loaded_sounds[sound_key]
                sound.play()
                # Wait for the sound to finish playing
                duration = sound.get_length()
                time.sleep(duration + 0.1)  # Add small buffer
            except Exception as e:
                logger.error(f"Error playing sound {sound_key}: {e}")

def play_sound_sequence(sound_keys, delay=0.1):
    """Play a sequence of sounds with delays between them"""
    for key in sound_keys:
        play_sound(key)
        time.sleep(delay)  # Add delay between sounds

# Parse command line arguments
parser = argparse.ArgumentParser(description='NiotoShield Face Security System')
parser.add_argument('--headless', action='store_true', help='Run in headless mode without GUI')
parser.add_argument('--led_pins', type=str, help='Comma-separated GPIO pin numbers for green,red LEDs (e.g. 22,27)')
args = parser.parse_args()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("face_security")

# GPIO Configuration
BUTTON_PIN = 17  # GPIO17 (pin 11)

# LED Configuration
GREEN_LED_PIN = None  # Will be set from command line args
RED_LED_PIN = None    # Will be set from command line args

# Parse LED pin arguments if provided
if args.led_pins:
    try:
        pins = args.led_pins.split(',')
        if len(pins) >= 2:
            GREEN_LED_PIN = int(pins[0])
            RED_LED_PIN = int(pins[1])
            logger.info(f"Using LED pins: Green={GREEN_LED_PIN}, Red={RED_LED_PIN}")
    except Exception as e:
        logger.error(f"Error parsing LED pins: {e}")

# Camera Configuration
CAMERA_DEVICE = '/dev/video0'  # USB camera device
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Anti-spoofing Configuration
MODEL_PATH = "EasyShield_V2.5.pt"
CONFIDENCE_THRESHOLD = 0.5
MAX_SCAN_FRAMES = 5
SCAN_DELAY = 150  # ms between captures

# Face recognition Configuration
EMBEDDINGS_FOLDER = "embeddings"
FACE_MATCH_THRESHOLD = 0.6

# Database notification types
NOTIFICATION_TYPES = {
    "SPOOF_DETECTED": "[ALERT] Security Alert: Possible spoofing attempt detected",
    "NO_FACE_MATCH": "[ALERT] Security Alert: Unrecognized person detected",
    "FACE_RECOGNIZED": "[ACCESS] Access granted: Authorized person verified"
}

# Add relay control configuration
RELAY_SERIAL_PORT = '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'
RELAY_BAUD_RATE = 9600
RELAY_TIMEOUT = 1.0

class FaceSecurity:
    def __init__(self):
        logger.info("Initializing NiotoShield Face Security System...")
        
        # Set headless mode
        self.headless = args.headless
        
        # Initialize state variables
        self.running = False
        self.button_pressed = False
        self.face_app = None
        self.model = None
        self.cap = None
        self.known_embeddings = []
        self.button_thread = None
        self.serial_port = None
        self.relay_on = False
        self.camera_status = False  # Track camera status for LED
        self.red_led_blink_thread = None  # Thread for blinking red LED
        self.is_blinking = False  # Flag to control blinking
        
        # Add scanning lock
        self.scanning_lock = threading.Lock()
        self.is_scanning = False
        
        # Initialize GPIO for LED indicators
        self.setup_leds()
        
        # Load environment variables for database access
        load_dotenv(override=True)
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.product_key = os.getenv('PRODUCT_KEY')
        
        if not all([self.supabase_url, self.supabase_key, self.product_key]):
            logger.error("Missing environment variables. Check .env file.")
            sys.exit(1)
        
        # Initialize face recognition
        self.initialize_face_recognition()
        
        # Load anti-spoofing model
        self.load_model()
        
        # Initialize camera
        self.initialize_camera()
        
        # Set up GPIO
        self.setup_gpio()
        
        # Initialize serial connection to Arduino
        self.initialize_serial()
        
        logger.info("System initialization complete. Ready for operation.")
        
    def setup_leds(self):
        """Set up GPIO pins for LED indicators"""
        if GREEN_LED_PIN is not None and RED_LED_PIN is not None:
            try:
                # Set up GPIO mode if not already set
                GPIO.setmode(GPIO.BCM)
                # Configure LED pins as outputs
                GPIO.setup(GREEN_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(RED_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
                logger.info(f"LED indicators configured: Green={GREEN_LED_PIN}, Red={RED_LED_PIN}")
                
                # Green LED starts OFF, will be turned on when system is fully running
                self.set_green_led(False)
                # Red LED starts OFF, will only turn on for camera issues
                self.set_red_led(False)
            except Exception as e:
                logger.error(f"Failed to initialize LED pins: {e}")
        else:
            logger.info("LED pins not specified, status indicators disabled")
    
    def set_green_led(self, state):
        """Set green LED state (True=on, False=off)"""
        if GREEN_LED_PIN is not None:
            try:
                GPIO.output(GREEN_LED_PIN, GPIO.HIGH if state else GPIO.LOW)
                logger.debug(f"Green LED set to {'ON' if state else 'OFF'}")
            except Exception as e:
                logger.error(f"Error setting green LED: {e}")
    
    def set_red_led(self, state):
        """Set red LED state (True=on, False=off)
        
        For the red LED:
        - ON (True) means camera issues
        - OFF (False) means camera working
        """
        if RED_LED_PIN is not None:
            try:
                GPIO.output(RED_LED_PIN, GPIO.HIGH if state else GPIO.LOW)
                logger.debug(f"Red LED set to {'ON' if state else 'OFF'}")
            except Exception as e:
                logger.error(f"Error setting red LED: {e}")
    
    def update_camera_led_status(self):
        """Update red LED based on current camera status
        
        - If camera_status is True (camera working), red LED should be OFF
        - If camera_status is False (camera issues), red LED should be ON
        """
        # Skip if blinking is active
        if self.is_blinking:
            return
            
        # Set red LED based on camera status
        # ON if camera issue (not self.camera_status)
        # OFF if camera working (self.camera_status)
        self.set_red_led(not self.camera_status)  # ON if camera issue, OFF if camera OK
    
    def blink_red_led(self):
        """Blink the red LED during scanning"""
        logger.debug("Red LED blinking started")
        
        # Cache original camera status at start of blinking
        original_camera_status = self.camera_status
        
        while self.is_blinking and self.running:
            try:
                if RED_LED_PIN is not None:
                    GPIO.output(RED_LED_PIN, GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(RED_LED_PIN, GPIO.LOW)
                    time.sleep(0.3)
            except Exception as e:
                logger.error(f"Error during LED blinking: {e}")
                break
        
        # Make sure LED returns to correct state when finished blinking
        if RED_LED_PIN is not None:
            try:
                # Get the current camera status (may have changed during blinking)
                current_camera_status = self.camera_status
                
                # Set LED: ON if camera issue (not camera_status), OFF if camera OK
                GPIO.output(RED_LED_PIN, GPIO.HIGH if not current_camera_status else GPIO.LOW)
                logger.debug(f"Red LED reset to {'ON' if not current_camera_status else 'OFF'} after blinking")
            except Exception as e:
                logger.error(f"Error resetting LED after blinking: {e}")
        
        logger.debug("Red LED blinking stopped")
    
    def start_red_led_blinking(self):
        """Start blinking the red LED in a separate thread"""
        if self.red_led_blink_thread is None or not self.red_led_blink_thread.is_alive():
            self.is_blinking = True
            self.red_led_blink_thread = threading.Thread(target=self.blink_red_led)
            self.red_led_blink_thread.daemon = True
            self.red_led_blink_thread.start()
            logger.debug("Red LED blink thread started")
    
    def stop_red_led_blinking(self):
        """Stop the red LED from blinking"""
        self.is_blinking = False
        if self.red_led_blink_thread is not None and self.red_led_blink_thread.is_alive():
            # Wait for thread to finish
            self.red_led_blink_thread.join(timeout=1.0)
            logger.debug("Red LED blink thread stopped")
        
        # Reset the LED to the correct state based on camera status
        self.set_red_led(not self.camera_status)  # ON if camera issue, OFF if camera OK
    
    def setup_gpio(self):
        """Set up GPIO for button input"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            logger.info(f"GPIO initialized. Button configured on GPIO{BUTTON_PIN}.")
        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")
            sys.exit(1)
    
    def initialize_face_recognition(self):
        """Initialize InsightFace for face recognition"""
        try:
            logger.info("Initializing face recognition system...")
            self.face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(320, 320))  # Use smaller detection size for Pi
            logger.info("Face recognition initialized successfully")
            
            # Load embeddings
            self.load_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize face recognition: {e}")
            sys.exit(1)
    
    def load_embeddings(self):
        """Load face embeddings from the embeddings folder"""
        try:
            if not os.path.exists(EMBEDDINGS_FOLDER):
                logger.error(f"Embeddings folder '{EMBEDDINGS_FOLDER}' not found")
                return
            
            metadata_path = os.path.join(EMBEDDINGS_FOLDER, "embeddings_metadata.json")
            if not os.path.exists(metadata_path):
                logger.error(f"Embeddings metadata file not found")
                return
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.known_embeddings = []
            for entry in metadata:
                embedding_path = entry["embedding_path"]
                if os.path.exists(embedding_path):
                    embedding = np.load(embedding_path)
                    self.known_embeddings.append({
                        "embedding": embedding,
                        "filename": entry["filename"],
                        "confidence": entry["confidence"]
                    })
            
            logger.info(f"Loaded {len(self.known_embeddings)} face embeddings")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
    
    def load_model(self):
        """Load the YOLO anti-spoofing model"""
        try:
            logger.info(f"Loading anti-spoofing model from {MODEL_PATH}...")
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found: {MODEL_PATH}")
                sys.exit(1)
            
            # Force garbage collection before loading model
            gc.collect()
                
            self.model = YOLO(MODEL_PATH)
            
            # Force to CPU for Raspberry Pi (unless you have a Pi with GPU)
            self.model.to('cpu')
            logger.info("Anti-spoofing model loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
    
    def initialize_camera(self):
        """Initialize the USB camera"""
        try:
            logger.info(f"Initializing USB camera from {CAMERA_DEVICE}...")
            
            # Force garbage collection before creating new camera object
            gc.collect()
            
            # Open the USB camera
            self.cap = cv2.VideoCapture(CAMERA_DEVICE)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open USB camera at {CAMERA_DEVICE}")
                self.camera_status = False
                self.update_camera_led_status()
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Read a test frame to verify camera works
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("Failed to read frame from USB camera")
                self.camera_status = False
                self.update_camera_led_status()
                return
            
            # Camera is working
            self.camera_status = True
            self.update_camera_led_status()
            logger.info("USB camera initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing USB camera: {e}")
            self.cap = None
            self.camera_status = False
            self.update_camera_led_status()

    # Simplified camera check and reset function
    def check_and_reset_camera(self):
        """Check camera connection and reset if needed"""
        if self.cap is None or not self.cap.isOpened():
            logger.warning("USB camera not available, trying to reinitialize")
            
            # Close the camera if it exists
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
                self.cap = None
            
            # Force garbage collection
            gc.collect()
            
            # Reinitialize camera
            try:
                self.cap = cv2.VideoCapture(CAMERA_DEVICE)
                if self.cap.isOpened():
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Update camera status
                    self.camera_status = True
                    self.update_camera_led_status()
                    logger.info("USB camera reinitialized successfully")
                    return True
                else:
                    self.camera_status = False
                    self.update_camera_led_status()
                    return False
            except Exception as e:
                logger.error(f"Error reinitializing USB camera: {e}")
                self.camera_status = False
                self.update_camera_led_status()
                return False
        
        return True  # Camera is already initialized and opened

    # Replace complex reconnection functions with simplified versions
    def reconnect_camera(self):
        """Simple wrapper for check_and_reset_camera"""
        return self.check_and_reset_camera()
    
    def camera_retry_connect(self):
        """For compatibility - just calls check_and_reset_camera"""
        self.check_and_reset_camera()

    def initialize_serial(self):
        """Initialize serial connection to the Arduino for relay control"""
        try:
            logger.info(f"Initializing serial connection to Arduino on {RELAY_SERIAL_PORT}...")
            self.serial_port = serial.Serial(RELAY_SERIAL_PORT, RELAY_BAUD_RATE, timeout=RELAY_TIMEOUT)
            time.sleep(2)  # Wait for Arduino to initialize
            logger.info("Serial connection established successfully")
            
            # Read initial message from Arduino
            if self.serial_port and self.serial_port.is_open:
                welcome = self.serial_port.read_all().decode('utf-8', errors='replace')
                if welcome:
                    logger.info(f"Arduino says: {welcome.strip()}")
        except Exception as e:
            logger.error(f"Failed to initialize serial connection: {e}")
            self.serial_port = None

    def send_relay_command(self, command):
        """Send command to control the relay"""
        if self.serial_port is None or not self.serial_port.is_open:
            logger.error("Serial port not available for relay control")
            return False
        
        try:
            # Add newline to command
            full_command = f"{command}\n"
            self.serial_port.write(full_command.encode('utf-8'))
            self.serial_port.flush()
            logger.info(f"Sent relay command: {command}")
            
            # Read response
            time.sleep(0.2)  # Short delay to allow Arduino to respond
            response = self.serial_port.read_all().decode('utf-8', errors='replace')
            if response:
                logger.info(f"Arduino response: {response.strip()}")
            
            # Update relay state
            if command == "relayon":
                self.relay_on = True
            elif command == "relayoff":
                self.relay_on = False
                
            return True
        except Exception as e:
            logger.error(f"Error sending relay command: {e}")
            return False

    def button_monitor(self):
        """Monitor the button for presses with debouncing and scan lock"""
        logger.info("Button monitoring started. Press the button to trigger a scan.")
        
        debounce_time = 0
        while self.running:
            if GPIO.input(BUTTON_PIN) and (time.time() - debounce_time) > 1:
                # If relay is on, turn it off first and then trigger a new scan
                if self.relay_on:
                    logger.info("Relay is ON and button pressed - turning relay OFF and starting new scan")
                    self.send_relay_command("relayoff")
                    debounce_time = time.time()
                    time.sleep(0.5)  # Short delay before proceeding with scan
                
                # Check if a scan is already in progress
                if self.is_scanning:
                    logger.warning("Scan already in progress - ignoring button press")
                    debounce_time = time.time()
                    continue
                
                # Try to acquire the scanning lock
                if self.scanning_lock.acquire(blocking=False):
                    try:
                        logger.info("Button pressed! Starting security scan...")
                        self.is_scanning = True
                        self.button_pressed = True
                        debounce_time = time.time()
                        
                        # Run the security scan in a separate thread
                        scan_thread = threading.Thread(target=self.run_security_scan)
                        scan_thread.start()
                    except Exception as e:
                        logger.error(f"Error starting scan: {e}")
                        self.is_scanning = False
                        self.button_pressed = False
                    finally:
                        self.scanning_lock.release()
                else:
                    logger.warning("System busy - ignoring button press")
            
            time.sleep(0.1)  # Reduce CPU usage
    
    def get_user_id(self):
        """Get the user ID associated with the product key from Supabase"""
        try:
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}"
            }
            
            url = f"{self.supabase_url}/rest/v1/products"
            params = {
                "product_key": f"eq.{self.product_key}",
                "select": "user_id"
            }
            
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200 or not response.json():
                logger.error(f"Could not find user ID for product key: {self.product_key}")
                return None
            
            return response.json()[0]['user_id']
        except Exception as e:
            logger.error(f"Error getting user ID: {e}")
            return None
    
    def upload_alert_image(self, frame):
        """Upload an alert image to Supabase storage bucket"""
        try:
            # Generate unique filename
            filename = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
            
            # Save frame temporarily
            temp_path = f"/tmp/{filename}"
            cv2.imwrite(temp_path, frame)
            
            # Prepare headers for Supabase storage API
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}"
            }
            
            # Upload to Supabase storage
            with open(temp_path, 'rb') as f:
                files = {'file': (filename, f, 'image/jpeg')}
                url = f"{self.supabase_url}/storage/v1/object/alert-pictures/{filename}"
                response = requests.post(url, headers=headers, files=files)
                
                if response.status_code in [200, 201]:
                    # Get public URL
                    public_url = f"{self.supabase_url}/storage/v1/object/public/alert-pictures/{filename}"
                    logger.info(f"Alert image uploaded successfully: {filename}")
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    return public_url
                else:
                    logger.error(f"Failed to upload alert image: {response.text}")
                    return None
                
        except Exception as e:
            logger.error(f"Error uploading alert image: {e}")
            return None

    def add_notification(self, message, image_url=None):
        """Add a notification to the Supabase notifications table with optional image URL"""
        try:
            user_id = self.get_user_id()
            if not user_id:
                logger.error("Failed to get user ID for notification")
                return False
            
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            notification_url = f"{self.supabase_url}/rest/v1/notifications"
            notification_data = {
                "user_id": user_id,
                "message": message,
                "is_read": False,
                "image_url": image_url
            }
            
            response = requests.post(notification_url, headers=headers, json=notification_data)
            
            if response.status_code in [200, 201, 204]:
                logger.info(f"Notification added: '{message}' with image: {image_url}")
                return True
            else:
                logger.error(f"Failed to add notification: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error adding notification: {e}")
            return False
    
    def get_face_crop(self, frame, face):
        """Extract face region with context from the frame"""
        x1, y1, x2, y2 = list(map(int, face.bbox))
        width = x2 - x1
        height = y2 - y1
        
        # Add padding to capture more context
        padding_factor = 0.5
        x_padding = int(width * padding_factor)
        y_padding = int(height * padding_factor)
        
        # Calculate new coordinates with padding
        x1_padded = max(0, x1 - x_padding)
        y1_padded = max(0, y1 - y_padding)
        x2_padded = min(frame.shape[1], x2 + x_padding)
        y2_padded = min(frame.shape[0], y2 + y_padding)
        
        # Extract padded face region
        face_img = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Resize to expected input size for anti-spoofing model
        try:
            if face_img.size > 0:
                resized_face = cv2.resize(face_img, (640, 640))
                return resized_face, (x1_padded, y1_padded, x2_padded, y2_padded)
        except Exception as e:
            logger.error(f"Error preparing face crop: {e}")
        
        return None, None
    
    def calculate_blur(self, image):
        """Calculate the blur level of an image using Laplacian variance"""
        if image is None or image.size == 0:
            return 0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def run_anti_spoofing(self, face_img):
        """Run anti-spoofing detection on a face image"""
        try:
            # Make sure model is loaded
            if self.model is None:
                logger.error("Anti-spoofing model not loaded")
                return False, 0.0
                
            # Use a with torch.no_grad() block to reduce memory usage
            with torch.no_grad():
                results = self.model(
                    face_img,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the prediction result (0 = real, 1 = fake)
                result = results[0]
                conf = float(result.boxes.conf[0])
                cls = int(result.boxes.cls[0])
                
                is_real = cls == 0  # Class 0 is real
                
                logger.info(f"Anti-spoofing result: {'REAL' if is_real else 'FAKE'} with confidence {conf:.2f}")
                return is_real, conf
            else:
                logger.warning("No anti-spoofing results detected")
                return False, 0.0
        except Exception as e:
            logger.error(f"Error in anti-spoofing detection: {e}")
            return False, 0.0
    
    def compare_face(self, face_embedding):
        """Compare face embedding with all stored embeddings"""
        if not self.known_embeddings:
            logger.warning("No known embeddings to compare with")
            return False, None, 0.0
        
        # Find the best match
        best_match = None
        best_similarity = -1
        
        for known in self.known_embeddings:
            similarity = np.dot(known["embedding"], face_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = known
        
        # Check if the similarity passes the threshold
        is_match = best_similarity >= FACE_MATCH_THRESHOLD
        
        if is_match:
            logger.info(f"Face recognized as {best_match['filename']} with similarity {best_similarity:.2f}")
        else:
            logger.info(f"No match found. Best similarity: {best_similarity:.2f} with {best_match['filename']}")
        
        return is_match, best_match, best_similarity
    
    def capture_frames(self):
        """Capture multiple frames with robust error handling and recovery"""
        logger.info(f"Starting frame capture sequence ({MAX_SCAN_FRAMES} frames)...")
        frames = []
        failed_reads = 0
        max_failed_reads = 3
        
        # Pre-allocate array for frames to avoid memory fragmentation
        try:
            # Force garbage collection before starting frame capture
            gc.collect()
            
            for i in range(MAX_SCAN_FRAMES):
                try:
                    # Check if we've already captured enough frames
                    if len(frames) >= MAX_SCAN_FRAMES:
                        break
                        
                    logger.info(f"Capturing frame {i+1}/{MAX_SCAN_FRAMES}")
                    
                    # Verify camera is available
                    if self.cap is None or not self.cap.isOpened():
                        logger.error("Camera not available for frame capture")
                        
                        # Only attempt reconnection if we don't have enough frames yet
                        if len(frames) < 3:  # At least 3 frames needed for reliable analysis
                            logger.info("Attempting camera reconnection during frame capture...")
                            reconnect_success = self.reconnect_camera()
                            
                            if not reconnect_success:
                                logger.error("Failed to reconnect camera during frame capture")
                                break
                        else:
                            # We already have some frames, just return what we have
                            logger.info(f"Already captured {len(frames)} frames, continuing with analysis")
                            break
                    
                    # Read frame with timeout protection
                    read_start = time.time()
                    read_timeout = 1.0  # 1 second timeout
                    
                    # Use try/except around frame read
                    try:
                        ret, frame = self.cap.read()
                        read_duration = time.time() - read_start
                        
                        # Check for timeout during read
                        if read_duration > read_timeout:
                            logger.warning(f"Frame read timeout ({read_duration:.2f}s)")
                            failed_reads += 1
                            continue
                    except Exception as e:
                        logger.error(f"Exception during frame read: {e}")
                        failed_reads += 1
                        
                        # Force garbage collection after error
                        gc.collect()
                        continue
                    
                    # Check read result
                    if not ret or frame is None or frame.size == 0:
                        failed_reads += 1
                        logger.error(f"Failed to capture frame (attempt {failed_reads}/{max_failed_reads})")
                        
                        if failed_reads >= max_failed_reads:
                            logger.error("Too many failed frame captures, trying camera reconnect")
                            # Try a final reconnection if we have too many failed reads
                            self.reconnect_camera()
                            failed_reads = 0
                            continue
                        
                        # Skip this frame
                        continue
                    
                    # Reset failed read counter on success
                    failed_reads = 0
                    
                    # Make a copy of the frame to avoid reference issues
                    try:
                        frame_copy = frame.copy()
                        frames.append(frame_copy)
                    except Exception as e:
                        logger.error(f"Error copying frame: {e}")
                        continue
                    
                    # Display debug frame
                    if not self.headless:
                        try:
                            debug_frame = frame.copy()
                            cv2.putText(debug_frame, f"Capturing {i+1}/{MAX_SCAN_FRAMES}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow("NiotoShield Security", debug_frame)
                            cv2.waitKey(1)
                        except Exception as e:
                            logger.error(f"Error displaying debug frame: {e}")
                    
                    # Delay between captures
                    time.sleep(SCAN_DELAY / 1000)
                    
                except Exception as e:
                    logger.error(f"Error in frame capture loop: {e}")
                    # Force garbage collection after any error
                    gc.collect()
                    # Short delay to recover
                    time.sleep(0.2)
            
            # Check if we have enough frames
            if len(frames) == 0:
                logger.error("Failed to capture any frames")
                return []
            
            logger.info(f"Captured {len(frames)} frames for analysis")
            return frames
            
        except Exception as e:
            logger.error(f"Critical error in frame capture: {e}")
            # Return any frames we managed to capture
            logger.info(f"Returning {len(frames)} frames despite error")
            return frames
    
    def run_security_scan(self):
        """Run the complete security scan process with robust error handling"""
        logger.info("Starting security scan process...")
        
        try:
            # Play scan start sound first, before any LED changes
            play_sound('scan_start')
            time.sleep(0.5)  # Wait for the sound to complete
            
            # Start blinking the red LED to indicate scanning
            self.start_red_led_blinking()
            
            # Check if camera is available
            if not self.check_and_reset_camera():
                logger.error("USB camera not available. Aborting security scan.")
                self.button_pressed = False
                self.add_notification("Security scan failed: Camera unavailable")
                return
            
            # Reset state
            self.button_pressed = False
            
            # Capture multiple frames
            frames = self.capture_frames()
            if not frames:
                logger.error("No frames captured")
                self.add_notification("Security scan failed: No frames captured")
                play_sound('no_face')
                return
            
            # Force garbage collection before processing
            gc.collect()
            
            # Process frames with error handling
            try:
                # Process frames for anti-spoofing
                processed_faces = []
                blur_scores = []
                rectangles = []
                
                for i, frame in enumerate(frames):
                    # Skip invalid frames
                    if frame is None or frame.size == 0:
                        continue
                        
                    # Display processing status
                    if not self.headless:
                        debug_frame = frame.copy()
                        cv2.putText(debug_frame, f"Processing {i+1}/{len(frames)}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("NiotoShield Security", debug_frame)
                        cv2.waitKey(1)
                    
                    # Detect faces
                    faces = self.face_app.get(frame)
                    
                    if not faces:
                        logger.warning(f"No faces detected in frame {i+1}")
                        continue
                    
                    # Use the face with highest detection score
                    face = max(faces, key=lambda x: x.det_score)
                    
                    # Get face crop for anti-spoofing
                    face_img, rect = self.get_face_crop(frame, face)
                    if face_img is not None:
                        # Calculate blur level
                        blur = self.calculate_blur(face_img)
                        
                        processed_faces.append({
                            "frame_index": i,
                            "face_img": face_img,
                            "face": face,
                            "blur": blur
                        })
                        blur_scores.append(blur)
                        rectangles.append(rect)
                        
                        logger.info(f"Frame {i+1}: Face detected with blur score {blur:.2f}")
                    
                    # Force garbage collection periodically
                    if i % 10 == 0:
                        gc.collect()
                
                if not processed_faces:
                    logger.error("No valid faces detected in any frame")
                    self.add_notification("Security scan failed: No faces detected")
                    play_sound('no_face')
                    return
                
                # 3. Run anti-spoofing on all faces
                real_count = 0
                fake_count = 0
                confidence_scores = []
                
                for i, processed in enumerate(processed_faces):
                    # Run anti-spoofing
                    is_real, confidence = self.run_anti_spoofing(processed["face_img"])
                    confidence_scores.append(confidence)
                    
                    if is_real:
                        real_count += 1
                    else:
                        fake_count += 1
                    
                    # Display result on frame
                    if not self.headless:
                        result_frame = frames[processed["frame_index"]].copy()
                        x1, y1, x2, y2 = rectangles[i]
                        color = (0, 255, 0) if is_real else (0, 0, 255)
                        label = f"REAL {confidence:.2f}" if is_real else f"FAKE {confidence:.2f}"
                        
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        cv2.imshow("NiotoShield Security", result_frame)
                        cv2.waitKey(500)  # Show each result for half a second
                
                # 4. Decision on anti-spoofing
                is_real_person = real_count > fake_count
                avg_confidence = mean(confidence_scores) if confidence_scores else 0
                
                logger.info(f"Anti-spoofing results: {real_count} real, {fake_count} fake")
                logger.info(f"Average confidence: {avg_confidence:.2f}")
                
                if not is_real_person:
                    # Spoof detected - upload image and send alert
                    logger.warning("SECURITY ALERT: Spoofing attempt detected!")
                    
                    # Upload the frame with the highest quality
                    best_frame = frames[processed_faces[0]["frame_index"]]
                    image_url = self.upload_alert_image(best_frame)
                    
                    # Send notification with image URL
                    self.add_notification(NOTIFICATION_TYPES["SPOOF_DETECTED"], image_url)
                    play_sound('spoof_detected')
                    
                    # Show final result
                    if not self.headless:
                        result_frame = frames[0].copy()
                        cv2.putText(result_frame, "SPOOF DETECTED", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        cv2.imshow("NiotoShield Security", result_frame)
                        cv2.waitKey(3000)  # Show for 3 seconds
                    return
                
                # 5. Find the least blurry face for recognition
                if blur_scores:
                    # Get the index of the face with the highest blur score (least blurry)
                    best_face_idx = blur_scores.index(max(blur_scores))
                    best_processed = processed_faces[best_face_idx]
                    
                    logger.info(f"Selected frame {best_processed['frame_index']+1} with blur score {best_processed['blur']:.2f} for recognition")
                    
                    # Get the face embedding for recognition
                    face_embedding = best_processed["face"].normed_embedding
                    
                    # 6. Compare with known faces
                    is_match, best_match, similarity = self.compare_face(face_embedding)
                    
                    if is_match:
                        # Face recognized - turn on relay
                        logger.info(f"SUCCESS: Face recognized as {best_match['filename']} with similarity {similarity:.2f}")
                        self.add_notification(NOTIFICATION_TYPES["FACE_RECOGNIZED"])
                        play_sound('access_granted')
                        
                        # Turn on the relay
                        if self.send_relay_command("relayon"):
                            logger.info("Relay turned ON after successful face recognition")
                            self.add_notification("Vehicle security system deactivated - Access granted")
                        else:
                            logger.error("Failed to turn on relay")
                        
                        # Show successful recognition
                        if not self.headless:
                            result_frame = frames[best_processed["frame_index"]].copy()
                            x1, y1, x2, y2 = rectangles[best_face_idx]
                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(result_frame, f"ACCESS GRANTED: {best_match['filename']}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.imshow("NiotoShield Security", result_frame)
                            cv2.waitKey(3000)  # Show for 3 seconds
                    else:
                        # Real person but not recognized - upload image and send alert
                        logger.warning(f"SECURITY ALERT: Unrecognized person detected. Best match: {best_match['filename']} ({similarity:.2f})")
                        
                        # Upload the best quality frame
                        best_frame = frames[best_processed["frame_index"]]
                        image_url = self.upload_alert_image(best_frame)
                        
                        # Send notification with image URL
                        self.add_notification(NOTIFICATION_TYPES["NO_FACE_MATCH"], image_url)
                        play_sound('unauthorized_face')
                        
                        # Show unrecognized face
                        if not self.headless:
                            result_frame = frames[best_processed["frame_index"]].copy()
                            x1, y1, x2, y2 = rectangles[best_face_idx]
                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            cv2.putText(result_frame, "ACCESS DENIED: UNRECOGNIZED", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            cv2.imshow("NiotoShield Security", result_frame)
                            cv2.waitKey(3000)  # Show for 3 seconds
            except Exception as e:
                logger.error(f"Error processing frames: {e}")
                self.add_notification("Security scan failed: Processing error")
                return
            
        except Exception as e:
            logger.error(f"Error during security scan: {e}")
            self.add_notification(f"Security scan error: {str(e)}")
        finally:
            # Stop blinking and reset LED to appropriate state
            self.stop_red_led_blinking()
            
            # Reset scanning state and button state
            self.is_scanning = False
            self.button_pressed = False
            # Force final garbage collection
            gc.collect()
            logger.info("Security scan completed")
    
    def start(self):
        """Start the security system"""
        logger.info("Starting NiotoShield Face Security System...")
        self.running = True
        
        # Play system start sound and wait for it to complete
        play_sound('system_start')
        time.sleep(0.5)  # Add delay after system start sound
        
        # Turn on green LED to indicate system is running
        self.set_green_led(True)
        
        # Make sure red LED matches camera status at startup
        self.update_camera_led_status()
        
        # Start button monitoring in a separate thread
        self.button_thread = threading.Thread(target=self.button_monitor)
        self.button_thread.daemon = True
        self.button_thread.start()
        
        # Create named window for display if not in headless mode
        if not self.headless:
            cv2.namedWindow("NiotoShield Security", cv2.WINDOW_NORMAL)
        
        logger.info("System running. Press the button to trigger a security scan.")
        self.add_notification("[SYSTEM] NiotoShield Security System activated")
        
        try:
            # Main loop
            while self.running:
                # Check camera periodically
                if not self.check_and_reset_camera():
                    # If camera check failed, wait a bit before retrying
                    time.sleep(1)
                    continue
                
                # Read frame for live view
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from USB camera")
                    # Camera might need resetting
                    self.check_and_reset_camera()
                    time.sleep(0.5)
                    continue
                
                # Frame read successful, update camera status
                if not self.camera_status:
                    self.camera_status = True
                    self.update_camera_led_status()
                
                # Display live view if not in headless mode
                if not self.headless:
                    cv2.putText(frame, "NiotoShield Security - Press button to scan", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                              (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("NiotoShield Security", frame)
                    
                    # Check for 'q' key to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit key pressed. Shutting down...")
                        self.running = False
                        break
                
                # Don't use 100% CPU
                time.sleep(0.03)  # ~30 FPS
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
        except MemoryError:
            logger.error("Memory error detected. Attempting to recover...")
            # Force garbage collection
            gc.collect()
            # Try to continue operation
            if not self.cap.isOpened():
                self.reconnect_camera()
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the security system"""
        self.running = False
        
        # Wait for any ongoing scan to complete
        if self.is_scanning:
            logger.info("Waiting for ongoing scan to complete...")
            with self.scanning_lock:
                pass
        
        # Turn off relay before shutting down
        if self.relay_on and self.serial_port and self.serial_port.is_open:
            logger.info("Turning off relay before shutdown")
            self.send_relay_command("relayoff")
        
        # Close serial port
        if self.serial_port and self.serial_port.is_open:
            logger.info("Closing serial port")
            self.serial_port.close()
        
        # Force cleanup of resources
        try:
            # Turn off LEDs - system is no longer running
            self.set_green_led(False)
            self.set_red_led(False)
            
            # Release resources
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Clean up OpenCV windows if not in headless mode
            if not self.headless:
                cv2.destroyAllWindows()
            
            # Clean up GPIO
            GPIO.cleanup()
            
            # Force garbage collection before exit
            gc.collect()
            
            logger.info("NiotoShield Face Security System shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    logger.info("Starting NiotoShield Face Security System")
    
    try:
        security_system = FaceSecurity()
        security_system.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # If using LEDs and the error is related to camera, exit with code 5
        if "camera" in str(e).lower() and (GREEN_LED_PIN is not None or RED_LED_PIN is not None):
            sys.exit(5)  # Special exit code for camera errors
        sys.exit(1)

if __name__ == "__main__":
    main()  