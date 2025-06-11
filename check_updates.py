#!/usr/bin/env python3
"""
Database Update Monitor
-----------------------
Checks for updates in the user profile and runs the sequence script if needed.
This script runs in the background, checking every 5 minutes.

Usage:
  python check_updates.py           # Run in background mode (default)
  python check_updates.py --console # Run with console output visible
"""

import os
import json
import time
import requests
import subprocess
import sys
import argparse
from dotenv import load_dotenv

# Time between checks (in seconds)
CHECK_INTERVAL = 300  # 5 minutes

def check_file_integrity():
    """
    Check if essential files and folders exist.
    Returns True if everything is in order, False if something is missing
    """
    user_data_exists = os.path.exists('user_data.json')
    embeddings_folder_exists = os.path.exists('embeddings')
    embeddings_has_files = False
    
    if embeddings_folder_exists:
        # Check if embeddings folder has at least one .npy file
        embeddings_files = [f for f in os.listdir('embeddings') if f.endswith('.npy')]
        embeddings_has_files = len(embeddings_files) > 0
    
    # Also check for embeddings metadata
    embeddings_metadata_exists = os.path.exists(os.path.join('embeddings', 'embeddings_metadata.json')) if embeddings_folder_exists else False
    
    return user_data_exists and embeddings_folder_exists and embeddings_has_files and embeddings_metadata_exists

def get_db_updated_at():
    """
    Fetch the updated_at timestamp from Supabase user_profiles table
    Returns the timestamp or None if there's an error
    """
    load_dotenv(override=True)
    
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    PRODUCT_KEY = os.getenv('PRODUCT_KEY')
    
    if not all([SUPABASE_URL, SUPABASE_KEY, PRODUCT_KEY]):
        print("⚠️ Missing required environment variables")
        return None
    
    try:
        # First get the user_id associated with the product key
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        # Query the products table to get the user_id
        url = f"{SUPABASE_URL}/rest/v1/products"
        params = {
            "product_key": f"eq.{PRODUCT_KEY}",
            "select": "user_id"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200 or not response.json():
            print(f"⚠️ Could not find user ID for product key: {PRODUCT_KEY}")
            return None
        
        user_id = response.json()[0]['user_id']
        
        # Now get the user profile with the updated_at timestamp
        profile_url = f"{SUPABASE_URL}/rest/v1/user_profiles"
        profile_params = {
            "id": f"eq.{user_id}",
            "select": "updated_at"
        }
        
        profile_response = requests.get(profile_url, headers=headers, params=profile_params)
        if profile_response.status_code != 200 or not profile_response.json():
            print(f"⚠️ Could not find user profile for ID: {user_id}")
            return None
        
        return profile_response.json()[0]['updated_at']
            
    except Exception as e:
        print(f"⚠️ Error checking database: {str(e)}")
        return None

def get_local_updated_at():
    """
    Get the updated_at timestamp from the local user_data.json file
    Returns the timestamp or None if there's an error
    """
    try:
        if not os.path.exists('user_data.json'):
            return None
            
        with open('user_data.json', 'r') as f:
            user_data = json.load(f)
            
        return user_data.get('updated_at')
    except Exception as e:
        print(f"⚠️ Error reading local data: {str(e)}")
        return None

def run_sequence():
    """
    Activate the virtual environment and run the sequence script
    """
    try:
        # Get the absolute path to run_sequence.py
        script_path = os.path.abspath('run_sequence.py')
        
        # Determine the correct activation command based on OS
        if os.name == 'nt':  # Windows
            activate_cmd = 'rasso\\Scripts\\activate'
            run_cmd = [sys.executable, script_path]
        else:  # Linux/Mac
            activate_cmd = 'source rasso/bin/activate'
            run_cmd = [sys.executable, script_path]
        
        # On Linux/Mac, we need to use shell=True and combine commands
        if os.name != 'nt':
            full_cmd = f"{activate_cmd} && {sys.executable} {script_path}"
            print(f"Running: {full_cmd}")
            subprocess.run(full_cmd, shell=True, check=True)
        else:
            # On Windows, we can activate and then run separately
            subprocess.run(activate_cmd, shell=True, check=True)
            subprocess.run(run_cmd, check=True)
            
        print(f"✅ Sequence completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"⚠️ Error running sequence: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Monitor for database updates')
    parser.add_argument('--console', action='store_true', help='Run with console output visible')
    args = parser.parse_args()
    
    console_mode = args.console
    
    if console_mode:
        # Print a more visible header in console mode
        print("\n" + "="*50)
        print("🔒 NiotoShield Monitor Service 🔒".center(50))
        print("="*50 + "\n")
    
    print(f"🔄 Starting update monitor at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Running in {'console' if console_mode else 'background'} mode")
    
    # If in console mode, block this terminal with a clear message
    if console_mode:
        print("\n⚠️ This terminal is now dedicated to the NiotoShield monitor.")
        print("⚠️ Please do not close this window for continuous protection.")
        print("⚠️ Press Ctrl+C to stop the service if needed.\n")
    
    while True:
        print(f"\n⏰ Checking for updates at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if files exist
        if not check_file_integrity():
            print("⚠️ Essential files are missing. Running sequence...")
            run_sequence()
        else:
            # Compare timestamps
            db_updated_at = get_db_updated_at()
            local_updated_at = get_local_updated_at()
            
            print(f"Database timestamp: {db_updated_at}")
            print(f"Local timestamp: {local_updated_at}")
            
            if db_updated_at is None or local_updated_at is None:
                print("⚠️ Could not compare timestamps. Running sequence to be safe...")
                run_sequence()
            elif db_updated_at != local_updated_at:
                print("🔄 Database update detected. Running sequence...")
                run_sequence()
            else:
                print("✅ No updates needed")
                
        # Wait for next check
        print(f"💤 Next check in {CHECK_INTERVAL // 60} minutes")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main() 