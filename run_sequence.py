import subprocess
import sys
import time
import os
import shutil
import json
import requests
from dotenv import load_dotenv

def add_notification(message):
    """Add a notification to the Supabase notifications table for the user"""
    # Load environment variables
    load_dotenv(override=True)
    
    # Get Supabase credentials
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    PRODUCT_KEY = os.getenv('PRODUCT_KEY')
    
    if not all([SUPABASE_URL, SUPABASE_KEY, PRODUCT_KEY]):
        print("⚠️ Cannot add notification - missing environment variables")
        return False
    
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
            return False
        
        user_id = response.json()[0]['user_id']
        
        # Now create the notification
        notification_url = f"{SUPABASE_URL}/rest/v1/notifications"
        notification_data = {
            "user_id": user_id,
            "message": message,
            "is_read": False
        }
        
        # Add the notification
        notification_response = requests.post(notification_url, headers=headers, json=notification_data)
        
        if notification_response.status_code in [200, 201, 204]:
            print(f"✅ Notification added: '{message}'")
            return True
        else:
            print(f"⚠️ Failed to add notification: {notification_response.text}")
            return False
            
    except Exception as e:
        print(f"⚠️ Error adding notification: {str(e)}")
        return False

def run_script(script_name):
    """Run a Python script and return True if successful, False otherwise"""
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    
    try:
        # Run the script and capture output
        process = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True,
            capture_output=True
        )
        
        # Print the output
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(process.stderr)
            
        print(f"\n✅ {script_name} completed successfully!")
        
        # Send notifications based on which script completed
        if script_name == "create_pair_key.py" and "Generating new RSA key pair" in process.stdout:
            # Only notify if we actually generated keys (not if they already existed)
            add_notification("NiotoShield pairing successful")
        elif script_name == "embedding_generator.py" and "Generated embedding" in process.stdout:
            # Only notify if embeddings were actually generated
            add_notification("User data synced securely")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}:")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("\nOutput:")
            print(e.stdout)
        if e.stderr:
            print("\nError:")
            print(e.stderr)
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error running {script_name}:")
        print(str(e))
        return False

def check_file_integrity():
    """
    Check if all essential files and folders exist.
    Returns True if everything is in order, False if cleanup needed
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
    
    all_present = user_data_exists and embeddings_folder_exists and embeddings_has_files and embeddings_metadata_exists
    
    if not all_present:
        print("\n⚠️ System integrity check found missing components:")
        if not user_data_exists:
            print("  ❌ user_data.json is missing")
        if not embeddings_folder_exists:
            print("  ❌ embeddings folder is missing")
        elif not embeddings_has_files:
            print("  ❌ embeddings folder exists but contains no .npy files")
        if not embeddings_metadata_exists:
            print("  ❌ embeddings_metadata.json is missing")
    
    return all_present

def cleanup_for_fresh_start():
    """
    Clean up any potentially corrupted files or folders before restarting 
    the process with a fresh download
    """
    print("\n🧹 Performing system cleanup for fresh start...")
    
    # Remove embeddings folder if it exists
    if os.path.exists('embeddings'):
        shutil.rmtree('embeddings')
        print("  ✓ Removed embeddings folder")
    
    # Remove decrypted_images folder if it exists
    if os.path.exists('decrypted_images'):
        shutil.rmtree('decrypted_images')
        print("  ✓ Removed decrypted_images folder")
    
    # Update user_data.json to force a new download
    # By setting updated_at to an old date or removing it
    if os.path.exists('user_data.json'):
        try:
            with open('user_data.json', 'r') as f:
                user_data = json.load(f)
            
            # Reset the updated_at field to force a new download
            user_data['updated_at'] = "2000-01-01T00:00:00.000000"
            
            with open('user_data.json', 'w') as f:
                json.dump(user_data, f, indent=2)
            
            print("  ✓ Reset user_data.json for fresh download")
        except Exception as e:
            # If there's any issue with the file, just remove it
            os.remove('user_data.json')
            print("  ✓ Removed corrupted user_data.json")
    
    print("✅ Cleanup completed successfully")

def main():
    # Define the sequence of scripts
    scripts = [
        "create_pair_key.py",
        "image_grabber.py",
        "embedding_generator.py"
    ]
    
    print("\n Starting script sequence...")
    
    # Check for file integrity first
    if not check_file_integrity():
        print("\n🚨 System integrity check failed! Some essential files are missing.")
        cleanup_for_fresh_start()
        print("\n🔄 Restarting sequence with a fresh download...")
    
    # Run each script in sequence
    for i, script in enumerate(scripts):
        # Add a small delay between scripts
        time.sleep(1)
        
        # Run the script
        success = run_script(script)
        
        # If the script failed, check if it's the embedding generator
        if not success:
            if script == "embedding_generator.py":
                print("\n🔍 Embedding generator failed. Checking for potential data corruption...")
                
                # If there's no decrypted_images folder but we have embeddings, this is normal
                if (not os.path.exists('decrypted_images') and 
                    os.path.exists('embeddings') and 
                    len([f for f in os.listdir('embeddings') if f.endswith('.npy')]) > 0):
                    print("✅ This is normal - embeddings exist but original images were cleaned up for security.")
                    print("✅ Continuing with face scanning...")
                    continue
                
                # Otherwise, clean up and restart from image_grabber.py
                print("\n⚠️ Data appears corrupted. Performing cleanup and restarting from image grabber...")
                cleanup_for_fresh_start()
                
                # Restart from image_grabber.py
                for restart_script in scripts[1:]:  # Skip create_pair_key.py
                    time.sleep(1)
                    restart_success = run_script(restart_script)
                    if not restart_success:
                        print(f"\n❌ Sequence stopped due to persistent error in {restart_script}")
                        sys.exit(1)
                
                # If we get here, the recovery was successful
                break
            else:
                # For other scripts, just stop the sequence
                print(f"\n❌ Sequence stopped due to error in {script}")
                sys.exit(1)
    
    print("\n✨ All scripts completed successfully!")

if __name__ == "__main__":
    main() 