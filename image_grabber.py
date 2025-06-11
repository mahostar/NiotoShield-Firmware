import os
import json
import shutil
import time
import requests
import base64
from supabase import create_client
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import sys

# Initialize rich console
console = Console()

# Load environment variables from .env file
load_dotenv(override=True)

# Supabase configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# S3 configuration from environment variables
S3_URL = os.getenv("S3_URL")

# Product key from environment variables
PRODUCT_KEY = os.getenv("PRODUCT_KEY")

# Local storage configuration
ENCRYPTED_IMAGES_FOLDER = 'encrypted_images'
DECRYPTED_IMAGES_FOLDER = 'decrypted_images'
USER_DATA_FILE = 'user_data.json'

def initialize_supabase():
    """Initialize Supabase client"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_user_id_from_product_key(supabase, product_key):
    """Get user ID associated with the product key"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Checking product key...", total=None)
        response = supabase.table('products').select('user_id').eq('product_key', product_key).execute()
    
    if not response.data:
        console.print(f"[red][ERROR] Product key '{product_key}' not found.[/red]")
        return None
    
    user_id = response.data[0].get('user_id')
    if not user_id:
        console.print(f"[red][ERROR] Product key '{product_key}' has no associated user.[/red]")
        return None
    
    return user_id

def get_user_profile(supabase, user_id):
    """Get user profile data"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Fetching user profile...", total=None)
        response = supabase.table('user_profiles').select('*').eq('id', user_id).execute()
    
    if not response.data:
        console.print(f"[red][ERROR] User profile for ID '{user_id}' not found.[/red]")
        return None
    
    return response.data[0]

def save_user_data(user_data):
    """Save user data to a JSON file, excluding image_urls"""
    user_info = user_data.copy()
    user_info.pop('image_urls', None)
    
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(user_info, f, indent=2)
    
    console.print(f"[green][INFO] User data saved to {USER_DATA_FILE}[/green]")

def download_encrypted_images(image_urls):
    """Download encrypted images to the local folder and return their paths"""
    if os.path.exists(ENCRYPTED_IMAGES_FOLDER):
        shutil.rmtree(ENCRYPTED_IMAGES_FOLDER)
    os.makedirs(ENCRYPTED_IMAGES_FOLDER)
    
    if not os.path.exists(DECRYPTED_IMAGES_FOLDER):
        os.makedirs(DECRYPTED_IMAGES_FOLDER)
    
    local_paths = []
    with Progress() as progress:
        download_task = progress.add_task("[cyan]Downloading encrypted images...", total=len(image_urls))
        
        for i, url in enumerate(image_urls):
            try:
                image_url = url if url.startswith('http') else f"{S3_URL}/{url}"
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                
                filename = os.path.basename(url) if '/' in url else f"encrypted_{i+1}.bin"
                local_path = os.path.join(ENCRYPTED_IMAGES_FOLDER, filename)
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                local_paths.append(local_path)
                console.print(f"[green][SUCCESS] Downloaded {filename}[/green]")
                
            except Exception as e:
                console.print(f"[red][ERROR] Failed to download {url}: {e}[/red]")
            
            progress.update(download_task, advance=1)
            time.sleep(0.5)  # Delay for visibility
    
    return local_paths

def decrypt_image(local_path, encrypted_key_base64, private_key, index):
    """Decrypt a single image using its encrypted AES key"""
    try:
        # Decrypt AES key
        encrypted_key = base64.b64decode(encrypted_key_base64)
        aes_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Read encrypted image
        with open(local_path, 'rb') as f:
            encrypted_data = f.read()
        
        iv = encrypted_data[:16]  # First 16 bytes are the IV
        encrypted_content = encrypted_data[16:]
        
        # Decrypt image with AES-CBC
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(encrypted_content) + decryptor.finalize()
        
        # Remove PKCS7 padding
        padding_length = decrypted_data[-1]
        if not (1 <= padding_length <= 16):
            raise ValueError("Invalid padding length")
        decrypted_data = decrypted_data[:-padding_length]
        
        # Determine file extension from magic numbers
        if decrypted_data.startswith(b'\xff\xd8\xff'):
            ext = '.jpg'
        elif decrypted_data.startswith(b'\x89PNG\r\n\x1a\n'):
            ext = '.png'
        elif decrypted_data.startswith(b'GIF89a') or decrypted_data.startswith(b'GIF87a'):
            ext = '.gif'
        else:
            ext = '.bin'  # Fallback for unknown types
        
        # Save decrypted image
        decrypted_path = os.path.join(DECRYPTED_IMAGES_FOLDER, f"decrypted_{index}{ext}")
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
        
        console.print(f"[green][SUCCESS] Decrypted image {index} to {decrypted_path}[/green]")
        
    except Exception as e:
        console.print(f"[red][ERROR] Failed to decrypt image {index}: {e}[/red]")

def check_if_data_is_current(user_profile):
    """Check if the local user data is up to date with the database"""
    try:
        if not os.path.exists(USER_DATA_FILE):
            return False
        
        with open(USER_DATA_FILE, 'r') as f:
            local_data = json.load(f)
            
        # Compare the updated_at timestamps
        local_updated_at = local_data.get('updated_at')
        db_updated_at = user_profile.get('updated_at')
        
        if not local_updated_at or not db_updated_at:
            return False
            
        # If the database timestamp is newer than our local timestamp, we need to update
        return local_updated_at >= db_updated_at
            
    except Exception as e:
        console.print(f"[yellow][WARNING] Error checking data currency: {e}[/yellow]")
        return False

def main():
    console.print("[cyan][INFO] Starting image grabber...[/cyan]")
    
    # Verify environment variables
    if not all([SUPABASE_URL, SUPABASE_KEY, PRODUCT_KEY, PRIVATE_KEY]):
        console.print("[red][ERROR] Missing environment variables. Check .env file.[/red]")
        return
    
    try:
        # Initialize Supabase client
        supabase = initialize_supabase()
        console.print("[green][INFO] Connected to Supabase.[/green]")
        
        # Load private key once
        private_key_bytes = base64.b64decode(PRIVATE_KEY)
        private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
        
        # Get user ID from product key
        user_id = get_user_id_from_product_key(supabase, PRODUCT_KEY)
        if not user_id:
            return
        
        console.print(f"[green][INFO] Found user ID: {user_id}[/green]")
        
        # Fetch user profile
        user_profile = get_user_profile(supabase, user_id)
        if not user_profile:
            return
            
        # Check if we need to update the data
        if check_if_data_is_current(user_profile):
            console.print("[green][INFO] Local data is up to date. No need to download images.[/green]")
            return
        
        image_urls = user_profile.get('image_urls', [])
        encrypted_keys_json = user_profile.get('images_encrypted_keys')
        
        if not image_urls or not encrypted_keys_json:
            console.print("[yellow][WARNING] No images or keys found in profile.[/yellow]")
            return
        
        # Parse encrypted keys
        try:
            encrypted_keys = json.loads(encrypted_keys_json)
        except json.JSONDecodeError:
            console.print("[red][ERROR] Invalid images_encrypted_keys JSON format.[/red]")
            return
        
        # Validate counts (1 to 6 images expected)
        num_images = len(image_urls)
        if num_images < 1 or num_images > 6:
            console.print(f"[red][ERROR] Expected 1-6 images, found {num_images}.[/red]")
            return
        if len(encrypted_keys) != num_images:
            console.print("[red][ERROR] Mismatch between images and keys.[/red]")
            return
        
        console.print(f"[cyan][INFO] Processing {num_images} encrypted images.[/cyan]")
        
        # Download images
        local_paths = download_encrypted_images(image_urls)
        if not local_paths:
            console.print("[red][ERROR] No images downloaded.[/red]")
            return
        
        # Decrypt each image
        for i, (local_path, encrypted_key) in enumerate(zip(local_paths, encrypted_keys), 1):
            decrypt_image(local_path, encrypted_key, private_key, i)
        
        # Automatically clean up encrypted_images folder
        if os.path.exists(ENCRYPTED_IMAGES_FOLDER):
            shutil.rmtree(ENCRYPTED_IMAGES_FOLDER)
            console.print("[green][INFO] Encrypted images folder cleaned up.[/green]")
        
        # Save user data
        save_user_data(user_profile)
        
    except Exception as e:
        console.print(f"[red][ERROR] Image grabber failed: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()