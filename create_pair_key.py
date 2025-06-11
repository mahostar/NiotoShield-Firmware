#!/usr/bin/env python3
"""
Fixed RSA Key Pair Generator for Supabase Products
--------------------------------------------------
This script properly generates a unique RSA key pair (public/private)
and stores the public key in Supabase while saving
the private key to the .env file.

Each product key will get its own unique key pair with proper
verification to ensure keys are not duplicated.
"""

import os
import base64
import time
import json
import hashlib
from pathlib import Path
import requests
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv, find_dotenv

# Maximum attempts for generating a unique key
MAX_ATTEMPTS = 5

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
PRODUCT_KEY = os.getenv('PRODUCT_KEY')  # Product key from .env

def verify_product_key():
    """
    Verify that the product key exists in Supabase and check if it has a public key
    
    Returns:
        dict: Product details including id, product_key, and possibly public_key
    """
    if not all([SUPABASE_URL, SUPABASE_KEY, PRODUCT_KEY]):
        raise ValueError("Missing required environment variables (SUPABASE_URL, SUPABASE_KEY, or PRODUCT_KEY)")

    print(f"Verifying product key: {PRODUCT_KEY}")
    
    # Query Supabase for the product
    url = f"{SUPABASE_URL}/rest/v1/products"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    params = {
        "product_key": f"eq.{PRODUCT_KEY}",
        "select": "id,product_key,public_key"
    }

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Failed to verify product key: {response.text}")
    
    products = response.json()
    if not products:
        raise Exception(f"Product key '{PRODUCT_KEY}' not found in database")
    
    return products[0]

def check_public_key_exists(public_key_base64):
    """
    Check if this public key already exists in the database
    
    Returns:
        bool: True if the key exists, False otherwise
    """
    url = f"{SUPABASE_URL}/rest/v1/products"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    params = {
        "public_key": f"eq.{public_key_base64}",
        "select": "id"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Warning: Failed to check for existing public key. Status: {response.status_code}")
            return False
        
        existing = response.json()
        return len(existing) > 0
    except Exception as e:
        print(f"Error checking for existing key: {str(e)}")
        return False

def generate_unique_key_pair():
    """
    Generate a unique RSA key pair with proper entropy and uniqueness verification
    
    Returns:
        tuple: (private_key_pem, public_key_pem, private_key_b64, public_key_b64)
               or (None, None, None, None) if generation fails
    """
    print("Generating new RSA key pair...")
    
    # Try multiple times to generate a unique key
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Generate RSA key pair with secure entropy
        private_key = rsa.generate_private_key(
            public_exponent=65537,  # Standard RSA exponent
            key_size=2048           # 2048-bit key size (industry standard)
        )
        
        # Extract the public key
        public_key = private_key.public_key()
        
        # Serialize keys to PEM format
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Generate SHA-256 hash of the key material to verify uniqueness
        key_hash = hashlib.sha256(private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )).hexdigest()
        
        print(f"Attempt {attempt}: Generated key with hash: {key_hash[:16]}...")
        
        # Convert to base64 for storage
        private_key_b64 = base64.b64encode(private_key_pem).decode('utf-8')
        public_key_b64 = base64.b64encode(public_key_pem).decode('utf-8')
        
        # Verify keys are different (they must be different)
        if private_key_b64 == public_key_b64:
            print(f"Attempt {attempt}: ERROR - Generated identical private and public keys. This should never happen.")
            continue
        
        # Check if this public key already exists in the database
        if check_public_key_exists(public_key_b64):
            print(f"Attempt {attempt}: Generated key already exists in database. Trying again...")
            time.sleep(0.5)  # Brief delay before retry
            continue
            
        # Key is good - return it
        return private_key_pem, public_key_pem, private_key_b64, public_key_b64
    
    # If we get here, we failed to generate a unique key
    print(f"ERROR: Failed to generate a unique key pair after {MAX_ATTEMPTS} attempts")
    return None, None, None, None

def save_private_key(private_key_b64):
    """Save the private key to the .env file"""
    if not dotenv_path:
        raise ValueError("Could not find .env file")
    
    print("Saving private key to .env file...")
    
    # Read existing .env content
    env_content = {}
    if os.path.exists(dotenv_path):
        with open(dotenv_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    if key != 'PRIVATE_KEY':  # Skip existing PRIVATE_KEY
                        env_content[key] = value
    
    # Add the private key to env content
    env_content['PRIVATE_KEY'] = private_key_b64
    
    # Write back to .env file
    with open(dotenv_path, 'w') as f:
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")
    
    print(f"Private key saved to {dotenv_path}")
    return True

def update_public_key(public_key_b64, product_id):
    """
    Update the public key in Supabase
    """
    if not all([SUPABASE_URL, SUPABASE_KEY, product_id]):
        raise ValueError("Missing required parameters for updating public key")
    
    # Prepare the request
    url = f"{SUPABASE_URL}/rest/v1/products"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    # Update the product record by ID
    params = {"id": f"eq.{product_id}"}
    data = {"public_key": public_key_b64}
    
    print(f"Updating public key in database for product ID: {product_id}...")
    response = requests.patch(url, headers=headers, json=data, params=params)
    
    if response.status_code != 204:  # Supabase returns 204 on successful PATCH
        raise Exception(f"Failed to update public key in Supabase: {response.text}")
    
    # Verify the update
    verify_url = f"{SUPABASE_URL}/rest/v1/products"
    verify_headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    verify_params = {
        "id": f"eq.{product_id}",
        "select": "id,product_key,public_key"
    }
    
    verify_response = requests.get(verify_url, headers=verify_headers, params=verify_params)
    if verify_response.status_code != 200:
        raise Exception("Failed to verify public key update")
    
    result = verify_response.json()
    if not result or not result[0].get('public_key'):
        raise Exception("Public key was not properly updated in database")
    
    updated_public_key = result[0].get('public_key')
    if updated_public_key != public_key_b64:
        raise Exception("Updated public key does not match the generated one")
    
    print(f"Public key successfully updated for product {PRODUCT_KEY}")
    return True

def main():
    try:
        # Verify product exists in database
        product = verify_product_key()
        product_id = product['id']
        
        print("\nProduct Details:")
        print(f"Product ID: {product_id}")
        print(f"Product Key: {product['product_key']}")
        
        # Check if product already has a public key
        if product.get('public_key'):
            print(f"Public Key: {product['public_key'][:20]}... (truncated)")
            print("\nNotice: Public key already exists for this product. Skipping key generation.")
            return
        
        print("\nNo public key found. Generating new RSA key pair...")
        
        # Generate a new key pair
        private_key_pem, public_key_pem, private_key_b64, public_key_b64 = generate_unique_key_pair()
        
        if not private_key_pem:
            print("Key generation failed. Exiting.")
            return
        
        # Save the private key
        print("\nSaving private key to .env file...")
        save_private_key(private_key_b64)
        
        # Update the public key in Supabase
        print("Updating public key in Supabase...")
        update_public_key(public_key_b64, product_id)
        
        print("\nKey pair successfully generated and stored!")
        print(f"Public key in database: {public_key_b64[:20]}... (truncated)")
        
        # Verify the keys are different
        if private_key_b64 == public_key_b64:
            print("\nWARNING: Private and public keys appear identical. This should never happen!")
        else:
            print("\nVerification successful: Private and public keys are different (as expected).")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 