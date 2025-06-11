#!/usr/bin/env python3
"""
Delete Product Public Key
-------------------------
This script asks for a product key and deletes its public key in the Supabase database.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

def verify_supabase_config():
    """Verify Supabase configuration is available"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Missing Supabase configuration!")
        print("Make sure you have SUPABASE_URL and SUPABASE_KEY in your .env file")
        return False
    return True

def get_product_by_key(product_key):
    """Find a product by its product key"""
    if not product_key:
        return None
    
    url = f"{SUPABASE_URL}/rest/v1/products"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    params = {
        "product_key": f"eq.{product_key}",
        "select": "id,product_key,public_key"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        products = response.json()
        if not products:
            return None
        
        return products[0]
    except Exception as e:
        print(f"Error fetching product: {str(e)}")
        return None

def delete_public_key(product_id):
    """Delete the public key for a product by setting it to null"""
    if not product_id:
        return False
    
    url = f"{SUPABASE_URL}/rest/v1/products"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    params = {
        "id": f"eq.{product_id}"
    }
    data = {
        "public_key": None  # Set to null in the database
    }
    
    try:
        response = requests.patch(url, headers=headers, json=data, params=params)
        
        # Supabase PATCH with return=minimal returns 204 No Content on success
        return response.status_code == 204
    except Exception as e:
        print(f"Error deleting public key: {str(e)}")
        return False

def main():
    # Verify Supabase configuration
    if not verify_supabase_config():
        return
    
    # Get product key from user
    product_key = input("Enter product key to delete public key: ").strip()
    
    if not product_key:
        print("Error: Product key cannot be empty")
        return
    
    # Fetch product details
    print(f"Looking up product key: {product_key}")
    product = get_product_by_key(product_key)
    
    if not product:
        print(f"Error: Product with key '{product_key}' not found")
        return
    
    # Display product details
    print("\nProduct Found:")
    print(f"ID: {product['id']}")
    print(f"Product Key: {product['product_key']}")
    
    has_public_key = 'public_key' in product and product['public_key'] is not None
    if has_public_key:
        print(f"Public Key: {product['public_key'][:20]}... (truncated)")
    else:
        print("Public Key: None (already deleted)")
    
    # Confirm deletion
    if not has_public_key:
        print("\nThis product has no public key to delete.")
        return
    
    confirm = input("\nAre you sure you want to delete the public key? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled")
        return
    
    # Delete public key
    print(f"Deleting public key for product {product_key}...")
    success = delete_public_key(product['id'])
    
    if success:
        print(f"SUCCESS: Public key for product {product_key} has been deleted!")
    else:
        print(f"ERROR: Failed to delete public key for product {product_key}")

if __name__ == "__main__":
    main() 