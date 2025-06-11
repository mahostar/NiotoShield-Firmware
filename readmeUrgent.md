# Cross-Language Encryption Implementation Guide (Python & Node.js)

## Overview
This guide details the implementation of a hybrid encryption system between a Node.js app and a Python-based Raspberry Pi system, using Supabase for data storage. The system uses RSA for key exchange and AES for image encryption.

## How It Works (Simplified)
1. **App Side (Node.js)**:
   - Generate random AES key for each image
   - Encrypt image with AES key
   - Encrypt AES key with RSA public key
   - Upload both encrypted image and encrypted AES key to Supabase

2. **Raspberry Pi Side (Python)**:
   - Download encrypted image and encrypted AES key from Supabase
   - Decrypt AES key using RSA private key
   - Use decrypted AES key to decrypt the image

## System Components
- **Raspberry Pi (Python)**: Handles key pair generation and image decryption
- **Mobile App (Node.js)**: Handles image encryption
- **Supabase**: Stores encrypted data and public keys

## Database Schema Update
Add the public key field to your products table:

```sql
alter table public.products
add column public_key text null;
```

## Implementation Steps

### 1. Raspberry Pi (Python) Setup

#### Required Python Packages
```bash
pip install cryptography requests python-dotenv
```

#### Key Generation Code (Python)
```python
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import base64
import os
import requests

def generate_key_pair():
    # Generate RSA key pair
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    # Serialize private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Serialize public key
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # Save private key securely
    with open("/etc/niotoshield/private_key.pem", "wb") as f:
        f.write(private_pem)
    os.chmod("/etc/niotoshield/private_key.pem", 0o600)

    # Return Base64 encoded public key
    return base64.b64encode(public_pem).decode('utf-8')
```

#### Decryption Code (Python)
```python
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes

def decrypt_image(encrypted_aes_key_base64, iv_base64, encrypted_image_base64):
    # Load private key
    with open('/etc/niotoshield/private_key.pem', 'rb') as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    # Decode from Base64
    encrypted_aes_key = base64.b64decode(encrypted_aes_key_base64)
    iv = base64.b64decode(iv_base64)
    encrypted_image = base64.b64decode(encrypted_image_base64)

    # Step 1: Decrypt the AES key using RSA private key
    aes_key = private_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Step 2: Use decrypted AES key to decrypt the image
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    return decryptor.update(encrypted_image) + decryptor.finalize()
```

### 2. Mobile App (Node.js) Setup

#### Required Node.js Packages
```bash
npm install crypto fs
```

#### Encryption Code (Node.js)
```javascript
const crypto = require('crypto');
const fs = require('fs');

async function encryptImage(imagePath, publicKeyBase64) {
    // Decode public key from Base64
    const publicKeyPem = Buffer.from(publicKeyBase64, 'base64').toString('utf-8');

    // Step 1: Generate random AES key and IV for image encryption
    const aesKey = crypto.randomBytes(32);  // This will be used to encrypt the image
    const iv = crypto.randomBytes(16);      // Initialization vector for AES

    // Step 2: Encrypt the image with AES key
    const imageData = fs.readFileSync(imagePath);
    const cipher = crypto.createCipheriv('aes-256-cbc', aesKey, iv);
    const encryptedImage = Buffer.concat([cipher.update(imageData), cipher.final()]);

    // Step 3: Encrypt the AES key with RSA public key
    const encryptedAESKey = crypto.publicEncrypt(
        {
            key: publicKeyPem,
            padding: crypto.constants.RSA_PKCS1_OAEP_PADDING,
            oaepHash: 'sha256',
        },
        aesKey
    );

    // Convert everything to Base64 for storage
    return {
        encryptedImage: encryptedImage.toString('base64'),
        encryptedKey: encryptedAESKey.toString('base64'),  // This is the AES key encrypted with RSA
        iv: iv.toString('base64')
    };
}
```

## Detailed Data Flow

1. **Initial Setup (One Time)**
   - Raspberry Pi generates RSA key pair (public & private)
   - Public key is stored in Supabase products table
   - Private key is stored securely on Raspberry Pi

2. **Image Encryption Flow (App Side)**
   - App retrieves RSA public key from Supabase using product key
   - For each image:
     1. Generate new random AES key
     2. Encrypt the image using this AES key
     3. Encrypt the AES key using RSA public key
     4. Upload to Supabase:
        - The encrypted image
        - The encrypted AES key
        - The IV (initialization vector)

3. **Image Decryption Flow (Raspberry Pi Side)**
   - For each image:
     1. Download from Supabase:
        - The encrypted image
        - The encrypted AES key
        - The IV
     2. Use RSA private key to decrypt the AES key
     3. Use decrypted AES key to decrypt the image

## Security Considerations

1. **Private Key Storage**
   - Store private key securely on Raspberry Pi
   - Use restricted permissions (chmod 600)
   - Never upload private key to Supabase or any external service

2. **Product Key**
   - Don't hardcode in source code
   - Store in environment variables or secure configuration

3. **Data Storage**
   - Store all encrypted data as Base64 strings
   - Include IV with each encrypted image
   - Store encrypted AES key separately

## Testing

1. Test the encryption/decryption process:
```bash
# On Raspberry Pi
python test_decryption.py

# On development machine
node test_encryption.js
```

2. Verify that:
   - Encrypted data can be stored in Supabase
   - Decrypted images match originals
   - Keys are properly exchanged

## Troubleshooting

Common issues and solutions:

1. **Padding Errors**
   - Ensure RSA padding is consistent (OAEP with SHA-256)
   - Verify AES key length (32 bytes for AES-256)

2. **Base64 Issues**
   - Check for proper encoding/decoding
   - Ensure no data truncation

3. **Key Format Problems**
   - Verify PEM format for RSA keys
   - Check key permissions on Raspberry Pi

## Support

For issues or questions:
1. Check key formats and encodings
2. Verify Supabase connection
3. Ensure proper permissions for key files
4. Test with small sample images first 


https://chatgpt.com/share/67c518e9-ea2c-8007-a0c3-7ccb13e47fc0

setup instruction sequance :
- create an account with the app using the product key
- open the kit (so it can create the pair keys)
- setup your profile inside the app