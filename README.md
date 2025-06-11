# Supabase Image Grabber with Face Recognition

This Python script connects to a Supabase database, checks for a specific product key, and downloads images associated with the corresponding user profile. It also generates face embeddings from the downloaded images and provides face recognition capabilities.

## Features

- Connects to Supabase database using product key authentication
- Downloads images from user profiles
- Tracks updates to user profiles and only downloads when changes are detected
- Creates a JSON file with user information
- Manages an images folder that is recreated when new images are available
- Uses .env file for secure credential storage
- Generates face embeddings from downloaded images
- Provides face recognition capabilities for new images

## Requirements

- Python 3.6 or higher
- Supabase account and project
- Internet connection
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone this repository or copy the files to your Raspberry Pi

2. Create a virtual environment named "rasb":

   **On Windows:**
   ```bash
   python -m venv rasb
   .\rasb\Scripts\activate
   ```

   **On Raspberry Pi/Linux:**
   ```bash
   python3 -m venv rasb
   source rasb/bin/activate
   ```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Configure the .env file:

The project uses a .env file to securely store credentials. Make sure the file contains the following variables:

```
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key

# S3 Configuration
S3_URL=your_s3_storage_url

# Product Configuration
PRODUCT_KEY=your_product_key
```

An example .env file is provided, but you should modify it with your own credentials if needed.

## Usage

### Image Grabber and Embedding Generation

Run the main script to download images and generate face embeddings:

```bash
python image_grabber.py
```

The script will:
1. Load configuration from the .env file
2. Connect to Supabase using the configured credentials
3. Check if the product key exists and is associated with a user
4. Retrieve the user's profile information
5. Delete and recreate the 'images' folder
6. Download all images from the user's profile to the 'images' folder
7. Generate face embeddings from the downloaded images
8. Clean up the original images
9. Save user information to 'user_data.json'
10. Track the last update time to avoid unnecessary downloads

### Face Recognition

To scan a new image against the stored face embeddings:

1. Place an image file named `for_scan` (with .jpg, .jpeg, or .png extension) in the project directory
2. Run the face scanner:
   ```bash
   python face_scanner.py
   ```

The scanner will:
1. Load all stored face embeddings
2. Analyze the new image for faces
3. Compare any found faces with the stored embeddings
4. Generate a visual result showing matches
5. Save the annotated image as 'scan_result.jpg'

## Configuration

The script is configured through the .env file which contains:
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_KEY: Your Supabase service role key
- S3_URL: Your Supabase storage URL
- PRODUCT_KEY: The product key for this hardware device

## Output

- `embeddings/` - Directory containing face embeddings and metadata
- `user_data.json` - JSON file with user profile information
- `last_update.txt` - File tracking the last update time
- `scan_result.jpg` - Result of the latest face scan (when using face_scanner.py)

## Automatic Updates

The script checks the 'updated_at' field in the user profile to determine if new content is available. If the profile has been updated since the last run, it will download new images and update the face embeddings.

## Security Notes

- The .env file contains sensitive information and should not be shared or committed to version control
- If deploying to multiple devices, each device should have its own .env file with the appropriate product key
- Consider adding .env to your .gitignore file if using git

## Face Recognition Notes

- The face recognition system uses InsightFace for accurate face detection and embedding generation
- Face embeddings are stored in the 'embeddings' folder along with metadata
- The system automatically manages embeddings, updating them when new images are downloaded
- A similarity threshold of 0.5 is used by default for face matching