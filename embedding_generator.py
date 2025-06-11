import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import json
import shutil
import sys
from rich.console import Console
from rich.progress import Progress

# Initialize rich console for better output
console = Console()

class EmbeddingGenerator:
    def __init__(self, decrypted_folder="decrypted_images", embeddings_folder="embeddings"):
        self.decrypted_folder = decrypted_folder
        self.embeddings_folder = embeddings_folder
        
        # Initialize Face Recognition model
        self.app = FaceAnalysis(name="buffalo_l", 
                              providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Create embeddings folder if it doesn't exist
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)

    def generate_embedding(self, image_path, output_name):
        """Generate embedding for a single image"""
        if not os.path.exists(image_path):
            console.print(f"[red][ERROR] Image not found: {image_path}[/red]")
            return None
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            console.print(f"[red][ERROR] Could not read image: {image_path}[/red]")
            return None

        # Detect face and extract embedding
        faces = self.app.get(image)

        if not faces:
            console.print(f"[yellow][WARNING] No face detected in {image_path}[/yellow]")
            return None

        if len(faces) > 1:
            console.print(f"[yellow][WARNING] Multiple faces detected in {image_path}. Using the most prominent face.[/yellow]")

        # Get the face with highest detection score
        face = max(faces, key=lambda x: x.det_score)
        embedding = face.normed_embedding

        # Save the embedding
        output_path = os.path.join(self.embeddings_folder, f"{output_name}.npy")
        np.save(output_path, embedding)
        
        console.print(f"[green][SUCCESS] Generated embedding for {image_path}[/green]")
        return {
            "filename": output_name,
            "confidence": float(face.det_score),
            "embedding_path": output_path
        }

    def process_decrypted_images(self):
        """Process all decrypted images in sequence (1-6)"""
        if not os.path.exists(self.decrypted_folder):
            console.print("[red][ERROR] Decrypted images folder not found.[/red]")
            return []

        results = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing decrypted images...", total=6)
            
            # Look for images named decrypted_1 through decrypted_6 with any image extension
            for i in range(1, 7):
                found = False
                for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                    image_path = os.path.join(self.decrypted_folder, f"decrypted_{i}{ext}")
                    if os.path.exists(image_path):
                        try:
                            result = self.generate_embedding(image_path, f"face_{i}")
                            if result:
                                results.append(result)
                            found = True
                            break
                        except Exception as e:
                            console.print(f"[red][ERROR] Processing decrypted_{i}{ext}: {str(e)}[/red]")
                
                if not found:
                    console.print(f"[yellow][INFO] No image found for decrypted_{i}[/yellow]")
                
                progress.update(task, advance=1)

        # Save metadata about the embeddings
        if results:
            metadata_path = os.path.join(self.embeddings_folder, "embeddings_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green][INFO] Saved metadata for {len(results)} embeddings[/green]")

        return results

    def cleanup_decrypted_images(self):
        """Delete the decrypted images folder after processing"""
        if os.path.exists(self.decrypted_folder):
            shutil.rmtree(self.decrypted_folder)
            console.print("[green][INFO] Cleaned up decrypted images folder[/green]")

def main():
    console.print("[cyan][INFO] Starting embedding generation process...[/cyan]")
    
    try:
        generator = EmbeddingGenerator()
        results = generator.process_decrypted_images()
        
        if results:
            console.print(f"\n[green][SUCCESS] Successfully generated {len(results)} embeddings[/green]")
            generator.cleanup_decrypted_images()
        else:
            console.print("\n[yellow][WARNING] No embeddings were generated[/yellow]")
        
    except Exception as e:
        console.print(f"\n[red][ERROR] Error during embedding generation: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 