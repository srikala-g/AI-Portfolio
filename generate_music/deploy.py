#!/usr/bin/env python3
"""
Deployment script for Hugging Face Spaces
This script helps deploy the AI Music Generator to Hugging Face Spaces
"""

import subprocess
import os
import sys

def check_huggingface_hub():
    """Check if huggingface_hub is installed"""
    try:
        import huggingface_hub
        print("✅ huggingface_hub is installed")
        return True
    except ImportError:
        print("❌ huggingface_hub not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        return True

def login_to_huggingface():
    """Login to Hugging Face"""
    print("🔐 Please login to Hugging Face...")
    subprocess.run(["huggingface-cli", "login"])

def create_space():
    """Create a new Hugging Face Space"""
    print("🚀 Creating Hugging Face Space...")
    
    space_name = input("Enter space name (e.g., 'ai-music-generator'): ").strip()
    if not space_name:
        space_name = "ai-music-generator"
    
    # Create space using huggingface_hub
    try:
        from huggingface_hub import HfApi, create_repo
        
        api = HfApi()
        
        # Create the space
        repo_url = create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        
        print(f"✅ Space created: https://huggingface.co/spaces/{space_name}")
        return space_name
        
    except Exception as e:
        print(f"❌ Error creating space: {e}")
        return None

def upload_files(space_name):
    """Upload files to the space"""
    print(f"📤 Uploading files to {space_name}...")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        
        # Upload main files
        files_to_upload = [
            "gradio_app.py",
            "requirements.txt",
            "README.md"
        ]
        
        for file in files_to_upload:
            if os.path.exists(file):
                print(f"📁 Uploading {file}...")
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=space_name,
                    repo_type="space"
                )
                print(f"✅ {file} uploaded successfully")
            else:
                print(f"⚠️ {file} not found, skipping...")
        
        print(f"🎉 Deployment complete! Visit: https://huggingface.co/spaces/{space_name}")
        
    except Exception as e:
        print(f"❌ Error uploading files: {e}")

def main():
    """Main deployment function"""
    print("🎵 AI Music Generator - Hugging Face Deployment")
    print("=" * 50)
    
    # Check dependencies
    if not check_huggingface_hub():
        return
    
    # Login to Hugging Face
    login_to_huggingface()
    
    # Create space
    space_name = create_space()
    if not space_name:
        return
    
    # Upload files
    upload_files(space_name)

if __name__ == "__main__":
    main()
