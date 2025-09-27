#!/usr/bin/env python3
"""
Startup script for the Fake News Detector web app
"""

import subprocess
import sys
import os

def start_app():
    """Start the Streamlit web app"""
    print("🚨 Starting Fake News Detector Web App")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run from the 2_fake-news-detector directory.")
        return
    
    print("✅ Starting Streamlit web application...")
    print("🌐 The app will open in your browser at: http://localhost:8501")
    print("📝 Press Ctrl+C to stop the application")
    print()
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    start_app()
