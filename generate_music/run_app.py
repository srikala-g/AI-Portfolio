#!/usr/bin/env python3
"""
Simple launcher script for the AI Music Generator Gradio app
"""

import subprocess
import sys
import os

def main():
    """Launch the Gradio music generator app"""
    print("🎵 Starting AI Music Generator...")
    print("📱 Web interface will be available at: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Change to the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Run the Gradio app
        subprocess.run([sys.executable, "gradio_app.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
