#!/usr/bin/env python3
"""
Launcher script for Meeting Minutes Generator

This script provides an easy way to run the application with different options.
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Meeting Minutes Generator Launcher")
    parser.add_argument(
        "--version", 
        choices=["simple", "full"], 
        default="simple",
        help="Choose version: 'simple' for lightweight models, 'full' for all models (default: simple)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Port to run the application on (default: 7860)"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create a public link for the application"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run tests instead of starting the application"
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Running tests...")
        os.system("python test_app.py")
        return
    
    # Choose the appropriate script
    if args.version == "simple":
        script_name = "meeting_minutes_simple.py"
        print("🚀 Starting Meeting Minutes Generator (Simple Version)")
        print("   Using lightweight models for better performance")
    else:
        script_name = "meeting_minutes_gradio.py"
        print("🚀 Starting Meeting Minutes Generator (Full Version)")
        print("   Using all available models")
    
    # Check if script exists
    if not os.path.exists(script_name):
        print(f"❌ Error: {script_name} not found!")
        print("Make sure you're in the correct directory.")
        sys.exit(1)
    
    # Prepare command
    cmd = f"python {script_name}"
    
    # Add port and share options
    if args.port != 7860:
        cmd += f" --port {args.port}"
    
    if args.share:
        cmd += " --share"
    
    print(f"📝 Command: {cmd}")
    print("🌐 The application will be available at:")
    print(f"   http://localhost:{args.port}")
    if args.share:
        print("   A public link will also be created")
    
    print("\n" + "="*50)
    print("Starting application...")
    print("="*50)
    
    # Run the application
    os.system(cmd)

if __name__ == "__main__":
    main()
