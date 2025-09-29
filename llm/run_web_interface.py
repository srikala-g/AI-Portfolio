#!/usr/bin/env python3
"""
Quick launcher for the Website Summary Tool web interface.

This script provides easy access to both command-line and web interfaces.
"""

import sys
import os

def main():
    print("Website Summary Tool Launcher")
    print("=" * 40)
    print("Choose your interface:")
    print("1. Web Interface (Gradio)")
    print("2. Command Line Interface")
    print("3. Usage Examples")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\nLaunching web interface...")
            os.system("python gradio_app.py")
            break
        elif choice == "2":
            print("\nLaunching command line interface...")
            os.system("python website_summary.py")
            break
        elif choice == "3":
            print("\nRunning usage examples...")
            os.system("python usage_example.py")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
