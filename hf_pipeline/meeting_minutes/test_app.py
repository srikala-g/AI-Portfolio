#!/usr/bin/env python3
"""
Test script for Meeting Minutes Generator

This script tests the basic functionality without running the full Gradio interface.
"""

import os
import sys
import tempfile
import warnings
from transformers import pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def test_whisper_loading():
    """Test if Whisper model can be loaded"""
    try:
        print("Testing Whisper model loading...")
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device="cpu"  # Use CPU for testing
        )
        print("✅ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return False

def test_llm_loading():
    """Test if LLM model can be loaded"""
    try:
        print("Testing LLM model loading...")
        llm_pipeline = pipeline(
            "text-generation",
            model="microsoft/Phi-3-mini-4k-instruct",
            device="cpu"  # Use CPU for testing
        )
        print("✅ LLM model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading LLM model: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    required_packages = [
        'torch',
        'transformers', 
        'gradio',
        'soundfile',
        'huggingface_hub'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_gradio_import():
    """Test if Gradio can be imported and basic functionality works"""
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
        
        # Test basic Gradio functionality
        with gr.Blocks() as demo:
            gr.Textbox("Test")
        print("✅ Gradio basic functionality works")
        return True
    except Exception as e:
        print(f"❌ Error with Gradio: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Meeting Minutes Generator")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Gradio Import", test_gradio_import),
        ("Whisper Loading", test_whisper_loading),
        ("LLM Loading", test_llm_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        print("\nTo run the application:")
        print("python meeting_minutes_simple.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("You may need to install missing dependencies or check your setup.")

if __name__ == "__main__":
    main()
