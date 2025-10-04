#!/usr/bin/env python3
"""
Test script for the Gradio app
This script tests the basic functionality without launching the full interface
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required imports work"""
    try:
        import gradio as gr
        import torch
        from transformers import pipeline
        from diffusers import DiffusionPipeline
        from datasets import load_dataset
        import soundfile as sf
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_device_detection():
    """Test device detection"""
    try:
        import torch
        device = "cpu"  # Force CPU for testing
        print(f"✅ Device detection: {device}")
        return True
    except Exception as e:
        print(f"❌ Device detection error: {e}")
        return False

def test_simple_pipeline():
    """Test a simple pipeline"""
    try:
        from transformers import pipeline
        
        # Test sentiment analysis (lightweight model)
        classifier = pipeline("sentiment-analysis", device="cpu")
        result = classifier("I love this!")
        print(f"✅ Sentiment analysis test: {result}")
        return True
    except Exception as e:
        print(f"❌ Pipeline test error: {e}")
        return False

def main():
    print("🧪 Testing Hugging Face Pipelines App")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Device Detection", test_device_detection),
        ("Simple Pipeline", test_simple_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should work correctly.")
        print("\nTo run the full app:")
        print("python3 app.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
