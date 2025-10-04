#!/usr/bin/env python3
"""
Test script for Synthetic Data Generator

This script tests the basic functionality without running the full Gradio interface.
"""

import os
import sys
import warnings
from transformers import pipeline
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

def test_dependencies():
    """Test if all required dependencies are available"""
    required_packages = [
        'torch',
        'transformers', 
        'gradio',
        'pandas',
        'numpy'
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
    """Test if Gradio can be imported"""
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

def test_text_generation():
    """Test text generation functionality"""
    try:
        print("Testing text generation...")
        text_pipeline = pipeline(
            "text-generation",
            model="gpt2",
            device="cpu"  # Use CPU for testing
        )
        
        # Test generation
        result = text_pipeline(
            "The future of AI is",
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
        
        print("✅ Text generation works")
        return True
    except Exception as e:
        print(f"❌ Error in text generation: {e}")
        return False

def test_tabular_generation():
    """Test tabular data generation"""
    try:
        print("Testing tabular data generation...")
        
        # Test customer data generation
        data = []
        for i in range(5):
            data.append({
                "customer_id": f"CUST_{i+1:06d}",
                "name": f"Customer {i+1}",
                "email": f"customer{i+1}@example.com",
                "age": np.random.randint(18, 80),
                "city": np.random.choice(["New York", "Los Angeles", "Chicago"])
            })
        
        # Test DataFrame creation
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} rows of tabular data")
        return True
    except Exception as e:
        print(f"❌ Error in tabular generation: {e}")
        return False

def test_export_functionality():
    """Test data export functionality"""
    try:
        print("Testing export functionality...")
        
        # Test data
        data = [
            {"id": 1, "text": "Sample text 1"},
            {"id": 2, "text": "Sample text 2"}
        ]
        
        # Test JSON export
        import json
        with open("test_export.json", "w") as f:
            json.dump(data, f, indent=2)
        
        # Test CSV export
        df = pd.DataFrame(data)
        df.to_csv("test_export.csv", index=False)
        
        # Clean up
        os.remove("test_export.json")
        os.remove("test_export.csv")
        
        print("✅ Export functionality works")
        return True
    except Exception as e:
        print(f"❌ Error in export functionality: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Synthetic Data Generator")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Gradio Import", test_gradio_import),
        ("Text Generation", test_text_generation),
        ("Tabular Generation", test_tabular_generation),
        ("Export Functionality", test_export_functionality)
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
        print("🎉 All tests passed! The generator should work correctly.")
        print("\nTo run the application:")
        print("python simple_generator.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("You may need to install missing dependencies or check your setup.")

if __name__ == "__main__":
    main()

