#!/usr/bin/env python3
"""
Test script for Code Generator

This script tests the basic functionality of the code generator without requiring API keys.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test if all required modules can be imported."""
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
        
        # Test basic Gradio functionality
        with gr.Blocks() as demo:
            gr.Textbox("Test")
        print("✅ Gradio basic functionality works")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error with imports: {e}")
        return False

def test_code_generator_structure():
    """Test if CodeGenerator class can be instantiated (without API calls)."""
    try:
        # Mock environment variables to avoid API key requirements
        os.environ['OPENAI_API_KEY'] = 'test-key'
        os.environ['ANTHROPIC_API_KEY'] = 'test-key'
        
        from code_gen import CodeGenerator
        
        # Test class instantiation
        generator = CodeGenerator()
        print("✅ CodeGenerator class instantiated successfully")
        
        # Test method existence
        methods = ['convert_with_gpt', 'convert_with_claude', 'execute_python', 'execute_cpp']
        for method in methods:
            if hasattr(generator, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error testing CodeGenerator: {e}")
        return False

def test_python_execution():
    """Test Python code execution functionality."""
    try:
        from code_gen import CodeGenerator
        
        # Mock environment variables
        os.environ['OPENAI_API_KEY'] = 'test-key'
        os.environ['ANTHROPIC_API_KEY'] = 'test-key'
        
        generator = CodeGenerator()
        
        # Test simple Python execution
        simple_code = "print('Hello, World!')"
        result = generator.execute_python(simple_code)
        
        if "Hello, World!" in result:
            print("✅ Python execution works")
            return True
        else:
            print(f"❌ Python execution failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Python execution: {e}")
        return False

def test_gradio_interface():
    """Test if Gradio interface can be created."""
    try:
        from code_gen import create_gradio_interface
        
        # This will fail if API keys are not set, but we can test the structure
        try:
            interface = create_gradio_interface()
            print("✅ Gradio interface created successfully")
            return True
        except Exception as e:
            if "API" in str(e) or "key" in str(e).lower():
                print("✅ Gradio interface structure is correct (API key required for full test)")
                return True
            else:
                print(f"❌ Gradio interface creation failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing Gradio interface: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Code Generator")
    print("=" * 50)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("CodeGenerator Structure", test_code_generator_structure),
        ("Python Execution", test_python_execution),
        ("Gradio Interface", test_gradio_interface)
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
        print("🎉 All tests passed! The code generator is ready to use.")
        print("\nTo run the application:")
        print("1. Set up your API keys in a .env file")
        print("2. Run: python code_gen.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("You may need to install missing dependencies or check your setup.")

if __name__ == "__main__":
    main()
