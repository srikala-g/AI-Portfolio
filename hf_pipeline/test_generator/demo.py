#!/usr/bin/env python3
"""
Demo script for the AI-Powered Unit Test Generator.

This script demonstrates how to use the TestGenerator programmatically.
"""

import os
from test_generator import TestGenerator

def demo_basic_usage():
    """Demonstrate basic usage of the TestGenerator."""
    print("🧪 AI-Powered Unit Test Generator Demo")
    print("=" * 50)
    
    # Initialize the generator
    generator = TestGenerator()
    
    # Sample Python code to test
    sample_code = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def divide(self, a, b):
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
'''
    
    print("📝 Sample Code:")
    print(sample_code)
    print("\n" + "=" * 50)
    
    # Analyze the code
    print("🔍 Code Analysis:")
    analysis = generator.analyze_code(sample_code)
    if 'error' in analysis:
        print(f"❌ Analysis failed: {analysis['error']}")
        return
    
    print(f"✅ Functions found: {[f['name'] for f in analysis['functions']]}")
    print(f"✅ Classes found: {[c['name'] for c in analysis['classes']]}")
    print(f"✅ Imports found: {analysis['imports']}")
    print(f"✅ Complexity score: {analysis['complexity']}")
    
    print("\n" + "=" * 50)
    
    # Generate tests with different models
    models_to_try = ["GPT", "Claude"]
    if generator.is_gemini_available():
        models_to_try.append("Gemini")
    
    for model in models_to_try:
        print(f"🤖 Generating tests with {model}:")
        print("-" * 30)
        
        try:
            test_code = ""
            for chunk in generator.generate_tests(sample_code, model, "pytest"):
                test_code += chunk
                if len(test_code) > 500:  # Limit output for demo
                    test_code += "\n# ... (truncated for demo) ..."
                    break
            
            print(test_code[:500] + "..." if len(test_code) > 500 else test_code)
            print(f"✅ Generated {len(test_code)} characters of test code")
            
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
        
        print("\n" + "-" * 30)
    
    print("\n🎉 Demo completed!")
    print("\nTo run the full web interface:")
    print("python test_generator.py")

def demo_interface_info():
    """Show information about the web interface."""
    print("\n🌐 Web Interface Information:")
    print("=" * 50)
    print("The TestGenerator includes a beautiful Gradio web interface with:")
    print("✅ Real-time test generation")
    print("✅ Multiple AI model support")
    print("✅ Interactive code analysis")
    print("✅ Test execution and results")
    print("✅ Support for pytest and unittest")
    print("\nTo launch the interface:")
    print("python test_generator.py")
    print("Then open: http://localhost:7860")

if __name__ == "__main__":
    # Set up environment variables for demo
    os.environ['OPENAI_API_KEY'] = 'test-key'
    os.environ['ANTHROPIC_API_KEY'] = 'test-key'
    os.environ['GOOGLE_API_KEY'] = 'test-key'
    
    try:
        demo_basic_usage()
        demo_interface_info()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
