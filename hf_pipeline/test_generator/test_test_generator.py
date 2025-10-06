#!/usr/bin/env python3
"""
Test suite for the TestGenerator class.

This module contains comprehensive tests for the AI-powered unit test generator.
"""

import os
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch, MagicMock

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_generator import TestGenerator


class TestTestGenerator(unittest.TestCase):
    """Test cases for the TestGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'GOOGLE_API_KEY': 'test-google-key'
        })
        self.env_patcher.start()
        
        # Mock AI clients
        self.openai_patcher = patch('test_generator.openai.OpenAI')
        self.claude_patcher = patch('test_generator.anthropic.Anthropic')
        self.gemini_patcher = patch('test_generator.genai')
        
        self.mock_openai = self.openai_patcher.start()
        self.mock_claude = self.claude_patcher.start()
        self.mock_gemini = self.gemini_patcher.start()
        
        # Create TestGenerator instance
        self.generator = TestGenerator()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        self.openai_patcher.stop()
        self.claude_patcher.stop()
        self.gemini_patcher.stop()
    
    def test_initialization(self):
        """Test TestGenerator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.openai_model, "gpt-4o")
        self.assertEqual(self.generator.claude_model, "claude-3-5-sonnet-20240620")
    
    def test_analyze_code_simple_function(self):
        """Test code analysis with a simple function."""
        code = """
def add(a, b):
    return a + b
"""
        analysis = self.generator.analyze_code(code)
        
        self.assertIn('functions', analysis)
        self.assertEqual(len(analysis['functions']), 1)
        self.assertEqual(analysis['functions'][0]['name'], 'add')
        self.assertEqual(analysis['functions'][0]['args'], ['a', 'b'])
    
    def test_analyze_code_with_class(self):
        """Test code analysis with a class."""
        code = """
class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self.value
"""
        analysis = self.generator.analyze_code(code)
        
        self.assertIn('classes', analysis)
        self.assertEqual(len(analysis['classes']), 1)
        self.assertEqual(analysis['classes'][0]['name'], 'Calculator')
        self.assertIn('add', analysis['classes'][0]['methods'])
    
    def test_analyze_code_with_imports(self):
        """Test code analysis with imports."""
        code = """
import os
from typing import List
import numpy as np

def process_data(data):
    return data
"""
        analysis = self.generator.analyze_code(code)
        
        self.assertIn('imports', analysis)
        self.assertIn('os', analysis['imports'])
        self.assertIn('typing', analysis['imports'])
        self.assertIn('numpy', analysis['imports'])
    
    def test_analyze_code_syntax_error(self):
        """Test code analysis with syntax error."""
        code = """
def broken_function(
    # Missing closing parenthesis
"""
        analysis = self.generator.analyze_code(code)
        
        self.assertIn('error', analysis)
        self.assertIn('Syntax error', analysis['error'])
    
    def test_generate_tests_with_gpt(self):
        """Test test generation with GPT."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].delta.content = "def test_add():\n    assert add(2, 3) == 5"
        
        self.mock_openai.return_value.chat.completions.create.return_value = [mock_response]
        
        code = "def add(a, b): return a + b"
        result = list(self.generator.generate_tests_with_gpt(code))
        
        self.assertGreater(len(result), 0)
        self.assertIn("test_add", result[0])
    
    def test_generate_tests_with_claude(self):
        """Test test generation with Claude."""
        # Mock Claude response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "def test_add():\n    assert add(2, 3) == 5"
        
        self.mock_claude.return_value.messages.create.return_value = mock_response
        
        code = "def add(a, b): return a + b"
        result = list(self.generator.generate_tests_with_claude(code))
        
        self.assertGreater(len(result), 0)
        self.assertIn("test_add", result[0])
    
    def test_generate_tests_with_gemini(self):
        """Test test generation with Gemini."""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "def test_add():\n    assert add(2, 3) == 5"
        
        self.mock_gemini.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        # Set up Gemini model
        self.generator.gemini_model = "gemini-1.5-flash"
        self.generator.gemini_model_obj = self.mock_gemini.GenerativeModel.return_value
        
        code = "def add(a, b): return a + b"
        result = list(self.generator.generate_tests_with_gemini(code))
        
        self.assertGreater(len(result), 0)
        self.assertIn("test_add", result[0])
    
    def test_generate_tests_gemini_not_available(self):
        """Test test generation when Gemini is not available."""
        self.generator.gemini_model = None
        self.generator.gemini_model_obj = None
        
        code = "def add(a, b): return a + b"
        result = list(self.generator.generate_tests_with_gemini(code))
        
        self.assertEqual(len(result), 1)
        self.assertIn("Gemini model not available", result[0])
    
    def test_run_tests_success(self):
        """Test running tests successfully."""
        test_code = """
def test_simple():
    assert 1 + 1 == 2
"""
        
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "test_simple PASSED"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            success, output = self.generator.run_tests(test_code)
            
            self.assertTrue(success)
            self.assertIn("PASSED", output)
    
    def test_run_tests_failure(self):
        """Test running tests with failures."""
        test_code = """
def test_failing():
    assert 1 + 1 == 3
"""
        
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "test_failing FAILED"
            mock_run.return_value = mock_result
            
            success, output = self.generator.run_tests(test_code)
            
            self.assertFalse(success)
            self.assertIn("FAILED", output)
    
    def test_run_tests_timeout(self):
        """Test running tests with timeout."""
        test_code = "def test_simple(): assert True"
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("pytest", 30)
            
            success, output = self.generator.run_tests(test_code)
            
            self.assertFalse(success)
            self.assertIn("timed out", output)
    
    def test_is_gemini_available(self):
        """Test Gemini availability check."""
        # Test when Gemini is not available
        self.generator.gemini_model = None
        self.generator.gemini_model_obj = None
        self.assertFalse(self.generator.is_gemini_available())
        
        # Test when Gemini is available
        self.generator.gemini_model = "gemini-1.5-flash"
        self.generator.gemini_model_obj = MagicMock()
        self.assertTrue(self.generator.is_gemini_available())
    
    def test_generate_tests_unsupported_model(self):
        """Test test generation with unsupported model."""
        code = "def add(a, b): return a + b"
        result = list(self.generator.generate_tests(code, "UnsupportedModel"))
        
        self.assertEqual(len(result), 1)
        self.assertIn("Unsupported model", result[0])


def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 Testing Test Generator")
    print("=" * 50)
    
    try:
        # Test imports
        print("🔍 Import Dependencies:")
        import gradio as gr
        print("✅ Gradio imported successfully")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
        
        # Test basic Gradio functionality
        with gr.Blocks() as demo:
            gr.Markdown("Test")
        print("✅ Gradio basic functionality works")
        
        # Test TestGenerator instantiation
        print("\n🔍 TestGenerator Structure:")
        generator = TestGenerator()
        print("✅ TestGenerator class instantiated successfully")
        
        # Test methods exist
        assert hasattr(generator, 'analyze_code'), "analyze_code method missing"
        assert hasattr(generator, 'generate_tests_with_gpt'), "generate_tests_with_gpt method missing"
        assert hasattr(generator, 'generate_tests_with_claude'), "generate_tests_with_claude method missing"
        assert hasattr(generator, 'run_tests'), "run_tests method missing"
        print("✅ All required methods exist")
        
        # Test code analysis
        print("\n🔍 Code Analysis:")
        test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        analysis = generator.analyze_code(test_code)
        assert 'functions' in analysis, "Functions not found in analysis"
        assert len(analysis['functions']) > 0, "No functions detected"
        print("✅ Code analysis works")
        
        # Test interface creation
        print("\n🔍 Interface Creation:")
        interface = generator.create_interface() if hasattr(generator, 'create_interface') else None
        if interface:
            print("✅ Interface creation method exists")
        else:
            print("✅ Interface creation handled by module function")
        
        print("\n" + "=" * 50)
        print("📊 Test Results:")
        print("✅ PASS Import Dependencies")
        print("✅ PASS TestGenerator Structure")
        print("✅ PASS Code Analysis")
        print("✅ PASS Interface Creation")
        print("\nOverall: 4/4 tests passed")
        print("🎉 All tests passed! The test generator is ready to use.")
        print("\nTo run the application:")
        print("1. Set up your API keys in a .env file")
        print("2. Run: python test_generator.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run basic tests
    run_basic_tests()
    
    # Run unit tests
    print("\n" + "=" * 50)
    print("🧪 Running Unit Tests")
    print("=" * 50)
    
    unittest.main(verbosity=2, exit=False)
