#!/usr/bin/env python3
"""
AI-Powered Unit Test Case Generator

A comprehensive tool that generates unit tests for Python code using state-of-the-art
language models (GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-Flash).

Author: AI Assistant
Date: 2024
"""

import os
import ast
import inspect
import logging
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Generator, Tuple
from pathlib import Path

import openai
import anthropic
import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestGenerator:
    """
    AI-powered unit test case generator that creates comprehensive test suites
    for Python code using multiple language models.
    """
    
    def __init__(self, openai_model: str = "gpt-4o", claude_model: str = "claude-3-5-sonnet-20240620", gemini_model: str = "gemini-1.5-flash"):
        """
        Initialize the TestGenerator with AI models.
        
        Args:
            openai_model: OpenAI model name
            claude_model: Anthropic Claude model name  
            gemini_model: Google Gemini model name
        """
        self._load_environment()
        self.openai_client = openai.OpenAI()
        self.claude_client = anthropic.Anthropic()
        self.openai_model = openai_model
        self.claude_model = claude_model
        self.gemini_model = gemini_model
        self.system_message = self._get_system_message()
        
        # Initialize Gemini (optional - only if API key is available)
        self.gemini_model = None
        self.gemini_model_obj = None
        
        try:
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key or google_api_key == 'your-key-if-not-using-env' or google_api_key == 'test-key':
                logger.info("GOOGLE_API_KEY not set or invalid, Gemini model will not be available")
            else:
                genai.configure(api_key=google_api_key)
                # Try different model names to find one that works
                model_names_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'models/gemini-1.5-flash']
                
                for model_name in model_names_to_try:
                    try:
                        test_model = genai.GenerativeModel(model_name)
                        # Test with a simple request
                        test_response = test_model.generate_content("Hello")
                        if test_response and test_response.text:
                            self.gemini_model = model_name
                            self.gemini_model_obj = test_model
                            logger.info(f"Gemini model '{model_name}' initialized successfully")
                            break
                    except Exception as model_error:
                        logger.debug(f"Model '{model_name}' not available: {model_error}")
                        continue
                
                if self.gemini_model is None:
                    logger.warning("No working Gemini model found with the provided API key")
                    
        except Exception as e:
            logger.info(f"Gemini model not available: {e}")
            # Don't treat this as an error - Gemini is optional
        
    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        try:
            load_dotenv(override=True)
            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
            os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
            os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')
            logger.info("Environment variables loaded successfully")
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
    
    def _get_system_message(self) -> str:
        """Get the system message for test generation."""
        return """You are an expert Python developer and testing specialist. Your task is to generate comprehensive unit tests for Python code.

Requirements:
1. Generate complete, runnable pytest test cases
2. Include edge cases, boundary conditions, and error scenarios
3. Use descriptive test names that explain what is being tested
4. Include proper imports and setup
5. Test both positive and negative cases
6. Include docstrings for test functions
7. Use fixtures where appropriate
8. Test all public methods and functions
9. Include parameterized tests for multiple scenarios
10. Ensure tests are isolated and independent

Format the output as clean, well-documented Python test code that follows pytest conventions."""

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze Python code to extract functions, classes, and dependencies.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            tree = ast.parse(code)
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'variables': [],
                'complexity': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'line_number': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'line_number': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)
            
            # Calculate complexity (simple metric)
            analysis['complexity'] = len(analysis['functions']) + len(analysis['classes']) * 2
            
            return analysis
            
        except SyntaxError as e:
            logger.error(f"Syntax error in code analysis: {e}")
            return {'error': f"Syntax error: {e}"}
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {'error': f"Analysis error: {e}"}

    def generate_tests_with_gpt(self, code: str, test_framework: str = "pytest") -> Generator[str, None, None]:
        """
        Generate unit tests using OpenAI GPT.
        
        Args:
            code: Python code to generate tests for
            test_framework: Testing framework to use (pytest, unittest)
            
        Yields:
            Generated test code chunks
        """
        try:
            analysis = self.analyze_code(code)
            if 'error' in analysis:
                yield f"❌ Code analysis failed: {analysis['error']}"
                return
            
            prompt = f"""Generate comprehensive unit tests for the following Python code using {test_framework}:

Code to test:
```python
{code}
```

Code analysis:
- Functions: {[f['name'] for f in analysis['functions']]}
- Classes: {[c['name'] for c in analysis['classes']]}
- Imports: {analysis['imports']}
- Complexity: {analysis['complexity']}

Generate complete, runnable test cases that cover:
1. All functions and methods
2. Edge cases and boundary conditions
3. Error scenarios and exceptions
4. Parameterized tests for multiple inputs
5. Mocking external dependencies
6. Setup and teardown where needed

Return only the test code, no explanations."""

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                temperature=0.3
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating tests with GPT: {e}")
            yield f"❌ Error with GPT test generation: {e}"

    def generate_tests_with_claude(self, code: str, test_framework: str = "pytest") -> Generator[str, None, None]:
        """
        Generate unit tests using Anthropic Claude.
        
        Args:
            code: Python code to generate tests for
            test_framework: Testing framework to use (pytest, unittest)
            
        Yields:
            Generated test code chunks
        """
        try:
            analysis = self.analyze_code(code)
            if 'error' in analysis:
                yield f"❌ Code analysis failed: {analysis['error']}"
                return
            
            prompt = f"""Generate comprehensive unit tests for the following Python code using {test_framework}:

Code to test:
```python
{code}
```

Code analysis:
- Functions: {[f['name'] for f in analysis['functions']]}
- Classes: {[c['name'] for c in analysis['classes']]}
- Imports: {analysis['imports']}
- Complexity: {analysis['complexity']}

Generate complete, runnable test cases that cover:
1. All functions and methods
2. Edge cases and boundary conditions
3. Error scenarios and exceptions
4. Parameterized tests for multiple inputs
5. Mocking external dependencies
6. Setup and teardown where needed

Return only the test code, no explanations."""

            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=4000,
                temperature=0.3,
                system=self.system_message,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Simulate streaming for Claude
            content = response.content[0].text
            chunk_size = 50
            for i in range(0, len(content), chunk_size):
                yield content[i:i + chunk_size]
                
        except Exception as e:
            logger.error(f"Error generating tests with Claude: {e}")
            yield f"❌ Error with Claude test generation: {e}"

    def generate_tests_with_gemini(self, code: str, test_framework: str = "pytest") -> Generator[str, None, None]:
        """
        Generate unit tests using Google Gemini.
        
        Args:
            code: Python code to generate tests for
            test_framework: Testing framework to use (pytest, unittest)
            
        Yields:
            Generated test code chunks
        """
        if not self.gemini_model or not self.gemini_model_obj:
            yield "❌ Gemini model not available. Please check your GOOGLE_API_KEY and ensure it's valid."
            return
            
        try:
            analysis = self.analyze_code(code)
            if 'error' in analysis:
                yield f"❌ Code analysis failed: {analysis['error']}"
                return
            
            prompt = f"""Generate comprehensive unit tests for the following Python code using {test_framework}:

Code to test:
```python
{code}
```

Code analysis:
- Functions: {[f['name'] for f in analysis['functions']]}
- Classes: {[c['name'] for c in analysis['classes']]}
- Imports: {analysis['imports']}
- Complexity: {analysis['complexity']}

Generate complete, runnable test cases that cover:
1. All functions and methods
2. Edge cases and boundary conditions
3. Error scenarios and exceptions
4. Parameterized tests for multiple inputs
5. Mocking external dependencies
6. Setup and teardown where needed

Return only the test code, no explanations."""

            response = self.gemini_model_obj.generate_content(prompt)
            
            if response and response.text:
                # Simulate streaming for Gemini
                content = response.text
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    yield content[i:i + chunk_size]
            else:
                yield "❌ No response from Gemini model"
                
        except Exception as e:
            logger.error(f"Error generating tests with Gemini: {e}")
            yield f"❌ Error with Gemini test generation: {e}"

    def generate_tests(self, code: str, model: str, test_framework: str = "pytest") -> Generator[str, None, None]:
        """
        Generate unit tests using the specified model.
        
        Args:
            code: Python code to generate tests for
            model: AI model to use (GPT, Claude, Gemini)
            test_framework: Testing framework to use
            
        Yields:
            Generated test code chunks
        """
        logger.info(f"Generating tests with {model} using {test_framework}")
        
        if model == "GPT":
            yield from self.generate_tests_with_gpt(code, test_framework)
        elif model == "Claude":
            yield from self.generate_tests_with_claude(code, test_framework)
        elif model == "Gemini":
            yield from self.generate_tests_with_gemini(code, test_framework)
        else:
            yield f"❌ Unsupported model: {model}"

    def run_tests(self, test_code: str, original_code: str = None) -> Tuple[bool, str]:
        """
        Run the generated tests and return results.
        
        Args:
            test_code: Generated test code
            original_code: Original code being tested (optional)
            
        Returns:
            Tuple of (success, output)
        """
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as test_file:
                test_file.write(test_code)
                test_file_path = test_file.name
            
            # Run pytest
            result = subprocess.run(
                ['python', '-m', 'pytest', test_file_path, '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(test_file_path)
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out"
        except Exception as e:
            return False, f"Error running tests: {e}"

    def is_gemini_available(self) -> bool:
        """Check if Gemini model is available."""
        return self.gemini_model is not None and self.gemini_model_obj is not None


def create_interface():
    """Create the Gradio interface for the test generator."""
    test_generator = TestGenerator()
    
    # Sample Python code for demonstration
    sample_code = '''def fibonacci(n):
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
        return result'''

    # Available models
    available_models = ["GPT", "Claude"]
    if test_generator.is_gemini_available():
        available_models.append("Gemini")

    with gr.Blocks(title="AI Test Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧪 AI-Powered Unit Test Generator")
        gr.Markdown("Generate comprehensive unit tests for your Python code using state-of-the-art AI models.")
        
        # Status message for Gemini
        if not test_generator.is_gemini_available():
            gr.Markdown("⚠️ **Note**: Gemini model is not available. Please set a valid `GOOGLE_API_KEY` to use Gemini.")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📝 Input Code")
                code_input = gr.Textbox(
                    label="Python Code to Test",
                    value=sample_code,
                    lines=20,
                    placeholder="Enter your Python code here...",
                    info="Paste your Python code that you want to generate tests for"
                )
                
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=available_models,
                        value=available_models[0],
                        label="AI Model",
                        info="Choose the AI model for test generation"
                    )
                    
                    framework_selector = gr.Dropdown(
                        choices=["pytest", "unittest"],
                        value="pytest",
                        label="Test Framework",
                        info="Choose the testing framework"
                    )
                
                generate_btn = gr.Button("🚀 Generate Tests", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("## 🧪 Generated Tests")
                test_output = gr.Textbox(
                    label="Generated Test Code",
                    lines=20,
                    placeholder="Generated tests will appear here...",
                    info="AI-generated unit test code"
                )
                
                with gr.Row():
                    run_tests_btn = gr.Button("▶️ Run Tests", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear", variant="stop")
        
        with gr.Row():
            gr.Markdown("## 📊 Test Results")
            test_results = gr.Textbox(
                label="Test Execution Results",
                lines=10,
                placeholder="Test results will appear here...",
                info="Results from running the generated tests"
            )
        
        # Event handlers
        def generate_tests(code, model, framework):
            if not code.strip():
                return "❌ Please enter some Python code to test."
            
            test_code = ""
            for chunk in test_generator.generate_tests(code, model, framework):
                test_code += chunk
                yield test_code
        
        def run_tests(test_code):
            if not test_code.strip():
                return "❌ No test code to run."
            
            success, output = test_generator.run_tests(test_code)
            status = "✅ Tests passed!" if success else "❌ Tests failed!"
            return f"{status}\n\n{output}"
        
        def clear_outputs():
            return "", "", ""
        
        # Connect events
        generate_btn.click(
            fn=generate_tests,
            inputs=[code_input, model_selector, framework_selector],
            outputs=test_output
        )
        
        run_tests_btn.click(
            fn=run_tests,
            inputs=test_output,
            outputs=test_results
        )
        
        clear_btn.click(
            fn=clear_outputs,
            outputs=[test_output, test_results]
        )
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
