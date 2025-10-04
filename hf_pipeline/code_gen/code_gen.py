#!/usr/bin/env python3
"""
Code Generator: Python to C++ Converter

This module provides functionality to convert Python code to high-performance C++ code
using AI models (GPT-4o and Claude-3.5-Sonnet). It includes a Gradio web interface
for interactive code conversion and execution.

Features:
- Convert Python code to optimized C++ code
- Support for multiple AI models (GPT, Claude)
- Real-time streaming of conversion process
- Execute both Python and C++ code
- Web interface for interactive usage

Author: Extracted from LLM Engineering Week 4 Day 3
"""

import os
import io
import sys
import subprocess
import logging
from typing import Generator, List, Dict, Any, Optional
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    A class to handle Python to C++ code conversion using AI models.
    
    This class provides methods to convert Python code to optimized C++ code
    using OpenAI GPT or Anthropic Claude models, with support for streaming
    and execution of both Python and C++ code.
    """
    
    def __init__(self, openai_model: str = "gpt-4o", claude_model: str = "claude-3-5-sonnet-20240620"):
        """
        Initialize the CodeGenerator with AI models.
        
        Args:
            openai_model (str): OpenAI model to use for conversion
            claude_model (str): Claude model to use for conversion
        """
        self._load_environment()
        self.openai_client = OpenAI()
        self.claude_client = anthropic.Anthropic()
        self.openai_model = openai_model
        self.claude_model = claude_model
        self.system_message = self._get_system_message()
        
    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        try:
            load_dotenv(override=True)
            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
            os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
            logger.info("Environment variables loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load environment variables: {e}")
            raise
    
    def _get_system_message(self) -> str:
        """
        Get the system message for AI models.
        
        Returns:
            str: System message for code conversion
        """
        return ("You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
                "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
                "The C++ response needs to produce an identical output in the fastest possible time.")
    
    def _get_user_prompt(self, python_code: str) -> str:
        """
        Generate user prompt for code conversion.
        
        Args:
            python_code (str): Python code to convert
            
        Returns:
            str: Formatted user prompt
        """
        return (f"Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
                f"Respond only with C++ code; do not explain your work other than a few comments. "
                f"Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
                f"{python_code}")
    
    def _get_messages(self, python_code: str) -> List[Dict[str, str]]:
        """
        Create messages for OpenAI API.
        
        Args:
            python_code (str): Python code to convert
            
        Returns:
            List[Dict[str, str]]: Messages for API call
        """
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self._get_user_prompt(python_code)}
        ]
    
    def _write_cpp_file(self, cpp_code: str, filename: str = "optimized.cpp") -> None:
        """
        Write C++ code to a file.
        
        Args:
            cpp_code (str): C++ code to write
            filename (str): Output filename
        """
        try:
            # Clean up the code by removing markdown code blocks
            cleaned_code = cpp_code.replace("```cpp", "").replace("```", "").strip()
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(cleaned_code)
            logger.info(f"C++ code written to {filename}")
        except Exception as e:
            logger.error(f"Failed to write C++ file: {e}")
            raise
    
    def convert_with_gpt(self, python_code: str) -> str:
        """
        Convert Python code to C++ using GPT model.
        
        Args:
            python_code (str): Python code to convert
            
        Returns:
            str: Generated C++ code
        """
        try:
            logger.info("Starting GPT conversion")
            stream = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=self._get_messages(python_code),
                stream=True
            )
            
            reply = ""
            for chunk in stream:
                fragment = chunk.choices[0].delta.content or ""
                reply += fragment
                print(fragment, end='', flush=True)
            
            self._write_cpp_file(reply)
            logger.info("GPT conversion completed")
            return reply
            
        except Exception as e:
            logger.error(f"GPT conversion failed: {e}")
            raise
    
    def convert_with_claude(self, python_code: str) -> str:
        """
        Convert Python code to C++ using Claude model.
        
        Args:
            python_code (str): Python code to convert
            
        Returns:
            str: Generated C++ code
        """
        try:
            logger.info("Starting Claude conversion")
            result = self.claude_client.messages.stream(
                model=self.claude_model,
                max_tokens=2000,
                system=self.system_message,
                messages=[{"role": "user", "content": self._get_user_prompt(python_code)}],
            )
            
            reply = ""
            with result as stream:
                for text in stream.text_stream:
                    reply += text
                    print(text, end="", flush=True)
            
            self._write_cpp_file(reply)
            logger.info("Claude conversion completed")
            return reply
            
        except Exception as e:
            logger.error(f"Claude conversion failed: {e}")
            raise
    
    def stream_gpt(self, python_code: str) -> Generator[str, None, None]:
        """
        Stream GPT conversion process.
        
        Args:
            python_code (str): Python code to convert
            
        Yields:
            str: Partial C++ code as it's generated
        """
        try:
            stream = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=self._get_messages(python_code),
                stream=True
            )
            
            reply = ""
            for chunk in stream:
                fragment = chunk.choices[0].delta.content or ""
                reply += fragment
                yield reply.replace('```cpp\n', '').replace('```', '')
                
        except Exception as e:
            logger.error(f"GPT streaming failed: {e}")
            raise
    
    def stream_claude(self, python_code: str) -> Generator[str, None, None]:
        """
        Stream Claude conversion process.
        
        Args:
            python_code (str): Python code to convert
            
        Yields:
            str: Partial C++ code as it's generated
        """
        try:
            result = self.claude_client.messages.stream(
                model=self.claude_model,
                max_tokens=2000,
                system=self.system_message,
                messages=[{"role": "user", "content": self._get_user_prompt(python_code)}],
            )
            
            reply = ""
            with result as stream:
                for text in stream.text_stream:
                    reply += text
                    yield reply.replace('```cpp\n', '').replace('```', '')
                    
        except Exception as e:
            logger.error(f"Claude streaming failed: {e}")
            raise
    
    def convert_code(self, python_code: str, model: str) -> Generator[str, None, None]:
        """
        Convert Python code to C++ using specified model.
        
        Args:
            python_code (str): Python code to convert
            model (str): Model to use ("GPT" or "Claude")
            
        Yields:
            str: Partial C++ code as it's generated
            
        Raises:
            ValueError: If model is not supported
        """
        if model == "GPT":
            yield from self.stream_gpt(python_code)
        elif model == "Claude":
            yield from self.stream_claude(python_code)
        else:
            raise ValueError(f"Unknown model: {model}. Supported models: GPT, Claude")
    
    def execute_python(self, python_code: str) -> str:
        """
        Execute Python code and capture output.
        
        Args:
            python_code (str): Python code to execute
            
        Returns:
            str: Execution output
        """
        try:
            logger.info("Executing Python code")
            output = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = output
            
            try:
                # Create a proper namespace for execution
                namespace = {}
                exec(python_code, namespace)
            finally:
                sys.stdout = original_stdout
            
            result = output.getvalue()
            logger.info("Python execution completed")
            return result
            
        except Exception as e:
            logger.error(f"Python execution failed: {e}")
            return f"Error executing Python code: {e}"
    
    def execute_cpp(self, cpp_code: str) -> str:
        """
        Compile and execute C++ code.
        
        Args:
            cpp_code (str): C++ code to compile and execute
            
        Returns:
            str: Execution output or error message
        """
        try:
            logger.info("Compiling and executing C++ code")
            self._write_cpp_file(cpp_code)
            
            # Compile C++ code with optimizations for M1 Mac
            compile_cmd = [
                "clang++", "-Ofast", "-std=c++17", 
                "-march=armv8.5-a", "-mtune=apple-m1", 
                "-mcpu=apple-m1", "-o", "optimized", "optimized.cpp"
            ]
            
            compile_result = subprocess.run(
                compile_cmd, 
                check=True, 
                text=True, 
                capture_output=True
            )
            
            # Execute the compiled program
            run_cmd = ["./optimized"]
            run_result = subprocess.run(
                run_cmd, 
                check=True, 
                text=True, 
                capture_output=True
            )
            
            logger.info("C++ execution completed")
            return run_result.stdout
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Compilation or execution error:\n{e.stderr}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            return error_msg


def create_gradio_interface() -> gr.Blocks:
    """
    Create Gradio web interface for code conversion.
    
    Returns:
        gr.Blocks: Gradio interface
    """
    try:
        code_generator = CodeGenerator()
        
        # Sample Python code for demonstration
        sample_python = """# Be careful to support large number sizes

def lcg(seed, a=1664525, c=1013904223, m=2**32):
    value = seed
    while True:
        value = (a * value + c) % m
        yield value
        
def max_subarray_sum(n, seed, min_val, max_val):
    lcg_gen = lcg(seed)
    random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
    max_sum = float('-inf')
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += random_numbers[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum

def total_max_subarray_sum(n, initial_seed, min_val, max_val):
    total_sum = 0
    lcg_gen = lcg(initial_seed)
    for _ in range(20):
        seed = next(lcg_gen)
        total_sum += max_subarray_sum(n, seed, min_val, max_val)
    return total_sum

# Parameters - using smaller values for demo
n = 1000          # Number of random numbers (reduced for demo)
initial_seed = 42  # Initial seed for the LCG
min_val = -10     # Minimum value of random numbers
max_val = 10      # Maximum value of random numbers

# Timing the function
import time
start_time = time.time()
result = total_max_subarray_sum(n, initial_seed, min_val, max_val)
end_time = time.time()

print("Total Maximum Subarray Sum (20 runs):", result)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))"""
        
        # CSS for styling
        css = """
        .python {background-color: #306998;}
        .cpp {background-color: #050;}
        """
        
        with gr.Blocks(css=css, title="Python to C++ Code Converter") as interface:
            gr.Markdown("# Convert Python Code to High-Performance C++")
            gr.Markdown("Use AI models to convert Python code to optimized C++ code with performance improvements.")
            
            with gr.Row():
                python_input = gr.Textbox(
                    label="Python Code:", 
                    value=sample_python, 
                    lines=15,
                    placeholder="Enter your Python code here..."
                )
                cpp_output = gr.Textbox(
                    label="Generated C++ Code:", 
                    lines=15,
                    placeholder="C++ code will appear here..."
                )
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    ["GPT", "Claude"], 
                    label="Select AI Model", 
                    value="GPT"
                )
                convert_btn = gr.Button("Convert to C++", variant="primary")
            
            with gr.Row():
                python_run_btn = gr.Button("Run Python Code", variant="secondary")
                cpp_run_btn = gr.Button("Run C++ Code", variant="secondary")
            
            with gr.Row():
                python_result = gr.TextArea(
                    label="Python Execution Result:", 
                    lines=8,
                    elem_classes=["python"]
                )
                cpp_result = gr.TextArea(
                    label="C++ Execution Result:", 
                    lines=8,
                    elem_classes=["cpp"]
                )
            
            # Event handlers
            convert_btn.click(
                code_generator.convert_code,
                inputs=[python_input, model_selector],
                outputs=[cpp_output]
            )
            
            python_run_btn.click(
                code_generator.execute_python,
                inputs=[python_input],
                outputs=[python_result]
            )
            
            cpp_run_btn.click(
                code_generator.execute_cpp,
                inputs=[cpp_output],
                outputs=[cpp_result]
            )
        
        logger.info("Gradio interface created successfully")
        return interface
        
    except Exception as e:
        logger.error(f"Failed to create Gradio interface: {e}")
        raise


def main() -> None:
    """
    Main function to run the code generator application.
    """
    try:
        logger.info("Starting Code Generator application")
        
        # Create and launch the interface
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True
        )
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
