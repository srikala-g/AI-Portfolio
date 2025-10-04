# Code Generator: Python to C++ Converter

A powerful AI-powered tool that converts Python code to high-performance C++ code using state-of-the-art language models (GPT-4o and Claude-3.5-Sonnet).

## 🚀 Features

- **AI-Powered Conversion**: Uses OpenAI GPT-4o and Anthropic Claude-3.5-Sonnet for code conversion
- **Real-time Streaming**: Watch the conversion process in real-time
- **Interactive Web Interface**: User-friendly Gradio interface
- **Code Execution**: Execute both Python and C++ code directly in the interface
- **Performance Optimization**: Generates optimized C++ code for maximum performance
- **Cross-platform Support**: Works on Mac, Windows, and Linux

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Anthropic API key
- C++ compiler (clang++ recommended)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hf_pipeline/code_gen
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## 🚀 Usage

### Command Line Interface

```python
from code_gen import CodeGenerator

# Initialize the generator
generator = CodeGenerator()

# Convert Python code to C++
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""

# Convert using GPT
cpp_code = generator.convert_with_gpt(python_code)
print(cpp_code)

# Convert using Claude
cpp_code = generator.convert_with_claude(python_code)
print(cpp_code)
```

### Web Interface

```bash
python code_gen.py
```

Then open your browser and go to `http://localhost:7860`

## 📖 API Reference

### CodeGenerator Class

#### `__init__(openai_model: str, claude_model: str)`
Initialize the CodeGenerator with AI models.

**Parameters:**
- `openai_model` (str): OpenAI model to use (default: "gpt-4o")
- `claude_model` (str): Claude model to use (default: "claude-3-5-sonnet-20240620")

#### `convert_with_gpt(python_code: str) -> str`
Convert Python code to C++ using GPT model.

**Parameters:**
- `python_code` (str): Python code to convert

**Returns:**
- `str`: Generated C++ code

#### `convert_with_claude(python_code: str) -> str`
Convert Python code to C++ using Claude model.

**Parameters:**
- `python_code` (str): Python code to convert

**Returns:**
- `str`: Generated C++ code

#### `convert_code(python_code: str, model: str) -> Generator[str, None, None]`
Convert Python code to C++ with streaming output.

**Parameters:**
- `python_code` (str): Python code to convert
- `model` (str): Model to use ("GPT" or "Claude")

**Yields:**
- `str`: Partial C++ code as it's generated

#### `execute_python(python_code: str) -> str`
Execute Python code and capture output.

**Parameters:**
- `python_code` (str): Python code to execute

**Returns:**
- `str`: Execution output

#### `execute_cpp(cpp_code: str) -> str`
Compile and execute C++ code.

**Parameters:**
- `cpp_code` (str): C++ code to compile and execute

**Returns:**
- `str`: Execution output or error message

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📝 Examples

### Example 1: Simple Function Conversion

**Python Code:**
```python
def calculate_pi(iterations):
    result = 0.0
    for i in range(iterations):
        result += 4 * ((-1) ** i) / (2 * i + 1)
    return result

print(calculate_pi(1000000))
```

**Generated C++ Code:**
```cpp
#include <iostream>
#include <cmath>

double calculate_pi(int iterations) {
    double result = 0.0;
    for (int i = 0; i < iterations; i++) {
        result += 4 * pow(-1, i) / (2 * i + 1);
    }
    return result;
}

int main() {
    std::cout << calculate_pi(1000000) << std::endl;
    return 0;
}
```

### Example 2: Performance Comparison

The tool is particularly useful for computationally intensive tasks where C++ can provide significant performance improvements over Python.

## 🔧 Configuration

### Model Selection

You can choose between different AI models:

- **GPT-4o**: Latest OpenAI model with excellent code generation
- **Claude-3.5-Sonnet**: Anthropic's model with strong reasoning capabilities

### Compiler Options

The C++ compiler options can be customized in the `execute_cpp` method:

```python
# For M1 Mac
compile_cmd = ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1", "-o", "optimized", "optimized.cpp"]

# For Intel Mac
compile_cmd = ["clang++", "-Ofast", "-std=c++17", "-march=native", "-o", "optimized", "optimized.cpp"]

# For Linux
compile_cmd = ["g++", "-Ofast", "-std=c++17", "-march=native", "-o", "optimized", "optimized.cpp"]
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your API keys are correctly set in the `.env` file
2. **Compilation Errors**: Ensure you have a C++ compiler installed
3. **Memory Issues**: For large code conversions, consider using smaller models

### Performance Tips

1. **Use appropriate models**: GPT-4o for complex code, Claude for simpler conversions
2. **Optimize input**: Clean and well-formatted Python code produces better C++
3. **Test thoroughly**: Always test the generated C++ code before production use

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- Anthropic for providing the Claude models
- Gradio for the excellent web interface framework
- The LLM Engineering community for inspiration and examples
