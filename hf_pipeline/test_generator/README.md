# 🧪 AI-Powered Unit Test Generator

A comprehensive tool that generates unit tests for Python code using state-of-the-art language models (GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-Flash).

## 🚀 Features

- **AI-Powered Test Generation**: Uses OpenAI GPT-4o, Anthropic Claude-3.5-Sonnet, and Google Gemini-1.5-Flash for test generation
- **Real-time Streaming**: Watch the test generation process in real-time
- **Interactive Web Interface**: User-friendly Gradio interface
- **Code Analysis**: Automatically analyzes Python code to understand functions, classes, and dependencies
- **Multiple Test Frameworks**: Supports both pytest and unittest frameworks
- **Test Execution**: Run generated tests directly in the interface
- **Comprehensive Coverage**: Generates tests for edge cases, boundary conditions, and error scenarios
- **Cross-platform Support**: Works on Mac, Windows, and Linux

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models)
- Google API key (for Gemini models) - Optional
- Internet connection for AI model access

## 🛠️ Installation

1. **Clone or download the project:**
   ```bash
   cd /Users/srikala/projects/AI-Portfolio/hf_pipeline/test_generator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project directory:
   ```bash
   # Required API keys
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Optional API key (for Gemini support)
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 🚀 Usage

### Command Line Interface

Run the test generator:
```bash
python test_generator.py
```

The application will start a web interface at `http://localhost:7860`.

### Web Interface

1. **Open your browser** and navigate to `http://localhost:7860`
2. **Paste your Python code** in the input area
3. **Select an AI model** (GPT, Claude, or Gemini)
4. **Choose a test framework** (pytest or unittest)
5. **Click "Generate Tests"** to create unit tests
6. **Click "Run Tests"** to execute the generated tests
7. **View results** in the test results area

### Programmatic Usage

```python
from test_generator import TestGenerator

# Initialize the generator
generator = TestGenerator()

# Analyze code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

analysis = generator.analyze_code(code)
print(f"Functions found: {[f['name'] for f in analysis['functions']]}")

# Generate tests
for test_chunk in generator.generate_tests(code, "GPT", "pytest"):
    print(test_chunk, end="")

# Run tests
test_code = """
def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
"""

success, output = generator.run_tests(test_code)
print(f"Tests passed: {success}")
```

## 🧪 Generated Test Features

The AI generates comprehensive test cases including:

### **Function Testing**
- ✅ **Basic functionality tests**
- ✅ **Edge case testing**
- ✅ **Boundary condition testing**
- ✅ **Error scenario testing**
- ✅ **Parameterized tests**

### **Class Testing**
- ✅ **Method testing**
- ✅ **Constructor testing**
- ✅ **Property testing**
- ✅ **Inheritance testing**

### **Advanced Features**
- ✅ **Mocking external dependencies**
- ✅ **Setup and teardown methods**
- ✅ **Fixture usage**
- ✅ **Exception testing**
- ✅ **Type validation**

## 📊 Supported AI Models

### **OpenAI GPT-4o** ✅
- **Model**: `gpt-4o`
- **Strengths**: Excellent code understanding, comprehensive test generation
- **Requirements**: Valid OpenAI API key

### **Anthropic Claude-3.5-Sonnet** ✅
- **Model**: `claude-3.5-sonnet-20240620`
- **Strengths**: High-quality reasoning, detailed test cases
- **Requirements**: Valid Anthropic API key

### **Google Gemini-1.5-Flash** ✅
- **Model**: `gemini-1.5-flash`
- **Strengths**: Fast generation, good coverage
- **Requirements**: Valid Google API key (optional)

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT models |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude models |
| `GOOGLE_API_KEY` | No | Google API key for Gemini models |

### Model Configuration

You can customize the models used:

```python
generator = TestGenerator(
    openai_model="gpt-4o",
    claude_model="claude-3-5-sonnet-20240620",
    gemini_model="gemini-1.5-flash"
)
```

## 🧪 Testing

Run the test suite:

```bash
# Run basic functionality tests
python test_test_generator.py

# Run with pytest (if available)
pytest test_test_generator.py -v
```

## 📁 Project Structure

```
test_generator/
├── test_generator.py          # Main application
├── test_test_generator.py     # Test suite
├── requirements.txt            # Dependencies
├── README.md                  # Documentation
└── .env                       # Environment variables (create this)
```

## 🎯 Example Usage

### Input Code
```python
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
```

### Generated Tests (Sample)
```python
import pytest
from unittest.mock import patch

def test_fibonacci_basic():
    """Test basic Fibonacci calculations."""
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5

def test_fibonacci_edge_cases():
    """Test edge cases for Fibonacci."""
    assert fibonacci(2) == 1
    assert fibonacci(10) == 55

def test_fibonacci_negative_input():
    """Test Fibonacci with negative input raises ValueError."""
    with pytest.raises(ValueError, match="n must be non-negative"):
        fibonacci(-1)

class TestCalculator:
    """Test cases for Calculator class."""
    
    def test_calculator_init(self):
        """Test calculator initialization."""
        calc = Calculator()
        assert calc.history == []
    
    def test_add_positive_numbers(self):
        """Test adding positive numbers."""
        calc = Calculator()
        result = calc.add(2, 3)
        assert result == 5
        assert "2 + 3 = 5" in calc.history
    
    def test_add_negative_numbers(self):
        """Test adding negative numbers."""
        calc = Calculator()
        result = calc.add(-2, -3)
        assert result == -5
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all required API keys are set in `.env` file
   - Check that API keys are valid and have sufficient credits

2. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Ensure Python 3.8+ is being used

3. **Test Execution Errors**
   - Ensure pytest is installed: `pip install pytest`
   - Check that generated test code is syntactically correct

4. **Gemini Model Not Available**
   - Set a valid `GOOGLE_API_KEY` in your `.env` file
   - The application works fine without Gemini (uses GPT and Claude)

### Getting Help

If you encounter issues:

1. **Check the logs** for detailed error messages
2. **Verify API keys** are correctly set
3. **Test with simple code** first
4. **Check internet connection** for AI model access

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **Anthropic** for Claude models  
- **Google** for Gemini models
- **Gradio** for the web interface framework
- **pytest** for the testing framework

---

**Happy Testing! 🧪✨**
