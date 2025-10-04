#!/bin/bash

echo "🚀 Installing Hugging Face Pipelines Dependencies"
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install basic requirements
echo "📦 Installing basic requirements..."
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install transformers datasets diffusers gradio soundfile accelerate safetensors

echo "✅ Installation complete!"
echo ""
echo "To test the installation, run:"
echo "python3 test_app.py"
echo ""
echo "To run the Gradio app, run:"
echo "python3 app.py"
