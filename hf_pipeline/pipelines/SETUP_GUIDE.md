# 🚀 Complete Setup Guide for Hugging Face Pipelines

This guide will help you set up and deploy your Hugging Face Pipelines demo both locally and on Hugging Face Spaces.

## 📁 Project Structure

```
hf_pipeline/
├── app.py                    # Main Gradio application
├── requirements_spaces.txt   # Dependencies for Hugging Face Spaces
├── README_spaces.md         # README for Hugging Face Spaces
├── DEPLOYMENT_GUIDE.md       # Detailed deployment instructions
├── install_dependencies.sh  # Local installation script
├── test_app.py              # Test script
└── pipelines/               # Original pipeline code
    └── pipelines.py
```

## 🏠 Local Setup

### Option 1: Quick Setup (Recommended)

```bash
cd /Users/srikala/projects/AI-Portfolio/hf_pipeline
./install_dependencies.sh
```

### Option 2: Manual Setup

```bash
# Install PyTorch (CPU version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip3 install transformers datasets diffusers gradio soundfile accelerate safetensors
```

### Test the Installation

```bash
python3 test_app.py
```

### Run the App Locally

```bash
python3 app.py
```

The app will be available at `http://localhost:7860`

## 🌐 Deploy to Hugging Face Spaces

### Step 1: Create a Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co)
2. Create a free account
3. Verify your email

### Step 2: Install Hugging Face CLI

```bash
pip install huggingface_hub[cli]
huggingface-cli login
```

### Step 3: Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name**: `hf-pipelines-demo`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free)
   - **Visibility**: Public

### Step 4: Upload Files

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy the necessary files
cp /Users/srikala/projects/AI-Portfolio/hf_pipeline/app.py .
cp /Users/srikala/projects/AI-Portfolio/hf_pipeline/requirements_spaces.txt requirements.txt
cp /Users/srikala/projects/AI-Portfolio/hf_pipeline/README_spaces.md README.md

# Deploy
git add .
git commit -m "Initial commit: Hugging Face Pipelines Demo"
git push
```

## 🎯 Features Overview

### 📝 Text Analysis
- **Sentiment Analysis**: Analyze emotional tone
- **Named Entity Recognition**: Extract entities

### ❓ Q&A & Summarization
- **Question Answering**: Answer questions from context
- **Text Summarization**: Create concise summaries

### 🌍 Translation & Classification
- **Translation**: English to French, Spanish, German, Italian
- **Zero-shot Classification**: Classify into custom categories

### ✍️ Text Generation
- **Text Generation**: Generate text from prompts

### 🎨 Image Generation
- **Image Generation**: Create images from text using Stable Diffusion

### 🎵 Text-to-Speech
- **Text-to-Speech**: Convert text to natural speech

## ⚡ Performance Optimizations

### CPU Optimization
- All models run on CPU for compatibility
- Model caching for faster subsequent runs
- Lazy loading to reduce memory usage
- Optimized inference steps for image generation

### Memory Management
- Pipeline caching to avoid reloading models
- Temporary file handling for audio output
- Error handling for memory constraints

## 🔧 Troubleshooting

### Common Issues

1. **"No module named 'torch'"**
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **"CUDA out of memory"**
   - The app is designed for CPU, but if you see this error, restart the app

3. **Slow model loading**
   - First-time downloads can take 5-10 minutes
   - Models are cached after first use

4. **Audio generation fails**
   - Check if `soundfile` is installed
   - Some models may require additional dependencies

### Performance Tips

- **First Run**: Be patient during model downloads
- **Memory**: Close other applications for better performance
- **Network**: Ensure stable internet for model downloads

## 📊 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Stable connection for model downloads

### Recommended
- **RAM**: 8GB+
- **CPU**: Multi-core processor
- **Storage**: 5GB+ free space

## 🚀 Next Steps

After successful setup:

1. **Test Locally**: Run `python3 app.py` and test all features
2. **Deploy to Spaces**: Follow the deployment guide
3. **Customize**: Modify the app for your specific needs
4. **Share**: Share your Space with others

## 📚 Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)

## 🆘 Support

If you encounter issues:

1. Check the logs in your Space (if deployed)
2. Test locally first to isolate issues
3. Review the troubleshooting section
4. Check Hugging Face Spaces documentation

## 🎉 Success!

Once everything is working, you'll have:
- A fully functional AI demo app
- Deployed on Hugging Face Spaces
- Accessible to anyone on the internet
- Showcasing various AI capabilities

Your app will be available at:
`https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
