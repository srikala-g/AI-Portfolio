# Website Summary Tool

A simple AI-powered tool to summarize websites using OpenAI GPT models.

## Features

- **Auto Mode**: Intelligently chooses the best scraping method
- **HTTP Scraping**: Fast scraping for static websites  
- **Selenium Scraping**: Handles JavaScript-rendered websites
- **Web Interface**: Clean, simple Gradio interface
- **Command Line**: Interactive CLI for batch processing

## Quick Start

### 1. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 2. Set up API Key
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Launch Web Interface
```bash
python3 gradio_app.py
```

### 4. Use Command Line
```bash
python3 website_summary.py
```

## Usage

### Web Interface
1. Enter a website URL
2. Choose scraping method (Auto Mode recommended)
3. Click Submit to get AI-generated summary

### Command Line
- Interactive menu with multiple options
- Batch processing of multiple URLs
- Method selection per URL

## Examples

```bash
# Web interface
python3 gradio_app.py

# Command line interface  
python3 website_summary.py

# Usage examples
python3 usage_example.py
```

## Files

- `website_summary.py` - Main module with CLI and Gradio support
- `gradio_app.py` - Standalone web interface
- `run_web_interface.py` - Easy launcher
- `usage_example.py` - Usage examples
- `requirements.txt` - Dependencies
