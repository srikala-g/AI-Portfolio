# Multi-Modal AI Interface

A comprehensive multi-modal AI interface using OpenAI's APIs for image generation, text-to-speech, and speech-to-text conversion.

## Features

### 🎨 Image Generation
- Generate high-quality images from text prompts using DALL-E-3
- Multiple image sizes: 1024x1024, 1792x1024, 1024x1792
- Quality options: Standard and HD
- Style options: Vivid and Natural

### 🔊 Text to Speech
- Convert text to natural-sounding speech
- Multiple voice options: alloy, echo, fable, onyx, nova, shimmer
- Two model options: tts-1 and tts-1-hd

### 🎤 Speech to Text
- Transcribe audio files to text using Whisper
- Support for various audio formats
- High accuracy transcription

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

Run the application:
```bash
python multi-modal.py
```

The interface will open in your browser at `http://localhost:7860`.

## API Usage

You can also use the `MultiModalAI` class directly in your code:

```python
from multi_modal import MultiModalAI

# Initialize the AI
ai = MultiModalAI()

# Generate an image
image = ai.generate_image("A beautiful sunset over mountains")

# Convert text to speech
audio_data = ai.text_to_speech("Hello, world!", voice="alloy")

# Transcribe audio
transcript = ai.speech_to_text("path/to/audio.wav")
```

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## Notes

- Image generation costs approximately $0.04 per image
- TTS costs vary by model and length
- STT costs vary by audio duration
- Make sure to monitor your OpenAI API usage
