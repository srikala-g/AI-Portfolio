# Meeting Minutes Generator

A powerful AI application that converts audio recordings into structured meeting minutes using Hugging Face models and Gradio interface.

## Features

- 🎤 **Audio Transcription**: Convert audio files to text using Hugging Face Whisper models
- 📝 **Meeting Minutes Generation**: Generate structured meeting minutes with action items
- 🎯 **Quick Generate**: One-click audio-to-minutes conversion
- 🔧 **Step-by-Step**: Manual control over transcription and minutes generation
- ⚙️ **Model Management**: Load and manage different AI models
- 🌐 **Web Interface**: Easy-to-use Gradio web interface

## Models Supported

### Speech-to-Text (Whisper)
- **Whisper Small**: Fast, lightweight transcription
- **Whisper Medium**: Balanced speed and accuracy
- **Whisper Large**: High accuracy (full version only)
- **Whisper Large v3**: Latest and most accurate (full version only)

### Language Models
- **Phi-3 Mini**: Microsoft's efficient small model
- **Gemma 2B**: Google's lightweight model
- **Llama 3.1 8B**: Meta's powerful model (full version only)
- **Llama 3.1 70B**: Meta's largest model (full version only)

## Installation

1. **Clone or download the files**:
   ```bash
   # Make sure you have the meeting_minutes_gradio.py or meeting_minutes_simple.py file
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up Hugging Face token** (for private models):
   ```bash
   # Login to Hugging Face
   huggingface-cli login
   ```

## Usage

### Quick Start

1. **Run the application**:
   ```bash
   python meeting_minutes_simple.py
   ```

2. **Open your browser** and go to `http://localhost:7860`

3. **Upload an audio file** (MP3, WAV, M4A, etc.)

4. **Click "Generate Meeting Minutes"** and wait for the AI to process

### Step-by-Step Process

1. **Transcribe Audio**: Upload audio and get transcription
2. **Generate Minutes**: Use the transcription to create structured minutes
3. **Review and Edit**: Make any necessary adjustments

### Model Management

- **Load Models**: Pre-load models for faster processing
- **Switch Models**: Change between different AI models
- **Monitor Status**: Check model loading status

## File Structure

```
meeting_minutes/
├── meeting_minutes_gradio.py      # Full-featured version with all models
├── meeting_minutes_simple.py      # Simplified version with smaller models
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Gradio
- SoundFile
- Hugging Face Hub

## Hardware Requirements

### Minimum (CPU)
- 8GB RAM
- Any modern CPU
- Uses smaller models for basic functionality

### Recommended (GPU)
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA support for faster processing

### Optimal (High-end GPU)
- 32GB+ RAM
- NVIDIA RTX 4090 or similar
- Can run largest models (Llama 70B, Whisper Large v3)

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- And most other common audio formats

## Example Output

The application generates structured meeting minutes including:

1. **Meeting Summary** (attendees, date, location)
2. **Key Discussion Points**
3. **Decisions Made**
4. **Action Items** (with owners and deadlines)
5. **Next Steps**

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use smaller models or reduce batch size
2. **Model Loading Errors**: Check internet connection and Hugging Face access
3. **Audio Processing**: Ensure audio file is not corrupted

### Performance Tips

1. **Use smaller models** for faster processing
2. **Pre-load models** to avoid loading delays
3. **Use GPU** when available for better performance
4. **Split long audio** into smaller chunks if needed

## Advanced Usage

### Custom Models
You can modify the model configurations in the code to use different models:

```python
WHISPER_MODELS = {
    "Custom Model": "your-username/your-whisper-model"
}

LLM_MODELS = {
    "Custom LLM": "your-username/your-llm-model"
}
```

### API Integration
The functions can be imported and used in other applications:

```python
from meeting_minutes_simple import transcribe_audio, generate_meeting_minutes

# Use in your own code
transcription = transcribe_audio("audio.mp3", "Whisper Small")
minutes = generate_meeting_minutes(transcription, "Phi-3 Mini")
```

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## License

This project is open source and available under the MIT License.
