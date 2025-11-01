#!/usr/bin/env python3
"""
Music Generation with AI using Transformers
Alternative implementation using Hugging Face transformers
"""

import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import soundfile as sf

# Try to import Jupyter-specific modules
try:
    from IPython.display import Audio, display
    from ipywidgets import Textarea, Button
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    print("Jupyter modules not available. Running in command-line mode.")


def load_model():
    """Load the pretrained MusicGen model from Hugging Face"""
    model_name = "facebook/musicgen-small"
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    return model, processor


def configure_model(model, duration=8):
    """Configure model generation parameters"""
    # Set generation parameters
    model.generation_config.max_new_tokens = int(duration * 50)  # Approximate tokens for duration
    return model


def generate_music(model, processor, prompt):
    """Generate music from text prompt"""
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=400)
    
    # Convert to numpy array
    audio_numpy = audio_values[0, 0].cpu().numpy()
    sample_rate = 32000  # MusicGen default sample rate
    
    return audio_numpy, sample_rate


def save_audio(audio_data, sample_rate, filename="generated_music.wav"):
    """Save generated audio to a file"""
    sf.write(filename, audio_data, sample_rate)
    print(f"Audio saved to: {filename}")


def create_interactive_ui(model, processor):
    """Create interactive UI for music generation"""
    if not JUPYTER_AVAILABLE:
        print("Interactive UI requires Jupyter notebook environment.")
        return None, None
        
    # Create a text area and a button
    description = Textarea(value='', placeholder='Give a music prompt', disabled=False, rows=4)
    generate_button = Button(description="Generate Tune")

    # A function to generate music as prompted
    def generate_tune(event):
        try:
            audio_data, sample_rate = generate_music(model, processor, description.value)
            display(Audio(audio_data, rate=sample_rate))
        except Exception as e:
            print(f"Error generating music: {e}")

    # Create a click event on the button
    generate_button.on_click(generate_tune)

    # Display the UI items
    display(description)
    display(generate_button)
    
    return description, generate_button


def main():
    """Main function to run the music generation demo"""
    print("Loading MusicGen model from Hugging Face...")
    model, processor = load_model()
    
    print("Configuring model parameters...")
    model = configure_model(model, duration=8)
    
    print("Model loaded successfully!")
    
    # Example: Generate a classic rock song
    print("Generating example music...")
    try:
        audio_data, sample_rate = generate_music(model, processor, 'classic rock song')
        print(f"Generated audio with sample rate: {sample_rate}")
        print(f"Audio shape: {audio_data.shape}")
        
        # Save the example audio
        save_audio(audio_data, sample_rate, "example_classic_rock.wav")
    except Exception as e:
        print(f"Error in example generation: {e}")
    
    # Create interactive UI
    print("Creating interactive UI...")
    try:
        description, generate_button = create_interactive_ui(model, processor)
        if JUPYTER_AVAILABLE:
            print("Interactive music generator is ready!")
            print("Enter a music prompt in the text area and click 'Generate Tune'")
        else:
            print("To use the interactive UI, run this script in a Jupyter notebook.")
            print("For command-line usage, you can call generate_music() function directly.")
    except Exception as e:
        print(f"Error creating UI: {e}")


if __name__ == "__main__":
    main()
