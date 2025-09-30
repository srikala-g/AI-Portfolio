"""
Multi-Modal AI Interface with OpenAI

This module provides a comprehensive multi-modal interface using OpenAI's APIs:
1. Prompt to Image Generation (DALL-E-3)
2. Text to Speech (TTS)
3. Speech to Text (STT)

Features:
- Image generation from text prompts
- Text-to-speech conversion with multiple voices
- Speech-to-text transcription
- Gradio web interface for easy interaction
"""

import os
import base64
import tempfile
import json
from io import BytesIO
from typing import Optional, Tuple
import gradio as gr
from openai import OpenAI
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MultiModalAI:
    """Multi-modal AI interface using OpenAI APIs."""
    
    def __init__(self):
        """Initialize the OpenAI client and check API key."""
        # Check if API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required. Please set it with: export OPENAI_API_KEY='your-key-here'")
        
        self.client = OpenAI(api_key=api_key)
        print(f"✅ OpenAI client initialized successfully")
        
        # Available TTS voices
        self.tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        # Default settings
        self.default_image_size = "1024x1024"
        self.default_image_quality = "standard"
        self.default_image_style = "vivid"
    
    def generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard", style: str = "vivid") -> Image.Image:
        """
        Generate an image from a text prompt using DALL-E-3.
        
        Args:
            prompt: Text description of the image to generate
            size: Image size (1024x1024, 1792x1024, or 1024x1792)
            quality: Image quality (standard or hd)
            style: Image style (vivid or natural)
            
        Returns:
            PIL Image object
        """
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1,
                response_format="b64_json"
            )
            
            # Decode the base64 image
            image_base64 = response.data[0].b64_json
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            return image
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def text_to_speech(self, text: str, voice: str = "alloy", model: str = "tts-1") -> bytes:
        """
        Convert text to speech using OpenAI's TTS API.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use (tts-1 or tts-1-hd)
            
        Returns:
            Audio data as bytes
        """
        try:
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            return response.content
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def speech_to_text(self, audio_file_path: str, model: str = "whisper-1") -> str:
        """
        Convert speech to text using OpenAI's Whisper API.
        
        Args:
            audio_file_path: Path to the audio file
            model: Whisper model to use (whisper-1)
            
        Returns:
            Transcribed text
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file
                )
            return transcript.text
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return f"Error: {e}"
    
    def process_image_generation(self, prompt: str, size: str, quality: str, style: str) -> Tuple[Optional[Image.Image], str]:
        """Process image generation with error handling."""
        if not prompt.strip():
            return None, "Please enter a prompt for image generation."
        
        image = self.generate_image(prompt, size, quality, style)
        if image:
            return image, "Image generated successfully!"
        else:
            return None, "Failed to generate image. Please check your prompt and try again."
    
    def process_text_to_speech(self, text: str, voice: str, model: str) -> Tuple[Optional[bytes], str, Optional[str]]:
        """Process text-to-speech with error handling."""
        if not text.strip():
            return None, "Please enter text to convert to speech.", None
        
        audio_data = self.text_to_speech(text, voice, model)
        if audio_data:
            # Save audio to temporary file for download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            return audio_data, "Audio generated successfully!", tmp_path
        else:
            return None, "Failed to generate audio. Please try again.", None
    
    def process_speech_to_text(self, audio_file) -> Tuple[str, str]:
        """Process speech-to-text with error handling."""
        if audio_file is None:
            return "", "Please upload an audio file."
        
        try:
            # audio_file is already a file path when using type="filepath"
            transcript = self.speech_to_text(audio_file)
            return transcript, "Transcription completed successfully!"
            
        except Exception as e:
            return "", f"Error processing audio: {e}"


def create_gradio_interface():
    """Create and return the Gradio interface."""
    ai = MultiModalAI()
    
    with gr.Blocks(title="Multi-Modal AI", theme=gr.themes.Default()) as interface:
        gr.Markdown("# Multi-Modal AI Interface")
        gr.Markdown("Generate images, convert text to speech, and transcribe audio using OpenAI.")
        
        with gr.Tabs():
            # Image Generation Tab
            with gr.Tab("Image Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_prompt = gr.Textbox(
                            label="Describe the image you want to generate",
                            placeholder="A serene mountain landscape at sunset",
                            lines=3
                        )
                        
                        with gr.Row():
                            image_size = gr.Dropdown(
                                choices=["1024x1024", "1792x1024", "1024x1792"],
                                value="1024x1024",
                                label="Size"
                            )
                            image_quality = gr.Dropdown(
                                choices=["standard", "hd"],
                                value="standard",
                                label="Quality"
                            )
                        
                        generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
                        image_status = gr.Textbox(label="Status", interactive=False, visible=False)
                    
                    with gr.Column(scale=1):
                        generated_image = gr.Image(label="Generated Image", type="pil", height=400)
            
            # Text-to-Speech Tab
            with gr.Tab("Text to Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(
                            label="Enter text to convert to speech",
                            placeholder="Hello! This is a test of text-to-speech conversion.",
                            lines=4
                        )
                        
                        with gr.Row():
                            tts_voice = gr.Dropdown(
                                choices=ai.tts_voices,
                                value="alloy",
                                label="Voice"
                            )
                            tts_model = gr.Dropdown(
                                choices=["tts-1", "tts-1-hd"],
                                value="tts-1",
                                label="Model"
                            )
                        
                        tts_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                        tts_status = gr.Textbox(label="Status", interactive=False, visible=False)
                    
                    with gr.Column(scale=1):
                        tts_audio = gr.Audio(label="Generated Audio", type="numpy")
                        download_audio = gr.DownloadButton("Download Audio", visible=True)
            
            # Speech-to-Text Tab
            with gr.Tab("Speech to Text"):
                with gr.Row():
                    with gr.Column(scale=1):
                        stt_audio = gr.Audio(
                            label="Upload audio file or record",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        stt_btn = gr.Button("Transcribe Audio", variant="primary", size="lg")
                        stt_status = gr.Textbox(label="Status", interactive=False, visible=False)
                    
                    with gr.Column(scale=1):
                        stt_text = gr.Textbox(
                            label="Transcribed Text",
                            lines=8,
                            interactive=True
                        )
        
        # Event handlers
        generate_btn.click(
            fn=ai.process_image_generation,
            inputs=[image_prompt, image_size, image_quality, gr.Dropdown(value="vivid", visible=False)],
            outputs=[generated_image, image_status]
        )
        
        tts_btn.click(
            fn=ai.process_text_to_speech,
            inputs=[tts_text, tts_voice, tts_model],
            outputs=[tts_audio, tts_status, download_audio]
        )
        
        stt_btn.click(
            fn=ai.process_speech_to_text,
            inputs=[stt_audio],
            outputs=[stt_text, stt_status]
        )
    
    return interface


def main():
    """Main function to launch the interface."""
    print("Starting Multi-Modal AI Interface...")
    
    interface = create_gradio_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
