#!/usr/bin/env python3
"""
Minimal AI Music Generator for Hugging Face Spaces
Simplified version to avoid deployment issues
"""

import gradio as gr
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import soundfile as sf
from datetime import datetime


class MusicGenerator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the pretrained MusicGen model"""
        if self.model_loaded:
            return "Model already loaded!"
            
        try:
            print("Loading MusicGen model...")
            model_name = "facebook/musicgen-small"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
            
            # Configure model
            self.model.generation_config.max_new_tokens = 400
            
            self.model_loaded = True
            print("Model loaded successfully!")
            return "✅ Model loaded successfully!"
            
        except Exception as e:
            error_msg = f"❌ Error loading model: {str(e)}"
            print(error_msg)
            return error_msg
    
    def generate_music(self, prompt, duration=8):
        """Generate music from text prompt"""
        if not self.model_loaded:
            return None, "Please load the model first!"
            
        if not prompt.strip():
            return None, "Please enter a music prompt!"
            
        try:
            print(f"Generating music for: {prompt}")
            
            # Process input
            inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
            
            # Generate music
            with torch.no_grad():
                audio_values = self.model.generate(**inputs, max_new_tokens=400)
            
            # Convert to numpy
            audio_numpy = audio_values[0, 0].cpu().numpy()
            sample_rate = 32000
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_music_{timestamp}.wav"
            sf.write(filename, audio_numpy, sample_rate)
            
            success_msg = f"✅ Music generated! Saved as {filename}"
            print(success_msg)
            
            return (sample_rate, audio_numpy), success_msg
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return None, error_msg


# Initialize generator
music_gen = MusicGenerator()


def create_interface():
    """Create simple Gradio interface"""
    
    with gr.Blocks(title="AI Music Generator") as demo:
        
        gr.Markdown("# 🎵 AI Music Generator")
        gr.Markdown("Generate music from text descriptions using AI")
        
        with gr.Row():
            load_btn = gr.Button("Load Model", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Music Description",
                    placeholder="e.g., 'Classic rock song with electric guitar'",
                    lines=2
                )
                duration = gr.Slider(
                    minimum=4,
                    maximum=12,
                    value=8,
                    step=1,
                    label="Duration (seconds)"
                )
                generate_btn = gr.Button("Generate Music", variant="primary")
                
            with gr.Column():
                audio_output = gr.Audio(label="Generated Music")
                output_status = gr.Textbox(label="Output Status", interactive=False)
        
        # Examples
        gr.Markdown("### Example Prompts:")
        examples = [
            "Classic rock song with electric guitar",
            "Jazz piano solo",
            "Electronic dance music",
            "Acoustic folk song",
            "Orchestral cinematic music"
        ]
        
        gr.Examples(examples=examples, inputs=prompt)
        
        # Event handlers
        load_btn.click(
            fn=music_gen.load_model,
            outputs=status
        )
        
        generate_btn.click(
            fn=music_gen.generate_music,
            inputs=[prompt, duration],
            outputs=[audio_output, output_status]
        )
        
        # Auto-load model
        demo.load(
            fn=music_gen.load_model,
            outputs=status
        )
    
    return demo


def main():
    """Main function"""
    print("Starting AI Music Generator...")
    
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    main()
