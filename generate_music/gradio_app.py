import gradio as gr
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
from datetime import datetime


class MusicGenerator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False
        
    def load_model(self):
        if self.model_loaded:
            return "Model already loaded!"
            
        try:
            model_name = "facebook/musicgen-small"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
            self.model.generation_config.max_new_tokens = 400
            self.model_loaded = True
            return "✅ Model loaded successfully!"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_music(self, prompt, duration=8):
        if not self.model_loaded:
            return None, "Please load the model first!"
            
        if not prompt.strip():
            return None, "Please enter a music prompt!"
            
        try:
            inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
            
            with torch.no_grad():
                audio_values = self.model.generate(**inputs, max_new_tokens=400)
            
            audio_numpy = audio_values[0, 0].cpu().numpy()
            sample_rate = 32000
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_music_{timestamp}.wav"
            sf.write(filename, audio_numpy, sample_rate)
            
            return (sample_rate, audio_numpy), f"✅ Music generated! Saved as {filename}"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"


music_gen = MusicGenerator()


def create_interface():
    with gr.Blocks() as demo:
        
        gr.Markdown("# 🎵 AI Music Generator")
        
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
        
        gr.Markdown("### Example Prompts:")
        examples = [
            "Classic rock song with electric guitar",
            "Jazz piano solo",
            "Electronic dance music",
            "Acoustic folk song",
            "Orchestral cinematic music"
        ]
        
        gr.Examples(examples=examples, inputs=prompt)
        
        load_btn.click(
            fn=music_gen.load_model,
            outputs=status
        )
        
        generate_btn.click(
            fn=music_gen.generate_music,
            inputs=[prompt, duration],
            outputs=[audio_output, output_status]
        )
        
        demo.load(
            fn=music_gen.load_model,
            outputs=status
        )
    
    return demo


def main():
    demo = create_interface()
    demo.launch()


if __name__ == "__main__":
    main()