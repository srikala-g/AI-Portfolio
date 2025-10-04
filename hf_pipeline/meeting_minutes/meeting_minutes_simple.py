# -*- coding: utf-8 -*-
"""
Simplified Meeting Minutes Generator with Hugging Face and Gradio

This is a lightweight version that uses smaller, more reliable models:
- Whisper for speech-to-text transcription
- Smaller language models for generating meeting minutes

Features:
- Upload audio files (MP3, WAV, etc.)
- Automatic transcription using Hugging Face Whisper
- Generate structured meeting minutes
- Simple Gradio interface
"""

import gradio as gr
import torch
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Device detection
if torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # Fallback to CPU

print(f"Using device: {device}")

# Model configurations - using smaller, more reliable models
WHISPER_MODELS = {
    "Whisper Small": "openai/whisper-small",
    "Whisper Medium": "openai/whisper-medium"
}

LLM_MODELS = {
    "Phi-3 Mini": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma 2B": "google/gemma-2-2b-it",
    "DistilBERT": "distilbert-base-uncased"
}

# Global variables for model caching
whisper_pipeline = None
llm_pipeline = None

def load_whisper_model(model_name):
    """Load Whisper model for speech-to-text"""
    global whisper_pipeline
    try:
        print(f"Loading Whisper model: {model_name}")
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device=device
        )
        return f"✅ Whisper model '{model_name}' loaded successfully on {device}"
    except Exception as e:
        return f"❌ Error loading Whisper model: {str(e)}"

def load_llm_model(model_name):
    """Load LLM model for text generation"""
    global llm_pipeline
    try:
        print(f"Loading LLM model: {model_name}")
        
        if model_name == "DistilBERT":
            # Use a simple text generation pipeline for DistilBERT
            llm_pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=device
            )
        else:
            # Use causal LM for other models
            llm_pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device=device
            )
        
        return f"✅ LLM model '{model_name}' loaded successfully on {device}"
    except Exception as e:
        return f"❌ Error loading LLM model: {str(e)}"

def transcribe_audio(audio_file, whisper_model):
    """Transcribe audio file to text"""
    if audio_file is None:
        return "Please upload an audio file first."
    
    try:
        # Load Whisper model if not already loaded
        global whisper_pipeline
        if whisper_pipeline is None:
            status = load_whisper_model(whisper_model)
            if "❌" in status:
                return status
        
        # Transcribe audio
        print("Starting transcription...")
        result = whisper_pipeline(audio_file)
        transcription = result["text"]
        
        return f"**Transcription Complete!**\n\n{transcription}"
    except Exception as e:
        return f"❌ Error during transcription: {str(e)}"

def generate_meeting_minutes(transcription, llm_model_name, meeting_context=""):
    """Generate structured meeting minutes from transcription"""
    if not transcription.strip():
        return "Please provide a transcription first."
    
    try:
        # Load LLM model if not already loaded
        global llm_pipeline
        if llm_pipeline is None:
            status = load_llm_model(llm_model_name)
            if "❌" in status:
                return status
        
        # Create prompt for meeting minutes generation
        prompt = f"""Create professional meeting minutes from this transcript:

Meeting Context: {meeting_context if meeting_context else "General meeting"}

Transcript:
{transcription}

Please create structured meeting minutes including:
1. Meeting Summary
2. Key Discussion Points  
3. Decisions Made
4. Action Items
5. Next Steps

Meeting Minutes:
"""
        
        # Generate response
        print("Generating meeting minutes...")
        result = llm_pipeline(
            prompt,
            max_length=len(prompt.split()) + 500,  # Add 500 words to the prompt
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated_text = result[0]['generated_text']
        # Remove the original prompt from the response
        meeting_minutes = generated_text[len(prompt):].strip()
        
        return f"**Meeting Minutes Generated!**\n\n{meeting_minutes}"
        
    except Exception as e:
        return f"❌ Error generating meeting minutes: {str(e)}"

def process_audio_to_minutes(audio_file, whisper_model, llm_model_name, meeting_context=""):
    """Complete pipeline: audio -> transcription -> meeting minutes"""
    if audio_file is None:
        return "Please upload an audio file first."
    
    try:
        # Step 1: Transcribe audio
        transcription_result = transcribe_audio(audio_file, whisper_model)
        if "❌" in transcription_result:
            return transcription_result
        
        # Extract just the transcription text
        if "**Transcription Complete!**" in transcription_result:
            transcription_text = transcription_result.split("**Transcription Complete!**\n\n")[1]
        else:
            transcription_text = transcription_result
        
        # Step 2: Generate meeting minutes
        minutes_result = generate_meeting_minutes(transcription_text, llm_model_name, meeting_context)
        
        return f"{transcription_result}\n\n---\n\n{minutes_result}"
        
    except Exception as e:
        return f"❌ Error in complete pipeline: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="Meeting Minutes Generator",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    """
) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">🎤 Meeting Minutes Generator</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Convert audio recordings to structured meeting minutes using AI</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("🚀 Quick Generate"):
            gr.Markdown("### Upload audio and generate meeting minutes in one step")
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Upload Audio File (Supported formats: MP3, WAV, M4A, etc.)",
                        type="filepath"
                    )
                    meeting_context_input = gr.Textbox(
                        label="Meeting Context (Optional)",
                        placeholder="e.g., Weekly team standup, Board meeting, Project review",
                        lines=2
                    )
                    
                with gr.Column(scale=1):
                    whisper_model_select = gr.Dropdown(
                        choices=list(WHISPER_MODELS.keys()),
                        value="Whisper Small",
                        label="Speech-to-Text Model (Choose Whisper model for transcription)"
                    )
                    llm_model_select = gr.Dropdown(
                        choices=list(LLM_MODELS.keys()),
                        value="Phi-3 Mini",
                        label="Language Model (Choose LLM for generating minutes)"
                    )
            
            generate_btn = gr.Button("🚀 Generate Meeting Minutes", variant="primary", size="lg")
            quick_output = gr.Markdown(label="Generated Meeting Minutes")
            
            generate_btn.click(
                process_audio_to_minutes,
                inputs=[audio_input, whisper_model_select, llm_model_select, meeting_context_input],
                outputs=quick_output
            )
        
        with gr.Tab("🔧 Step by Step"):
            gr.Markdown("### Process audio step by step for more control")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Step 1: Transcribe Audio")
                    audio_input_step = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                    whisper_model_step = gr.Dropdown(
                        choices=list(WHISPER_MODELS.keys()),
                        value="Whisper Small",
                        label="Whisper Model"
                    )
                    transcribe_btn = gr.Button("🎤 Transcribe Audio", variant="secondary")
                    transcription_output = gr.Markdown(label="Transcription")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Step 2: Generate Minutes")
                    transcription_input = gr.Textbox(
                        label="Paste Transcription Here",
                        placeholder="Or use the transcription from Step 1",
                        lines=8
                    )
                    meeting_context_step = gr.Textbox(
                        label="Meeting Context",
                        placeholder="e.g., Weekly team standup",
                        lines=2
                    )
                    llm_model_step = gr.Dropdown(
                        choices=list(LLM_MODELS.keys()),
                        value="Phi-3 Mini",
                        label="Language Model"
                    )
                    generate_minutes_btn = gr.Button("📝 Generate Minutes", variant="primary")
                    minutes_output = gr.Markdown(label="Meeting Minutes")
            
            # Connect step-by-step functions
            transcribe_btn.click(
                transcribe_audio,
                inputs=[audio_input_step, whisper_model_step],
                outputs=transcription_output
            )
            
            generate_minutes_btn.click(
                generate_meeting_minutes,
                inputs=[transcription_input, llm_model_step, meeting_context_step],
                outputs=minutes_output
            )
        
        with gr.Tab("⚙️ Model Management"):
            gr.Markdown("### Load and manage AI models")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Load Whisper Model")
                    whisper_load_model = gr.Dropdown(
                        choices=list(WHISPER_MODELS.keys()),
                        value="Whisper Small",
                        label="Select Whisper Model"
                    )
                    load_whisper_btn = gr.Button("Load Whisper Model", variant="secondary")
                    whisper_status = gr.Textbox(label="Whisper Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### Load Language Model")
                    llm_load_model = gr.Dropdown(
                        choices=list(LLM_MODELS.keys()),
                        value="Phi-3 Mini",
                        label="Select Language Model"
                    )
                    load_llm_btn = gr.Button("Load Language Model", variant="secondary")
                    llm_status = gr.Textbox(label="LLM Status", interactive=False)
            
            # Connect model loading functions
            load_whisper_btn.click(
                load_whisper_model,
                inputs=[whisper_load_model],
                outputs=whisper_status
            )
            
            load_llm_btn.click(
                load_llm_model,
                inputs=[llm_load_model],
                outputs=llm_status
            )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>Powered by Hugging Face Transformers and Gradio</p>
        <p>Supports various audio formats and generates professional meeting minutes</p>
    </div>
    """)

if __name__ == "__main__":
    # Optional: Login to Hugging Face (uncomment if you have HF token)
    # login(token="your_hf_token_here")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True to create a public link
    )
