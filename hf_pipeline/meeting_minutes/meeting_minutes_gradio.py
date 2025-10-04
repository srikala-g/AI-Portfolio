# -*- coding: utf-8 -*-
"""
Meeting Minutes Generator with Hugging Face and Gradio Interface

This application converts audio files to meeting minutes using Hugging Face models:
- Whisper for speech-to-text transcription
- Llama for generating structured meeting minutes

Features:
- Upload audio files (MP3, WAV, etc.)
- Automatic transcription using Hugging Face Whisper
- Generate structured meeting minutes with action items
- Professional Gradio interface
"""

import gradio as gr
import torch
import tempfile
import os
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    BitsAndBytesConfig
)
from huggingface_hub import login
import soundfile as sf

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

# Model configurations
WHISPER_MODELS = {
    "Whisper Small": "openai/whisper-small",
    "Whisper Medium": "openai/whisper-medium", 
    "Whisper Large": "openai/whisper-large-v2",
    "Whisper Large v3": "openai/whisper-large-v3"
}

LLM_MODELS = {
    "Llama 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Llama 3.1 70B": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Phi-3 Mini": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma 2B": "google/gemma-2-2b-it",
    "Qwen2 7B": "Qwen/Qwen2-7B-Instruct"
}

# Global variables for model caching
whisper_pipeline = None
llm_model = None
llm_tokenizer = None

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
    global llm_model, llm_tokenizer
    try:
        print(f"Loading LLM model: {model_name}")
        
        # Configure quantization for memory efficiency
        if device.type == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quant_config = None
        
        # Load tokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
        # Load model
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device.type == "cuda" else None,
            quantization_config=quant_config,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
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
        global llm_model, llm_tokenizer
        if llm_model is None or llm_tokenizer is None:
            status = load_llm_model(llm_model_name)
            if "❌" in status:
                return status
        
        # Create system message for meeting minutes generation
        system_message = """You are an expert assistant that creates professional meeting minutes from transcripts. 
        Generate structured minutes including:
        1. Meeting Summary (attendees, date, location if available)
        2. Key Discussion Points
        3. Decisions Made
        4. Action Items (with owners and deadlines)
        5. Next Steps
        
        Format the output in clear markdown with proper headings."""
        
        user_prompt = f"""Below is a meeting transcript. Please create professional meeting minutes in markdown format.

Meeting Context: {meeting_context if meeting_context else "General meeting"}

Transcript:
{transcription}"""
        
        # Prepare messages for chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        inputs = llm_tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to(device)
        
        # Generate response
        print("Generating meeting minutes...")
        with torch.no_grad():
            outputs = llm_model.generate(
                inputs,
                max_new_tokens=2000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id
            )
        
        # Decode response
        response = llm_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return f"**Meeting Minutes Generated!**\n\n{response}"
        
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
        transcription_text = transcription_result.split("**Transcription Complete!**\n\n")[1]
        
        # Step 2: Generate meeting minutes
        minutes_result = generate_meeting_minutes(transcription_text, llm_model_name, meeting_context)
        
        return f"{transcription_result}\n\n---\n\n{minutes_result}"
        
    except Exception as e:
        return f"❌ Error in complete pipeline: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="Meeting Minutes Generator",
    theme=gr.themes.Glass(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    ),
    css="""
    .gradio-container {
        max-width: 1200px !important;
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
        with gr.Tab("🎯 Quick Generate", id="quick-generate"):
            gr.Markdown("### Upload audio and generate meeting minutes in one step")
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        info="Supported formats: MP3, WAV, M4A, etc."
                    )
                    meeting_context_input = gr.Textbox(
                        label="Meeting Context (Optional)",
                        placeholder="e.g., Weekly team standup, Board meeting, Project review",
                        lines=2
                    )
                    
                with gr.Column(scale=1):
                    whisper_model_select = gr.Dropdown(
                        choices=list(WHISPER_MODELS.keys()),
                        value="Whisper Medium",
                        label="Speech-to-Text Model",
                        info="Choose Whisper model for transcription"
                    )
                    llm_model_select = gr.Dropdown(
                        choices=list(LLM_MODELS.keys()),
                        value="Llama 3.1 8B",
                        label="Language Model",
                        info="Choose LLM for generating minutes"
                    )
            
            generate_btn = gr.Button("🚀 Generate Meeting Minutes", variant="primary", size="lg")
            quick_output = gr.Markdown(label="Generated Meeting Minutes", lines=20)
            
            generate_btn.click(
                process_audio_to_minutes,
                inputs=[audio_input, whisper_model_select, llm_model_select, meeting_context_input],
                outputs=quick_output
            )
        
        with gr.Tab("🔧 Step by Step", id="step-by-step"):
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
                        value="Whisper Medium",
                        label="Whisper Model"
                    )
                    transcribe_btn = gr.Button("🎤 Transcribe Audio", variant="secondary")
                    transcription_output = gr.Markdown(label="Transcription", lines=10)
                
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
                        value="Llama 3.1 8B",
                        label="Language Model"
                    )
                    generate_minutes_btn = gr.Button("📝 Generate Minutes", variant="primary")
                    minutes_output = gr.Markdown(label="Meeting Minutes", lines=15)
            
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
        
        with gr.Tab("⚙️ Model Management", id="model-management"):
            gr.Markdown("### Load and manage AI models")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Load Whisper Model")
                    whisper_load_model = gr.Dropdown(
                        choices=list(WHISPER_MODELS.keys()),
                        value="Whisper Medium",
                        label="Select Whisper Model"
                    )
                    load_whisper_btn = gr.Button("Load Whisper Model", variant="secondary")
                    whisper_status = gr.Textbox(label="Whisper Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### Load Language Model")
                    llm_load_model = gr.Dropdown(
                        choices=list(LLM_MODELS.keys()),
                        value="Llama 3.1 8B",
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
