# -*- coding: utf-8 -*-
"""
Synthetic Data Generator with Multiple Model Support

This application generates synthetic data using various AI models:
- Text generation (stories, articles, conversations)
- Tabular data (CSV datasets with realistic patterns)
- Image generation (photos, artwork, diagrams)
- Time series data (financial, sensor data)

Features:
- Multiple model options for each data type
- Customizable parameters and prompts
- Export to various formats (CSV, JSON, images)
- Batch generation capabilities
- Professional Gradio interface
"""

import gradio as gr
import torch
import pandas as pd
import numpy as np
import json
import os
import tempfile
import warnings
from datetime import datetime, timedelta
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from diffusers import DiffusionPipeline
import random
from typing import List, Dict, Any, Optional

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
TEXT_MODELS = {
    "GPT-2 Small": "gpt2",
    "GPT-2 Medium": "gpt2-medium",
    "GPT-2 Large": "gpt2-large",
    "Phi-3 Mini": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma 2B": "google/gemma-2-2b-it",
    "Llama 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "DistilGPT-2": "distilgpt2"
}

IMAGE_MODELS = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Kandinsky 2.2": "kandinsky-community/kandinsky-2-2-decoder"
}

# Global variables for model caching
text_pipeline = None
image_pipeline = None

def load_text_model(model_name):
    """Load text generation model"""
    global text_pipeline
    try:
        print(f"Loading text model: {model_name}")
        
        if "instruct" in model_name.lower() or "phi" in model_name.lower():
            # Use causal LM for instruction-tuned models
            text_pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device=device
            )
        else:
            # Use standard text generation pipeline
            text_pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=device
            )
        
        return f"✅ Text model '{model_name}' loaded successfully on {device}"
    except Exception as e:
        return f"❌ Error loading text model: {str(e)}"

def load_image_model(model_name):
    """Load image generation model"""
    global image_pipeline
    try:
        print(f"Loading image model: {model_name}")
        
        # Set appropriate dtype based on device
        if device.type == "cuda":
            torch_dtype = torch.float16
            variant = "fp16"
        else:
            torch_dtype = torch.float32
            variant = None
        
        image_pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant
        ).to(device)
        
        return f"✅ Image model '{model_name}' loaded successfully on {device}"
    except Exception as e:
        return f"❌ Error loading image model: {str(e)}"

def generate_text_data(prompt, model_name, num_samples, max_length, temperature):
    """Generate synthetic text data"""
    if not prompt.strip():
        return "Please provide a prompt for text generation."
    
    try:
        # Load model if not already loaded
        global text_pipeline
        if text_pipeline is None:
            status = load_text_model(model_name)
            if "❌" in status:
                return status
        
        generated_texts = []
        
        for i in range(num_samples):
            # Generate text
            result = text_pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=text_pipeline.tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            generated_texts.append({
                "id": i + 1,
                "prompt": prompt,
                "generated_text": generated_text,
                "model": model_name,
                "timestamp": datetime.now().isoformat()
            })
        
        return generated_texts
        
    except Exception as e:
        return f"❌ Error generating text data: {str(e)}"

def generate_tabular_data(data_type, num_rows, columns_config):
    """Generate synthetic tabular data"""
    try:
        data = []
        
        if data_type == "Customer Data":
            for i in range(num_rows):
                data.append({
                    "customer_id": f"CUST_{i+1:06d}",
                    "name": f"Customer {i+1}",
                    "email": f"customer{i+1}@example.com",
                    "age": random.randint(18, 80),
                    "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
                    "income": random.randint(30000, 150000),
                    "purchase_frequency": random.choice(["Low", "Medium", "High"]),
                    "satisfaction_score": round(random.uniform(1.0, 5.0), 1)
                })
        
        elif data_type == "Sales Data":
            for i in range(num_rows):
                data.append({
                    "sale_id": f"SALE_{i+1:06d}",
                    "product": random.choice(["Laptop", "Phone", "Tablet", "Headphones", "Camera"]),
                    "quantity": random.randint(1, 10),
                    "price": round(random.uniform(50.0, 2000.0), 2),
                    "date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
                    "salesperson": f"Salesperson {random.randint(1, 20)}",
                    "region": random.choice(["North", "South", "East", "West"]),
                    "commission": round(random.uniform(0.05, 0.15), 3)
                })
        
        elif data_type == "Sensor Data":
            base_temp = 20.0
            for i in range(num_rows):
                timestamp = datetime.now() - timedelta(minutes=i*5)
                # Simulate temperature variation
                temp = base_temp + 5 * np.sin(i * 0.1) + random.uniform(-2, 2)
                data.append({
                    "timestamp": timestamp.isoformat(),
                    "sensor_id": f"SENSOR_{random.randint(1, 10):02d}",
                    "temperature": round(temp, 2),
                    "humidity": round(random.uniform(30, 80), 1),
                    "pressure": round(random.uniform(1000, 1020), 1),
                    "location": random.choice(["Building A", "Building B", "Building C"]),
                    "status": random.choice(["Normal", "Warning", "Critical"])
                })
        
        elif data_type == "Financial Data":
            for i in range(num_rows):
                data.append({
                    "transaction_id": f"TXN_{i+1:08d}",
                    "account_id": f"ACC_{random.randint(1000, 9999)}",
                    "amount": round(random.uniform(-5000, 5000), 2),
                    "transaction_type": random.choice(["Deposit", "Withdrawal", "Transfer", "Payment"]),
                    "date": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
                    "merchant": random.choice(["Amazon", "Walmart", "Gas Station", "Restaurant", "Online Store"]),
                    "category": random.choice(["Food", "Transportation", "Shopping", "Entertainment", "Utilities"]),
                    "balance": round(random.uniform(0, 100000), 2)
                })
        
        return data
        
    except Exception as e:
        return f"❌ Error generating tabular data: {str(e)}"

def generate_image_data(prompt, model_name, num_images, width, height, steps):
    """Generate synthetic image data"""
    if not prompt.strip():
        return "Please provide a prompt for image generation."
    
    try:
        # Load model if not already loaded
        global image_pipeline
        if image_pipeline is None:
            status = load_image_model(model_name)
            if "❌" in status:
                return status
        
        generated_images = []
        
        for i in range(num_images):
            # Generate image
            image = image_pipeline(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=7.5,
                width=width,
                height=height
            ).images[0]
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name)
                generated_images.append({
                    "id": i + 1,
                    "prompt": prompt,
                    "image_path": tmp_file.name,
                    "model": model_name,
                    "timestamp": datetime.now().isoformat()
                })
        
        return generated_images
        
    except Exception as e:
        return f"❌ Error generating image data: {str(e)}"

def export_data(data, format_type, filename):
    """Export generated data to various formats"""
    try:
        if format_type == "JSON":
            with open(f"{filename}.json", "w") as f:
                json.dump(data, f, indent=2)
            return f"✅ Data exported to {filename}.json"
        
        elif format_type == "CSV":
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df.to_csv(f"{filename}.csv", index=False)
                return f"✅ Data exported to {filename}.csv"
            else:
                return "❌ No data to export"
        
        elif format_type == "TXT":
            with open(f"{filename}.txt", "w") as f:
                for item in data:
                    if isinstance(item, dict):
                        f.write(f"ID: {item.get('id', 'N/A')}\n")
                        f.write(f"Text: {item.get('generated_text', 'N/A')}\n")
                        f.write("-" * 50 + "\n")
                    else:
                        f.write(str(item) + "\n")
            return f"✅ Data exported to {filename}.txt"
        
        else:
            return "❌ Unsupported export format"
        
    except Exception as e:
        return f"❌ Error exporting data: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="Synthetic Data Generator",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1400px !important;
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
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">🎲 Synthetic Data Generator</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Generate realistic synthetic data using various AI models</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📝 Text Generation"):
            gr.Markdown("### Generate synthetic text data using language models")
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Write a story about...",
                        lines=3
                    )
                    text_model = gr.Dropdown(
                        choices=list(TEXT_MODELS.keys()),
                        value="GPT-2 Small",
                        label="Language Model"
                    )
                    
                    with gr.Row():
                        num_text_samples = gr.Slider(1, 10, value=3, label="Number of Samples")
                        max_length = gr.Slider(50, 500, value=150, label="Max Length")
                        temperature = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                    
                    text_generate_btn = gr.Button("🚀 Generate Text", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    text_output = gr.Markdown(label="Generated Text")
                    text_export_format = gr.Dropdown(["JSON", "TXT"], value="JSON", label="Export Format")
                    text_export_btn = gr.Button("💾 Export Data", variant="secondary")
                    text_export_status = gr.Textbox(label="Export Status", interactive=False)
            
            text_generate_btn.click(
                generate_text_data,
                inputs=[text_prompt, text_model, num_text_samples, max_length, temperature],
                outputs=text_output
            )
            
            text_export_btn.click(
                export_data,
                inputs=[text_output, text_export_format, gr.Textbox(value="text_data", visible=False)],
                outputs=text_export_status
            )
        
        with gr.Tab("📊 Tabular Data"):
            gr.Markdown("### Generate structured tabular data")
            
            with gr.Row():
                with gr.Column(scale=1):
                    data_type = gr.Dropdown(
                        choices=["Customer Data", "Sales Data", "Sensor Data", "Financial Data"],
                        value="Customer Data",
                        label="Data Type"
                    )
                    num_rows = gr.Slider(10, 1000, value=100, label="Number of Rows")
                    
                    tabular_generate_btn = gr.Button("🚀 Generate Data", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    tabular_output = gr.Dataframe(label="Generated Data", height=400)
                    tabular_export_format = gr.Dropdown(["CSV", "JSON"], value="CSV", label="Export Format")
                    tabular_export_btn = gr.Button("💾 Export Data", variant="secondary")
                    tabular_export_status = gr.Textbox(label="Export Status", interactive=False)
            
            tabular_generate_btn.click(
                generate_tabular_data,
                inputs=[data_type, num_rows, gr.Textbox(visible=False)],
                outputs=tabular_output
            )
            
            tabular_export_btn.click(
                export_data,
                inputs=[tabular_output, tabular_export_format, gr.Textbox(value="tabular_data", visible=False)],
                outputs=tabular_export_status
            )
        
        with gr.Tab("🖼️ Image Generation"):
            gr.Markdown("### Generate synthetic images using diffusion models")
            
            with gr.Row():
                with gr.Column(scale=2):
                    image_prompt = gr.Textbox(
                        label="Image Prompt",
                        placeholder="A beautiful landscape with mountains and lakes",
                        lines=2
                    )
                    image_model = gr.Dropdown(
                        choices=list(IMAGE_MODELS.keys()),
                        value="Stable Diffusion v1.5",
                        label="Image Model"
                    )
                    
                    with gr.Row():
                        num_images = gr.Slider(1, 5, value=2, label="Number of Images")
                        width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                        height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                        steps = gr.Slider(10, 50, value=20, label="Inference Steps")
                    
                    image_generate_btn = gr.Button("🚀 Generate Images", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    image_output = gr.Gallery(label="Generated Images", height=400)
                    image_export_btn = gr.Button("💾 Download Images", variant="secondary")
                    image_export_status = gr.Textbox(label="Export Status", interactive=False)
            
            image_generate_btn.click(
                generate_image_data,
                inputs=[image_prompt, image_model, num_images, width, height, steps],
                outputs=image_output
            )
        
        with gr.Tab("⚙️ Model Management"):
            gr.Markdown("### Load and manage AI models")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Load Text Model")
                    text_load_model = gr.Dropdown(
                        choices=list(TEXT_MODELS.keys()),
                        value="GPT-2 Small",
                        label="Select Text Model"
                    )
                    load_text_btn = gr.Button("Load Text Model", variant="secondary")
                    text_status = gr.Textbox(label="Text Model Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### Load Image Model")
                    image_load_model = gr.Dropdown(
                        choices=list(IMAGE_MODELS.keys()),
                        value="Stable Diffusion v1.5",
                        label="Select Image Model"
                    )
                    load_image_btn = gr.Button("Load Image Model", variant="secondary")
                    image_status = gr.Textbox(label="Image Model Status", interactive=False)
            
            # Connect model loading functions
            load_text_btn.click(
                load_text_model,
                inputs=[text_load_model],
                outputs=text_status
            )
            
            load_image_btn.click(
                load_image_model,
                inputs=[image_load_model],
                outputs=image_status
            )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>Powered by Hugging Face Transformers and Gradio</p>
        <p>Generate realistic synthetic data for testing, training, and research</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True to create a public link
    )

