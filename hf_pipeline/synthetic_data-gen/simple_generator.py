# -*- coding: utf-8 -*-
"""
Simple Synthetic Data Generator

A lightweight version of the synthetic data generator that focuses on:
- Text generation with GPT-2
- Tabular data generation
- Simple Gradio interface
- Easy to use and deploy
"""

import gradio as gr
import torch
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from transformers import pipeline
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Device detection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Global variables
text_pipeline = None

def load_text_model():
    """Load GPT-2 model for text generation"""
    global text_pipeline
    try:
        print("Loading GPT-2 model...")
        text_pipeline = pipeline(
            "text-generation",
            model="gpt2",
            device=device
        )
        return "✅ GPT-2 model loaded successfully"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def generate_text(prompt, num_samples, max_length, temperature):
    """Generate synthetic text data"""
    if not prompt.strip():
        return "Please provide a prompt."
    
    try:
        # Load model if not already loaded
        if text_pipeline is None:
            status = load_text_model()
            if "❌" in status:
                return status
        
        generated_texts = []
        
        for i in range(num_samples):
            result = text_pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=text_pipeline.tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            generated_texts.append({
                "id": i + 1,
                "prompt": prompt,
                "generated_text": generated_text,
                "timestamp": datetime.now().isoformat()
            })
        
        return generated_texts
        
    except Exception as e:
        return f"❌ Error generating text: {str(e)}"

def generate_customer_data(num_rows):
    """Generate synthetic customer data"""
    try:
        data = []
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
        purchase_freq = ["Low", "Medium", "High"]
        
        for i in range(num_rows):
            data.append({
                "customer_id": f"CUST_{i+1:06d}",
                "name": f"Customer {i+1}",
                "email": f"customer{i+1}@example.com",
                "age": random.randint(18, 80),
                "city": random.choice(cities),
                "income": random.randint(30000, 150000),
                "purchase_frequency": random.choice(purchase_freq),
                "satisfaction_score": round(random.uniform(1.0, 5.0), 1),
                "registration_date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
            })
        
        return data
        
    except Exception as e:
        return f"❌ Error generating customer data: {str(e)}"

def generate_sales_data(num_rows):
    """Generate synthetic sales data"""
    try:
        data = []
        products = ["Laptop", "Phone", "Tablet", "Headphones", "Camera", "Watch", "Speaker", "Mouse", "Keyboard", "Monitor"]
        regions = ["North", "South", "East", "West", "Central"]
        
        for i in range(num_rows):
            data.append({
                "sale_id": f"SALE_{i+1:06d}",
                "product": random.choice(products),
                "quantity": random.randint(1, 10),
                "price": round(random.uniform(50.0, 2000.0), 2),
                "date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
                "salesperson": f"Salesperson {random.randint(1, 20)}",
                "region": random.choice(regions),
                "commission": round(random.uniform(0.05, 0.15), 3),
                "customer_id": f"CUST_{random.randint(1, 1000):06d}"
            })
        
        return data
        
    except Exception as e:
        return f"❌ Error generating sales data: {str(e)}"

def generate_sensor_data(num_rows):
    """Generate synthetic sensor data"""
    try:
        data = []
        locations = ["Building A", "Building B", "Building C", "Warehouse", "Office"]
        statuses = ["Normal", "Warning", "Critical"]
        
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
                "location": random.choice(locations),
                "status": random.choice(statuses),
                "battery_level": round(random.uniform(20, 100), 1)
            })
        
        return data
        
    except Exception as e:
        return f"❌ Error generating sensor data: {str(e)}"

def export_data(data, format_type, filename):
    """Export data to various formats"""
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
        
        else:
            return "❌ Unsupported format"
        
    except Exception as e:
        return f"❌ Error exporting: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="Simple Synthetic Data Generator",
    theme=gr.themes.Soft()
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">🎲 Simple Synthetic Data Generator</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Generate realistic synthetic data quickly and easily</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📝 Text Generation"):
            gr.Markdown("### Generate synthetic text using GPT-2")
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Write a story about a robot who...",
                        lines=3
                    )
                    
                    with gr.Row():
                        num_text_samples = gr.Slider(1, 5, value=2, label="Number of Samples")
                        max_length = gr.Slider(50, 300, value=100, label="Max Length")
                        temperature = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                    
                    text_generate_btn = gr.Button("🚀 Generate Text", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    text_output = gr.Markdown(label="Generated Text")
                    text_export_format = gr.Dropdown(["JSON", "TXT"], value="JSON", label="Export Format")
                    text_export_btn = gr.Button("💾 Export", variant="secondary")
                    text_export_status = gr.Textbox(label="Export Status", interactive=False)
            
            text_generate_btn.click(
                generate_text,
                inputs=[text_prompt, num_text_samples, max_length, temperature],
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
                        choices=["Customer Data", "Sales Data", "Sensor Data"],
                        value="Customer Data",
                        label="Data Type"
                    )
                    num_rows = gr.Slider(10, 500, value=100, label="Number of Rows")
                    
                    tabular_generate_btn = gr.Button("🚀 Generate Data", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    tabular_output = gr.Dataframe(label="Generated Data", height=400)
                    tabular_export_format = gr.Dropdown(["CSV", "JSON"], value="CSV", label="Export Format")
                    tabular_export_btn = gr.Button("💾 Export", variant="secondary")
                    tabular_export_status = gr.Textbox(label="Export Status", interactive=False)
            
            def generate_tabular_data_wrapper(data_type, num_rows):
                if data_type == "Customer Data":
                    return generate_customer_data(num_rows)
                elif data_type == "Sales Data":
                    return generate_sales_data(num_rows)
                elif data_type == "Sensor Data":
                    return generate_sensor_data(num_rows)
                else:
                    return []
            
            tabular_generate_btn.click(
                generate_tabular_data_wrapper,
                inputs=[data_type, num_rows],
                outputs=tabular_output
            )
            
            tabular_export_btn.click(
                export_data,
                inputs=[tabular_output, tabular_export_format, gr.Textbox(value="tabular_data", visible=False)],
                outputs=tabular_export_status
            )
        
        with gr.Tab("⚙️ Model Management"):
            gr.Markdown("### Load and manage AI models")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Load Text Model")
                    load_text_btn = gr.Button("Load GPT-2 Model", variant="secondary")
                    text_status = gr.Textbox(label="Model Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### System Info")
                    gr.Markdown(f"**Device:** {device}")
                    gr.Markdown("**Model:** GPT-2 Small")
                    gr.Markdown("**Status:** Ready to load")
            
            load_text_btn.click(
                load_text_model,
                outputs=text_status
            )
    
    gr.HTML("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>Powered by Hugging Face Transformers and Gradio</p>
        <p>Simple, fast, and reliable synthetic data generation</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


