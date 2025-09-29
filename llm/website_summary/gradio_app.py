#!/usr/bin/env python3
"""
Gradio Web Interface for Website Summary Tool

This file provides a standalone Gradio web interface for the website summarization tool.
Run with: python gradio_app.py
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from website_summary module
from website_summary import (
    WebsiteSummarizer, 
    SELENIUM_AVAILABLE, 
    GRADIO_AVAILABLE
)

# Import gradio directly to avoid any import issues
import gradio as gr

# Load environment variables
load_dotenv()

def gradio_summarize(url, method, model_provider, progress=gr.Progress()):
    """
    Gradio interface function for website summarization with progress tracking.
    
    Args:
        url (str): URL to summarize
        method (str): Scraping method ('HTTP Scraping', 'Selenium Scraping', 'Auto Mode')
        model_provider (str): Model provider ('OpenAI', 'Ollama 3.2')
        progress: Gradio progress tracker
        
    Returns:
        tuple: (summary, status_message)
    """
    if not url.strip():
        return "Please enter a URL to summarize.", "❌ No URL provided"
    
    # Add https:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    start_time = time.time()
    
    try:
        # Convert model provider selection
        if model_provider == "OpenAI":
            provider = "openai"
        elif model_provider == "Ollama 3.2":
            provider = "ollama"
        else:
            return f"❌ Unknown model provider: {model_provider}", "❌ Invalid model selection"
        
        # Show initial progress
        progress(0.1, desc="Initializing...")
        
        summarizer = WebsiteSummarizer(model_provider=provider)
        
        # Show scraping progress
        progress(0.3, desc="Scraping website...")
        
        if method == "HTTP Scraping":
            summary = summarizer.summarize(url)
        elif method == "Selenium Scraping":
            if not SELENIUM_AVAILABLE:
                return "❌ **Selenium not available**\n\nPlease install with: `pip install selenium`", "❌ Selenium not installed"
            progress(0.4, desc="Running Selenium browser...")
            summary = summarizer.summarize_with_selenium(url, headless=True)
        elif method == "Auto Mode":
            summary = summarizer.summarize_auto(url)
        else:
            return f"❌ Unknown method: {method}", "❌ Invalid method"
        
        # Show AI processing progress
        progress(0.7, desc="AI processing...")
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Format response time
        if response_time < 1:
            time_str = f"{response_time:.2f} seconds"
        else:
            time_str = f"{response_time:.1f} seconds"
        
        # Complete progress
        progress(1.0, desc="Complete!")
        
        # Format final output
        output = f"**URL:** {url}\n\n**Model:** {model_provider}\n\n**Response Time:** {time_str}\n\n**Summary:**\n{summary}"
        status = f"✅ Completed in {time_str}"
        
        return output, status
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        error_msg = f"❌ **Error after {response_time:.1f}s:** {str(e)}"
        return error_msg, f"❌ Failed after {response_time:.1f}s"


def create_interface():
    """Create the Gradio interface with custom layout."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not available. Install with: pip install gradio")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        with gr.Blocks(title="Website Summary Tool", theme="soft") as interface:
            gr.Markdown("# Website Summary Tool")
            gr.Markdown("**Error:** OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        return interface
    
    # Create the custom layout interface
    with gr.Blocks(title="Website Summary Tool", theme="soft") as interface:
        # Add custom CSS for compact output
        interface.css = """
        .compact-output {
            font-size: 14px !important;
            line-height: 1.4 !important;
        }
        .compact-output h1, .compact-output h2, .compact-output h3 {
            font-size: 16px !important;
            margin: 8px 0 !important;
        }
        """
        
        gr.Markdown("# Website Summary Tool")
        gr.Markdown("AI-powered website summarization. Enter a URL and get an intelligent summary.")
        
        with gr.Row():
            # Left side - Input controls
            with gr.Column(scale=2):
                url_input = gr.Textbox(
                    label="Website URL",
                    placeholder="Enter a URL to summarize",
                    lines=1
                )
                method_dropdown = gr.Dropdown(
                    choices=["Auto Mode", "HTTP Scraping", "Selenium Scraping"],
                    value="Auto Mode",
                    label="Scraping Method"
                )
                model_dropdown = gr.Dropdown(
                    choices=["OpenAI", "Ollama 3.2"],
                    value="OpenAI",
                    label="AI Model"
                )
                submit_btn = gr.Button("Summarize", variant="primary")
            
            # Right side - Examples
            with gr.Column(scale=1):
                gr.Markdown("### Examples")
                example_buttons = [
                    gr.Button("OpenAI.com", size="sm"),
                    gr.Button("CNN.com", size="sm"),
                    gr.Button("Anthropic.com", size="sm")
                ]
                
                # Example URLs, methods, and models
                example_data = [
                    ("https://openai.com", "Selenium Scraping", "OpenAI"),
                    ("https://cnn.com", "HTTP Scraping", "Ollama 3.2"),
                    ("https://anthropic.com", "Auto Mode", "OpenAI")
                ]
                
                # Connect example buttons to inputs
                for i, (url, method, model) in enumerate(example_data):
                    example_buttons[i].click(
                        lambda u=url, m=method, mod=model: (u, m, mod),
                        outputs=[url_input, method_dropdown, model_dropdown]
                    )
        
        # Bottom - Output
        with gr.Row():
            with gr.Column(scale=3):
                output = gr.Markdown(
                    label="Summary",
                    show_copy_button=True,
                    elem_classes="compact-output"
                )
            with gr.Column(scale=1):
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready to summarize..."
                )
        
        # Connect submit button
        submit_btn.click(
            gradio_summarize,
            inputs=[url_input, method_dropdown, model_dropdown],
            outputs=[output, status]
        )
        
        # Also allow Enter key submission
        url_input.submit(
            gradio_summarize,
            inputs=[url_input, method_dropdown, model_dropdown],
            outputs=[output, status]
        )
    
    return interface


def main():
    """Launch the Gradio web interface."""
    if not GRADIO_AVAILABLE:
        print("Gradio not available. Install with: pip install gradio")
        return
    
    try:
        interface = create_interface()
        print("Launching Website Summary Tool...")
        print("Open your browser to the local URL shown below")
        
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        print(f"Error launching interface: {e}")


if __name__ == "__main__":
    main()
