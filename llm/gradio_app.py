#!/usr/bin/env python3
"""
Gradio Web Interface for Website Summary Tool

This file provides a standalone Gradio web interface for the website summarization tool.
Run with: python gradio_app.py
"""

import os
import sys
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

def gradio_summarize(url, method):
    """
    Gradio interface function for website summarization.
    
    Args:
        url (str): URL to summarize
        method (str): Scraping method ('HTTP Scraping', 'Selenium Scraping', 'Auto Mode')
        
    Returns:
        str: Summary of the website or error message
    """
    if not url.strip():
        return "Please enter a URL to summarize."
    
    # Add https:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        summarizer = WebsiteSummarizer()
        
        if method == "HTTP Scraping":
            summary = summarizer.summarize(url)
        elif method == "Selenium Scraping":
            if not SELENIUM_AVAILABLE:
                return "❌ **Selenium not available**\n\nPlease install with: `pip install selenium`"
            summary = summarizer.summarize_with_selenium(url, headless=True)
        elif method == "Auto Mode":
            summary = summarizer.summarize_auto(url)
        else:
            return f"❌ Unknown method: {method}"
        
        return f"**URL:** {url}\n\n**Summary:**\n{summary}"
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


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
                    label="Method"
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
                
                # Example URLs and methods
                example_data = [
                    ("https://openai.com", "Selenium Scraping"),
                    ("https://cnn.com", "HTTP Scraping"),
                    ("https://anthropic.com", "Auto Mode")
                ]
                
                # Connect example buttons to inputs
                for i, (url, method) in enumerate(example_data):
                    example_buttons[i].click(
                        lambda u=url, m=method: (u, m),
                        outputs=[url_input, method_dropdown]
                    )
        
        # Bottom - Output
        with gr.Row():
            output = gr.Markdown(
                label="Summary",
                show_copy_button=True,
                elem_classes="compact-output"
            )
        
        # Connect submit button
        submit_btn.click(
            gradio_summarize,
            inputs=[url_input, method_dropdown],
            outputs=output
        )
        
        # Also allow Enter key submission
        url_input.submit(
            gradio_summarize,
            inputs=[url_input, method_dropdown],
            outputs=output
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
