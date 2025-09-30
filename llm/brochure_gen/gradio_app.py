"""
Gradio Web Interface for Brochure Generator

A user-friendly web interface for generating company brochures with multi-shot prompting.
"""

import gradio as gr
import time
import os
import signal
import subprocess
import threading
from brochure_generator import BrochureGenerator

# Global variable to track generation state
generation_active = False


def cleanup_ports():
    """Clean up any existing Gradio processes on common ports."""
    try:
        # Find and kill processes on common Gradio ports
        ports = [7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868]
        for port in ports:
            try:
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"🧹 Cleaned up process {pid} on port {port}")
            except:
                pass
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")


def generate_brochure(company_name, url, tone, company_type, progress=gr.Progress()):
    """
    Generate a brochure using the BrochureGenerator with progress tracking.
    
    Args:
        company_name (str): Name of the company
        url (str): Company website URL
        tone (str): Tone of the brochure ('Professional' or 'Humorous')
        company_type (str): Type of company ('tech', 'gallery', 'service')
        progress: Gradio progress tracker
        
    Returns:
        tuple: (brochure_content, status_message)
    """
    global generation_active
    
    if not company_name.strip() or not url.strip():
        return "Please enter both company name and URL.", "❌ Missing information"
    
    # Add https:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    start_time = time.time()
    generation_active = True
    
    try:
        # Show initial progress
        progress(0.1, desc="Setup...")
        status = "🔄 Initializing..."
        
        # Check if stopped
        if not generation_active:
            return "❌ Generation stopped by user.", "⏹️ Stopped"
        
        # Create generator
        generator = BrochureGenerator()
        
        # Set tone
        if tone == "Humorous":
            generator.set_humorous_tone()
        else:
            generator.set_professional_tone()
        
        # Show scraping progress
        progress(0.3, desc="Website Analysis...")
        status = "🌐 Analyzing website..."
        
        # Check if stopped
        if not generation_active:
            return "❌ Generation stopped by user.", "⏹️ Stopped"
        
        # Show link analysis progress
        progress(0.5, desc="Link Analysis...")
        status = "🔗 Finding pages..."
        
        # Check if stopped
        if not generation_active:
            return "❌ Generation stopped by user.", "⏹️ Stopped"
        
        # Show AI generation progress
        progress(0.7, desc="AI Generation...")
        status = "🤖 AI Processing..."
        
        # Generate brochure with multi-shot prompting
        if company_type != "auto":
            brochure = generator.demonstrate_multi_shot_prompting(company_name, url, company_type)
        else:
            brochure = generator.create_brochure(company_name, url)
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Format response time
        if response_time < 1:
            time_str = f"{response_time:.2f} seconds"
        else:
            time_str = f"{response_time:.1f} seconds"
        
        # Complete progress
        progress(1.0, desc="Finished!")
        status = f"✅ Complete! ({time_str})"
        generation_active = False
        
        # Format final output
        output = f"# {company_name} Brochure\n\n**Generated in:** {time_str}\n**Tone:** {tone}\n**Company Type:** {company_type}\n\n---\n\n{brochure}"
        
        return output, status
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        error_msg = f"❌ **Error after {response_time:.1f}s:** {str(e)}"
        status = f"❌ Error ({response_time:.1f}s)"
        generation_active = False
        return error_msg, status


def stop_generation():
    """Stop the current generation process."""
    global generation_active
    generation_active = False
    return "⏹️ Stopping generation..."


def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .compact-output {
        font-size: 14px !important;
        line-height: 1.4 !important;
    }
    .compact-output h1, .compact-output h2, .compact-output h3 {
        font-size: 16px !important;
        margin: 8px 0 !important;
    }
    .example-button {
        margin: 2px !important;
    }
    .status-bar {
        background-color: #f0f0f0 !important;
        border: 1px solid #ddd !important;
        border-radius: 4px !important;
        padding: 6px !important;
        font-size: 11px !important;
        color: #333 !important;
        min-height: 20px !important;
        max-height: 30px !important;
    }
    .status-bar:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25) !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Brochure Generator") as interface:
        gr.Markdown("# 🚀 AI-Powered Brochure Generator")
        gr.Markdown("Generate professional company brochures with multi-shot prompting and enhanced AI performance.")
        
        with gr.Row():
            # Left side - Input controls
            with gr.Column(scale=2):
                gr.Markdown("## 📋 Company Information")
                
                company_input = gr.Textbox(
                    label="Company Name",
                    placeholder="Enter company name (e.g., OpenAI, Art Gallery, Tech Startup)",
                    lines=1
                )
                
                url_input = gr.Textbox(
                    label="Website URL",
                    placeholder="https://company-website.com",
                    lines=1
                )
                
                company_type_dropdown = gr.Dropdown(
                    choices=["auto", "tech", "gallery", "service"],
                    value="auto",
                    label="Company Type",
                    info="Choose company type for optimized examples (auto = AI decides)"
                )
                
                tone_dropdown = gr.Dropdown(
                    choices=["Professional", "Humorous"],
                    value="Professional",
                    label="Tone"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Generate Brochure", variant="primary", size="lg", scale=2)
                    status_display = gr.Textbox(
                        label="Status",
                        value="Ready...",
                        interactive=False,
                        scale=2,
                        show_label=True,
                        elem_classes="status-bar"
                    )
            
            # Right side - Examples and Info
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Examples")
                
                # Example buttons
                example_buttons = [
                    gr.Button("Tech Company", size="sm", elem_classes="example-button"),
                    gr.Button("Art Gallery", size="sm", elem_classes="example-button"),
                    gr.Button("Service Business", size="sm", elem_classes="example-button")
                ]
                
                # Example data
                example_data = [
                    ("OpenAI", "https://openai.com", "tech", "Professional"),
                    ("Art Gallery", "https://www.undertheyellowtree.com", "gallery", "Professional"),
                    ("Consulting Firm", "https://mckinsey.com", "service", "Professional")
                ]
                
                # Connect example buttons
                for i, (company, url, company_type, tone) in enumerate(example_data):
                    example_buttons[i].click(
                        lambda c=company, u=url, ct=company_type, t=tone: (c, u, ct, t),
                        outputs=[company_input, url_input, company_type_dropdown, tone_dropdown]
                    )
                
                gr.Markdown("### 🎯 Multi-Shot Prompting")
                gr.Markdown("""
                **Enhanced AI Performance:**
                - Better link analysis
                - Improved brochure structure
                - Company-specific examples
                - Consistent formatting
                """)
        
        # Output section
        with gr.Row():
            output = gr.Markdown(
                label="Generated Brochure",
                show_copy_button=True,
                elem_classes="compact-output"
            )
        
        # Connect submit button
        submit_btn.click(
            generate_brochure,
            inputs=[company_input, url_input, tone_dropdown, company_type_dropdown],
            outputs=[output, status_display]
        )
        
        # Also allow Enter key submission
        url_input.submit(
            generate_brochure,
            inputs=[company_input, url_input, tone_dropdown, company_type_dropdown],
            outputs=[output, status_display]
        )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        ### 🚀 Features
        - **Multi-Shot Prompting**: Enhanced AI performance with multiple examples
        - **Smart Link Analysis**: AI-powered identification of relevant pages
        - **Company Type Optimization**: Specialized examples for different industries
        - **Professional & Humorous Tones**: Choose the right style for your audience
        - **Real-time Progress**: Track generation progress with live updates
        """)
    
    return interface


def main():
    """Launch the Gradio web interface."""
    try:
        # Clean up any existing processes
        cleanup_ports()
        
        interface = create_interface()
        print("🚀 Starting Brochure Generator with Multi-Shot Prompting...")
        print("📋 Features:")
        print("  • Multi-shot prompting for enhanced AI performance")
        print("  • Smart link analysis with multiple examples")
        print("  • Company type optimization (tech, gallery, service)")
        print("  • Professional and humorous tones")
        print("  • Real-time progress tracking")
        print("  • No hardcoded URLs - fully dynamic")
        print("\n🌐 Opening web interface...")
        
        # Try different ports to avoid conflicts
        ports_to_try = [7860, 7864, 7865, 7866, 7867, 7868]
        
        for port in ports_to_try:
            try:
                print(f"🌐 Trying port {port}...")
                interface.launch(
                    share=False,
                    server_name="0.0.0.0",
                    server_port=port,
                    show_error=True
                )
                break
            except Exception as e:
                if "address already in use" in str(e):
                    print(f"❌ Port {port} is busy, trying next port...")
                    continue
                else:
                    raise e
        else:
            print("❌ All ports are busy. Please try again later.")
    except Exception as e:
        print(f"Error launching interface: {e}")


if __name__ == "__main__":
    main()
