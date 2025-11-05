"""
Gradio Interface for Mental Health Counseling Chatbot
"""
import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System message for the assistant
SYSTEM_MESSAGE = """You serve as a supportive and honest psychology and psychotherapy assistant. Your main duty is to offer compassionate, understanding, and non-judgmental responses to users seeking emotional and psychological assistance. Respond with empathy and exhibit active listening skills. Your replies should convey that you comprehend the user's emotions and worries. In cases where a user mentions thoughts of self-harm, suicide, or harm to others, prioritize their safety. Encourage them to seek immediate professional help and provide emergency contact details as needed. It's important to note that you are not a licensed medical professional. Refrain from diagnosing or prescribing treatments. Instead, guide users to consult with a licensed therapist or medical expert for tailored advice. Never store or disclose any personal information shared by users. Uphold their privacy at all times. Avoid taking sides or expressing personal viewpoints. Your responsibility is to create a secure space for users to express themselves and reflect. Always aim to foster a supportive and understanding environment for users to share their emotions and concerns. Above all, prioritize their well-being and safety."""

# Initialize OpenAI client
def get_client():
    """Initialize and return OpenAI client"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    return OpenAI(api_key=api_key)


def chat_with_bot(message, history, model_name):
    """
    Chat function for Gradio interface
    
    Args:
        message: Current user message
        history: Chat history (list of tuples)
        model_name: Name of the model to use
    
    Returns:
        Updated history
    """
    if not message:
        return history
    
    try:
        client = get_client()
        
        # Build messages list with system message
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        # Get response from OpenAI
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        
        assistant_response = completion.choices[0].message.content
        
        # Update history
        history.append((message, assistant_response))
        
        return history
    
    except Exception as e:
        error_message = f"Error: {str(e)}. Please check your API key and try again."
        history.append((message, error_message))
        return history


def create_interface():
    """Create and return Gradio interface"""
    
    # Default model - can be changed to fine-tuned model once available
    default_model = "gpt-3.5-turbo"
    
    with gr.Blocks(title="Mental Health Counseling Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌱 Mental Health Counseling Chatbot
            
            Welcome to your supportive mental health counseling assistant. I'm here to provide compassionate, 
            understanding, and non-judgmental responses to help you with your emotional and psychological concerns.
            
            **Important Disclaimer:** This chatbot is not a licensed medical professional. It cannot diagnose 
            or prescribe treatments. For professional medical advice, please consult with a licensed therapist 
            or medical expert.
            
            If you're experiencing thoughts of self-harm, suicide, or harm to others, please seek immediate 
            professional help or contact emergency services.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                    avatar_images=(None, "🤖")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Conversation", variant="secondary")
            
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
                    value=default_model,
                    label="Model",
                    info="Select the model to use"
                )
                
                gr.Markdown(
                    """
                    ### 💡 Tips for Better Conversations
                    
                    - Be open and honest about your feelings
                    - Ask specific questions for better guidance
                    - Remember this is a supportive space
                    - Seek professional help when needed
                    """
                )
                
                gr.Markdown(
                    """
                    ### 🆘 Emergency Resources
                    
                    - **National Suicide Prevention Lifeline:** 988
                    - **Crisis Text Line:** Text HOME to 741741
                    - **Emergency Services:** 911
                    """
                )
        
        # Event handlers
        msg.submit(
            fn=chat_with_bot,
            inputs=[msg, chatbot, model_dropdown],
            outputs=chatbot
        ).then(
            fn=lambda: "",
            outputs=msg
        )
        
        submit_btn.click(
            fn=chat_with_bot,
            inputs=[msg, chatbot, model_dropdown],
            outputs=chatbot
        ).then(
            fn=lambda: "",
            outputs=msg
        )
        
        clear_btn.click(
            fn=lambda: [],
            outputs=chatbot
        )
        
        gr.Markdown(
            """
            ---
            **Privacy Notice:** Your conversations are private. No personal information is stored or shared.
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )

