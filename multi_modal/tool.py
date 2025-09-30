"""
Multi-Modal AI Assistant for Airline Customer Support

This module implements a multi-modal AI assistant that can:
- Chat with customers using text
- Generate images of destinations using DALL-E-3
- Generate speech using OpenAI's TTS
- Use tools to get ticket prices

Extracted from day5.ipynb notebook.
"""

import os
import json
import base64
import tempfile
import subprocess
import time
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play
from IPython.display import Audio, display
import simpleaudio as sa


class MultiModalAIAssistant:
    """Multi-modal AI assistant for airline customer support."""
    
    def __init__(self):
        """Initialize the assistant with API keys and models."""
        load_dotenv(override=True)
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            print(f"OpenAI API Key exists and begins {self.openai_api_key[:8]}")
        else:
            print("OpenAI API Key not set")
            
        self.model = "gpt-4o-mini"
        self.openai = OpenAI()
        
        # System message for the assistant
        self.system_message = ("You are a helpful assistant for an Airline called FlightAI. "
                              "Give short, courteous answers, no more than 1 sentence. "
                              "Always be accurate. If you don't know the answer, say so.")
        
        # Ticket prices database
        self.ticket_prices = {
            "london": "$799", 
            "paris": "$899", 
            "tokyo": "$1400", 
            "berlin": "$499"
        }
        
        # Define the price function tool
        self.price_function = {
            "name": "get_ticket_price",
            "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "The city that the customer wants to travel to",
                    },
                },
                "required": ["destination_city"],
                "additionalProperties": False
            }
        }
        
        self.tools = [{"type": "function", "function": self.price_function}]
    
    def get_ticket_price(self, destination_city):
        """Get ticket price for a destination city."""
        print(f"Tool get_ticket_price called for {destination_city}")
        city = destination_city.lower()
        return self.ticket_prices.get(city, "Unknown")
    
    def handle_tool_call(self, message):
        """Handle tool calls from the LLM."""
        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get('destination_city')
        price = self.get_ticket_price(city)
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call.id
        }
        return response, city
    
    def artist(self, city):
        """Generate an image representing a vacation in the given city using DALL-E-3."""
        image_response = self.openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))
    
    def talker_mac(self, message):
        """Generate speech using OpenAI's TTS (Mac version)."""
        response = self.openai.audio.speech.create(
            model="tts-1",
            voice="onyx",    # Also, try replacing onyx with alloy
            input=message
        )
        
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play(audio)
    
    def talker_pc_variation1(self, message):
        """Generate speech using OpenAI's TTS (PC Variation 1)."""
        response = self.openai.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=message)

        audio_stream = BytesIO(response.content)
        output_filename = "output_audio.mp3"
        with open(output_filename, "wb") as f:
            f.write(audio_stream.read())

        # Play the generated audio
        display(Audio(output_filename, autoplay=True))
    
    def play_audio(self, audio_segment):
        """Play audio using ffplay (PC Variation 2)."""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "temp_audio.wav")
        try:
            audio_segment.export(temp_path, format="wav")
            time.sleep(3)  # Student Dominic found that this was needed
            subprocess.call([
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-hide_banner",
                temp_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    
    def talker_pc_variation2(self, message):
        """Generate speech using OpenAI's TTS (PC Variation 2)."""
        response = self.openai.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Also, try replacing onyx with alloy
            input=message
        )
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        self.play_audio(audio)
    
    def talker_pc_variation3(self, message):
        """Generate speech using OpenAI's TTS (PC Variation 3)."""
        # Set a custom directory for temporary files on Windows
        custom_temp_dir = os.path.expanduser("~/Documents/temp_audio")
        os.environ['TEMP'] = custom_temp_dir  # You can also use 'TMP' if necessary
        
        # Create the folder if it doesn't exist
        if not os.path.exists(custom_temp_dir):
            os.makedirs(custom_temp_dir)
        
        response = self.openai.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Also, try replacing onyx with alloy
            input=message
        )
        
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play(audio)
    
    def talker_pc_variation4(self, message):
        """Generate speech using OpenAI's TTS (PC Variation 4 with simpleaudio)."""
        response = self.openai.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Also, try replacing onyx with alloy
            input=message
        )
        
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")

        # Create a temporary file in a folder where you have write permissions
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=os.path.expanduser("~/Documents")) as temp_audio_file:
            temp_file_name = temp_audio_file.name
            audio.export(temp_file_name, format="wav")
        
        # Load and play audio using simpleaudio
        wave_obj = sa.WaveObject.from_wave_file(temp_file_name)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait for playback to finish

        # Clean up the temporary file afterward
        os.remove(temp_file_name)
    
    def chat(self, history):
        """Main chat function that handles the conversation and tool calls."""
        messages = [{"role": "system", "content": self.system_message}] + history
        response = self.openai.chat.completions.create(model=self.model, messages=messages, tools=self.tools)
        image = None
        
        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            response, city = self.handle_tool_call(message)
            messages.append(message)
            messages.append(response)
            image = self.artist(city)
            response = self.openai.chat.completions.create(model=self.model, messages=messages)
            
        reply = response.choices[0].message.content
        history += [{"role": "assistant", "content": reply}]

        # Comment out or delete the next line if you'd rather skip Audio for now..
        # self.talker_mac(reply)  # Choose the appropriate talker method for your system
        
        return history, image
    
    def simple_chat(self, message, history):
        """Simple chat function without tools."""
        messages = [{"role": "system", "content": self.system_message}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content
    
    def chat_with_tools(self, message, history):
        """Chat function with tool support."""
        messages = [{"role": "system", "content": self.system_message}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(model=self.model, messages=messages, tools=self.tools)

        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            response, city = self.handle_tool_call(message)
            messages.append(message)
            messages.append(response)
            response = self.openai.chat.completions.create(model=self.model, messages=messages)
        
        return response.choices[0].message.content
    
    def create_gradio_interface(self):
        """Create and launch the Gradio interface."""
        with gr.Blocks() as ui:
            with gr.Row():
                chatbot = gr.Chatbot(height=500, type="messages")
                image_output = gr.Image(height=500)
            with gr.Row():
                entry = gr.Textbox(label="Chat with our AI Assistant:")
            with gr.Row():
                clear = gr.Button("Clear")

            def do_entry(message, history):
                history += [{"role": "user", "content": message}]
                return "", history

            entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
                self.chat, inputs=chatbot, outputs=[chatbot, image_output]
            )
            clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

        return ui
    
    def launch_simple_interface(self):
        """Launch a simple chat interface."""
        return gr.ChatInterface(fn=self.simple_chat, type="messages").launch()
    
    def launch_tools_interface(self):
        """Launch a chat interface with tools."""
        return gr.ChatInterface(fn=self.chat_with_tools, type="messages").launch()


def main():
    """Main function to run the multi-modal AI assistant."""
    assistant = MultiModalAIAssistant()
    
    # Create and launch the full interface
    ui = assistant.create_gradio_interface()
    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
