"""
Resume Chatbot with Push Notifications
=====================================

A Gradio-based interactive resume chatbot that allows visitors to have conversations
with an AI representation of Srikala Gangi Reddy. The chatbot can answer questions
about career background, skills, and experience, while also collecting user information
and recording unanswered questions via push notifications.

Features:
- Interactive chat interface using Gradio
- PDF resume parsing and LinkedIn profile integration
- Push notification system for user engagement tracking
- Tool-based function calling for data collection
- Professional AI persona representation

Usage Instructions:
------------------
1. Prerequisites:
   - Install required dependencies: pip install -r requirements.txt
   - Set up environment variables in .env file:
     * OPENAI_API_KEY: Your OpenAI API key
     * PUSHOVER_TOKEN: Your Pushover app token
     * PUSHOVER_USER: Your Pushover user key

2. Required Files:
   - data/resume.pdf: Your resume in PDF format
   - data/summary.txt: A text summary of your background

3. Running the Application:
   - Execute: python app.py
   - The Gradio interface will launch in your browser
   - Share the public URL to allow others to interact with your resume chatbot

4. Functionality:
   - Users can ask questions about your background and experience
   - The chatbot will attempt to answer based on your resume and summary
   - Unknown questions are recorded via push notifications
   - User contact information is collected and sent via push notifications
   - Professional conversation steering towards email contact

5. Customization:
   - Modify the 'name' variable in the Me class to change the persona
   - Update the system prompt to adjust conversation style
   - Add or modify tools for different data collection needs
"""

# Implementation with PUSH notifications
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import logging
from typing import Optional


load_dotenv(override=True)

# Configure logging for Hugging Face Spaces
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging with both console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()  # Console output for Hugging Face Spaces
    ]
)
logger = logging.getLogger(__name__)

def push(text: str) -> bool:
    """
    Send a push notification via Pushover API.
    
    Args:
        text: The message text to send
        
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    try:
        # Check if required environment variables are set
        token = os.getenv("PUSHOVER_TOKEN")
        user = os.getenv("PUSHOVER_USER")
        
        if not token or not user:
            logger.warning("Pushover credentials not configured. Skipping notification.")
            return False
            
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token,
                "user": user,
                "message": text,
            },
            timeout=10  # Add timeout to prevent hanging
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info("Push notification sent successfully")
            return True
        else:
            logger.error(f"Pushover API error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("Pushover API request timed out")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Pushover API")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Pushover API request failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending push notification: {str(e)}")
        return False


def record_user_details(email, name="Name not provided", notes="not provided"):
    """
    Record user details and attempt to send push notification.
    Returns success status regardless of push notification outcome.
    """
    try:
        message = f"Recording {name} with email {email} and notes {notes}"
        push_success = push(message)
        
        if push_success:
            logger.info(f"User details recorded and notification sent: {name} ({email})")
        else:
            logger.warning(f"User details recorded but notification failed: {name} ({email})")
            
        return {"recorded": "ok", "notification_sent": push_success}
        
    except Exception as e:
        logger.error(f"Error recording user details: {str(e)}")
        # Still return success to not break the chat flow
        return {"recorded": "ok", "notification_sent": False, "error": str(e)}

def record_unknown_question(question):
    """
    Record unknown question and attempt to send push notification.
    Returns success status regardless of push notification outcome.
    """
    try:
        message = f"Recording unknown question: {question}"
        push_success = push(message)
        
        if push_success:
            logger.info(f"Unknown question recorded and notification sent: {question}")
        else:
            logger.warning(f"Unknown question recorded but notification failed: {question}")
            
        return {"recorded": "ok", "notification_sent": push_success}
        
    except Exception as e:
        logger.error(f"Error recording unknown question: {str(e)}")
        # Still return success to not break the chat flow
        return {"recorded": "ok", "notification_sent": False, "error": str(e)}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Srikala Gangi Reddy"
        reader = PdfReader("data/resume.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("data/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and resume profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. \
Do not reveal  personal information like phone number, address, etc. \
You are however allowed to share public urls like linkedin, github and the website. When you share my website, please specify that it is my art profile." 


        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## Resume Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()

    # Example questions to help users get started
    examples = [
        "What is your background in technology?",
        "What are your key skills and areas of expertise?",
        "Tell me about your art portfolio and creative work"
    ]

    demo = gr.ChatInterface(
        fn=me.chat,
        type="messages",
        title="Srikala Gangi Reddy: Interactive Resume",
        description=(
            "Hello! I'm Srikala Gangi Reddy. You can have a conversation with me about my background and experience in technology and art. "
            "Feel free to ask me about my skills, projects, or career journey. Click on the example questions below to get started!"
        ),
        examples=examples,
        theme="soft",
    )
demo.launch(share=True)
    