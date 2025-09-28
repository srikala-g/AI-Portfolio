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
from typing import Optional


load_dotenv(override=True)


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
            return True
        else:
            return False
            
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.RequestException as e:
        return False
    except Exception as e:
        return False


def record_user_details(email, name="Name not provided", notes="not provided"):
    """
    Record user details and attempt to send push notification.
    Returns success status regardless of push notification outcome.
    """
    try:
        message = f"Recording {name} with email {email} and notes {notes}"
        push_success = push(message)
        
            
        return {"recorded": "ok", "notification_sent": push_success}
        
    except Exception as e:
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
        
            
        return {"recorded": "ok", "notification_sent": push_success}
        
    except Exception as e:
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
        try:
            self.openai = OpenAI()
            self.name = "Srikala Gangi Reddy"
            
            # Load resume PDF
            try:
                reader = PdfReader("data/resume.pdf")
                self.linkedin = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        self.linkedin += text
            except Exception as pdf_error:
                self.linkedin = f"Error loading resume: {str(pdf_error)}"
            
            # Load summary
            try:
                with open("data/summary.txt", "r", encoding="utf-8") as f:
                    self.summary = f.read()
            except Exception as summary_error:
                self.summary = f"Error loading summary: {str(summary_error)}"
                
        except Exception as init_error:
            # Fallback initialization for critical errors
            self.name = "Srikala Gangi Reddy"
            self.linkedin = f"Initialization error: {str(init_error)}"
            self.summary = f"Initialization error: {str(init_error)}"
            self.openai = None


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                print(f"Tool called: {tool_name}", flush=True)
                tool = globals().get(tool_name)
                result = tool(**arguments) if tool else {}
                results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
            except Exception as tool_error:
                error_result = {
                    "role": "tool",
                    "content": json.dumps({
                        "error": f"Tool execution failed: {str(tool_error)}",
                        "tool_name": tool_call.function.name,
                        "error_type": type(tool_error).__name__
                    }),
                    "tool_call_id": tool_call.id
                }
                results.append(error_result)
        return results
    
    def system_prompt(self):
#         system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
# particularly questions related to {self.name}'s career, background, skills and experience. \
# Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
# You are given a summary of {self.name}'s background and resume profile which you can use to answer questions. \
# Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
# If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
# If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. \
# Do not reveal  personal information like phone number, address, etc. \
# You are however allowed to share public urls like linkedin, github and the website. When you share my website, please specify that it is my art profile. \
# Make sure the descriptive text clearly communicates the link's purpose. \
# IMPORTANT: When referencing URLs from the resume, use the EXACT URLs as they appear in the resume content. Do not modify, truncate, or change any URLs. Do not wrap the urls.\
# When asked about skills list Agile project management followed by AI skills. \
# When asked about next role do not mention desginations or role names explicitly, describe the role, be open and flexible and empahsize on adaptability and learning. \
# When asked about AI projects guide them to the git respository in the resume. Always emphasize on learning, experimentation adaptability in AI projects. Do not makeup project experience in AI at companies I worked for."

        system_prompt = f"""
        You are acting as {self.name}. You are answering questions on {self.name}'s website, 
        particularly questions related to {self.name}'s career, background, skills, and experience. 
        Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. 
        You are given a summary of {self.name}'s background and resume profile which you can use to answer questions. 

        Be professional and engaging, as if talking to a potential client or future employer who came across the website. 
        If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. 
        If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. 
        Do not reveal personal information like phone number, address, etc. 

        You are allowed to share public URLs like LinkedIn, GitHub, and the website. 
        **CRITICAL URL FORMATTING RULES:**
        1. NEVER show raw URLs like http://example.com
        2. NEVER use @ symbols before URLs
        3. ALWAYS use Markdown format: [descriptive text](exact_url)
        4. Use these exact formats:
           - For art portfolio: [Art Portfolio](http://undertheyellowtree.com)
           - For AI projects: [AI Projects](https://github.com/srikala-g/AI-Portfolio)
           - For LinkedIn: [LinkedIn](https://www.linkedin.com/in/srikala-gangi-reddy/)
        5. Use the EXACT URLs from the resume content, do not modify them
        6. For mobile compatibility, use simple link text like 'here' or 'this link'  

        When asked about skills, list Agile project management followed by AI skills.  
        When asked about next role, do not mention designations or role names explicitly; describe the role, be open and flexible, and emphasize adaptability and learning.  
        When asked about AI projects, guide them to the GitHub repository in the resume. Always emphasize learning, experimentation, and adaptability in AI projects. Do not make up project experience in AI at companies you worked for.
        For social media links show LinkedIn, AI Projects and Art Portfolio in that order
        """

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## Resume Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def truncate_response(self, response, max_length=200):
        """
        Truncate long responses for mobile display with expandable option.
        Only applies truncation on mobile devices.
        Preserves URLs and ensures clean content boundaries.
        """
        if len(response) <= max_length:
            return response
        
        import re
        
        # Check if there are URLs in the response
        url_pattern = r'https?://[^\s\)]+|\[[^\]]+\]\([^\)]+\)'
        urls = re.findall(url_pattern, response)
        
        # If URLs are present, find a safe truncation point
        if urls:
            # Find the last complete URL before max_length
            safe_truncation_point = 0
            for url in urls:
                url_start = response.find(url)
                url_end = url_start + len(url)
                if url_end <= max_length:
                    safe_truncation_point = max(safe_truncation_point, url_end)
            
            # If we found a safe point after a URL, use it
            if safe_truncation_point > max_length * 0.3:
                truncated = response[:safe_truncation_point]
            else:
                # No safe truncation point found, don't truncate
                return response
        else:
            # No URLs, find a good sentence boundary
            truncated = response[:max_length]
            
            # Look for complete sentences first
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            
            sentence_end = max(last_period, last_exclamation, last_question)
            
            if sentence_end > max_length * 0.6:
                truncated = response[:sentence_end + 1]
            else:
                # Look for word boundaries
                last_space = truncated.rfind(' ')
                if last_space > max_length * 0.7:
                    truncated = response[:last_space]
                else:
                    # No good break point, don't truncate
                    return response
        
        # Get the remaining text
        remaining_text = response[len(truncated):].strip()
        
        # Don't show "read more" if remaining text is too short or just punctuation
        if len(remaining_text) < 30 or remaining_text in ['.', '!', '?', '...', '(', ')', '.', 'o']:
            return response
        
        # Return with mobile detection using CSS classes
        return f"""<div class="response-container">
<div class="desktop-response">{response}</div>
<div class="mobile-response">{truncated}<details><summary>read more...</summary><div>{remaining_text}</div></details></div>
</div>"""

    def chat(self, message, history):
        try:
            # Check if OpenAI client is available
            if self.openai is None:
                return "**Service Unavailable**: OpenAI client not initialized. Please check the configuration and try again."
            
            messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
            done = False
            while not done:
                try:
                    response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
                    if response.choices[0].finish_reason=="tool_calls":
                        message = response.choices[0].message
                        tool_calls = message.tool_calls
                        results = self.handle_tool_call(tool_calls)
                        messages.append(message)
                        messages.extend(results)
                    else:
                        done = True
                except Exception as api_error:
                    return f"**API Error**: {str(api_error)}\n\nPlease try again or contact support if the issue persists."
            
            # Truncate long responses for mobile users
            full_response = response.choices[0].message.content
            return self.truncate_response(full_response)
            
        except Exception as e:
            error_details = f"""
**Error Details:**
- **Error Type**: {type(e).__name__}
- **Error Message**: {str(e)}
- **User Message**: {message[:100]}{'...' if len(message) > 100 else ''}
- **History Length**: {len(history) if history else 0}

**Troubleshooting:**
1. Check if OpenAI API key is configured
2. Verify internet connection
3. Try refreshing the page
4. Contact support if issue persists

**Technical Details for Support:**
```python
# Error occurred in chat method
# Timestamp: {__import__('datetime').datetime.now()}
# Error: {repr(e)}
```
            """
            return error_details
    

if __name__ == "__main__":
    try:
        me = Me()

        # Example questions to help users get started
        examples = [
            "Tell me about your background and key skills in technology.",
            "What roles is your profile suitable for?",
            "How does your art experience shape your work in technology and leadership?"
        ]

        demo = gr.ChatInterface(
            fn=me.chat,
            title="Srikala Gangi Reddy: Interactive Resume",
            description=(
                "Hello! I'm Srikala Gangi Reddy. You can have a conversation with me about my background and experience in technology and art. "
                "Feel free to ask me about my skills, projects, or career journey.\n\n"
                "**Note:** The application must only be used when the status indicates Running."
            ),
            examples=examples,
            theme="soft",
            type="messages",
            css="""
            .gradio-container {
                max-width: 100% !important;
                margin: 0 auto !important;
            }
            .chat-message {
                word-wrap: break-word !important;
                max-width: 100% !important;
            }
            /* Mobile/Desktop response handling */
            .response-container {
                position: relative;
            }
            .desktop-response {
                display: block;
            }
            .mobile-response {
                display: none;
            }
            details {
                margin-top: 8px;
                padding: 0;
                border: none;
                background: transparent;
            }
            details[open] {
                display: block;
            }
            details:not([open]) {
                display: block;
            }
            details:not([open]) > div {
                display: none;
            }
            summary {
                cursor: pointer;
                font-weight: 600;
                color: #007bff;
                background: #f8f9fa;
                padding: 6px 12px;
                border-radius: 20px;
                border: 1px solid #e9ecef;
                display: inline-block;
                font-size: 13px;
                transition: all 0.2s ease;
            }
            summary:hover {
                color: #0056b3;
                background: #e3f2fd;
                border-color: #007bff;
            }
            details[open] summary {
                background: #e3f2fd;
                color: #0056b3;
            }
            details > div {
                margin-top: 8px;
                padding: 12px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 3px solid #007bff;
                color: #333 !important;
            }
            /* Dark mode support for expanded content */
            @media (prefers-color-scheme: dark) {
                details > div {
                    background: #2d3748 !important;
                    color: #e2e8f0 !important;
                    border-left-color: #4299e1 !important;
                }
                .mobile-response details > div {
                    background: #2d3748 !important;
                    color: #e2e8f0 !important;
                }
            }
            @media (max-width: 768px) {
                .gradio-container {
                    padding: 10px !important;
                }
                .chat-message {
                    font-size: 14px !important;
                }
                summary {
                    font-size: 12px;
                    padding: 5px 10px;
                }
                /* Show mobile response, hide desktop response on mobile */
                .desktop-response {
                    display: none !important;
                }
                .mobile-response {
                    display: block !important;
                }
            }
            /* Dark mode support for mobile */
            @media (prefers-color-scheme: dark) and (max-width: 768px) {
                .mobile-response {
                    color: #e2e8f0 !important;
                }
                .mobile-response summary {
                    background: #4a5568 !important;
                    color: #e2e8f0 !important;
                    border-color: #718096 !important;
                }
                .mobile-response summary:hover {
                    background: #2d3748 !important;
                    color: #4299e1 !important;
                }
            }
            @media (min-width: 769px) {
                /* Show desktop response, hide mobile response on desktop */
                .desktop-response {
                    display: block !important;
                }
                .mobile-response {
                    display: none !important;
                }
            }
            """
        )
        demo.launch()
    except Exception as startup_error:
        print(f"**Startup Error**: {str(startup_error)}")
        print(f"Error Type: {type(startup_error).__name__}")
        print("Please check your configuration and try again.")
        # Create a fallback interface that shows the error
        try:
            error_demo = gr.Interface(
                fn=lambda x: f"**Application Error**: {str(startup_error)}\n\nPlease contact support.",
                inputs="text",
                outputs="text",
                title="Srikala Gangi Reddy: Interactive Resume (Error Mode)",
                description="The application encountered an error during startup. Please contact support."
            )
            error_demo.launch()
        except:
            print("Unable to create fallback interface.")
    