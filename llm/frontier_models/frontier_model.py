#!/usr/bin/env python3
"""
Frontier Model APIs - Extracted from day1.ipynb

This script demonstrates how to use various frontier model APIs including:
- OpenAI (GPT models)
- Anthropic (Claude models) 
- Google (Gemini models)
- DeepSeek models

It includes examples of:
- Basic API calls with different models
- Streaming responses
- Multi-model conversations
- Business-relevant use cases
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from IPython.display import Markdown, display, update_display

# Optional import for Google Gemini (may cause issues on some systems)
try:
    import google.generativeai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google generativeai not available - skipping Gemini examples")

def setup_api_keys():
    """Load and validate API keys from environment variables"""
    load_dotenv(override=True)
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    
    # Print key status
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")
        
    if anthropic_api_key:
        print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
    else:
        print("Anthropic API Key not set")

    if google_api_key:
        print(f"Google API Key exists and begins {google_api_key[:8]}")
    else:
        print("Google API Key not set")
        
    if deepseek_api_key:
        print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
    else:
        print("DeepSeek API Key not set")
    
    return openai_api_key, anthropic_api_key, google_api_key, deepseek_api_key

def initialize_clients(openai_api_key, anthropic_api_key, google_api_key, deepseek_api_key):
    """Initialize API clients for all available services"""
    clients = {}
    
    # OpenAI client
    if openai_api_key:
        clients['openai'] = OpenAI()
    
    # Anthropic client
    if anthropic_api_key:
        clients['anthropic'] = anthropic.Anthropic()
    
    # Google client (if available)
    if google_api_key and GOOGLE_AVAILABLE:
        try:
            google.generativeai.configure()
            clients['google'] = google.generativeai
        except Exception as e:
            print(f"Google client setup failed: {e}")
    
    # DeepSeek client (using OpenAI client with different base URL)
    if deepseek_api_key:
        clients['deepseek'] = OpenAI(
            api_key=deepseek_api_key, 
            base_url="https://api.deepseek.com"
        )
    
    return clients

def joke_examples(clients):
    """Demonstrate joke telling with different models"""
    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"
    
    prompts = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    print("=== JOKE EXAMPLES ===\n")
    
    # GPT-4o-mini
    if 'openai' in clients:
        print("GPT-4o-mini:")
        try:
            completion = clients['openai'].chat.completions.create(
                model='gpt-4o-mini', 
                messages=prompts
            )
            print(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50 + "\n")
    
    # GPT-4.1-mini with temperature
    if 'openai' in clients:
        print("GPT-4.1-mini (temperature=0.7):")
        try:
            completion = clients['openai'].chat.completions.create(
                model='gpt-4.1-mini',
                messages=prompts,
                temperature=0.7
            )
            print(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50 + "\n")
    
    # GPT-4.1-nano
    if 'openai' in clients:
        print("GPT-4.1-nano:")
        try:
            completion = clients['openai'].chat.completions.create(
                model='gpt-4.1-nano',
                messages=prompts
            )
            print(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50 + "\n")
    
    # GPT-4.1
    if 'openai' in clients:
        print("GPT-4.1:")
        try:
            completion = clients['openai'].chat.completions.create(
                model='gpt-4.1',
                messages=prompts,
                temperature=0.4
            )
            print(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50 + "\n")
    
    # Claude 4.0 Sonnet
    if 'anthropic' in clients:
        print("Claude 4.0 Sonnet:")
        try:
            message = clients['anthropic'].messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                temperature=0.7,
                system=system_message,
                messages=[{"role": "user", "content": user_prompt}]
            )
            print(message.content[0].text)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50 + "\n")
    
    # Claude with streaming
    if 'anthropic' in clients:
        print("Claude 4.0 Sonnet (streaming):")
        try:
            result = clients['anthropic'].messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                temperature=0.7,
                system=system_message,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            with result as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50 + "\n")
    
    # Gemini via OpenAI client
    if 'openai' in clients and google_api_key:
        print("Gemini 2.5 Flash (via OpenAI client):")
        try:
            gemini_via_openai_client = OpenAI(
                api_key=google_api_key, 
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            
            response = gemini_via_openai_client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=prompts
            )
            print(response.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50 + "\n")

def serious_business_question(clients):
    """Demonstrate a serious business question with streaming"""
    print("=== SERIOUS BUSINESS QUESTION ===\n")
    
    prompts = [
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown"},
        {"role": "user", "content": "How do I decide if a business problem is suitable for an LLM solution? Please respond in Markdown."}
    ]
    
    if 'openai' in clients:
        print("GPT-4.1-mini response (streaming):")
        try:
            stream = clients['openai'].chat.completions.create(
                model='gpt-4.1-mini',
                messages=prompts,
                temperature=0.7,
                stream=True
            )
            
            reply = ""
            for chunk in stream:
                reply += chunk.choices[0].delta.content or ''
                print(chunk.choices[0].delta.content or '', end='', flush=True)
            print("\n")
        except Exception as e:
            print(f"Error: {e}")

def deepseek_examples(clients):
    """Demonstrate DeepSeek models with challenging questions"""
    if 'deepseek' not in clients:
        print("DeepSeek API key not available - skipping DeepSeek examples")
        return
    
    print("=== DEEPSEEK EXAMPLES ===\n")
    
    # Basic DeepSeek Chat
    prompts = [
        {"role": "system", "content": "You are an assistant that is great at telling jokes"},
        {"role": "user", "content": "Tell a light-hearted joke for an audience of Data Scientists"}
    ]
    
    print("DeepSeek Chat:")
    try:
        response = clients['deepseek'].chat.completions.create(
            model="deepseek-chat",
            messages=prompts,
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
    print("\n" + "="*50 + "\n")
    
    # Challenging question
    challenge = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "How many words are there in your answer to this prompt"}
    ]
    
    print("DeepSeek Chat with challenging question:")
    try:
        stream = clients['deepseek'].chat.completions.create(
            model="deepseek-chat",
            messages=challenge,
            stream=True
        )
        
        reply = ""
        for chunk in stream:
            reply += chunk.choices[0].delta.content or ''
            print(chunk.choices[0].delta.content or '', end='', flush=True)
        
        print(f"\nNumber of words: {len(reply.split(' '))}")
    except Exception as e:
        print(f"Error: {e}")
    print("\n" + "="*50 + "\n")
    
    # DeepSeek Reasoner (if available)
    print("DeepSeek Reasoner:")
    try:
        response = clients['deepseek'].chat.completions.create(
            model="deepseek-reasoner",
            messages=challenge
        )
        
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        
        print("Reasoning:")
        print(reasoning_content)
        print("\nFinal Answer:")
        print(content)
        print(f"Number of words: {len(content.split(' '))}")
    except Exception as e:
        print(f"Error: {e}")

def adversarial_conversation(clients):
    """Demonstrate a conversation between two chatbots with different personalities"""
    if 'openai' not in clients or 'anthropic' not in clients:
        print("Need both OpenAI and Anthropic API keys for adversarial conversation")
        return
    
    print("=== ADVERSARIAL CONVERSATION ===\n")
    
    gpt_model = "gpt-4.1-mini"
    claude_model = "claude-3-5-haiku-latest"
    
    gpt_system = ("You are a chatbot who is very argumentative; "
                 "you disagree with anything in the conversation and you challenge everything, in a snarky way.")
    
    claude_system = ("You are a very polite, courteous chatbot. You try to agree with "
                    "everything the other person says, or find common ground. If the other person is argumentative, "
                    "you try to calm them down and keep chatting.")
    
    gpt_messages = ["Hi there"]
    claude_messages = ["Hi"]
    
    def call_gpt():
        messages = [{"role": "system", "content": gpt_system}]
        for gpt, claude in zip(gpt_messages, claude_messages):
            messages.append({"role": "assistant", "content": gpt})
            messages.append({"role": "user", "content": claude})
        
        completion = clients['openai'].chat.completions.create(
            model=gpt_model,
            messages=messages
        )
        return completion.choices[0].message.content
    
    def call_claude():
        messages = []
        for gpt, claude_message in zip(gpt_messages, claude_messages):
            messages.append({"role": "user", "content": gpt})
            messages.append({"role": "assistant", "content": claude_message})
        messages.append({"role": "user", "content": gpt_messages[-1]})
        
        message = clients['anthropic'].messages.create(
            model=claude_model,
            system=claude_system,
            messages=messages,
            max_tokens=500
        )
        return message.content[0].text
    
    print(f"GPT:\n{gpt_messages[0]}\n")
    print(f"Claude:\n{claude_messages[0]}\n")
    
    for i in range(5):
        try:
            gpt_next = call_gpt()
            print(f"GPT:\n{gpt_next}\n")
            gpt_messages.append(gpt_next)
            
            claude_next = call_claude()
            print(f"Claude:\n{claude_next}\n")
            claude_messages.append(claude_next)
        except Exception as e:
            print(f"Error in conversation: {e}")
            break

def main():
    """Main function to run all examples"""
    print("Frontier Model APIs Demo")
    print("=" * 50)
    
    # Setup
    openai_api_key, anthropic_api_key, google_api_key, deepseek_api_key = setup_api_keys()
    clients = initialize_clients(openai_api_key, anthropic_api_key, google_api_key, deepseek_api_key)
    
    # Run examples
    joke_examples(clients)
    serious_business_question(clients)
    deepseek_examples(clients)
    adversarial_conversation(clients)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
