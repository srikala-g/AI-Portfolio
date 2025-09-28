---
title: interactive_resume
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
license: mit
short_description: Interactive AI resume chatbot for Srikala Gangi Reddy
---

# Resume Chatbot - Srikala Gangi Reddy

An interactive AI-powered resume chatbot that allows visitors to have conversations with an AI representation of Srikala Gangi Reddy. The chatbot can answer questions about career background, skills, and experience.

## Features

- 🤖 **Interactive Chat Interface**: Powered by Gradio for seamless user experience
- 📄 **PDF Resume Integration**: Automatically parses and incorporates resume content
- 🔔 **Push Notifications**: Tracks user engagement and unanswered questions
- 🛠️ **Tool-based Function Calling**: Collects user information and records interactions
- 🎨 **Professional AI Persona**: Represents Srikala's professional background authentically

## How It Works

1. **Ask Questions**: Users can ask about Srikala's background, skills, projects, and career journey
2. **AI Responses**: The chatbot provides detailed answers based on resume and professional summary
3. **Contact Collection**: Encourages meaningful conversations and email collection
4. **Question Tracking**: Records questions that couldn't be answered for continuous improvement

## Technology Stack

- **Framework**: Gradio for the web interface
- **AI Model**: OpenAI GPT-4o-mini for intelligent responses
- **PDF Processing**: PyPDF for resume parsing
- **Notifications**: Pushover API for engagement tracking
- **Deployment**: Hugging Face Spaces

## Usage

Simply start a conversation by asking questions like:
- "Tell me about your background and key skills in technology"
- "What roles is your profile suitable for?"
- "How does your art experience shape your work in technology and leadership?"

## Environment Variables

The following environment variables need to be configured:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PUSHOVER_TOKEN`: Your Pushover app token (optional)
- `PUSHOVER_USER`: Your Pushover user key (optional)

## Files Required

- `data/resume.pdf`: Resume in PDF format
- `data/summary.txt`: Professional background summary

## License

MIT License - Feel free to use this as a template for your own resume chatbot!