---
title: Hugging Face Pipelines Demo
emoji: 🤗
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: Interactive demo of various Hugging Face pipelines for NLP, CV, and audio tasks
---

# 🤗 Hugging Face Pipelines Demo

An interactive web application showcasing various Hugging Face pipelines for natural language processing, computer vision, and audio generation tasks.

## Features

### 📝 Text Analysis
- **Sentiment Analysis**: Analyze the emotional tone of text
- **Named Entity Recognition**: Extract entities like people, places, organizations

### ❓ Q&A & Summarization  
- **Question Answering**: Answer questions based on provided context
- **Text Summarization**: Generate concise summaries of longer texts

### 🌍 Translation & Classification
- **Translation**: Translate text between different languages (English to French, Spanish, German, Italian)
- **Zero-shot Classification**: Classify text into custom categories without training data

### ✍️ Text Generation
- **Text Generation**: Generate new text based on prompts using GPT-style models

### 🎨 Image Generation
- **Image Generation**: Create images from text descriptions using Stable Diffusion

### 🎵 Text-to-Speech
- **Text-to-Speech**: Convert text to natural-sounding speech

## Technical Details

- **Framework**: Gradio 4.0+
- **Models**: Various Hugging Face transformer and diffusion models
- **Device**: Optimized for CPU execution
- **Caching**: Pipeline caching for better performance

## Usage

1. Select a tab for the task you want to perform
2. Enter your input text/prompt
3. Click the corresponding button to generate results
4. View the output in the interface

## Model Information

The app uses various pre-trained models from Hugging Face Hub:
- Sentiment Analysis: `distilbert-base-uncased-finetuned-sst-2-english`
- Named Entity Recognition: `dbmdz/bert-large-cased-finetuned-conll03-english`
- Question Answering: `distilbert-base-cased-distilled-squad`
- Text Summarization: `facebook/bart-large-cnn`
- Translation: Various Helsinki-NLP models
- Zero-shot Classification: `facebook/bart-large-mnli`
- Text Generation: `gpt2`
- Image Generation: `runwayml/stable-diffusion-v1-5`
- Text-to-Speech: `microsoft/speecht5_tts`

## Performance Notes

- All models run on CPU for compatibility
- First-time model loading may take a few minutes
- Image generation is optimized for CPU with reduced inference steps
- Audio generation uses pre-trained speaker embeddings

## License

This project is open source and available under the MIT License.
