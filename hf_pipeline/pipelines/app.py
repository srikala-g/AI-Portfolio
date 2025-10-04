import gradio as gr
import torch
import os
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
import tempfile
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Device detection - dynamic backend detection
if torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon GPU (M1, M2, M3, M4)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU (Hugging Face, Colab, etc.)
else:
    device = torch.device("cpu")   # Fallback

if device.type == "cpu":
    print("Note: Running on CPU. For better performance, consider using a GPU-enabled environment.")
elif device.type == "mps":
    print("Note: Using Apple Silicon GPU acceleration.")
elif device.type == "cuda":
    print("Note: Using NVIDIA GPU acceleration.")

# Initialize pipelines (lazy loading to avoid memory issues)
pipelines_cache = {}

def get_pipeline(task, model=None, **kwargs):
    """Get or create a pipeline with caching"""
    cache_key = f"{task}_{model}" if model else task
    
    if cache_key not in pipelines_cache:
        try:
            if model:
                pipelines_cache[cache_key] = pipeline(task, model=model, device=device, **kwargs)
            else:
                pipelines_cache[cache_key] = pipeline(task, device=device, **kwargs)
        except Exception as e:
            print(f"Error loading {task} pipeline: {e}")
            return None, None
    
    # Get the actual model name being used
    model_name = pipelines_cache[cache_key].model.config.name_or_path if hasattr(pipelines_cache[cache_key].model, 'config') else "Unknown"
    return pipelines_cache[cache_key], model_name

def sentiment_analysis(text, selected_model="distilbert-base-uncased-finetuned-sst-2-english"):
    """Analyze sentiment of input text"""
    if not text.strip():
        return "Please enter some text to analyze."
    
    try:
        classifier, model_name = get_pipeline("sentiment-analysis", model=selected_model)
        if classifier is None:
            return f"Error: Could not load sentiment analysis model: {selected_model}"
        
        result = classifier(text)
        return f"**Model:** {model_name}\n**Device:** {device}\n\n**Result:**\nLabel: {result[0]['label']}\nConfidence: {result[0]['score']:.3f}"
    except Exception as e:
        return f"Error: {str(e)}"

def named_entity_recognition(text, selected_model="dbmdz/bert-large-cased-finetuned-conll03-english"):
    """Extract named entities from text"""
    if not text.strip():
        return "Please enter some text to analyze."
    
    try:
        ner, model_name = get_pipeline("ner", model=selected_model, grouped_entities=True)
        if ner is None:
            return f"Error: Could not load NER model: {selected_model}"
        
        result = ner(text)
        if not result:
            return f"**Model:** {model_name}\n**Device:** {device}\n\nNo named entities found."
        
        entities = []
        for entity in result:
            entities.append(f"{entity['entity_group']}: {entity['word']} (confidence: {entity['score']:.3f})")
        
        return f"**Model:** {model_name}\n**Device:** {device}\n\n**Entities Found:**\n" + "\n".join(entities)
    except Exception as e:
        return f"Error: {str(e)}"

def question_answering(question, context, selected_model="distilbert-base-cased-distilled-squad"):
    """Answer questions based on context"""
    if not question.strip() or not context.strip():
        return "Please provide both a question and context."
    
    try:
        qa, model_name = get_pipeline("question-answering", model=selected_model)
        if qa is None:
            return f"Error: Could not load Q&A model: {selected_model}"
        
        result = qa(question=question, context=context)
        return f"**Model:** {model_name}\n**Device:** {device}\n\n**Answer:** {result['answer']}\n**Confidence:** {result['score']:.3f}"
    except Exception as e:
        return f"Error: {str(e)}"

def text_summarization(text, selected_model="facebook/bart-large-cnn"):
    """Summarize long text"""
    if not text.strip():
        return "Please enter some text to summarize."
    
    if len(text.split()) < 10:
        return "Text is too short for summarization. Please provide at least 10 words."
    
    try:
        summarizer, model_name = get_pipeline("summarization", model=selected_model)
        if summarizer is None:
            return f"Error: Could not load summarization model: {selected_model}"
        
        # Limit text length for CPU processing
        max_length = min(1000, len(text))
        text_input = text[:max_length]
        
        summary = summarizer(text_input, max_length=100, min_length=30, do_sample=False)
        return f"**Model:** {model_name}\n**Device:** {device}\n\n**Summary:**\n{summary[0]['summary_text']}"
    except Exception as e:
        return f"Error: {str(e)}"

def translation(text, target_language, selected_model="Helsinki-NLP/opus-mt-en-fr"):
    """Translate text to target language"""
    if not text.strip():
        return "Please enter some text to translate."
    
    try:
        # Map language names to pipeline names
        lang_map = {
            "French": "translation_en_to_fr",
            "Spanish": "translation_en_to_es", 
            "German": "translation_en_to_de",
            "Italian": "translation_en_to_it"
        }
        
        if target_language not in lang_map:
            return f"Translation to {target_language} not supported. Available: {', '.join(lang_map.keys())}"
        
        # Use selected model if it's a specific translation model, otherwise use language mapping
        if "opus-mt-en-" in selected_model or "m2m100" in selected_model:
            translator, model_name = get_pipeline("translation", model=selected_model)
        else:
            translator, model_name = get_pipeline(lang_map[target_language])
        
        if translator is None:
            return f"Error: Could not load translation model: {selected_model}"
        
        result = translator(text)
        return f"**Model:** {model_name}\n**Device:** {device}\n\n**Translation:**\n{result[0]['translation_text']}"
    except Exception as e:
        return f"Error: {str(e)}"

def zero_shot_classification(text, labels, selected_model="facebook/bart-large-mnli"):
    """Classify text into custom categories"""
    if not text.strip() or not labels.strip():
        return "Please provide both text and classification labels (comma-separated)."
    
    try:
        classifier, model_name = get_pipeline("zero-shot-classification", model=selected_model)
        if classifier is None:
            return f"Error: Could not load classification model: {selected_model}"
        
        label_list = [label.strip() for label in labels.split(",")]
        result = classifier(text, candidate_labels=label_list)
        
        output = f"**Model:** {model_name}\n**Device:** {device}\n\n**Predicted:** {result['labels'][0]}\n**Confidence:** {result['scores'][0]:.3f}\n\n**All Scores:**\n"
        for label, score in zip(result['labels'], result['scores']):
            output += f"- {label}: {score:.3f}\n"
        
        return output
    except Exception as e:
        return f"Error: {str(e)}"

def text_generation(prompt, max_length=50, selected_model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """Generate text based on prompt"""
    if not prompt.strip():
        return "Please enter a prompt for text generation."
    
    try:
        # Model mapping
        model_map = {
            "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct", 
            "google/gemma-2-2b-it": "google/gemma-2-2b-it",
            "Qwen/Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
            "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1"
        }
        
        model_name = model_map.get(selected_model, selected_model)
        generator, actual_model_name = get_pipeline("text-generation", model=model_name)
        
        if generator is None:
            return f"Error: Could not load text generation model: {model_name}"
        
        result = generator(prompt, max_length=max_length, do_sample=True, temperature=0.7)
        return f"**Model:** {actual_model_name}\n**Device:** {device}\n\n**Generated Text:**\n{result[0]['generated_text']}"
    except Exception as e:
        return f"Error: {str(e)}"

def image_generation(prompt, selected_model="runwayml/stable-diffusion-v1-5"):
    """Generate image from text prompt"""
    if not prompt.strip():
        return None, "Please enter a prompt for image generation."
    
    try:
        model_name = selected_model
        
        # Set appropriate dtype based on device
        if device.type == "cuda":
            torch_dtype = torch.float16
            variant = "fp16"
        else:
            torch_dtype = torch.float32
            variant = None
        
        image_gen = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant
        ).to(device)
        
        # Adjust steps based on device for optimal performance
        if device.type == "cpu":
            num_steps = 20  # Reduced for CPU
        elif device.type == "mps":
            num_steps = 30  # Moderate for Apple Silicon
        else:  # CUDA
            num_steps = 50  # Full quality for NVIDIA GPU
        
        image = image_gen(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.5
        ).images[0]
        
        return image, f"**Model:** {model_name}\n**Device:** {device}\n\nImage generated successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"

def text_to_speech(text, selected_model="microsoft/speecht5_tts"):
    """Convert text to speech"""
    if not text.strip():
        return None, "Please enter some text to convert to speech."
    
    try:
        model_name = selected_model
        synthesiser = get_pipeline("text-to-speech", model_name)
        if synthesiser is None:
            return None, "Error: Could not load TTS model"
        
        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
        speech = synthesiser(
            text, 
            forward_params={"speaker_embeddings": speaker_embedding}
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, speech["audio"], samplerate=speech["sampling_rate"])
            return tmp_file.name, f"**Model:** {model_name}\n**Device:** {device}\n\nAudio generated successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface with professional theme
with gr.Blocks(
    title="AI Pipeline Studio for open source", 
    theme=gr.themes.Glass(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    ),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .tab-content {
        padding: 1.5rem;
    }
    """
) as demo:
    
    # Professional header
    gr.HTML("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">AI Pipeline Studio</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Professional AI Tools for Text, Image, and Audio Processing</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("Text Analysis", id="text-analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Sentiment Analysis")
                    sentiment_input = gr.Textbox(
                        label="Text Input", 
                        placeholder="Enter text to analyze sentiment...",
                        lines=3
                    )
                    sentiment_model = gr.Dropdown(
                        choices=[
                            "distilbert-base-uncased-finetuned-sst-2-english",
                            "cardiffnlp/twitter-roberta-base-sentiment-latest",
                            "nlptown/bert-base-multilingual-uncased-sentiment"
                        ],
                        label="Model Selection",
                        value="distilbert-base-uncased-finetuned-sst-2-english",
                        info="Choose a sentiment analysis model"
                    )
                    sentiment_btn = gr.Button("Analyze", variant="primary", size="lg")
                    sentiment_output = gr.Textbox(label="Analysis Result", interactive=False, lines=2)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Entity Recognition")
                    ner_input = gr.Textbox(
                        label="Text Input", 
                        placeholder="Enter text to extract entities...",
                        lines=3
                    )
                    ner_model = gr.Dropdown(
                        choices=[
                            "dbmdz/bert-large-cased-finetuned-conll03-english",
                            "microsoft/DialoGPT-medium",
                            "xlm-roberta-large-finetuned-conll03-english"
                        ],
                        label="Model Selection",
                        value="dbmdz/bert-large-cased-finetuned-conll03-english",
                        info="Choose an NER model"
                    )
                    ner_btn = gr.Button("Extract", variant="primary", size="lg")
                    ner_output = gr.Textbox(label="Entities Found", interactive=False, lines=4)
        
        with gr.Tab("Q&A & Summarization", id="qa-summary"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Question Answering")
                    qa_question = gr.Textbox(label="Question", placeholder="What is your question?")
                    qa_context = gr.Textbox(label="Context", placeholder="Provide relevant context...", lines=4)
                    qa_model = gr.Dropdown(
                        choices=[
                            "distilbert-base-cased-distilled-squad",
                            "deepset/roberta-base-squad2",
                            "microsoft/DialoGPT-medium"
                        ],
                        label="Model Selection",
                        value="distilbert-base-cased-distilled-squad",
                        info="Choose a Q&A model"
                    )
                    qa_btn = gr.Button("Get Answer", variant="primary", size="lg")
                    qa_output = gr.Textbox(label="Answer", interactive=False, lines=3)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Text Summarization")
                    summary_input = gr.Textbox(label="Text to Summarize", placeholder="Enter long text to summarize...", lines=6)
                    summary_model = gr.Dropdown(
                        choices=[
                            "facebook/bart-large-cnn",
                            "google/pegasus-xsum",
                            "t5-base"
                        ],
                        label="Model Selection",
                        value="facebook/bart-large-cnn",
                        info="Choose a summarization model"
                    )
                    summary_btn = gr.Button("Summarize", variant="primary", size="lg")
                    summary_output = gr.Textbox(label="Summary", interactive=False, lines=4)
        
        with gr.Tab("Translation & Classification", id="translation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Translation")
                    trans_input = gr.Textbox(label="Text to Translate", placeholder="Enter text to translate...", lines=3)
                    trans_lang = gr.Dropdown(
                        choices=["French", "Spanish", "German", "Italian"], 
                        label="Target Language",
                        value="French"
                    )
                    trans_model = gr.Dropdown(
                        choices=[
                            "Helsinki-NLP/opus-mt-en-fr",
                            "Helsinki-NLP/opus-mt-en-es",
                            "Helsinki-NLP/opus-mt-en-de",
                            "Helsinki-NLP/opus-mt-en-it",
                            "facebook/m2m100_418M"
                        ],
                        label="Model Selection",
                        value="Helsinki-NLP/opus-mt-en-fr",
                        info="Choose a translation model"
                    )
                    trans_btn = gr.Button("Translate", variant="primary", size="lg")
                    trans_output = gr.Textbox(label="Translation", interactive=False, lines=3)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Text Classification")
                    class_input = gr.Textbox(label="Text to Classify", placeholder="Enter text to classify...", lines=3)
                    class_labels = gr.Textbox(
                        label="Categories", 
                        placeholder="technology, sports, politics",
                        info="Enter categories separated by commas"
                    )
                    class_model = gr.Dropdown(
                        choices=[
                            "facebook/bart-large-mnli",
                            "microsoft/DialoGPT-medium",
                            "roberta-base",
                            "distilbert-base-uncased",
                            "xlm-roberta-base"
                        ],
                        label="Model Selection",
                        value="facebook/bart-large-mnli",
                        info="Choose a classification model"
                    )
                    class_btn = gr.Button("Classify", variant="primary", size="lg")
                    class_output = gr.Textbox(label="Classification", interactive=False, lines=4)
        
        with gr.Tab("Text Generation", id="generation"):
            with gr.Column():
                gr.Markdown("### AI Text Generation")
                gen_input = gr.Textbox(
                    label="Prompt", 
                    placeholder="Enter your prompt to generate text...",
                    lines=3
                )
                gen_model = gr.Dropdown(
                    choices=[
                        "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        "microsoft/Phi-3-mini-4k-instruct",
                        "google/gemma-2-2b-it",
                        "Qwen/Qwen2-7B-Instruct",
                        "mistralai/Mixtral-8x7B-Instruct-v0.1"
                    ],
                    label="Model Selection",
                    value="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    info="Choose a language model for text generation"
                )
                with gr.Row():
                    gen_length = gr.Slider(20, 100, value=50, label="Max Length", info="Maximum number of tokens to generate")
                    gen_btn = gr.Button("Generate", variant="primary", size="lg")
                gen_output = gr.Textbox(label="Generated Text", interactive=False, lines=6)
        
        with gr.Tab("Image Generation", id="image"):
            with gr.Column():
                gr.Markdown("### AI Image Generation")
                img_prompt = gr.Textbox(
                    label="Image Description", 
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                img_model = gr.Dropdown(
                    choices=[
                        "runwayml/stable-diffusion-v1-5",
                        "stabilityai/stable-diffusion-2-1",
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        "CompVis/stable-diffusion-v1-4",
                        "runwayml/stable-diffusion-inpainting"
                    ],
                    label="Model Selection",
                    value="runwayml/stable-diffusion-v1-5",
                    info="Choose an image generation model"
                )
                img_btn = gr.Button("Generate Image", variant="primary", size="lg")
                img_output = gr.Image(label="Generated Image", height=400)
                img_status = gr.Textbox(label="Model & Device Info", interactive=False, lines=3)
        
        with gr.Tab("Text-to-Speech", id="tts"):
            with gr.Column():
                gr.Markdown("### AI Voice Generation")
                tts_input = gr.Textbox(
                    label="Text to Convert", 
                    placeholder="Enter text to convert to speech...",
                    lines=4
                )
                tts_model = gr.Dropdown(
                    choices=[
                        "microsoft/speecht5_tts",
                        "facebook/wav2vec2-base-960h",
                        "facebook/hubert-base-ls960",
                        "microsoft/DialoGPT-medium",
                        "facebook/wav2vec2-large-960h-lv60-self"
                    ],
                    label="Model Selection",
                    value="microsoft/speecht5_tts",
                    info="Choose a text-to-speech model"
                )
                tts_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                tts_output = gr.Audio(label="Generated Audio", type="filepath")
                tts_status = gr.Textbox(label="Model & Device Info", interactive=False, lines=3)
    
    # Connect all the functions
    sentiment_btn.click(sentiment_analysis, inputs=[sentiment_input, sentiment_model], outputs=sentiment_output)
    ner_btn.click(named_entity_recognition, inputs=[ner_input, ner_model], outputs=ner_output)
    qa_btn.click(question_answering, inputs=[qa_question, qa_context, qa_model], outputs=qa_output)
    summary_btn.click(text_summarization, inputs=[summary_input, summary_model], outputs=summary_output)
    trans_btn.click(translation, inputs=[trans_input, trans_lang, trans_model], outputs=trans_output)
    class_btn.click(zero_shot_classification, inputs=[class_input, class_labels, class_model], outputs=class_output)
    gen_btn.click(text_generation, inputs=[gen_input, gen_length, gen_model], outputs=gen_output)
    img_btn.click(image_generation, inputs=[img_prompt, img_model], outputs=[img_output, img_status])
    tts_btn.click(text_to_speech, inputs=[tts_input, tts_model], outputs=[tts_output, tts_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
