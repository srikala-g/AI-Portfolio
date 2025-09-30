# Brochure Generator

An AI-powered tool that generates professional company brochures by analyzing website content and extracting relevant information.

## Features

- **Website Analysis**: Automatically scrapes and analyzes company websites
- **Smart Link Detection**: AI-powered identification of relevant pages (About, Careers, etc.)
- **Multi-Shot Prompting**: Enhanced AI performance with multiple examples for better results
- **Professional Brochures**: Generates marketing materials for customers, investors, and recruits
- **Tone Control**: Choose between professional or humorous brochure styles
- **Streaming Output**: Real-time brochure generation with typewriter effect
- **Multiple LLM Calls**: Combines multiple AI calls for comprehensive analysis
- **Company Type Examples**: Specialized examples for tech companies, art galleries, and service businesses

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd brochure_gen
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Web Interface (Recommended)

Launch the Gradio web interface:

```bash
python gradio_app.py
```

Then open your browser to `http://localhost:7863`

**Interface Features:**
- **No Hardcoded URLs**: Fully dynamic input system
- **Company Type Selection**: Choose from tech, gallery, service, or auto-detect
- **Tone Control**: Professional or humorous brochures
- **Example Buttons**: Quick-start examples for different company types
- **Real-time Progress**: Live progress tracking during generation
- **Multi-shot Prompting**: Enhanced AI performance with examples

### Command Line Interface

```python
from brochure_generator import BrochureGenerator

# Create generator
generator = BrochureGenerator()

# Set tone (optional)
generator.set_professional_tone()  # or generator.set_humorous_tone()

# Generate brochure
brochure = generator.create_brochure("HuggingFace", "https://huggingface.co")
print(brochure)
```

### Streaming Output

For real-time generation:

```python
# Stream the brochure generation
for chunk in generator.stream_brochure("HuggingFace", "https://huggingface.co"):
    print(chunk, end="", flush=True)
```

### Run the Example

```bash
python brochure_generator.py
```

## API Reference

### BrochureGenerator Class

#### Methods

- `create_brochure(company_name, url)`: Generate a complete brochure
- `stream_brochure(company_name, url)`: Stream brochure generation
- `get_links(url)`: Get relevant links from a website
- `get_all_details(url)`: Get combined content from main page and links
- `set_professional_tone()`: Set professional tone for generation
- `set_humorous_tone()`: Set humorous tone for generation

### Website Class

- `get_contents()`: Get formatted website contents
- `links`: List of extracted links from the website

## Examples

### Professional Brochure
```python
generator = BrochureGenerator()
generator.set_professional_tone()
brochure = generator.create_brochure("HuggingFace", "https://huggingface.co")
```

### Humorous Brochure
```python
generator = BrochureGenerator()
generator.set_humorous_tone()
brochure = generator.create_brochure("OpenAI", "https://openai.com")
```

### Streaming Generation
```python
generator = BrochureGenerator()
for chunk in generator.stream_brochure("Company Name", "https://company-website.com"):
    print(chunk, end="", flush=True)
```

### Multi-Shot Prompting
```python
generator = BrochureGenerator()

# Demonstrate multi-shot prompting for different company types
brochure = generator.demonstrate_multi_shot_prompting(
    "Tech Company", 
    "https://tech-company.com", 
    company_type="tech"
)

# For art galleries
brochure = generator.demonstrate_multi_shot_prompting(
    "Art Gallery", 
    "https://gallery.com", 
    company_type="gallery"
)

# For service companies
brochure = generator.demonstrate_multi_shot_prompting(
    "Service Company", 
    "https://service-company.com", 
    company_type="service"
)
```

## How It Works

1. **Website Scraping**: Extracts content and links from the main website
2. **Multi-Shot Link Analysis**: Uses GPT-4o-mini with multiple examples to identify relevant pages
3. **Content Aggregation**: Combines content from main page and relevant linked pages
4. **Multi-Shot Brochure Generation**: Uses AI with multiple examples to create professional brochures
5. **Company Type Optimization**: Applies specialized examples based on company type (tech, gallery, service)

## Business Applications

- **Marketing**: Generate company brochures for prospective clients
- **Investor Relations**: Create materials for potential investors
- **Recruitment**: Develop content for attracting talent
- **Content Generation**: Automate marketing material creation

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for website scraping

## Dependencies

- `requests`: HTTP requests for website scraping
- `beautifulsoup4`: HTML parsing
- `openai`: OpenAI API client
- `python-dotenv`: Environment variable management

## Files

- `brochure_generator.py` - Main module with core functionality
- `gradio_app.py` - Web interface with Gradio
- `requirements.txt` - Dependencies
- `README.md` - Documentation

## License

MIT License
