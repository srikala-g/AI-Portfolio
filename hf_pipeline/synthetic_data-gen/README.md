# 🎲 Synthetic Data Generator

A powerful AI-powered application for generating realistic synthetic data using various machine learning models. Perfect for testing, training, research, and data augmentation.

## ✨ Features

### 📝 Text Generation
- **Multiple Language Models**: GPT-2, Phi-3, Gemma, Llama, and more
- **Customizable Parameters**: Temperature, length, number of samples
- **Export Options**: JSON, TXT formats
- **Use Cases**: Stories, articles, conversations, product descriptions

### 📊 Tabular Data Generation
- **Customer Data**: Names, emails, demographics, purchase behavior
- **Sales Data**: Products, transactions, salesperson info, regions
- **Sensor Data**: Temperature, humidity, pressure with realistic patterns
- **Financial Data**: Transactions, accounts, amounts, categories
- **Export Options**: CSV, JSON formats

### 🖼️ Image Generation
- **Diffusion Models**: Stable Diffusion, Kandinsky
- **Customizable**: Size, steps, prompts
- **Multiple Images**: Batch generation
- **Export Options**: PNG, JPG formats

### ⚙️ Model Management
- **Dynamic Loading**: Load models on demand
- **Memory Efficient**: Automatic model caching
- **Device Support**: CPU, CUDA, Apple Silicon
- **Status Monitoring**: Track model loading status

## 🚀 Quick Start

### Installation

1. **Clone or download the files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python synthetic_data_generator.py
   ```

4. **Open your browser** and go to `http://localhost:7860`

### Basic Usage

1. **Choose Data Type**: Select from Text, Tabular, or Image generation
2. **Configure Parameters**: Set model, samples, and other options
3. **Generate Data**: Click the generate button
4. **Export Results**: Download data in your preferred format

## 📋 Data Types Supported

### Text Data
- **Stories**: Creative writing, narratives
- **Articles**: News, blog posts, technical content
- **Conversations**: Chat logs, customer service
- **Product Descriptions**: E-commerce listings
- **Code**: Programming examples, documentation

### Tabular Data
- **Customer Data**: Demographics, behavior patterns
- **Sales Data**: Transactions, products, regions
- **Sensor Data**: IoT devices, environmental monitoring
- **Financial Data**: Banking, transactions, accounts
- **Medical Data**: Patient records, test results
- **E-commerce Data**: Orders, products, reviews

### Image Data
- **Photographs**: Landscapes, portraits, objects
- **Artwork**: Paintings, illustrations, designs
- **Diagrams**: Technical drawings, flowcharts
- **Logos**: Branding, marketing materials
- **Screenshots**: UI mockups, interfaces

## 🎯 Use Cases

### Testing & Development
- **Unit Testing**: Generate test data for applications
- **Load Testing**: Create large datasets for performance testing
- **A/B Testing**: Generate variations for experiments
- **Mock Data**: Replace real data during development

### Machine Learning
- **Data Augmentation**: Increase training dataset size
- **Synthetic Training**: Create labeled data for models
- **Privacy Protection**: Generate anonymized datasets
- **Bias Testing**: Create diverse datasets for fairness testing

### Research & Analysis
- **Simulation**: Model real-world scenarios
- **Benchmarking**: Compare algorithms on synthetic data
- **Prototype Development**: Test ideas without real data
- **Academic Research**: Generate datasets for studies

## ⚙️ Configuration

### Model Selection
Choose the best model for your needs:

**Text Generation**:
- **GPT-2 Small**: Fast, lightweight
- **GPT-2 Medium**: Balanced performance
- **Phi-3 Mini**: Microsoft's efficient model
- **Gemma 2B**: Google's lightweight model
- **Llama 3.1 8B**: Meta's powerful model

**Image Generation**:
- **Stable Diffusion v1.5**: Reliable, well-tested
- **Stable Diffusion v2.1**: Improved quality
- **Stable Diffusion XL**: High resolution
- **Kandinsky 2.2**: Alternative approach

### Performance Tips

1. **Use smaller models** for faster generation
2. **Pre-load models** to avoid loading delays
3. **Use GPU** when available for better performance
4. **Batch generation** for multiple samples
5. **Adjust parameters** for quality vs speed trade-offs

## 📁 File Structure

```
synthetic_data_generator/
├── synthetic_data_generator.py    # Main application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── examples/                     # Example outputs (optional)
    ├── text_samples/
    ├── tabular_samples/
    └── image_samples/
```

## 🔧 Advanced Usage

### Custom Data Types
You can extend the generator by adding new data types:

```python
def generate_custom_data(data_type, num_rows, config):
    # Your custom data generation logic
    pass
```

### API Integration
Use the functions in your own applications:

```python
from synthetic_data_generator import generate_text_data, generate_tabular_data

# Generate text
text_data = generate_text_data("Write a story about...", "GPT-2 Small", 5, 200, 0.8)

# Generate tabular data
tabular_data = generate_tabular_data("Customer Data", 1000, {})
```

### Batch Processing
Generate large datasets efficiently:

```python
# Generate 1000 customer records
customer_data = generate_tabular_data("Customer Data", 1000, {})

# Export to CSV
export_data(customer_data, "CSV", "customers")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory**: Use smaller models or reduce batch size
2. **Model Loading Errors**: Check internet connection and Hugging Face access
3. **Slow Generation**: Use GPU when available, pre-load models
4. **Export Errors**: Check file permissions and disk space

### Performance Optimization

1. **Model Caching**: Models are cached after first load
2. **Device Selection**: Automatically uses best available device
3. **Memory Management**: Models are loaded/unloaded as needed
4. **Batch Processing**: Generate multiple samples efficiently

## 📊 Example Outputs

### Text Generation
```json
{
  "id": 1,
  "prompt": "Write a story about...",
  "generated_text": "Once upon a time, in a distant land...",
  "model": "GPT-2 Small",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Tabular Data
```csv
customer_id,name,email,age,city,income
CUST_000001,Customer 1,customer1@example.com,25,New York,75000
CUST_000002,Customer 2,customer2@example.com,34,Los Angeles,85000
```

### Image Generation
- High-quality images based on text prompts
- Multiple formats (PNG, JPG)
- Customizable resolution and style

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the generator.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Hugging Face for providing the models and infrastructure
- Gradio for the excellent web interface framework
- The open-source community for the amazing tools and libraries


