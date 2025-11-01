# Malaria Detection using Deep Learning

This project implements malaria detection using both TensorFlow and PyTorch deep learning models. The application includes data preprocessing, model training, evaluation, and visualization capabilities.

## Features

- **Multiple Model Architectures**: Simple CNN, ResNet-like models, and Transfer Learning approaches
- **Dual Framework Support**: Both TensorFlow and PyTorch implementations
- **Data Preprocessing**: Image resizing, normalization, augmentation, and grayscale conversion
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, classification reports
- **Visualization**: Training history plots, sample image displays, and preprocessing step visualization

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python app.py
```

## Model Architectures

### TensorFlow Models
- **Simple CNN**: Basic convolutional neural network with 3 Conv2D layers
- **ResNet-like**: Custom residual network with skip connections
- **Transfer Learning**: Pre-trained MobileNetV2 with custom classifier

### PyTorch Models
- **Simple CNN**: Basic CNN with 2 convolutional layers
- **ResNet**: Custom ResNet implementation with residual blocks
- **Transfer Learning**: Pre-trained MobileNetV2 with custom classifier

## Data Requirements

The application can work with malaria cell images in two ways:

### Option 1: Local Dataset
Create a folder structure as follows:
```
cell_images/
├── parasitized/
│   ├── image1.png
│   └── ...
└── uninfected/
    ├── image1.png
    └── ...
```

### Option 2: TensorFlow Datasets (Automatic Fallback)
If no local dataset is found, the application will automatically download and use the malaria dataset from TensorFlow Datasets.

## Testing

Test the dataset loading functionality:
```bash
python test_dataset.py
```

## Output

The application will:
1. Load and preprocess the malaria dataset
2. Train multiple models (TensorFlow and PyTorch)
3. Evaluate model performance
4. Generate visualizations including:
   - Sample images from the dataset
   - Confusion matrices
   - ROC curves
   - Training history plots
   - Classification reports

## Notes

- The application uses only 10% of the dataset for demonstration purposes
- Training epochs are set to 2 for quick execution (can be increased for better performance)
- GPU support is available if CUDA-compatible hardware is present
- All models are saved and can be loaded for inference

## Web Interface (Gradio)

A user-friendly web interface is available for easy image classification:

```bash
python gradio_app.py
```

This will launch a web interface where you can:
- Upload cell images through a web browser
- Get instant predictions on whether cells are parasitized or uninfected
- View confidence scores and probability distributions
- Access the interface from any device on your network

**Features:**
- Interactive image upload
- Real-time predictions
- Visual probability charts
- User-friendly interface
- Network-accessible (configured to run on `0.0.0.0:7860`)

**Note:** For best results, train a model using `app.py` first and save it. You can then modify `gradio_app.py` to load your trained model by specifying the `model_path` parameter.

## Dependencies

See `requirements.txt` for the complete list of required packages.
