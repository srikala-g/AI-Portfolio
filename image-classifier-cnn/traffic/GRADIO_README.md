# 🚦 Traffic Sign Recognition - Gradio Web App

A modern, interactive web application for traffic sign recognition using Gradio and a trained CNN model.

## Features

- **Interactive Web Interface**: Easy-to-use web interface built with Gradio
- **Real-time Predictions**: Instant traffic sign recognition with confidence scores
- **Auto-crop Preprocessing**: Automatically detects and crops traffic signs from images
- **Top 3 Predictions**: Shows the most likely traffic sign classifications
- **Visual Confidence Scores**: Bar chart visualization of prediction confidence
- **Multiple Preprocessing Options**: Choose between auto-crop or simple resize
- **Example Images**: Built-in examples to test the model

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model File**:
   Ensure `model/model-50-dc.h5` exists in the traffic directory.

## Usage

### Running the Gradio App

```bash
python gradio_app.py
```

The app will launch at `http://localhost:7860` by default.

### Using the Interface

1. **Upload Image**: Click "Upload Traffic Sign Image" to select an image
2. **Choose Preprocessing**: Select between "Auto-crop" or "Simple resize"
3. **Get Results**: The app automatically analyzes the image and shows:
   - Predicted traffic sign class
   - Confidence score
   - Top 3 predictions
   - Processed image
   - Confidence visualization

### Supported Image Formats

- PNG
- JPEG/JPG
- BMP
- PPM (German Traffic Sign Benchmark format)

## Model Information

- **Model**: CNN with dropout and data augmentation
- **Classes**: 43 different traffic sign types
- **Input Size**: 30x30 pixels
- **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)

### Traffic Sign Classes

The model can recognize 43 different traffic signs including:
- Speed limits (20, 30, 50, 60, 70, 80, 100, 120 km/h)
- Traffic regulations (Stop, Yield, No entry, etc.)
- Warning signs (Curves, Road work, Pedestrians, etc.)
- Directional signs (Turn left/right, Ahead only, etc.)

## Preprocessing Methods

### Auto-crop (Recommended)
- Automatically detects traffic sign boundaries using edge detection
- Crops the sign from the background
- Best for images with multiple objects or complex backgrounds

### Simple Resize
- Uses the entire uploaded image
- Resizes to 30x30 pixels
- Best for images that are already cropped to show only the sign

## API Usage

You can also use the prediction functions programmatically:

```python
from gradio_app import predict_traffic_sign, auto_crop_and_preprocess

# Load and predict
result_text, processed_img, top_3 = predict_traffic_sign(
    image_path, 
    preprocessing_method="Auto-crop"
)
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `model/model-50-dc.h5` exists
2. **No contours detected**: Try the "Simple resize" preprocessing method
3. **Low confidence**: Ensure the image shows a clear traffic sign

### Performance Tips

- Use high-quality, well-lit images
- Ensure traffic signs are clearly visible
- Try both preprocessing methods for best results

## Development

### File Structure
```
traffic/
├── gradio_app.py          # Main Gradio application
├── model/
│   └── model-50-dc.h5     # Trained CNN model
├── requirements.txt        # Python dependencies
├── GRADIO_README.md       # This documentation
└── gtsrb/                 # Example images
```

### Customization

To modify the interface:
1. Edit `gradio_app.py`
2. Modify the `create_interface()` function
3. Add new components or change the layout

## Comparison with Other Interfaces

| Feature | Gradio App | Tkinter GUI | Command Line |
|---------|------------|-------------|--------------|
| Web Interface | ✅ | ❌ | ❌ |
| Auto-crop | ✅ | ✅ | ✅ |
| Confidence Plot | ✅ | ❌ | ❌ |
| Top 3 Predictions | ✅ | ❌ | ✅ |
| Easy Sharing | ✅ | ❌ | ❌ |
| Mobile Friendly | ✅ | ❌ | ❌ |

## License

This project is part of the AI Portfolio collection. See the main repository for license information.
