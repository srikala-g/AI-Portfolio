import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
import os
from PIL import Image

# Load the model
MODEL_PATH = "model/model-50-dc.h5"
model = load_model(MODEL_PATH)

# Traffic sign class names (43 classes)
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all restrictions", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
]

def auto_crop_and_preprocess(img, target_size=(30, 30)):
    """Auto-crop and preprocess image for traffic sign detection"""
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        orig = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # If no contours found, use the whole image
            cropped = orig
        else:
            # Find the largest contour (assumed to be the sign)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding to the bounding box
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            # Crop the region of interest
            cropped = orig[y:y+h, x:x+w]
        
        # Convert to RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        # Resize to model's input size
        resized = cv2.resize(cropped_rgb, target_size)
        
        # Normalize and expand dims for prediction
        img_array = resized.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, cropped_rgb
        
    except Exception as e:
        print(f"Error in auto_crop_and_preprocess: {e}")
        # Fallback: simple resize
        if isinstance(img, Image.Image):
            img = np.array(img)
        resized = cv2.resize(img, target_size)
        img_array = resized.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img

def simple_preprocess(img, target_size=(30, 30)):
    """Simple preprocessing without auto-crop"""
    try:
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        resized = cv2.resize(img, target_size)
        
        # Normalize and expand dims
        img_array = resized.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
        
    except Exception as e:
        print(f"Error in simple_preprocess: {e}")
        return None, None

def predict_traffic_sign(image, preprocessing_method="Auto-crop"):
    """Predict traffic sign from uploaded image"""
    try:
        if image is None:
            return "Please upload an image", None, None
        
        # Choose preprocessing method
        if preprocessing_method == "Auto-crop":
            img_array, processed_img = auto_crop_and_preprocess(image)
        else:
            img_array, processed_img = simple_preprocess(image)
        
        if img_array is None:
            return "Error processing image", None, None
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Get top prediction
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        predicted_class = CLASS_NAMES[class_idx]
        
        # Get top 3 predictions
        top_3_indices = prediction[0].argsort()[-3:][::-1]
        top_3_results = []
        for i in top_3_indices:
            top_3_results.append({
                'class': CLASS_NAMES[i],
                'confidence': float(prediction[0][i])
            })
        
        # Create results text
        result_text = f"**Predicted Traffic Sign:** {predicted_class}\n"
        result_text += f"**Confidence:** {confidence:.3f}\n\n"
        result_text += "**Top 3 Predictions:**\n"
        for i, result in enumerate(top_3_results, 1):
            result_text += f"{i}. {result['class']}: {result['confidence']:.3f}\n"
        
        return result_text, processed_img, top_3_results
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

def create_confidence_plot(top_3_results):
    """Create a bar plot of top 3 predictions"""
    if not top_3_results:
        return None
    
    classes = [result['class'] for result in top_3_results]
    confidences = [result['confidence'] for result in top_3_results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(classes)), confidences, color=['#2E8B57', '#4682B4', '#CD853F'])
    plt.xlabel('Traffic Sign Classes')
    plt.ylabel('Confidence Score')
    plt.title('Top 3 Predictions Confidence Scores')
    plt.xticks(range(len(classes)), [f"{i+1}" for i in range(len(classes))], rotation=0)
    
    # Add confidence values on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{conf:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Traffic Sign Recognition", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚦 Traffic Sign Recognition System
        
        Upload an image of a traffic sign (JPG, PNG, or PPM format) and get instant recognition results with confidence scores.
        The system can identify 43 different types of traffic signs using a deep learning model.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                image_input = gr.Image(
                    label="Upload Traffic Sign Image (supports JPG, PNG, PPM formats)",
                    type="pil",
                    height=300
                )
                
                preprocessing_dropdown = gr.Dropdown(
                    choices=["Auto-crop", "Simple resize"],
                    value="Auto-crop",
                    label="Preprocessing Method",
                    info="Auto-crop: Automatically detects and crops the sign. Simple resize: Uses the entire image."
                )
                
                predict_btn = gr.Button("🔍 Analyze Traffic Sign", variant="primary", size="lg")
                
                # Example images
                gr.Markdown("### 📸 Example Images")
                # Get current directory and build example paths (using PNG versions for web compatibility)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                example_paths = [
                    os.path.join(current_dir, "examples", "example1.png"),
                    os.path.join(current_dir, "examples", "example2.png"),
                    os.path.join(current_dir, "examples", "example3.png")
                ]
                # Filter to only include existing files
                valid_examples = [[path] for path in example_paths if os.path.exists(path)]
                if valid_examples:
                    gr.Examples(
                        examples=valid_examples,
                        inputs=image_input,
                        label="Try these examples"
                    )
                else:
                    gr.Markdown("*No example images found*")
            
            with gr.Column(scale=1):
                # Output components
                result_text = gr.Markdown(label="Prediction Results")
                
                processed_image = gr.Image(
                    label="Processed Image",
                    height=300
                )
                
                confidence_plot = gr.Plot(
                    label="Confidence Scores"
                )
        
        # Event handlers
        def analyze_image(image, preprocessing):
            result_text, processed_img, top_3 = predict_traffic_sign(image, preprocessing)
            plot = create_confidence_plot(top_3) if top_3 else None
            return result_text, processed_img, plot
        
        predict_btn.click(
            fn=analyze_image,
            inputs=[image_input, preprocessing_dropdown],
            outputs=[result_text, processed_image, confidence_plot]
        )
        
        # Auto-predict on image upload
        image_input.change(
            fn=analyze_image,
            inputs=[image_input, preprocessing_dropdown],
            outputs=[result_text, processed_image, confidence_plot]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **About this model:**
        - Trained on GTSRB (German Traffic Sign Recognition Benchmark) dataset
        - 43 different traffic sign classes
        - Input size: 30x30 pixels
        - Model: CNN with dropout and data augmentation
        - Supports JPG, PNG, and PPM image formats
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
