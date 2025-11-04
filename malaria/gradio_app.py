#!/usr/bin/env python3
"""
Gradio User Interface for Malaria Detection
==========================================

A web-based interface for malaria detection using deep learning models.
Users can upload images and get predictions on whether cells are parasitized or uninfected.
"""

import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from PIL import Image
import gradio as gr

# Suppress INFO & WARNING logs from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Global variables for model and preprocessing
MODEL = None
MODEL_INFO = None
INPUT_SHAPE = (64, 64, 3)
PREDICT_FN = None  # Compiled prediction function for speed


def create_cnn_model(input_shape):
    """Create a simple CNN model for malaria detection."""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_resnet_model(input_shape):
    """Create a ResNet-like model for malaria detection."""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolutional layers
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)
    
    # Residual block 1
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])
    
    # Residual block 2
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])
    
    # Final layers
    x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model


def create_mobilenetv2_model(input_shape):
    """Create a MobileNetV2 model for malaria detection using transfer learning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model initially
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model


def load_or_create_model(model_path=None, model_type="CNN"):
    """Load a pre-trained model or create a new one.
    
    Args:
        model_path: Path to saved model file (optional)
        model_type: Type of model to create ("CNN" or "ResNet")
    """
    global MODEL, INPUT_SHAPE
    
    # Default model info structure
    if model_type == "ResNet":
        model_info = {
            'name': 'ResNet Model',
            'architecture': 'ResNet-like with Residual Blocks',
            'layers': [
                'Conv2D(32, 3x3) + ReLU',
                'Conv2D(64, 3x3) + ReLU',
                'MaxPooling2D(3x3)',
                'Residual Block 1 (skip connections)',
                'Residual Block 2 (skip connections)',
                'GlobalAveragePooling2D',
                'Dense(256) + ReLU + Dropout(0.5)',
                'Dense(1) + Sigmoid'
            ],
            'input_shape': INPUT_SHAPE,
            'trained': False
        }
    elif model_type == "MobileNetV2":
        model_info = {
            'name': 'MobileNetV2 Model',
            'architecture': 'Transfer Learning (MobileNetV2)',
            'layers': [
                'MobileNetV2 Base (ImageNet weights, frozen)',
                'GlobalAveragePooling2D',
                'Dense(128) + ReLU',
                'Dropout(0.2)',
                'Dense(1) + Sigmoid'
            ],
            'input_shape': INPUT_SHAPE,
            'trained': False
        }
    else:  # CNN
        model_info = {
            'name': 'CNN Model',
            'architecture': 'Sequential',
            'layers': [
                'Conv2D(32, 3x3) + ReLU',
                'MaxPooling2D(2x2)',
                'Conv2D(64, 3x3) + ReLU',
                'MaxPooling2D(2x2)',
                'Conv2D(64, 3x3) + ReLU',
                'Flatten',
                'Dense(64) + ReLU',
                'Dense(1) + Sigmoid'
            ],
            'input_shape': INPUT_SHAPE,
            'trained': False
        }
    
    if model_path and os.path.exists(model_path):
        try:
            MODEL = tf.keras.models.load_model(model_path)
            INPUT_SHAPE = MODEL.input_shape[1:]  # Get input shape from model
            # Try to detect model type from loaded model
            if 'resnet' in model_path.lower() or 'residual' in str(MODEL.summary()).lower():
                model_info['name'] = 'ResNet Model'
                model_info['architecture'] = 'ResNet-like with Residual Blocks'
            elif 'mobilenet' in model_path.lower():
                model_info['name'] = 'MobileNetV2 Model'
                model_info['architecture'] = 'Transfer Learning (MobileNetV2)'
            model_info['trained'] = True
            model_info['saved_path'] = model_path
            model_info['source'] = f'Loaded from {model_path}'
            print(f"Model loaded from {model_path}")
            return MODEL, model_info
        except Exception as e:
            print(f"Error loading model: {e}. Creating new model...")
    
    # Create a new model based on type
    if model_type == "ResNet":
        MODEL = create_resnet_model(INPUT_SHAPE)
    elif model_type == "MobileNetV2":
        MODEL = create_mobilenetv2_model(INPUT_SHAPE)
    else:  # CNN
        MODEL = create_cnn_model(INPUT_SHAPE)
    
    # Compile the model
    MODEL.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    model_info['source'] = f'Created new {model_type} model (untrained)'
    print(f"New {model_type} model created (not trained - for demonstration)")
    return MODEL, model_info


def preprocess_image(image):
    """Preprocess an image for model prediction."""
    if image is None:
        return None
    
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is in the correct format (RGB)
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    # Resize to model input shape
    image_resized = tf.image.resize(image, INPUT_SHAPE[:2])
    
    # Normalize pixel values
    image_normalized = tf.image.per_image_standardization(image_resized)
    
    # Add batch dimension
    image_batch = tf.expand_dims(image_normalized, axis=0)
    
    return image_batch


def predict_malaria(image, progress=None):
    """Predict if an image shows parasitized or uninfected cells."""
    global MODEL, MODEL_INFO
    
    print("\n" + "="*70)
    print("PREDICT_MALARIA CALLED")
    print("="*70)
    print(f"Image type: {type(image)}")
    print(f"Image is None: {image is None}")
    print(f"MODEL is None: {MODEL is None}")
    print(f"MODEL_INFO is None: {MODEL_INFO is None}")
    
    steps_log = []
    
    try:
        # Helper to update progress
        def update_progress(p, desc):
            if progress:
                try:
                    progress(p, desc=desc)
                    print(f"[PROGRESS] {p*100:.1f}% - {desc}")
                except Exception as e:
                    print(f"[PROGRESS ERROR] Failed to update progress: {e}")
        
        # Step 0: Initialize
        print("\n[STEP 1/6] Initializing analysis...")
        update_progress(0, "Initializing...")
        steps_log.append("="*60)
        steps_log.append("MALARIA DETECTION - ANALYSIS STARTED")
        steps_log.append("="*60)
        steps_log.append("\n🔵 Step 1/6: Initializing analysis...")
        
        # Step 1: Load model if needed
        if MODEL is None or MODEL_INFO is None:
            print("[STEP 2/6] Loading/Creating model...")
            update_progress(0.05, "Loading model...")
            steps_log.append("\n📦 Step 2/6: Loading/Creating model...")
            MODEL, MODEL_INFO = load_or_create_model()
            print(f"✓ Model loaded successfully")
            print(f"  - Name: {MODEL_INFO['name']}")
            print(f"  - Architecture: {MODEL_INFO['architecture']}")
            print(f"  - Input shape: {MODEL_INFO['input_shape']}")
            print(f"  - Trained: {MODEL_INFO['trained']}")
            if MODEL is None:
                raise ValueError("Failed to load or create model")
            steps_log.append(f"   ✓ Model loaded: {MODEL_INFO['name']}")
            steps_log.append(f"   ✓ Architecture: {MODEL_INFO['architecture']}")
            steps_log.append(f"   ✓ Input shape: {MODEL_INFO['input_shape']}")
            steps_log.append(f"   ✓ Status: {'Trained' if MODEL_INFO['trained'] else 'Untrained (demo)'}")
        else:
            print("[STEP 2/6] Model already loaded, skipping load...")
            print(f"  - Model: {MODEL_INFO['name']}")
            print(f"  - Architecture: {MODEL_INFO['architecture']}")
            print(f"  - Trained: {MODEL_INFO['trained']}")
            steps_log.append(f"\n✓ Model Status:")
            steps_log.append(f"   • Model: {MODEL_INFO['name']}")
            steps_log.append(f"   • Architecture: {MODEL_INFO['architecture']}")
            steps_log.append(f"   • Input Shape: {MODEL_INFO['input_shape']}")
            steps_log.append(f"   • Status: {'✅ Trained Model' if MODEL_INFO['trained'] else '⚠️ Untrained (Demo)'}")
            steps_log.append(f"   • Model loaded and ready for inference")
        
        if image is None:
            print("ERROR: Image is None")
            error_msg = "❌ Please upload an image to begin analysis."
            return ("No image provided", error_msg, None, "No image provided")
        
        # Step 2: Validate image
        print("\n[STEP 3/6] Validating image...")
        update_progress(0.1, "Validating image...")
        steps_log.append("\n✅ Step 3/6: Validating image...")
        
        if not isinstance(image, (Image.Image, np.ndarray)):
            print(f"ERROR: Invalid image type: {type(image)}")
            error_msg = "❌ Invalid image format. Please upload a valid image file."
            return ("Invalid image format", error_msg, None, "Invalid format")
        
        image_shape = image.size if isinstance(image, Image.Image) else image.shape
        print(f"✓ Image validated - Size: {image_shape}")
        steps_log.append(f"   ✓ Image validated")
        steps_log.append(f"   ✓ Image size: {image_shape}")
        
        # Step 3: Preprocess image
        print("\n[STEP 4/6] Preprocessing image...")
        update_progress(0.2, "Preprocessing image...")
        steps_log.append("\n🔄 Step 4/6: Preprocessing image...")
        steps_log.append("   • Converting to RGB format...")
        steps_log.append("   • Resizing to 64x64 pixels...")
        steps_log.append("   • Normalizing pixel values...")
        
        processed_image = preprocess_image(image)
        print(f"✓ Preprocessing complete - Processed image shape: {processed_image.shape if processed_image is not None else 'None'}")
        
        if processed_image is None:
            print("ERROR: Preprocessing returned None")
            error_msg = "❌ Failed to preprocess image. Please try a different image."
            return ("Error processing image", error_msg, None, "Preprocessing failed")
        
        steps_log.append("   ✓ Image preprocessing complete")
        
        # Step 4: Run model prediction (this is usually the slowest part)
        print("\n[STEP 5/6] Running model prediction...")
        update_progress(0.4, "Running model prediction...")
        steps_log.append("\n🧠 Step 5/6: Running deep learning model...")
        if MODEL_INFO:
            steps_log.append(f"   • Model: {MODEL_INFO['name']}")
            steps_log.append(f"   • Architecture: {MODEL_INFO['architecture']}")
            steps_log.append(f"   • Input Shape: {MODEL_INFO['input_shape']}")
            steps_log.append(f"   • Status: {'Trained Model' if MODEL_INFO['trained'] else 'Untrained (Demo)'}")
        steps_log.append("   • Processing through neural network layers...")
        steps_log.append("   • Executing inference...")
        
        # Validate MODEL before using
        if MODEL is None:
            print("ERROR: MODEL is None - cannot make prediction")
            raise ValueError("Model is None - cannot make prediction")
        
        print(f"  - Model input shape: {MODEL.input_shape}")
        print(f"  - Processed image shape: {processed_image.shape}")
        print("  - Executing model inference...")
        update_progress(0.6, "Computing prediction...")
        
        # Use direct model call for faster inference
        # This is faster than MODEL.predict() for single images
        prediction = MODEL(processed_image, training=False)
        # Model output directly represents Parasitized/Infected probability
        probability = float(prediction[0][0])
        print(f"✓ Model prediction completed - Infected (parasitized) probability: {probability:.6f}")
        
        update_progress(0.85, "Processing results...")
        
        steps_log.append("   ✓ Model prediction completed")
        steps_log.append(f"   ✓ Infected (parasitized) probability: {probability:.6f}")
        
        # Step 5: Process results
        print("\n[STEP 6/6] Processing results...")
        steps_log.append("\n📊 Step 6/6: Processing results...")
        
        # Determine class and confidence
        # probability now always represents Parasitized/Infected probability
        if probability >= 0.5:
            label = "Parasitized"
            confidence = probability
            status = "⚠️ MALARIA DETECTED"
            color = "#FF0000"  # Red
        else:
            label = "Uninfected"
            confidence = 1.0 - probability
            status = "✅ NO MALARIA"
            color = "#00FF00"  # Green
        
        print(f"  - Classification: {label}")
        print(f"  - Confidence: {confidence:.2%}")
        steps_log.append(f"   • Classification: {label}")
        steps_log.append(f"   • Confidence: {confidence:.2%}")
        
        # Format output
        result_text = f"""
        <div style="text-align: center; padding: 20px;">
            <h2 style="color: {color}; font-size: 24px;">{status}</h2>
            <p style="font-size: 18px;">Classification: <strong>{label}</strong></p>
            <p style="font-size: 16px;">Confidence: <strong>{confidence:.2%}</strong></p>
        </div>
        """
        
        # Create visualization
        # probability now always represents Parasitized/Infected probability
        prob_parasitized = probability * 100
        prob_uninfected = (1 - probability) * 100
        
        # Final status with complete breakdown
        steps_log.append("\n" + "="*50)
        steps_log.append("✅ ANALYSIS COMPLETE!")
        steps_log.append("="*50)
        if MODEL_INFO:
            steps_log.append(f"\n📋 Model Information:")
            steps_log.append(f"   • Name: {MODEL_INFO['name']}")
            steps_log.append(f"   • Architecture: {MODEL_INFO['architecture']}")
            steps_log.append(f"   • Input Shape: {MODEL_INFO['input_shape']}")
            steps_log.append(f"   • Status: {'Trained Model' if MODEL_INFO['trained'] else 'Untrained (Demo)'}")
        steps_log.append(f"\n🔍 Prediction Results:")
        steps_log.append(f"   • Classification: {label}")
        steps_log.append(f"   • Confidence: {confidence:.2%}")
        steps_log.append(f"   • Probability (Parasitized): {prob_parasitized:.2f}%")
        steps_log.append(f"   • Probability (Uninfected): {prob_uninfected:.2f}%")
        
        final_status = "\n".join(steps_log)
        print("\n✓ Analysis complete - Preparing return values...")
        
        # Format probability data for BarPlot (expects pandas DataFrame)
        probability_data = pd.DataFrame([
            {"Label": "Parasitized", "Probability": prob_parasitized},
            {"Label": "Uninfected", "Probability": prob_uninfected}
        ])
        print(f"  - Probability data formatted as DataFrame:")
        print(f"    {probability_data}")
        
        update_progress(1.0, "Complete!")
        print("="*70)
        print("PREDICT_MALARIA COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return_tuple = (result_text, 
                       final_status, 
                       probability_data, 
                       f"Classification: {label}\nConfidence: {confidence:.2%}\n\nProbability Breakdown:\n• Parasitized: {prob_parasitized:.2f}%\n• Uninfected: {prob_uninfected:.2f}%")
        
        print(f"[RETURN] Returning tuple with {len(return_tuple)} elements")
        return return_tuple
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("\n" + "="*70)
        print("ERROR in predict_malaria:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("\nFull traceback:")
        print(error_details)
        print("="*70 + "\n")
        error_status = f"❌ ERROR OCCURRED:\n{str(e)}\n\nDetailed error:\n{error_details}"
        steps_log.append(f"\n❌ Error: {str(e)}")
        error_html = f"<div style='color: red; padding: 20px;'>Error: {str(e)}<br><br>Check the status log for details.</div>"
        return (error_html, 
                "\n".join(steps_log) + "\n" + error_status, 
                None, 
                f"Error: {str(e)}")


def augment_train_images_func(sample, img_size=(64, 64)):
    """Augment training images."""
    image = sample["image"]
    label = sample["label"]
    image = tf.image.resize(image, img_size)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    image = tf.image.per_image_standardization(image)
    return image, label

def augment_val_test_images_func(sample, img_size=(64, 64)):
    """Augment validation/test images."""
    image = sample["image"]
    label = sample["label"]
    image = tf.image.resize(image, img_size)
    image = tf.image.per_image_standardization(image)
    return image, label

def train_model(data_path, epochs, batch_size, train_ratio, val_ratio, model_type="CNN"):
    """Train the malaria detection model.
    
    Args:
        data_path: Path to training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        model_type: Type of model ("CNN" or "ResNet")
    """
    global MODEL, MODEL_INFO
    
    steps_log = []
    
    try:
        steps_log.append("🔵 Step 1/8: Initializing training...")
        steps_log.append(f"   • Model type: {model_type}")
        steps_log.append(f"   • Epochs: {epochs}")
        steps_log.append(f"   • Batch size: {batch_size}")
        steps_log.append(f"   • Train ratio: {train_ratio}")
        steps_log.append(f"   • Validation ratio: {val_ratio}")
        time.sleep(0.2)
        
        # Load dataset
        steps_log.append("\n📦 Step 2/8: Loading dataset...")
        steps_log.append(f"   • Data path: {data_path}")
        
        # Suppress dataset loading messages
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            if not os.path.exists(data_path):
                steps_log.append("   ⚠️ Local path not found, using TensorFlow Datasets...")
                malaria_builder = tfds.builder("malaria")
                malaria_builder.download_and_prepare()
                malaria_dataset = malaria_builder.as_dataset(split="train")
            else:
                try:
                    malaria_folder = tfds.ImageFolder(data_path)
                    dataset_dict = malaria_folder.as_dataset()
                    
                    if isinstance(dataset_dict, dict):
                        if 'train' in dataset_dict:
                            malaria_dataset = dataset_dict['train']
                            steps_log.append("   ✓ Using 'train' split from dataset")
                        elif len(dataset_dict) > 0:
                            # Get the first available split
                            malaria_dataset = list(dataset_dict.values())[0]
                            split_name = list(dataset_dict.keys())[0]
                            steps_log.append(f"   ✓ Using '{split_name}' split from dataset")
                        else:
                            raise ValueError("Dataset dictionary is empty. No splits found.")
                    else:
                        malaria_dataset = dataset_dict
                        steps_log.append("   ✓ Using dataset directly")
                        
                except Exception as e:
                    steps_log.append(f"   ⚠️ Error loading local dataset: {str(e)}")
                    steps_log.append("   • Falling back to TensorFlow Datasets...")
                    malaria_builder = tfds.builder("malaria")
                    malaria_builder.download_and_prepare()
                    malaria_dataset = malaria_builder.as_dataset(split="train")
        
        # Validate dataset
        if malaria_dataset is None:
            raise ValueError("Failed to load dataset. Please check your data path or internet connection.")
        
        dataset_size = len(malaria_dataset)
        if dataset_size == 0:
            raise ValueError("Dataset is empty. Please check your data path contains valid images.")
        
        steps_log.append(f"   ✓ Dataset loaded: {dataset_size} samples")
        time.sleep(0.2)
        
        # Split dataset
        steps_log.append("\n✂️ Step 3/8: Splitting dataset...")
        malaria_dataset = malaria_dataset.shuffle(buffer_size=10000)
        malaria_dataset = malaria_dataset.take(len(malaria_dataset) // 10)  # Use 10% for demo
        
        train_size = int(train_ratio * len(malaria_dataset))
        val_size = int(val_ratio * len(malaria_dataset))
        test_size = len(malaria_dataset) - train_size - val_size
        
        train_dataset = malaria_dataset.take(train_size)
        val_test_dataset = malaria_dataset.skip(train_size)
        val_dataset = val_test_dataset.skip(val_size)
        test_dataset = val_test_dataset.take(test_size)
        
        steps_log.append(f"   ✓ Train: {train_size}, Val: {val_size}, Test: {test_size}")
        time.sleep(0.2)
        
        # Preprocess and augment
        steps_log.append("\n🔄 Step 4/8: Preprocessing and augmenting images...")
        IMG_SIZE = (64, 64)
        augmented_train = train_dataset.map(lambda x: augment_train_images_func(x, IMG_SIZE)).batch(batch_size)
        resized_val = val_dataset.map(lambda x: augment_val_test_images_func(x, IMG_SIZE)).batch(batch_size)
        steps_log.append("   ✓ Images preprocessed and batched")
        time.sleep(0.2)
        
        # Get input shape
        steps_log.append("\n⚙️ Step 5/8: Setting up model...")
        for image_batch, _ in augmented_train.take(1):
            input_shape = image_batch[0].shape
            break
        
        # Create or reset model based on type
        if model_type == "ResNet":
            MODEL = create_resnet_model(input_shape)
            model_name = 'ResNet Model'
            model_architecture = 'ResNet-like with Residual Blocks'
        elif model_type == "MobileNetV2":
            MODEL = create_mobilenetv2_model(input_shape)
            model_name = 'MobileNetV2 Model'
            model_architecture = 'Transfer Learning (MobileNetV2)'
        else:  # CNN
            MODEL = create_cnn_model(input_shape)
            model_name = 'CNN Model'
            model_architecture = 'Sequential'
        
        MODEL.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy', 'precision', 'recall']
        )
        
        MODEL_INFO = {
            'name': model_name,
            'architecture': model_architecture,
            'input_shape': input_shape,
            'trained': True,
            'source': f'Trained via Gradio interface ({model_type})'
        }
        
        steps_log.append(f"   ✓ Model created with input shape: {input_shape}")
        time.sleep(0.2)
        
        # Train model
        steps_log.append(f"\n🚀 Step 6/8: Training model for {epochs} epochs...")
        steps_log.append("   • This may take several minutes...")
        
        # Suppress all training output to keep terminal clean
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        
        # Redirect both stdout and stderr to suppress all training messages
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Train model silently
            history = MODEL.fit(
                augmented_train,
                validation_data=resized_val,
                epochs=epochs,
                verbose=0  # No console output
            )
        
        # Log epoch results
        for epoch in range(epochs):
            epoch_num = epoch + 1
            if epoch_num <= len(history.history['loss']):
                steps_log.append(f"   • Epoch {epoch_num}/{epochs} - Loss: {history.history['loss'][epoch]:.4f}, "
                                f"Acc: {history.history['accuracy'][epoch]:.4f}, "
                                f"Val Loss: {history.history['val_loss'][epoch]:.4f}, "
                                f"Val Acc: {history.history['val_accuracy'][epoch]:.4f}")
        
        steps_log.append("   ✓ Training completed!")
        time.sleep(0.2)
        
        # Save model
        steps_log.append("\n💾 Step 7/8: Saving model...")
        if model_type == "MobileNetV2":
            model_path = "malaria_model_mobilenetv2_trained.h5"
        else:
            model_path = f"malaria_model_{model_type.lower()}_trained.h5"
        MODEL.save(model_path)
        steps_log.append(f"   ✓ Model saved to: {model_path}")
        
        # Update MODEL_INFO with saved path
        MODEL_INFO['saved_path'] = model_path
        MODEL_INFO['trained'] = True
        MODEL_INFO['source'] = f'Saved as {model_path}'
        
        time.sleep(0.2)
        
        # Final summary
        steps_log.append("\n" + "="*50)
        steps_log.append("✅ TRAINING COMPLETE!")
        steps_log.append("="*50)
        steps_log.append(f"\n📊 Final Metrics:")
        steps_log.append(f"   • Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        steps_log.append(f"   • Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        steps_log.append(f"   • Final Training Loss: {history.history['loss'][-1]:.4f}")
        steps_log.append(f"   • Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        steps_log.append(f"\n💾 Model saved to: {model_path}")
        steps_log.append("\nYou can now use this trained model in the Testing tab!")
        
        final_log = "\n".join(steps_log)
        return final_log, history.history['accuracy'][-1], history.history['val_accuracy'][-1]
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        steps_log.append(f"\n❌ ERROR: {str(e)}")
        steps_log.append(f"\nDetails:\n{error_details}")
        return "\n".join(steps_log), 0.0, 0.0


def create_interface():
    """Create the Gradio interface with Training and Testing tabs."""
    # Load or create model (check for saved trained model first)
    global MODEL, MODEL_INFO
    
    # Check if a trained model exists (prefer CNN, then ResNet, then MobileNetV2)
    saved_model_path_cnn = "malaria_model_cnn_trained.h5"
    saved_model_path_resnet = "malaria_model_resnet_trained.h5"
    saved_model_path_mobilenet = "malaria_model_mobilenetv2_trained.h5"
    saved_model_path_legacy = "malaria_model_trained.h5"  # Legacy path
    
    if os.path.exists(saved_model_path_cnn):
        print(f"Found saved CNN model at {saved_model_path_cnn}, loading...")
        MODEL, MODEL_INFO = load_or_create_model(saved_model_path_cnn, "CNN")
    elif os.path.exists(saved_model_path_resnet):
        print(f"Found saved ResNet model at {saved_model_path_resnet}, loading...")
        MODEL, MODEL_INFO = load_or_create_model(saved_model_path_resnet, "ResNet")
    elif os.path.exists(saved_model_path_mobilenet):
        print(f"Found saved MobileNetV2 model at {saved_model_path_mobilenet}, loading...")
        MODEL, MODEL_INFO = load_or_create_model(saved_model_path_mobilenet, "MobileNetV2")
    elif os.path.exists(saved_model_path_legacy):
        print(f"Found legacy saved model at {saved_model_path_legacy}, loading...")
        MODEL, MODEL_INFO = load_or_create_model(saved_model_path_legacy, "CNN")
    else:
        MODEL, MODEL_INFO = load_or_create_model(None, "CNN")
    
    print(f"Model initialized: {MODEL_INFO['name']}, Trained: {MODEL_INFO['trained']}")
    
    # Warm up and compile the model for faster inference
    print("Warming up model and optimizing for inference...")
    try:
        dummy_input = tf.zeros((1, 64, 64, 3))
        # First call to compile the graph
        _ = MODEL(dummy_input, training=False)
        # Second call should be faster (warmed up)
        _ = MODEL(dummy_input, training=False)
        print("✓ Model warmed up and optimized successfully!")
    except Exception as e:
        print(f"⚠ Warning: Could not warm up model: {e}")
    
    # Create Gradio interface
    with gr.Blocks(title="Malaria Detection System", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🔬 Malaria Detection System
            ### Deep Learning-Based Cell Image Classification
            
            Train your own model or test with pre-trained models to classify blood cells as **Parasitized** (infected with malaria) 
            or **Uninfected** (healthy).
            """
        )
        
        with gr.Tabs() as tabs:
            # ========== TRAINING TAB ==========
            with gr.Tab("🏋️ Training"):
                gr.Markdown("### Train a Malaria Detection Model")
                gr.Markdown(
                    """
                    **Instructions:**
                    - Specify your dataset path (or leave default to use TensorFlow Datasets)
                    - Set training parameters (epochs, batch size, split ratios)
                    - Click "Start Training" to begin
                    - Monitor progress in the training log
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Configuration")
                        data_path_input = gr.Textbox(
                            label="Dataset Path",
                            value="cell_images/",
                            placeholder="cell_images/ or leave default for TensorFlow Datasets",
                            lines=1
                        )
                        
                        model_type_train = gr.Dropdown(
                            choices=["CNN", "ResNet", "MobileNetV2"],
                            value="CNN",
                            label="Model Architecture",
                            info="Select the model architecture for training: CNN (Simple), ResNet (with Residual Blocks), or MobileNetV2 (Transfer Learning)"
                        )
                        
                        gr.Markdown("### Training Parameters")
                        with gr.Row():
                            epochs_input = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=5,
                                step=1,
                                label="Epochs"
                            )
                            batch_size_input = gr.Slider(
                                minimum=16,
                                maximum=128,
                                value=32,
                                step=16,
                                label="Batch Size"
                            )
                        
                        with gr.Row():
                            train_ratio_input = gr.Slider(
                                minimum=0.5,
                                maximum=0.9,
                                value=0.7,
                                step=0.05,
                                label="Train Ratio"
                            )
                            val_ratio_input = gr.Slider(
                                minimum=0.05,
                                maximum=0.3,
                                value=0.15,
                                step=0.05,
                                label="Validation Ratio"
                            )
                        
                        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                        
                        train_status = gr.Textbox(
                            label="Training Status & Log",
                            value="Ready to train. Configure parameters and click 'Start Training'.",
                            interactive=False,
                            lines=25,
                            max_lines=30
                        )
                    
                    with gr.Column(scale=1):
                        model_info_display = gr.Textbox(
                            label="Current Model Information",
                            value=f"Model: {MODEL_INFO['name']} | Architecture: {MODEL_INFO['architecture']} | Status: {'Trained' if MODEL_INFO['trained'] else 'Untrained'}",
                            interactive=False,
                            lines=4
                        )
                        
                        train_metrics = gr.Markdown(
                            label="Training Metrics",
                            value="Training metrics will appear here after training completes."
                        )
                        
                        with gr.Row():
                            train_acc_display = gr.Number(
                                label="Final Training Accuracy",
                                value=0.0,
                                precision=4
                            )
                            val_acc_display = gr.Number(
                                label="Final Validation Accuracy",
                                value=0.0,
                                precision=4
                            )
            
            # ========== TESTING TAB ==========
            with gr.Tab("🔍 Testing"):
                gr.Markdown("### Test the Malaria Detection Model")
                
                # Display Model Name prominently
                with gr.Row():
                    # Check if saved model path exists
                    saved_path_display = ""
                    if MODEL_INFO.get('saved_path'):
                        if os.path.exists(MODEL_INFO['saved_path']):
                            saved_path_display = f"<br>📁 Saved Model: <strong>{MODEL_INFO['saved_path']}</strong>"
                        else:
                            saved_path_display = f"<br>📁 Saved Model: <strong>{MODEL_INFO['saved_path']}</strong> (file not found)"
                    
                    model_name_display = gr.Markdown(
                        f"""
                        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                            <h3 style="margin: 0; color: #2c3e50;">🤖 Current Model: <strong>{MODEL_INFO['name']}</strong></h3>
                            <p style="margin: 5px 0 0 0; color: #7f8c8d;">
                                Architecture: {MODEL_INFO['architecture']} | 
                                Status: <strong>{'✅ Trained' if MODEL_INFO['trained'] else '⚠️ Untrained (Demo)'}</strong>
                                {saved_path_display}
                            </p>
                        </div>
                        """
                    )
                
                gr.Markdown(
                    """
                    **Instructions:**
                    - Upload an image of a blood cell
                    - Click "Analyze Image" to get predictions
                    - Results show classification and confidence scores
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Set initial dropdown value based on loaded model
                        initial_model_type = "ResNet" if "ResNet" in MODEL_INFO.get('name', '') else ("MobileNetV2" if "MobileNetV2" in MODEL_INFO.get('name', '') else "CNN")
                        model_type_test = gr.Dropdown(
                            choices=["CNN", "ResNet", "MobileNetV2"],
                            value=initial_model_type,
                            label="Model Architecture",
                            info="Select the model architecture for prediction"
                        )
                        
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Cell Image",
                            height=400
                        )
                        predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                        test_status_text = gr.Textbox(
                            label="Status & Progress Log",
                            value="Ready - Upload an image and click 'Analyze Image' to begin",
                            interactive=False,
                            lines=20,
                            max_lines=25
                        )
                        # Create initial model info text with saved path
                        saved_info = ""
                        if MODEL_INFO.get('saved_path'):
                            if os.path.exists(MODEL_INFO['saved_path']):
                                saved_info = f" | Saved as: {MODEL_INFO['saved_path']}"
                            else:
                                saved_info = f" | Saved as: {MODEL_INFO['saved_path']} (not found)"
                        
                        test_model_info_text = gr.Textbox(
                            label="Model Information",
                            value=f"Model: {MODEL_INFO['name']} | Architecture: {MODEL_INFO['architecture']} | Status: {'Trained' if MODEL_INFO['trained'] else 'Untrained (Demo)'}{saved_info}",
                            interactive=False,
                            lines=3
                        )
                        
                        gr.Markdown(
                            """
                            **Note:** Make sure to train a model in the Training tab first for accurate predictions.
                            """
                        )
                    
                    with gr.Column(scale=1):
                        result_html = gr.HTML(label="Prediction Result")
                        probability_plot = gr.BarPlot(
                            x="Label",
                            y="Probability",
                            title="Classification Probabilities",
                            y_lim=[0, 100],
                            height=300
                        )
                        result_text = gr.Textbox(
                            label="Detailed Results",
                            lines=3,
                            interactive=False
                        )
        
        def update_model_info_after_training():
            """Update model info display after training."""
            global MODEL_INFO
            return f"Model: {MODEL_INFO['name']} | Architecture: {MODEL_INFO['architecture']} | Status: {'Trained' if MODEL_INFO['trained'] else 'Untrained'}"

        def update_test_model_info():
            """Update test tab model info."""
            global MODEL_INFO
            saved_info = ""
            if MODEL_INFO.get('saved_path'):
                if os.path.exists(MODEL_INFO['saved_path']):
                    saved_info = f" | Saved as: {MODEL_INFO['saved_path']}"
                else:
                    saved_info = f" | Saved as: {MODEL_INFO['saved_path']} (not found)"
            return f"Model: {MODEL_INFO['name']} | Architecture: {MODEL_INFO['architecture']} | Status: {'Trained' if MODEL_INFO['trained'] else 'Untrained (Demo)'}{saved_info}"
        
        def update_model_name_display():
            """Update the prominent model name display in testing tab."""
            global MODEL_INFO
            saved_path_display = ""
            if MODEL_INFO.get('saved_path') and os.path.exists(MODEL_INFO['saved_path']):
                saved_path_display = f"<br>📁 Saved Model: <strong>{MODEL_INFO['saved_path']}</strong>"
            elif MODEL_INFO.get('saved_path'):
                saved_path_display = f"<br>📁 Saved Model: <strong>{MODEL_INFO['saved_path']}</strong> (file not found)"
            
            return f"""
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                <h3 style="margin: 0; color: #2c3e50;">🤖 Current Model: <strong>{MODEL_INFO['name']}</strong></h3>
                <p style="margin: 5px 0 0 0; color: #7f8c8d;">
                    Architecture: {MODEL_INFO['architecture']} | 
                    Status: <strong>{'✅ Trained' if MODEL_INFO['trained'] else '⚠️ Untrained (Demo)'}</strong>
                    {saved_path_display}
                </p>
            </div>
            """
        
        def switch_model_for_testing(model_type):
            """Switch to the selected model type for testing."""
            global MODEL, MODEL_INFO
            try:
                # Check for saved model of this type first
                if model_type == "MobileNetV2":
                    saved_model_path = "malaria_model_mobilenetv2_trained.h5"
                else:
                    saved_model_path = f"malaria_model_{model_type.lower()}_trained.h5"
                if os.path.exists(saved_model_path):
                    print(f"Loading saved {model_type} model from {saved_model_path}")
                    MODEL, MODEL_INFO = load_or_create_model(saved_model_path, model_type)
                else:
                    print(f"Creating new {model_type} model for testing")
                    MODEL, MODEL_INFO = load_or_create_model(None, model_type)
                
                # Warm up the model
                try:
                    dummy_input = tf.zeros((1, 64, 64, 3))
                    _ = MODEL(dummy_input, training=False)
                    print(f"✓ {model_type} model warmed up")
                except Exception as e:
                    print(f"⚠ Warning: Could not warm up model: {e}")
                
                # Update model info display
                saved_info = ""
                if MODEL_INFO.get('saved_path'):
                    if os.path.exists(MODEL_INFO['saved_path']):
                        saved_info = f" | Saved as: {MODEL_INFO['saved_path']}"
                    else:
                        saved_info = f" | Saved as: {MODEL_INFO['saved_path']} (not found)"
                
                return f"Model: {MODEL_INFO['name']} | Architecture: {MODEL_INFO['architecture']} | Status: {'Trained' if MODEL_INFO['trained'] else 'Untrained (Demo)'}{saved_info}"
            except Exception as e:
                print(f"Error switching model: {e}")
                import traceback
                print(traceback.format_exc())
                return f"Error switching to {model_type} model: {str(e)}"
        
        # Set up training
        train_btn.click(
            fn=train_model,
            inputs=[data_path_input, epochs_input, batch_size_input, train_ratio_input, val_ratio_input, model_type_train],
            outputs=[train_status, train_acc_display, val_acc_display]
        ).then(
            fn=update_model_info_after_training,
            outputs=[model_info_display]
        ).then(
            fn=update_test_model_info,
            outputs=[test_model_info_text]
        ).then(
            fn=update_model_name_display,
            outputs=[model_name_display]
        )
        
        # Set up prediction - only on button click to avoid conflicts
        def predict_with_progress(image, progress=gr.Progress()):
            """Wrapper to add progress tracking."""
            print("\n" + "="*70)
            print("PREDICT_WITH_PROGRESS CALLED")
            print("="*70)
            try:
                # Call the main prediction function with progress
                # Progress updates are handled inside predict_malaria
                result = predict_malaria(image, progress)
                
                print("\n[PREDICT_WITH_PROGRESS] Got result from predict_malaria")
                print(f"  - Result type: {type(result)}")
                print(f"  - Result length: {len(result) if isinstance(result, (tuple, list)) else 'N/A'}")
                if isinstance(result, (tuple, list)) and len(result) == 4:
                    print(f"  - Result[0] (HTML): {type(result[0])}, length: {len(str(result[0])) if result[0] else 0}")
                    print(f"  - Result[1] (Status): {type(result[1])}, length: {len(str(result[1])) if result[1] else 0}")
                    print(f"  - Result[2] (Plot): {type(result[2])}, value: {result[2]}")
                    print(f"  - Result[3] (Text): {type(result[3])}, length: {len(str(result[3])) if result[3] else 0}")
                
                print("[PREDICT_WITH_PROGRESS] Returning result to Gradio...")
                print("="*70 + "\n")
                
                # Ensure we return exactly what Gradio expects
                if isinstance(result, (tuple, list)) and len(result) == 4:
                    return tuple(result)  # Explicitly convert to tuple
                else:
                    print(f"WARNING: Unexpected result format: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                    return result
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"\n[ERROR] in predict_with_progress: {str(e)}")
                print(error_details)
                error_html = f"<div style='color: red; padding: 20px;'>Error: {str(e)}</div>"
                error_status = f"Error occurred: {str(e)}\n\n{error_details}"
                return (error_html, error_status, None, f"Error: {str(e)}")
        
        # Connect model type dropdown to switch models
        model_type_test.change(
            fn=switch_model_for_testing,
            inputs=model_type_test,
            outputs=[test_model_info_text]
        ).then(
            fn=update_model_name_display,
            outputs=[model_name_display]
        )
        
        predict_btn.click(
            fn=predict_with_progress,
            inputs=image_input,
            outputs=[result_html, test_status_text, probability_plot, result_text],
            show_progress=True
        )
        
        # Optional: Allow prediction on image change (comment out if causing issues)
        # image_input.upload(
        #     fn=predict_malaria,
        #     inputs=image_input,
        #     outputs=[result_html, test_status_text, probability_plot, result_text],
        #     show_progress=True
        # )
    
    return app


def main():
    """Main function to launch the Gradio app."""
    print("Initializing Malaria Detection Interface...")
    
    # Create and launch the interface
    app = create_interface()
    
    print("\n" + "="*50)
    print("Malaria Detection Gradio Interface")
    print("="*50)
    print("\nStarting server...")
    print("The interface will open in your default browser.")
    print("You can also access it at the local URL shown below.\n")
    
    app.launch(
        server_name="0.0.0.0",  # Allow access from network
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True to create a public link
        show_error=True
    )


if __name__ == "__main__":
    main()

