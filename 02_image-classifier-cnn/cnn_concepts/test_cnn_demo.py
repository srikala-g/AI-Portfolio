#!/usr/bin/env python3
"""
Test script for CNN Power Demonstration
This script tests all the components before running the full notebook
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    try:
        print("✅ NumPy:", np.__version__)
        print("✅ TensorFlow:", tf.__version__)
        print("✅ OpenCV:", cv2.__version__)
        print("✅ Matplotlib:", plt.matplotlib.__version__)
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test loading datasets"""
    print("\n🔍 Testing dataset loading...")
    try:
        # Test MNIST
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print(f"✅ MNIST loaded: {x_train.shape}, {y_train.shape}")
        
        # Test Fashion-MNIST
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        print(f"✅ Fashion-MNIST loaded: {x_train.shape}, {y_train.shape}")
        
        # Test CIFAR-10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print(f"✅ CIFAR-10 loaded: {x_train.shape}, {y_train.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return False

def test_cnn_creation():
    """Test CNN model creation"""
    print("\n🔍 Testing CNN model creation...")
    try:
        # Create a simple CNN
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        print("✅ CNN model created and compiled successfully!")
        print(f"✅ Model parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"❌ CNN creation error: {e}")
        return False

def test_quick_training():
    """Test quick training on small dataset"""
    print("\n🔍 Testing quick training...")
    try:
        # Load small subset of MNIST
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Use only first 1000 samples for quick test
        x_train = x_train[:1000].reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train = to_categorical(y_train[:1000], 10)
        
        # Create simple model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train for 1 epoch
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        
        print("✅ Quick training successful!")
        print(f"✅ Final accuracy: {history.history['accuracy'][-1]:.4f}")
        return True
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_visualization():
    """Test visualization capabilities"""
    print("\n🔍 Testing visualization...")
    try:
        # Create a simple plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Plot')
        plt.close(fig)  # Close to avoid display issues
        
        print("✅ Visualization working!")
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting CNN Power Demo Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_cnn_creation,
        test_quick_training,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The CNN demo should work perfectly.")
        print("\n📝 Next steps:")
        print("1. Open cnn_power_demo.ipynb in Cursor IDE")
        print("2. Make sure Python interpreter is set to the virtual environment")
        print("3. Run the notebook cells one by one")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
