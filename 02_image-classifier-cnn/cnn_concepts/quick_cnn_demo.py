#!/usr/bin/env python3
"""
Quick CNN Power Demonstration
This script shows the key visualizations and results without full training
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_advanced_cnn(input_shape, num_classes):
    """Advanced CNN with dropout and batch normalization"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def demonstrate_cnn_power():
    """Demonstrate CNN power across multiple datasets"""
    
    print("🚀 Starting CNN Power Demonstration...")
    print("=" * 60)
    
    # Load datasets
    print("📊 Loading datasets...")
    
    # MNIST (Handwritten digits)
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    x_train_mnist = x_train_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test_mnist = x_test_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train_mnist = to_categorical(y_train_mnist, 10)
    y_test_mnist = to_categorical(y_test_mnist, 10)
    
    # Fashion-MNIST (Clothing items)
    (x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()
    x_train_fashion = x_train_fashion.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test_fashion = x_test_fashion.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train_fashion = to_categorical(y_train_fashion, 10)
    y_test_fashion = to_categorical(y_test_fashion, 10)
    
    # CIFAR-10 (Natural images)
    (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
    x_train_cifar = x_train_cifar.astype('float32') / 255.0
    x_test_cifar = x_test_cifar.astype('float32') / 255.0
    y_train_cifar = to_categorical(y_train_cifar, 10)
    y_test_cifar = to_categorical(y_test_cifar, 10)
    
    datasets = {
        'MNIST': (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist, (28, 28, 1)),
        'Fashion-MNIST': (x_train_fashion, y_train_fashion, x_test_fashion, y_test_fashion, (28, 28, 1)),
        'CIFAR-10': (x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar, (32, 32, 3))
    }
    
    results = {}
    
    for dataset_name, (x_train, y_train, x_test, y_test, input_shape) in datasets.items():
        print(f"\n{'='*50}")
        print(f"🧠 Training CNN on {dataset_name}")
        print(f"{'='*50}")
        
        # Create and compile model
        model = create_advanced_cnn(input_shape, 10)
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train model (reduced epochs for demo)
        print(f"⏳ Training {dataset_name} (this may take a few minutes)...")
        history = model.fit(x_train, y_train,
                          epochs=3,  # Reduced for demo
                          batch_size=128,
                          validation_split=0.2,
                          verbose=1)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        results[dataset_name] = {
            'accuracy': test_acc,
            'history': history,
            'model': model
        }
        
        print(f"\n✅ {dataset_name} Test Accuracy: {test_acc:.4f}")
    
    return results

def visualize_cnn_performance(results):
    """Visualize CNN performance across datasets"""
    
    print("\n📈 Creating visualizations...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    datasets = list(results.keys())
    accuracies = [results[dataset]['accuracy'] for dataset in datasets]
    
    bars = axes[0, 0].bar(datasets, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('CNN Test Accuracy Across Datasets', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 2. Training history - Loss
    for i, (dataset, result) in enumerate(results.items()):
        axes[0, 1].plot(result['history'].history['loss'], 
                       label=f'{dataset} - Training', linestyle='-', alpha=0.8)
        axes[0, 1].plot(result['history'].history['val_loss'], 
                       label=f'{dataset} - Validation', linestyle='--', alpha=0.8)
    
    axes[0, 1].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training history - Accuracy
    for i, (dataset, result) in enumerate(results.items()):
        axes[1, 0].plot(result['history'].history['accuracy'], 
                       label=f'{dataset} - Training', linestyle='-', alpha=0.8)
        axes[1, 0].plot(result['history'].history['val_accuracy'], 
                       label=f'{dataset} - Validation', linestyle='--', alpha=0.8)
    
    axes[1, 0].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Model complexity comparison
    model_params = []
    for dataset in datasets:
        total_params = results[dataset]['model'].count_params()
        model_params.append(total_params)
    
    bars = axes[1, 1].bar(datasets, model_params, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 1].set_title('Model Complexity (Parameters)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Parameters')
    for i, v in enumerate(model_params):
        axes[1, 1].text(i, v + max(model_params) * 0.01, f'{v:,}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizations created successfully!")

def demonstrate_real_world_applications():
    """Show real-world CNN applications"""
    
    print("\n🌍 Real-World CNN Applications")
    print("=" * 50)
    
    applications = {
        'Medical Imaging': {
            'description': 'CNNs excel at detecting diseases in X-rays, MRIs, and CT scans',
            'accuracy': '95%+ in cancer detection',
            'impact': 'Saves lives through early diagnosis'
        },
        'Autonomous Vehicles': {
            'description': 'CNNs power object detection, lane recognition, and traffic sign classification',
            'accuracy': '99%+ in object detection',
            'impact': 'Enables safe self-driving cars'
        },
        'Security & Surveillance': {
            'description': 'Facial recognition, anomaly detection, and behavior analysis',
            'accuracy': '98%+ in facial recognition',
            'impact': 'Enhances security systems'
        },
        'Agriculture': {
            'description': 'Crop disease detection, yield prediction, and soil analysis',
            'accuracy': '90%+ in disease detection',
            'impact': 'Increases crop yields and reduces waste'
        },
        'Entertainment': {
            'description': 'Content recommendation, image generation, and video analysis',
            'accuracy': '85%+ in content matching',
            'impact': 'Personalizes user experiences'
        }
    }
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = list(applications.keys())
    accuracies = [float(app['accuracy'].split('%')[0]) for app in applications.values()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    bars = ax.barh(categories, accuracies, color=colors)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Real-World CNN Applications and Their Performance', 
                fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2, 
               f'{acc:.0f}%', va='center', fontweight='bold')
    
    # Add descriptions
    for i, (category, app) in enumerate(applications.items()):
        ax.text(0.5, i, f"{app['description']}\nImpact: {app['impact']}", 
               transform=ax.get_yaxis_transform(), 
               va='center', ha='left', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Real-world applications visualization created!")

def main():
    """Main demonstration function"""
    print("🎯 CNN Power Demonstration")
    print("=" * 60)
    
    # Run the demonstration
    results = demonstrate_cnn_power()
    
    # Create visualizations
    visualize_cnn_performance(results)
    
    # Show real-world applications
    demonstrate_real_world_applications()
    
    print("\n🎉 CNN Power Demonstration Complete!")
    print("=" * 60)
    print("Key Takeaways:")
    print("• CNNs achieve high accuracy across multiple datasets")
    print("• Performance improves with more complex architectures")
    print("• Real-world applications span multiple industries")
    print("• Automatic feature learning eliminates manual engineering")

if __name__ == "__main__":
    main()
