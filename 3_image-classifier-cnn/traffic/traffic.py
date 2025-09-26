"""
Traffic Sign Recognition CNN Model

This script implements a Convolutional Neural Network (CNN) for traffic sign recognition
using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model uses
a multi-layer CNN architecture with convolutional layers, max pooling, and dense layers
to classify traffic signs into 43 different categories.

Features:
- Loads traffic sign images from organized directory structure
- Preprocesses images (resize to 30x30, normalize to [0,1])
- Implements CNN with 2 convolutional layers + 2 dense layers
- Uses categorical crossentropy loss and Adam optimizer
- Supports model training and evaluation
- Saves trained model to file

Usage:
    python traffic.py data_directory [model.h5]
    
Arguments:
    data_directory: Path to directory containing traffic sign images
                   Expected structure: data_directory/0/, data_directory/1/, ..., data_directory/42/
                   Each subdirectory contains images for that traffic sign category
    model.h5: (Optional) Filename to save the trained model
    
Example:
    python traffic.py gtsrb/ traffic_model.h5
    
Requirements:
    - TensorFlow
    - OpenCV (cv2)
    - NumPy
    - scikit-learn
    - Traffic sign dataset with 43 categories (0-42)
"""

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    # Loop over each category directory
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_dir):
            continue
        # Loop over each image file in the category directory
        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            # Read image using cv2
            img = cv2.imread(filepath)
            if img is None:
                continue
            # Resize image to IMG_WIDTH x IMG_HEIGHT
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # Normalize image to range [0, 1]
            img = img.astype("float32") / 255.0

            images.append(img)
            labels.append(category)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
