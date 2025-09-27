"""
Handwritten Digit Recognition Application

This application provides an interactive interface for recognizing handwritten digits
using a trained neural network model. Users can draw digits on a 28x28 grid and get
real-time predictions from the model.

Features:
- Interactive drawing canvas (28x28 pixel grid)
- Real-time digit classification
- Reset functionality to clear the canvas
- Visual feedback with grayscale drawing
- Model compatibility checking

Requirements:
- TensorFlow 2.x
- Pygame
- NumPy
- A trained Keras model file (.h5 or .keras format)

Usage:
    python recognition.py <model_path>
    
    Example:
    python recognition.py model.h5

Controls:
- Mouse: Click and drag to draw digits on the grid
- Reset button: Clear the canvas and start over
- Classify button: Get prediction for the drawn digit

Note: The model should be trained on 28x28 grayscale images for best results.
The application expects the model to output probabilities for digits 0-9.
"""

import numpy as np
import pygame
import sys
import tensorflow as tf
import time

# Check command-line arguments
if len(sys.argv) != 2:
    sys.exit("Usage: python recognition.py model")

# Try to load model with compatibility settings
try:
    model = tf.keras.models.load_model(sys.argv[1], compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("This model may be incompatible with TensorFlow 2.19.0")
    print("You may need to retrain the model or use a different version")
    sys.exit(1)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Start pygame
pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)

# Fonts
OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)

ROWS, COLS = 28, 28

OFFSET = 20
CELL_SIZE = 10

handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None

while True:

    # Check if game quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    # Check for mouse press
    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # Draw each grid cell
    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # If cell has been written on, darken cell
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            # Draw blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # Reset button
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify button
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # Reset drawing
    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None

    # Generate classification
    if mouse and classifyButton.collidepoint(mouse):
        try:
            # Prepare input data
            input_data = np.array(handwriting).reshape(1, 28, 28, 1)
            prediction = model.predict(input_data, verbose=0)
            classification = prediction.argmax()
        except Exception as e:
            print(f"Prediction error: {e}")
            classification = None

    # Show classification if one exists
    if classification is not None:
        classificationText = largeFont.render(str(classification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            100
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()
