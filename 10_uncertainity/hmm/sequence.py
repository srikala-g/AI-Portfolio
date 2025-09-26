"""
Hidden Markov Model (HMM) Weather Prediction Demo

This script demonstrates how to use a pre-trained Hidden Markov Model to predict
weather conditions (sunny or rainy) based on umbrella observations over time.

The model uses the following logic:
- If it's sunny: 20% chance of umbrella, 80% chance of no umbrella
- If it's rainy: 90% chance of umbrella, 10% chance of no umbrella
- Weather transitions: sunny→sunny (80%), sunny→rainy (20%), rainy→sunny (30%), rainy→rainy (70%)

Usage Instructions:
1. Run this script directly: python sequence.py
2. The script will output the most likely weather sequence for each day
3. Modify the 'observations' list to test different umbrella patterns
4. The model will predict the underlying weather states that best explain the observations

Example Output:
    sun
    sun
    rain
    rain
    rain
    rain
    rain
    sun
    sun

The model uses the Viterbi algorithm to find the most likely sequence of hidden
states (weather conditions) given the observed data (umbrella usage).
"""

from model import model

# Observed data
observations = [
    "umbrella",
    "umbrella",
    "no umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "no umbrella",
    "no umbrella"
]

# Predict underlying states
predictions = model.predict(observations)
for prediction in predictions:
    print(model.states[prediction].name)
