"""
Hidden Markov Model (HMM) for Weather Prediction

This module implements a simple Hidden Markov Model to predict weather conditions
(sunny or rainy) based on umbrella observations. The model uses the pomegranate
library to create a probabilistic model that can infer hidden states from observations.

Model Components:
- States: sun, rain (hidden weather conditions)
- Observations: umbrella, no umbrella (visible evidence)
- Transition probabilities: Weather state transitions between days
- Emission probabilities: Likelihood of observations given weather states

Usage Instructions:
1. Import and run the script to create the HMM model
2. Use model.predict() to predict most likely weather sequence from observations
3. Use model.predict_proba() to get probability distributions for each state
4. Use model.log_probability() to calculate likelihood of observation sequences

Example:
    # Predict weather from umbrella observations
    observations = ["umbrella", "no umbrella", "umbrella"]
    predictions = model.predict(observations)
    print(f"Predicted weather: {predictions}")
    
    # Get probability distributions
    proba = model.predict_proba(observations)
    print(f"State probabilities: {proba}")
"""

from pomegranate import *

# Observation model for each state
sun = DiscreteDistribution({
    "umbrella": 0.2,
    "no umbrella": 0.8
})

rain = DiscreteDistribution({
    "umbrella": 0.9,
    "no umbrella": 0.1
})

states = [sun, rain]

# Transition model
transitions = numpy.array(
    [[0.8, 0.2], # Tomorrow's predictions if today = sun
     [0.3, 0.7]] # Tomorrow's predictions if today = rain
)

# Starting probabilities
starts = numpy.array([0.5, 0.5])

# Create the model
model = HiddenMarkovModel.from_matrix(
    transitions, states, starts,
    state_names=["sun", "rain"]
)
model.bake()
