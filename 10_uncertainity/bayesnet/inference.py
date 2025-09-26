"""
Bayesian Network Inference Module

This module performs probabilistic inference on a Bayesian network model using the pomegranate library.
It demonstrates how to calculate conditional probabilities and make predictions based on observed evidence.

The Bayesian network models a scenario with the following variables:
- Rain: Weather conditions (none, light, heavy)
- Maintenance: Track maintenance status (yes, no) - depends on rain
- Train: Train punctuality (on time, delayed) - depends on rain and maintenance
- Appointment: Whether appointment is attended (attend, miss) - depends on train

Key Features:
- Conditional probability inference using predict_proba()
- Evidence-based prediction for all network variables
- Integration with pomegranate's Bayesian network framework

Usage Instructions:
1. Ensure you have the model.py file in the same directory (contains the Bayesian network definition)
2. Install required dependencies: pip install pomegranate
3. Run the script: python inference.py
4. The script will output probability distributions for all variables given the evidence

Example Output:
    rain
        none: 0.0000
        light: 0.0000
        heavy: 0.0000
    maintenance
        yes: 0.0000
        no: 0.0000
    train
        on time: 0.0000
        delayed: 1.0000
    appointment
        attend: 0.6000
        miss: 0.4000

Dependencies:
- pomegranate: For Bayesian network modeling and inference
- model: Local module containing the Bayesian network definition

Note: This script demonstrates inference given evidence that the train is delayed.
You can modify the evidence dictionary to test different scenarios.
"""

from model import model

# Calculate predictions
predictions = model.predict_proba({
    "train": "delayed"
})

# Print predictions for each node
for node, prediction in zip(model.states, predictions):
    if isinstance(prediction, str):
        print(f"{node.name}: {prediction}")
    else:
        print(f"{node.name}")
        for value, probability in prediction.parameters[0].items():
            print(f"    {value}: {probability:.4f}")
