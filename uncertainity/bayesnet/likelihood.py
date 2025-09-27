"""
Bayesian Network Likelihood Calculator

This script demonstrates how to calculate the probability of a specific observation
in a Bayesian network that models the relationship between weather conditions,
train maintenance, train punctuality, and appointment attendance.

The network structure:
- Rain (none/light/heavy) → affects both maintenance and train punctuality
- Maintenance (yes/no) → affects train punctuality
- Train (on time/delayed) → affects appointment attendance
- Appointment (attend/miss) → final outcome

Usage Instructions:
1. Ensure you have the required dependencies installed:
   pip install pomegranate

2. Run the script to see the probability calculation:
   python likelihood.py

3. To calculate probabilities for different scenarios, modify the observation list:
   - Format: [["rain", "maintenance", "train", "appointment"]]
   - Example scenarios:
     * ["none", "no", "on time", "attend"] - No rain, no maintenance, train on time, attend appointment
     * ["heavy", "yes", "delayed", "miss"] - Heavy rain, maintenance done, train delayed, miss appointment
     * ["light", "no", "on time", "attend"] - Light rain, no maintenance, train on time, attend appointment

4. The output will be a probability value between 0 and 1 representing the likelihood
   of that specific combination of events occurring.
"""

from model import model

# Calculate probability for a given observation
# Format: [["rain", "maintenance", "train", "appointment"]]
# This example: no rain, no maintenance, train on time, attend appointment
probability = model.probability([["none", "no", "on time", "attend"]])

print(f"Probability of observation: {probability}")
