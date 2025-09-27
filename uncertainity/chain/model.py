"""
Weather Markov Chain Model

This script implements a simple Markov chain model to simulate weather patterns
using the Pomegranate library. The model represents a two-state weather system
(sunny and rainy) with defined transition probabilities.

Model Description:
- States: "sun" (sunny) and "rain" (rainy)
- Starting probabilities: 50% chance for each state
- Transition probabilities:
  * If sunny today: 80% chance sunny tomorrow, 20% chance rainy tomorrow
  * If rainy today: 30% chance sunny tomorrow, 70% chance rainy tomorrow

Usage Instructions:
1. Install required dependency: pip install pomegranate
2. Run the script: python model.py
3. The script will output 50 sampled weather states from the Markov chain
4. Each output represents the weather for consecutive days

Example Output:
['sun', 'sun', 'rain', 'rain', 'sun', 'sun', 'sun', 'rain', ...]

The model can be extended by:
- Modifying transition probabilities in the ConditionalProbabilityTable
- Changing the number of samples (currently 50)
- Adding more weather states (e.g., cloudy, stormy)
- Adjusting starting probabilities
"""

from pomegranate import *

# Define starting probabilities
start = DiscreteDistribution({
    "sun": 0.5,
    "rain": 0.5
})

# Define transition model
transitions = ConditionalProbabilityTable([
    ["sun", "sun", 0.8],
    ["sun", "rain", 0.2],
    ["rain", "sun", 0.3],
    ["rain", "rain", 0.7]
], [start])

# Create Markov chain
model = MarkovChain([start, transitions])

# Sample 50 states from chain
print(model.sample(50))
