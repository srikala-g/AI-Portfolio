"""
Bayesian Network Sampling Module

This module implements sampling methods for Bayesian networks using the pomegranate library.
It provides functionality to generate samples from a Bayesian network model and perform
rejection sampling for conditional probability queries.

Key Features:
- Forward sampling from Bayesian network models
- Rejection sampling for conditional probability estimation
- Integration with pomegranate's ConditionalProbabilityTable

Usage Instructions:
1. Ensure you have a valid Bayesian network model imported from 'model.py'
2. Call generate_sample() to generate a single sample from the network
3. For rejection sampling, iterate over multiple samples and filter based on evidence
4. Use Counter from collections to analyze the distribution of sampled values

Example:
    # Generate a single sample
    sample = generate_sample()
    print(sample)
    
    # Perform rejection sampling for P(Appointment | Train=delayed)
    N = 10000
    data = []
    for i in range(N):
        sample = generate_sample()
        if sample["train"] == "delayed":
            data.append(sample["appointment"])
    print(Counter(data))

Dependencies:
- pomegranate: For Bayesian network modeling
- collections.Counter: For analyzing sample distributions
- model: Local module containing the Bayesian network definition
"""

import pomegranate

from collections import Counter

from model import model

def generate_sample():

    # Mapping of random variable name to sample generated
    sample = {}

    # Mapping of distribution to sample generated
    parents = {}

    # Loop over all states, assuming topological order
    for state in model.states:

        # If we have a non-root node, sample conditional on parents
        if isinstance(state.distribution, pomegranate.ConditionalProbabilityTable):
            sample[state.name] = state.distribution.sample(parent_values=parents)

        # Otherwise, just sample from the distribution alone
        else:
            sample[state.name] = state.distribution.sample()

        # Keep track of the sampled value in the parents mapping
        parents[state.distribution] = sample[state.name]

    # Return generated sample
    return sample

# Rejection sampling
# Compute distribution of Appointment given that train is delayed
N = 10000
data = []
for i in range(N):
    sample = generate_sample()
    if sample["train"] == "delayed":
        data.append(sample["appointment"])
print(Counter(data))

