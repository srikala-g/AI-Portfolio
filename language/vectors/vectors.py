"""
Word Vector Similarity Calculator

This module provides functionality for working with word embeddings and calculating
semantic similarity between words using cosine distance. It loads pre-trained word
vectors from a text file and provides utilities to find the most similar words
to a given embedding.

Key Features:
- Load word vectors from a text file (words.txt)
- Calculate cosine distance between word embeddings
- Find the closest word to a given embedding
- Find the top 10 most similar words to a given embedding

Input Requirements:
- words.txt: A text file containing word vectors where each line has:
  - First column: the word
  - Remaining columns: the vector components (space-separated floats)
  - Example format: "word 0.1 0.2 0.3 ... 0.9"

Usage:
    # Load the module (words.txt must be in the same directory)
    import vectors
    
    # Find closest word to a given embedding
    closest = vectors.closest_word(some_embedding)
    
    # Find top 10 similar words
    similar = vectors.closest_words(some_embedding)
    
    # Calculate distance between two word embeddings
    dist = vectors.distance(embedding1, embedding2)
"""

from scipy.spatial.distance import cosine

import math
import numpy as np

with open("words.txt") as f:
    words = dict()
    for line in f:
        row = line.split()
        word = row[0]
        vector = np.array([float(x) for x in row[1:]])
        words[word] = vector


def distance(w1, w2):
    return cosine(w1, w2)


def closest_words(embedding):
    distances = {
        w: distance(embedding, words[w])
        for w in words
    }
    return sorted(distances, key=lambda w: distances[w])[:10]


def closest_word(embedding):
    return closest_words(embedding)[0]

