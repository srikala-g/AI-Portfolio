"""
N-gram Frequency Analyzer

This script analyzes text corpora to find the most frequent n-grams (sequences of n consecutive words).
It processes all text files in a given directory, tokenizes the content, and returns the top 10 most
common n-gram sequences.

INPUT INSTRUCTIONS:
    Usage: python ngrams.py <n> <corpus_directory>
    
    Parameters:
    - n: The size of n-grams to analyze (e.g., 1 for unigrams, 2 for bigrams, 3 for trigrams)
    - corpus_directory: Path to directory containing text files to analyze
    
    Example:
    python ngrams.py 2 ./text_files/
    
    This will find the most common bigrams (2-word sequences) in all text files within ./text_files/

FEATURES:
    - Processes multiple text files in a directory
    - Tokenizes text using NLTK
    - Filters to keep only words containing alphabetic characters
    - Converts all text to lowercase for consistent analysis
    - Returns top 10 most frequent n-grams with their counts

REQUIREMENTS:
    - NLTK library for text processing
    - Text files in the specified directory
"""

from collections import Counter

import math
import nltk
import os
import sys


def main():
    """Calculate top term frequencies for a corpus of documents."""

    if len(sys.argv) != 3:
        sys.exit("Usage: python ngrams.py n corpus")
    print("Loading data...")

    n = int(sys.argv[1])
    corpus = load_data(sys.argv[2])

    # Compute n-grams
    ngrams = Counter(nltk.ngrams(corpus, n))

    # Print most common n-grams
    for ngram, freq in ngrams.most_common(10):
        print(f"{freq}: {ngram}")


def load_data(directory):
    contents = []

    # Read all files and extract words
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            contents.extend([
                word.lower() for word in
                nltk.word_tokenize(f.read())
                if any(c.isalpha() for c in word)
            ])
    return contents


if __name__ == "__main__":
    main()
