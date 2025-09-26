"""
Sentiment Analysis Tool

This script performs sentiment analysis on text using a Naive Bayes classifier.
It trains a model on positive and negative text samples, then classifies new text input.

Features:
- Loads training data from positive and negative text files
- Extracts word features from documents
- Trains a Naive Bayes classifier using NLTK
- Classifies new text and provides probability scores for each sentiment

Input Instructions:
- Run the script with: python sentiment.py <corpus_directory>
- The corpus directory should contain:
  - positives.txt: One positive text sample per line
  - negatives.txt: One negative text sample per line
- After training, enter text to classify when prompted

Example usage:
    python sentiment.py ./corpus
    s: This movie is amazing!
    Output: Positive: 0.8234, Negative: 0.1766
"""

import nltk
import os
import sys


def main():

    # Read data from files
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py corpus")
    positives, negatives = load_data(sys.argv[1])

    # Create a set of all words
    words = set()
    for document in positives:
        words.update(document)
    for document in negatives:
        words.update(document)

    # Extract features from text
    training = []
    training.extend(generate_features(positives, words, "Positive"))
    training.extend(generate_features(negatives, words, "Negative"))

    # Classify a new sample
    classifier = nltk.NaiveBayesClassifier.train(training)
    s = input("s: ")
    result = (classify(classifier, s, words))
    for key in result.samples():
        print(f"{key}: {result.prob(key):.4f}")


def extract_words(document):
    return set(
        word.lower() for word in nltk.word_tokenize(document)
        if any(c.isalpha() for c in word)
    )


def load_data(directory):
    result = []
    for filename in ["positives.txt", "negatives.txt"]:
        with open(os.path.join(directory, filename)) as f:
            result.append([
                extract_words(line)
                for line in f.read().splitlines()
            ])
    return result


def generate_features(documents, words, label):
    features = []
    for document in documents:
        features.append(({
            word: (word in document)
            for word in words
        }, label))
    return features


def classify(classifier, document, words):
    document_words = extract_words(document)
    features = {
        word: (word in document_words)
        for word in words
    }
    return classifier.prob_classify(features)


if __name__ == "__main__":
    main()
