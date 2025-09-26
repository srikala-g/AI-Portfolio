"""
Context-Free Grammar (CFG) Parser Demo

This script demonstrates basic natural language parsing using NLTK's Context-Free Grammar.
It defines a simple grammar with rules for sentence structure (S -> NP VP) and parses
user input sentences to generate parse trees.

Grammar Rules:
- S -> NP VP (Sentence consists of Noun Phrase + Verb Phrase)
- NP -> D N | N (Noun Phrase can be Determiner + Noun or just Noun)
- VP -> V | V NP (Verb Phrase can be Verb or Verb + Noun Phrase)
- D -> "the" | "a" (Determiners)
- N -> "she" | "city" | "car" (Nouns)
- V -> "saw" | "walked" (Verbs)

Usage: Run the script and enter a sentence when prompted. The parser will attempt
to generate a parse tree or indicate if parsing is not possible.
"""

import nltk

grammar = nltk.CFG.fromstring("""
    S -> NP VP

    NP -> D N | N
    VP -> V | V NP

    D -> "the" | "a"
    N -> "she" | "city" | "car"
    V -> "saw" | "walked"
""")

parser = nltk.ChartParser(grammar)

sentence = input("Sentence: ").split()
try:
    for tree in parser.parse(sentence):
        tree.pretty_print()
except ValueError:
    print("No parse tree possible.")
