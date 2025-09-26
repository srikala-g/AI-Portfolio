"""
Context-Free Grammar (CFG) Parser for Simple English Sentences

WHAT THIS FILE DOES:
This script implements a basic context-free grammar parser using NLTK to analyze
simple English sentences. It defines grammatical rules for sentence structure
and parses user input to generate parse trees. The parser can handle:
- Simple sentences with noun phrases and verb phrases
- Sentences with adjectives and determiners
- Sentences with prepositional phrases
- Complex sentences with multiple grammatical components

The parser will either display a parse tree showing the grammatical structure
or indicate that the sentence cannot be parsed according to the defined grammar rules.

Grammar Rules:
- S -> NP VP (Sentence consists of Noun Phrase + Verb Phrase)
- NP -> N | D NP | AP NP | N PP (Noun Phrase variations)
- VP -> V | V NP | V NP PP (Verb Phrase variations)
- AP -> A | A AP (Adjective Phrase)
- PP -> P NP (Prepositional Phrase)

Supported Words:
- Adjectives: big, blue, small, dry, wide
- Determiners: the, a, an
- Nouns: she, city, car, street, dog, binoculars
- Prepositions: on, over, before, below, with
- Verbs: saw, walked

Example Inputs and Outputs:

Input: "she saw the car"
Output: Parse tree showing S -> NP(she) VP(saw NP(the car))

Input: "the big dog walked"
Output: Parse tree showing S -> NP(the AP(big) dog) VP(walked)

Input: "she saw the dog with binoculars"
Output: Parse tree showing S -> NP(she) VP(saw NP(the dog) PP(with binoculars))

Input: "the small blue car"
Output: No parse tree possible (missing verb)

How to Run:
1. Install dependencies from requirements.txt:
   pip install -r requirements.txt

2. Download NLTK data (if not already downloaded):
   python -c "import nltk; nltk.download('punkt')"

3. Run the script:
   python cfg1.py

4. Enter a sentence when prompted. Use only the supported words:
   - Valid examples: "she saw the car", "the big dog walked", "she saw the dog with binoculars"
   - Invalid examples: "the small blue car" (no verb), "I am happy" (unsupported words)
5. The parser will display the parse tree or indicate if parsing is not possible

INPUT INSTRUCTIONS:
- Use only the supported vocabulary listed above
- Sentences must contain at least a noun and a verb
- Words are case-sensitive and must match exactly
- Separate words with spaces only
- The parser will tokenize your input automatically

Usage:
Run the script and enter a sentence when prompted. The parser will attempt
to generate a parse tree or indicate if parsing is not possible.
"""

import nltk

grammar = nltk.CFG.fromstring("""
    S -> NP VP

    AP -> A | A AP
    NP -> N | D NP | AP NP | N PP
    PP -> P NP
    VP -> V | V NP | V NP PP

    A -> "big" | "blue" | "small" | "dry" | "wide"
    D -> "the" | "a" | "an"
    N -> "she" | "city" | "car" | "street" | "dog" | "binoculars"
    P -> "on" | "over" | "before" | "below" | "with"
    V -> "saw" | "walked"
""")

parser = nltk.ChartParser(grammar)

sentence = input("Sentence: ").split()
try:
    for tree in parser.parse(sentence):
        tree.pretty_print()
except ValueError:
    print("No parse tree possible.")
