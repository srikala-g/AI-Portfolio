"""
Natural Language Parser using Context-Free Grammar

This script implements a natural language parser using NLTK's ChartParser with a 
custom Context-Free Grammar (CFG). It can parse sentences and extract noun phrase 
chunks from the parsed syntax trees.

Features:
- Parses sentences using a predefined grammar with terminals and non-terminals
- Extracts noun phrase chunks from parsed syntax trees
- Supports both file input and interactive sentence input
- Preprocesses text by converting to lowercase and filtering non-alphabetic words

Input Instructions:
1. Run with a text file: python parser.py <filename>
   - The file should contain a single sentence to parse
2. Run interactively: python parser.py
   - Enter a sentence when prompted
3. Example sentences that work with the grammar:
   - "Holmes sat down and lit his pipe"
   - "The little red door opened"
   - "She smiled and walked home"

The parser will display the syntax tree and extract noun phrase chunks.
"""

import nltk
import sys

# nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# NONTERMINALS = """
# S -> N V
# """

NONTERMINALS = """
S -> NP VP

AP -> Adj | Adj AP
AdvP -> Adv | Adv AdvP | AdvP CP
NP -> N | Det NP | AP NP | N PP |N CP | N AdvP
PP -> P NP | P Det NP
VP -> V | V NP | V NP PP | V PP | V AdvP
CP -> Conj S | Conj VP
"""
# Holmes sat down and lit his pipe.

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")

        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = [word for word in words if any(char.isalpha() for char in word)]
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            # Make sure this NP does not contain other NPs
            if not any(child.label() == "NP" for child in subtree.subtrees(lambda t: t != subtree)):
                chunks.append(subtree)
    return chunks


if __name__ == "__main__":
    main()
