"""
Markov Chain Text Generator

WHAT THIS FILE DOES:
This script generates new text using Markov chains based on a provided text file.
It analyzes the input text to learn patterns of word sequences and then generates
new sentences that follow similar patterns. The generator creates 5 new sentences
that mimic the style and structure of the original text.

The Markov model learns:
- Which words commonly follow other words
- Sentence structure patterns
- Writing style characteristics
- Common phrases and transitions

INPUT INSTRUCTIONS:
- Provide a text file as a command line argument
- The text file should contain substantial content (at least a few paragraphs)
- More text generally produces better results
- The file should be plain text (.txt format)
- Ensure the file exists and is readable

USAGE:
python generator.py <input_file.txt>

Example:
python generator.py sample.txt

OUTPUT:
The script will generate and display 5 new sentences based on the input text.
Each sentence will be separated by a blank line for readability.

REQUIREMENTS:
- markovify library (install with: pip install markovify)
- Python 3.x
- A text file with sufficient content for training
"""

import markovify
import sys

# Read text from file
if len(sys.argv) != 2:
    sys.exit("Usage: python generator.py sample.txt")
with open(sys.argv[1]) as f:
    text = f.read()

# Train model
text_model = markovify.Text(text)

# Generate sentences
print()
for i in range(5):
    print(text_model.make_sentence())
    print()
