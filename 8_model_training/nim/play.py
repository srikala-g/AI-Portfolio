"""
Nim Game AI Player

This script trains a Q-learning AI agent to play the classic Nim game and then allows
you to play against the trained AI. The AI learns optimal strategies through self-play
and becomes increasingly difficult to beat.

Summary:
    - Trains an AI agent using Q-learning with 10,000 self-play games
    - Launches an interactive game where you can play against the trained AI
    - The AI learns to play optimally and will typically win if it goes first

Usage Instructions:
    1. Run this script: python play.py
    2. The AI will train for a few moments (10,000 games)
    3. Once training is complete, you'll see the game board with piles
    4. Take turns choosing a pile and how many objects to remove
    5. The last player to remove an object wins
    6. Type 'quit' at any time to exit

Game Rules:
    - Players alternate turns
    - On your turn, choose a pile and remove 1 or more objects from it
    - You must remove at least 1 object
    - You can only remove objects from one pile per turn
    - The player who removes the last object wins

Tips:
    - The AI is trained to play optimally, so it's quite challenging
    - Try to force the AI into losing positions
    - Pay attention to the XOR sum of pile sizes for optimal play
"""

from nim import train, play

# Train the AI with 10,000 self-play games
ai = train(10000)

# Start playing against the trained AI
play(ai)
