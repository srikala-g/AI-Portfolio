"""
Knights and Knaves Logic Puzzle Solver

This module implements a logical reasoning system to solve classic "Knights and Knaves" puzzles
using propositional logic. The puzzles involve characters who are either:
- Knights: Always tell the truth
- Knaves: Always lie

The module defines four different puzzle scenarios and uses model checking to determine
the identity (knight or knave) of each character based on their statements.

Puzzles included:
- Puzzle 0: A says "I am both a knight and a knave" (paradox)
- Puzzle 1: A says "We are both knaves", B says nothing
- Puzzle 2: A says "We are the same kind", B says "We are of different kinds"
- Puzzle 3: Complex scenario with A, B, and C making various statements

Usage:
    python puzzle.py

The program will solve all four puzzles and display the logical conclusions about
each character's identity (knight or knave).

Dependencies:
    - logic.py: Contains the logical operators and model_check function
"""

from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # A is either a knight or a knave
    Or(AKnight, AKnave),
    # A cannot be both a knight and a knave
    Not(And(AKnight, AKnave)),
    # If A is a knight, then the statement is true
    Implication(AKnight, And(AKnight, AKnave)),
    # If A is a knave, then the statement is false
    Implication(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),

    # A said "We are both knaves"
    Implication(AKnight, And(AKnave, BKnave)),   # If A is a knight, he tells the truth
    Implication(AKnave, Not(And(AKnave, BKnave)))  # If A is a knave, he lies
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."

same_kind = Or(And(AKnight, BKnight), And(AKnave, BKnave))
different_kind = Or(And(AKnight, BKnave), And(AKnave, BKnight))

knowledge2 = And(
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),

    # A says: "We are the same kind"
    Implication(AKnight, same_kind),
    Implication(AKnave, Not(same_kind)),

    # B says: "We are of different kinds"
    Implication(BKnight, different_kind),
    Implication(BKnave, Not(different_kind))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."

A_said_knave = Symbol("A said 'I am a knave'")

knowledge3 = And(
    # Base: everyone is either knight or knave, not both
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave),
    Not(And(CKnight, CKnave)),

    # A cannot logically say "I am a knave"
    # Because it causes a paradox regardless of A's identity
    Not(A_said_knave),

    # B says: "A said 'I am a knave'"
    Implication(BKnight, A_said_knave),
    Implication(BKnave, Not(A_said_knave)),

    # B says: "C is a knave"
    Implication(BKnight, CKnave),
    Implication(BKnave, Not(CKnave)),

    # C says: "A is a knight"
    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight))
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
