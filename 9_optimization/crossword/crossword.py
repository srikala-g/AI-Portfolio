"""
Crossword Puzzle Solver - Constraint Satisfaction Problem Implementation

This module implements a crossword puzzle solver using constraint satisfaction techniques.
It provides classes to represent crossword variables and the overall crossword structure.

Classes:
    Variable: Represents a word slot in the crossword (either across or down)
    Crossword: Main class that parses crossword structure and manages variables

Key Features:
    - Parses crossword structure from text files (using '_' for white cells)
    - Identifies all word slots (variables) in the crossword
    - Computes overlaps between intersecting words
    - Provides neighbor detection for constraint propagation

Usage Instructions:
    1. Create a structure file (e.g., 'structure.txt') with the crossword layout:
       - Use '_' for white cells where letters go
       - Use any other character (or space) for black cells
       - Example structure:
         _ _ _ _ _
         _ _ _ _ _
         _ _ _ _ _
         _ _ _ _ _
         _ _ _ _ _

    2. Create a words file (e.g., 'words.txt') with one word per line:
       - Example:
         CAT
         DOG
         BIRD
         FISH

    3. Initialize the crossword:
       crossword = Crossword('structure.txt', 'words.txt')

    4. Access variables and overlaps:
       - crossword.variables: Set of all word slots
       - crossword.overlaps: Dictionary of variable overlaps
       - crossword.neighbors(var): Get overlapping variables for a given variable

Example:
    # Create crossword from files
    puzzle = Crossword('my_crossword.txt', 'word_list.txt')
    
    # Get all variables
    for var in puzzle.variables:
        print(f"Variable: {var}")
    
    # Check overlaps between variables
    for v1, v2 in puzzle.overlaps:
        overlap = puzzle.overlaps[v1, v2]
        if overlap:
            print(f"{v1} overlaps {v2} at positions {overlap}")

File Format Requirements:
    - Structure file: Text file with '_' for white cells, other chars for black cells
    - Words file: Text file with one word per line (case insensitive)
"""

class Variable():

    ACROSS = "across"
    DOWN = "down"

    def __init__(self, i, j, direction, length):
        """Create a new variable with starting point, direction, and length."""
        self.i = i
        self.j = j
        self.direction = direction
        self.length = length
        self.cells = []
        for k in range(self.length):
            self.cells.append(
                (self.i + (k if self.direction == Variable.DOWN else 0),
                 self.j + (k if self.direction == Variable.ACROSS else 0))
            )

    def __hash__(self):
        return hash((self.i, self.j, self.direction, self.length))

    def __eq__(self, other):
        return (
            (self.i == other.i) and
            (self.j == other.j) and
            (self.direction == other.direction) and
            (self.length == other.length)
        )

    def __str__(self):
        return f"({self.i}, {self.j}) {self.direction} : {self.length}"

    def __repr__(self):
        direction = repr(self.direction)
        return f"Variable({self.i}, {self.j}, {direction}, {self.length})"


class Crossword():

    def __init__(self, structure_file, words_file):

        # Determine structure of crossword
        with open(structure_file) as f:
            contents = f.read().splitlines()
            self.height = len(contents)
            self.width = max(len(line) for line in contents)

            self.structure = []
            for i in range(self.height):
                row = []
                for j in range(self.width):
                    if j >= len(contents[i]):
                        row.append(False)
                    elif contents[i][j] == "_":
                        row.append(True)
                    else:
                        row.append(False)
                self.structure.append(row)

        # Save vocabulary list
        with open(words_file) as f:
            self.words = set(f.read().upper().splitlines())

        # Determine variable set
        self.variables = set()
        for i in range(self.height):
            for j in range(self.width):

                # Vertical words
                starts_word = (
                    self.structure[i][j]
                    and (i == 0 or not self.structure[i - 1][j])
                )
                if starts_word:
                    length = 1
                    for k in range(i + 1, self.height):
                        if self.structure[k][j]:
                            length += 1
                        else:
                            break
                    if length > 1:
                        self.variables.add(Variable(
                            i=i, j=j,
                            direction=Variable.DOWN,
                            length=length
                        ))

                # Horizontal words
                starts_word = (
                    self.structure[i][j]
                    and (j == 0 or not self.structure[i][j - 1])
                )
                if starts_word:
                    length = 1
                    for k in range(j + 1, self.width):
                        if self.structure[i][k]:
                            length += 1
                        else:
                            break
                    if length > 1:
                        self.variables.add(Variable(
                            i=i, j=j,
                            direction=Variable.ACROSS,
                            length=length
                        ))

        # Compute overlaps for each word
        # For any pair of variables v1, v2, their overlap is either:
        #    None, if the two variables do not overlap; or
        #    (i, j), where v1's ith character overlaps v2's jth character
        self.overlaps = dict()
        for v1 in self.variables:
            for v2 in self.variables:
                if v1 == v2:
                    continue
                cells1 = v1.cells
                cells2 = v2.cells
                intersection = set(cells1).intersection(cells2)
                if not intersection:
                    self.overlaps[v1, v2] = None
                else:
                    intersection = intersection.pop()
                    self.overlaps[v1, v2] = (
                        cells1.index(intersection),
                        cells2.index(intersection)
                    )

    def neighbors(self, var):
        """Given a variable, return set of overlapping variables."""
        return set(
            v for v in self.variables
            if v != var and self.overlaps[v, var]
        )
