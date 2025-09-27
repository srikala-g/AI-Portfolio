"""
Scheduling Problem Solver using Constraint Programming

This script solves a scheduling problem where 7 entities (A, B, C, D, E, F, G) 
need to be assigned to 3 time slots (Monday, Tuesday, Wednesday) with the 
constraint that certain pairs of entities cannot be scheduled on the same day.

The problem uses constraint programming to find all valid solutions where:
- Each entity is assigned exactly one day
- Constrained pairs cannot be on the same day
- All possible valid schedules are generated

Usage Instructions:
1. Install required dependency: pip install python-constraint
2. Run the script: python schedule1.py
3. The script will output all valid scheduling solutions
4. Each solution shows which day each entity (A-G) is assigned to

Example Output:
{'A': 'Monday', 'B': 'Tuesday', 'C': 'Wednesday', 'D': 'Monday', 'E': 'Wednesday', 'F': 'Monday', 'G': 'Tuesday'}

Constraints:
- A cannot be with B or C
- B cannot be with A, C, D, or E  
- C cannot be with A, B, E, or F
- D cannot be with B or E
- E cannot be with B, C, D, F, or G
- F cannot be with C, E, or G
- G cannot be with E or F
"""

from constraint import *

problem = Problem()

# Add variables
problem.addVariables(
    ["A", "B", "C", "D", "E", "F", "G"],
    ["Monday", "Tuesday", "Wednesday"]
)

# Add constraints
CONSTRAINTS = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "E"),
    ("C", "F"),
    ("D", "E"),
    ("E", "F"),
    ("E", "G"),
    ("F", "G")
]
for x, y in CONSTRAINTS:
    problem.addConstraint(lambda x, y: x != y, (x, y))

# Solve problem
for solution in problem.getSolutions():
    print(solution)
