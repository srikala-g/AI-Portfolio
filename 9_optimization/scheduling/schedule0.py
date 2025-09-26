"""
SCHEDULING PROBLEM SOLVER - NAIVE BACKTRACKING APPROACH

SUMMARY:
This module implements a constraint satisfaction problem (CSP) solver for scheduling
using a naive backtracking search algorithm. It solves the problem of assigning 
variables (A, B, C, D, E, F, G) to time slots (Monday, Tuesday, Wednesday) such that
no two variables connected by constraints are assigned to the same time slot.

The algorithm uses:
- Backtracking search without heuristics or inference
- Simple variable selection (first unassigned variable)
- Constraint checking to ensure no conflicts
- Three possible values: Monday, Tuesday, Wednesday

USAGE INSTRUCTIONS:
1. Run the script directly: python schedule0.py
2. The script will automatically solve the scheduling problem
3. Output will be a dictionary showing the assignment of variables to days
4. If no solution exists, the output will be None

EXAMPLE OUTPUT:
{'A': 'Monday', 'B': 'Tuesday', 'C': 'Wednesday', 'D': 'Monday', 
 'E': 'Wednesday', 'F': 'Monday', 'G': 'Tuesday'}

CONSTRAINTS:
The problem has 11 constraints connecting variables that cannot be scheduled
on the same day. The algorithm ensures no two connected variables share
the same time slot.

VARIABLES: A, B, C, D, E, F, G
VALUES: Monday, Tuesday, Wednesday
"""

VARIABLES = ["A", "B", "C", "D", "E", "F", "G"]
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


def backtrack(assignment):
    """Runs backtracking search to find an assignment."""

    # Check if assignment is complete
    if len(assignment) == len(VARIABLES):
        return assignment

    # Try a new variable
    var = select_unassigned_variable(assignment)
    for value in ["Monday", "Tuesday", "Wednesday"]:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        if consistent(new_assignment):
            result = backtrack(new_assignment)
            if result is not None:
                return result
    return None


def select_unassigned_variable(assignment):
    """Chooses a variable not yet assigned, in order."""
    for variable in VARIABLES:
        if variable not in assignment:
            return variable
    return None


def consistent(assignment):
    """Checks to see if an assignment is consistent."""
    for (x, y) in CONSTRAINTS:

        # Only consider arcs where both are assigned
        if x not in assignment or y not in assignment:
            continue

        # If both have same value, then not consistent
        if assignment[x] == assignment[y]:
            return False

    # If nothing inconsistent, then assignment is consistent
    return True


solution = backtrack(dict())
print(solution)
