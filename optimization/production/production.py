"""
Production Optimization Problem Solver

This script solves a linear programming optimization problem for production planning.
It finds the optimal allocation of resources (x1 and x2) to maximize profit while
satisfying resource constraints.

Problem Formulation:
- Objective: Maximize profit = 50x₁ + 80x₂
- Constraint 1: 5x₁ + 2x₂ ≤ 20 (resource limitation)
- Constraint 2: -10x₁ - 12x₂ ≤ -90 (minimum production requirement)

Usage Instructions:
1. Ensure scipy is installed: pip install scipy
2. Run the script: python production.py
3. The script will output the optimal values for x1 and x2 (hours)
4. If no feasible solution exists, it will display "No solution"

Expected Output:
- X1: [optimal hours for resource 1]
- X2: [optimal hours for resource 2]

Note: This is a minimization problem converted to standard form for scipy.optimize.linprog
"""

import scipy.optimize

# Objective Function: 50x_1 + 80x_2
# Constraint 1: 5x_1 + 2x_2 <= 20
# Constraint 2: -10x_1 + -12x_2 <= -90

result = scipy.optimize.linprog(
    [50, 80],  # Cost function: 50x_1 + 80x_2
    A_ub=[[5, 2], [-10, -12]],  # Coefficients for inequalities
    b_ub=[20, -90],  # Constraints for inequalities: 20 and -90
)

if result.success:
    print(f"X1: {round(result.x[0], 2)} hours")
    print(f"X2: {round(result.x[1], 2)} hours")
else:
    print("No solution")
