"""
Search Algorithm Utilities

This module provides fundamental data structures for implementing search algorithms,
specifically designed for graph traversal and pathfinding problems.

Classes:
    Node: Represents a node in a search tree with state, parent, and action information
    StackFrontier: Implements a LIFO (Last In, First Out) frontier using a stack
    QueueFrontier: Implements a FIFO (First In, First Out) frontier using a queue

Usage:
    # Create a node
    node = Node(state="initial_state", parent=None, action=None)
    
    # Use StackFrontier for depth-first search
    stack = StackFrontier()
    stack.add(node)
    if not stack.empty():
        next_node = stack.remove()
    
    # Use QueueFrontier for breadth-first search  
    queue = QueueFrontier()
    queue.add(node)
    if not queue.empty():
        next_node = queue.remove()
    
    # Check if frontier contains a specific state
    if stack.contains_state("target_state"):
        print("State found in frontier")

Note: This module is typically used in conjunction with search algorithms like
BFS, DFS, A*, etc. The choice between StackFrontier and QueueFrontier determines
the search strategy (depth-first vs breadth-first).
"""

class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node
